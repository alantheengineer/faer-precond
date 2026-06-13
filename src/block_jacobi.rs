//! Block-Jacobi preconditioner.
//!
//! `M = blkdiag(A_1, ..., A_p)` where each `A_k` is the dense diagonal block
//! `A[off_k..off_{k+1}, off_k..off_{k+1}]` of the source matrix. Each block is
//! factored with partial-pivoted LU once at construction; subsequent applies
//! are pure dense triangular solves and contain no heap allocation.

use core::fmt::Debug;

use dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::{
    Conj, MatMut, MatRef, Par,
    linalg::lu::partial_pivoting::{factor as plu_factor, solve as plu_solve},
    matrix_free::{BiLinOp, BiPrecond, LinOp, Precond},
    perm::PermRef,
    prelude::ReborrowMut,
};
use faer_traits::ComplexField;
use faer_traits::math_utils::{abs2, copy, zero};

/// Error produced when constructing a [`BlockJacobiPrecond`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockJacobiError {
    /// The source matrix was not square.
    NonSquareMatrix { nrows: usize, ncols: usize },
    /// Less than two offsets were supplied (need at least one block).
    EmptyBlocks,
    /// The first offset must be `0`.
    BlockOffsetsMustStartAtZero { first: usize },
    /// The last offset must equal the matrix dimension.
    BlockOffsetsMustEndAtDim { last: usize, dim: usize },
    /// Offsets must be strictly increasing — blocks of size 0 are rejected.
    BlockOffsetsNotStrictlyIncreasing {
        index: usize,
        prev: usize,
        curr: usize,
    },
    /// One of the diagonal blocks was singular (numerically rank-deficient).
    SingularBlock { block_index: usize },
}

impl core::fmt::Display for BlockJacobiError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonSquareMatrix { nrows, ncols } => {
                write!(f, "matrix must be square but is {nrows}x{ncols}")
            }
            Self::EmptyBlocks => f.write_str("at least one block is required"),
            Self::BlockOffsetsMustStartAtZero { first } => {
                write!(f, "block offsets must start at 0 but start at {first}")
            }
            Self::BlockOffsetsMustEndAtDim { last, dim } => {
                write!(
                    f,
                    "block offsets must end at the matrix dimension {dim} but end at {last}"
                )
            }
            Self::BlockOffsetsNotStrictlyIncreasing { index, prev, curr } => {
                write!(
                    f,
                    "block offsets must be strictly increasing: offset[{index}]={curr} <= offset[{}]={prev}",
                    index - 1
                )
            }
            Self::SingularBlock { block_index } => {
                write!(f, "diagonal block {block_index} is singular")
            }
        }
    }
}

impl core::error::Error for BlockJacobiError {}

/// Block-Jacobi preconditioner: `M^{-1} y = blkdiag(A_k^{-1}) y`.
///
/// The factorisation is computed once at construction. `apply`, `apply_in_place`,
/// and the transpose/adjoint variants perform only dense triangular solves and
/// permutation applications — they require no heap allocation and all scratch
/// flows through the [`MemStack`] provided by faer's trait interface.
///
/// # Storage
///
/// All `p` LU factors are packed contiguously in column-major order into a
/// single `Vec<T>`, and the partial-pivoting permutations are packed into two
/// `Vec<usize>` of total length `n`. This keeps the working set cache-friendly
/// during `apply`.
#[derive(Debug, Clone)]
pub struct BlockJacobiPrecond<T> {
    n: usize,
    block_offsets: Vec<usize>,
    factor_offsets: Vec<usize>,
    factors: Vec<T>,
    perm_fwd: Vec<usize>,
    perm_inv: Vec<usize>,
    max_block_size: usize,
}

impl<T> BlockJacobiPrecond<T> {
    /// Dimension `n` of the preconditioner (sum of block sizes).
    #[inline]
    pub fn dim(&self) -> usize {
        self.n
    }

    /// `true` if the preconditioner has dimension zero.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Number of diagonal blocks.
    #[inline]
    pub fn block_count(&self) -> usize {
        self.block_offsets.len().saturating_sub(1)
    }

    /// Slice of length `block_count() + 1` giving the row/column offsets of each block.
    #[inline]
    pub fn block_offsets(&self) -> &[usize] {
        &self.block_offsets
    }

    /// Size of the largest block — useful for sizing external scratch buffers.
    #[inline]
    pub fn max_block_size(&self) -> usize {
        self.max_block_size
    }
}

impl<T: ComplexField> BlockJacobiPrecond<T> {
    /// Build a block-Jacobi preconditioner from `a` and a block partition.
    ///
    /// `block_offsets` must satisfy `block_offsets[0] == 0`, the values must
    /// be strictly increasing, and `block_offsets[block_offsets.len() - 1]`
    /// must equal `a.nrows()`.
    ///
    /// # Errors
    ///
    /// Returns [`BlockJacobiError::NonSquareMatrix`] if `a` is not square,
    /// validation errors for ill-formed `block_offsets`, and
    /// [`BlockJacobiError::SingularBlock`] if any diagonal block is rank-deficient.
    pub fn try_new(a: MatRef<'_, T>, block_offsets: &[usize]) -> Result<Self, BlockJacobiError> {
        if a.nrows() != a.ncols() {
            return Err(BlockJacobiError::NonSquareMatrix {
                nrows: a.nrows(),
                ncols: a.ncols(),
            });
        }
        let n = a.nrows();

        if block_offsets.len() < 2 {
            return Err(BlockJacobiError::EmptyBlocks);
        }
        if block_offsets[0] != 0 {
            return Err(BlockJacobiError::BlockOffsetsMustStartAtZero {
                first: block_offsets[0],
            });
        }
        if *block_offsets.last().unwrap() != n {
            return Err(BlockJacobiError::BlockOffsetsMustEndAtDim {
                last: *block_offsets.last().unwrap(),
                dim: n,
            });
        }
        for window in block_offsets.windows(2).enumerate() {
            let (i, w) = window;
            if w[1] <= w[0] {
                return Err(BlockJacobiError::BlockOffsetsNotStrictlyIncreasing {
                    index: i + 1,
                    prev: w[0],
                    curr: w[1],
                });
            }
        }

        let nblocks = block_offsets.len() - 1;

        // Compute factor offsets and total factor storage.
        let mut factor_offsets = Vec::with_capacity(nblocks + 1);
        factor_offsets.push(0);
        let mut total_vals = 0usize;
        let mut max_block_size = 0usize;
        for i in 0..nblocks {
            let size = block_offsets[i + 1] - block_offsets[i];
            max_block_size = max_block_size.max(size);
            total_vals += size * size;
            factor_offsets.push(total_vals);
        }

        // Allocate packed buffers. All allocations live for the lifetime of the
        // preconditioner; no allocation happens after construction.
        let mut factors: Vec<T> = (0..total_vals).map(|_| zero::<T>()).collect();
        let mut perm_fwd: Vec<usize> = vec![0; n];
        let mut perm_inv: Vec<usize> = vec![0; n];

        // One-shot factorisation scratch sized for the largest block.
        let factor_scratch = plu_factor::lu_in_place_scratch::<usize, T>(
            max_block_size,
            max_block_size,
            Par::Seq,
            Default::default(),
        );
        let mut factor_buf = MemBuffer::new(factor_scratch);

        for k in 0..nblocks {
            let start = block_offsets[k];
            let size = block_offsets[k + 1] - start;

            // Copy block into the packed factor buffer (column-major).
            let factor_range = factor_offsets[k]..factor_offsets[k + 1];
            let block_slice = &mut factors[factor_range];
            let mut block = MatMut::<T>::from_column_major_slice_mut(block_slice, size, size);
            for j in 0..size {
                for i in 0..size {
                    *block.rb_mut().get_mut(i, j) = copy(a.get(start + i, start + j));
                }
            }

            // Factor in place. The permutation is local to the block.
            let perm_slice_fwd = &mut perm_fwd[start..start + size];
            let perm_slice_inv = &mut perm_inv[start..start + size];
            let _ = plu_factor::lu_in_place::<usize, T>(
                block.rb_mut(),
                perm_slice_fwd,
                perm_slice_inv,
                Par::Seq,
                MemStack::new(&mut factor_buf),
                Default::default(),
            );

            // Detect singular blocks: any zero on the diagonal of U.
            let factor_view = MatRef::<T>::from_column_major_slice(
                &factors[factor_offsets[k]..factor_offsets[k + 1]],
                size,
                size,
            );
            for i in 0..size {
                if abs2(factor_view.get(i, i)) == zero::<T::Real>() {
                    return Err(BlockJacobiError::SingularBlock { block_index: k });
                }
            }
        }

        Ok(Self {
            n,
            block_offsets: block_offsets.to_vec(),
            factor_offsets,
            factors,
            perm_fwd,
            perm_inv,
            max_block_size,
        })
    }

    /// Worst-case scratch required to apply this preconditioner.
    #[inline]
    fn solve_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        plu_solve::solve_in_place_scratch::<usize, T>(self.max_block_size, rhs_ncols, par)
    }

    #[inline]
    fn apply_blocks(
        &self,
        mut rhs: MatMut<'_, T>,
        conj: Conj,
        transpose: bool,
        par: Par,
        stack: &mut MemStack,
    ) {
        assert_eq!(
            rhs.nrows(),
            self.n,
            "rhs row count must match preconditioner dimension"
        );

        let nblocks = self.block_count();
        for k in 0..nblocks {
            let start = self.block_offsets[k];
            let size = self.block_offsets[k + 1] - start;

            let factor_slice = &self.factors[self.factor_offsets[k]..self.factor_offsets[k + 1]];
            let lu = MatRef::<T>::from_column_major_slice(factor_slice, size, size);

            let perm = unsafe {
                PermRef::<'_, usize>::new_unchecked(
                    &self.perm_fwd[start..start + size],
                    &self.perm_inv[start..start + size],
                    size,
                )
            };

            let rhs_block = rhs.rb_mut().subrows_mut(start, size);
            if transpose {
                plu_solve::solve_transpose_in_place_with_conj(
                    lu, lu, perm, conj, rhs_block, par, stack,
                );
            } else {
                plu_solve::solve_in_place_with_conj(lu, lu, perm, conj, rhs_block, par, stack);
            }
        }
    }
}

impl<T> LinOp<T> for BlockJacobiPrecond<T>
where
    T: ComplexField + Debug + Sync,
{
    fn apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        self.solve_scratch(rhs_ncols, par)
    }

    fn nrows(&self) -> usize {
        self.n
    }

    fn ncols(&self) -> usize {
        self.n
    }

    fn apply(&self, mut out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
        assert_eq!(
            out.nrows(),
            self.n,
            "out row count must match preconditioner dimension"
        );
        assert_eq!(
            rhs.nrows(),
            self.n,
            "rhs row count must match preconditioner dimension"
        );
        assert_eq!(
            out.ncols(),
            rhs.ncols(),
            "out and rhs must have the same number of columns"
        );
        out.copy_from(rhs);
        self.apply_blocks(out, Conj::No, false, par, stack);
    }

    fn conj_apply(
        &self,
        mut out: MatMut<'_, T>,
        rhs: MatRef<'_, T>,
        par: Par,
        stack: &mut MemStack,
    ) {
        assert_eq!(
            out.nrows(),
            self.n,
            "out row count must match preconditioner dimension"
        );
        assert_eq!(
            rhs.nrows(),
            self.n,
            "rhs row count must match preconditioner dimension"
        );
        assert_eq!(
            out.ncols(),
            rhs.ncols(),
            "out and rhs must have the same number of columns"
        );
        out.copy_from(rhs);
        self.apply_blocks(out, Conj::Yes, false, par, stack);
    }
}

impl<T> Precond<T> for BlockJacobiPrecond<T>
where
    T: ComplexField + Debug + Sync,
{
    fn apply_in_place_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        self.solve_scratch(rhs_ncols, par)
    }

    fn apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
        self.apply_blocks(rhs, Conj::No, false, par, stack);
    }

    fn conj_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
        self.apply_blocks(rhs, Conj::Yes, false, par, stack);
    }
}

impl<T> BiLinOp<T> for BlockJacobiPrecond<T>
where
    T: ComplexField + Debug + Sync,
{
    fn transpose_apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        self.solve_scratch(rhs_ncols, par)
    }

    fn transpose_apply(
        &self,
        mut out: MatMut<'_, T>,
        rhs: MatRef<'_, T>,
        par: Par,
        stack: &mut MemStack,
    ) {
        assert_eq!(
            out.nrows(),
            self.n,
            "out row count must match preconditioner dimension"
        );
        assert_eq!(
            rhs.nrows(),
            self.n,
            "rhs row count must match preconditioner dimension"
        );
        assert_eq!(
            out.ncols(),
            rhs.ncols(),
            "out and rhs must have the same number of columns"
        );
        out.copy_from(rhs);
        self.apply_blocks(out, Conj::No, true, par, stack);
    }

    fn adjoint_apply(
        &self,
        mut out: MatMut<'_, T>,
        rhs: MatRef<'_, T>,
        par: Par,
        stack: &mut MemStack,
    ) {
        assert_eq!(
            out.nrows(),
            self.n,
            "out row count must match preconditioner dimension"
        );
        assert_eq!(
            rhs.nrows(),
            self.n,
            "rhs row count must match preconditioner dimension"
        );
        assert_eq!(
            out.ncols(),
            rhs.ncols(),
            "out and rhs must have the same number of columns"
        );
        out.copy_from(rhs);
        self.apply_blocks(out, Conj::Yes, true, par, stack);
    }
}

impl<T> BiPrecond<T> for BlockJacobiPrecond<T>
where
    T: ComplexField + Debug + Sync,
{
    fn transpose_apply_in_place_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        self.solve_scratch(rhs_ncols, par)
    }

    fn transpose_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
        self.apply_blocks(rhs, Conj::No, true, par, stack);
    }

    fn adjoint_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
        self.apply_blocks(rhs, Conj::Yes, true, par, stack);
    }
}

#[cfg(test)]
mod tests {
    use core::mem::MaybeUninit;

    use super::*;
    use faer::{
        Mat, MatRef, mat,
        matrix_free::{BiLinOp, LinOp, Precond},
    };

    fn with_stack(req: StackReq, f: impl FnOnce(&mut MemStack)) {
        let nbytes = req.unaligned_bytes_required().max(1);
        let mut buf = vec![MaybeUninit::<u8>::uninit(); nbytes].into_boxed_slice();
        f(MemStack::new(&mut buf));
    }

    fn assert_close(lhs: MatRef<'_, f64>, rhs: MatRef<'_, f64>, tol: f64) {
        assert_eq!(lhs.nrows(), rhs.nrows());
        assert_eq!(lhs.ncols(), rhs.ncols());
        for j in 0..lhs.ncols() {
            for i in 0..lhs.nrows() {
                let diff = (*lhs.get(i, j) - *rhs.get(i, j)).abs();
                assert!(
                    diff <= tol,
                    "mismatch at ({i}, {j}): lhs={}, rhs={}, diff={diff}",
                    *lhs.get(i, j),
                    *rhs.get(i, j),
                );
            }
        }
    }

    /// 5x5 block-diagonal with a 2x2 block and a 3x3 block (plus off-diagonal
    /// noise that the preconditioner should ignore).
    fn test_matrix() -> Mat<f64> {
        mat![
            [4.0, 1.0, 7.0, 9.0, 0.0],
            [2.0, 3.0, 0.0, 0.0, 8.0],
            [9.0, 5.0, 6.0, 1.0, 2.0],
            [1.0, 1.0, 3.0, 5.0, 1.0],
            [3.0, 0.0, 2.0, 1.0, 4.0],
        ]
    }

    #[test]
    fn builds_from_matrix() {
        let a = test_matrix();
        let pc = BlockJacobiPrecond::try_new(a.as_ref(), &[0, 2, 5]).unwrap();
        assert_eq!(pc.dim(), 5);
        assert_eq!(pc.block_count(), 2);
        assert_eq!(pc.max_block_size(), 3);
        assert_eq!(pc.block_offsets(), &[0, 2, 5]);
    }

    #[test]
    fn rejects_non_square() {
        let a = Mat::<f64>::from_fn(3, 4, |i, j| (i + j) as f64);
        let err = BlockJacobiPrecond::try_new(a.as_ref(), &[0, 3]).unwrap_err();
        assert_eq!(
            err,
            BlockJacobiError::NonSquareMatrix { nrows: 3, ncols: 4 }
        );
    }

    #[test]
    fn rejects_bad_offsets() {
        let a = Mat::<f64>::identity(4, 4);
        assert!(matches!(
            BlockJacobiPrecond::try_new(a.as_ref(), &[]).unwrap_err(),
            BlockJacobiError::EmptyBlocks
        ));
        assert!(matches!(
            BlockJacobiPrecond::try_new(a.as_ref(), &[1, 4]).unwrap_err(),
            BlockJacobiError::BlockOffsetsMustStartAtZero { first: 1 }
        ));
        assert!(matches!(
            BlockJacobiPrecond::try_new(a.as_ref(), &[0, 3]).unwrap_err(),
            BlockJacobiError::BlockOffsetsMustEndAtDim { last: 3, dim: 4 }
        ));
        assert!(matches!(
            BlockJacobiPrecond::try_new(a.as_ref(), &[0, 2, 2, 4]).unwrap_err(),
            BlockJacobiError::BlockOffsetsNotStrictlyIncreasing { .. }
        ));
    }

    #[test]
    fn rejects_singular_block() {
        let mut a = Mat::<f64>::identity(4, 4);
        // Make the 2x2 trailing block singular.
        *a.as_mut().get_mut(2, 2) = 0.0;
        *a.as_mut().get_mut(2, 3) = 1.0;
        *a.as_mut().get_mut(3, 2) = 0.0;
        *a.as_mut().get_mut(3, 3) = 0.0;
        let err = BlockJacobiPrecond::try_new(a.as_ref(), &[0, 2, 4]).unwrap_err();
        assert_eq!(err, BlockJacobiError::SingularBlock { block_index: 1 });
    }

    #[test]
    fn apply_inverts_block_diagonal_part() {
        // Pure block-diagonal A so that M = A and applying M^{-1} to A x recovers x.
        let a = mat![
            [4.0, 1.0, 0.0, 0.0, 0.0],
            [2.0, 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 6.0, 1.0, 2.0],
            [0.0, 0.0, 3.0, 5.0, 1.0],
            [0.0, 0.0, 2.0, 1.0, 4.0],
        ];
        let pc = BlockJacobiPrecond::try_new(a.as_ref(), &[0, 2, 5]).unwrap();

        let x = mat![
            [1.0, -2.0],
            [2.0, 1.0],
            [3.0, 0.5],
            [-1.0, 2.0],
            [0.5, -1.0_f64],
        ];
        let b = &a * &x;

        let mut out = Mat::<f64>::zeros(5, 2);
        with_stack(pc.apply_scratch(b.ncols(), Par::Seq), |stack| {
            pc.apply(out.as_mut(), b.as_ref(), Par::Seq, stack);
        });
        assert_close(out.as_ref(), x.as_ref(), 1e-12);
    }

    #[test]
    fn apply_in_place_matches_apply() {
        let a = test_matrix();
        let pc = BlockJacobiPrecond::try_new(a.as_ref(), &[0, 2, 5]).unwrap();

        let rhs = mat![
            [1.0, 4.0],
            [2.0, 5.0],
            [3.0, 6.0],
            [4.0, 7.0],
            [5.0, 8.0_f64],
        ];

        let mut out = Mat::<f64>::zeros(5, 2);
        with_stack(pc.apply_scratch(rhs.ncols(), Par::Seq), |stack| {
            pc.apply(out.as_mut(), rhs.as_ref(), Par::Seq, stack);
        });

        let mut inplace = rhs.to_owned();
        with_stack(pc.apply_in_place_scratch(rhs.ncols(), Par::Seq), |stack| {
            pc.apply_in_place(inplace.as_mut(), Par::Seq, stack);
        });

        assert_close(out.as_ref(), inplace.as_ref(), 1e-12);
    }

    #[test]
    fn transpose_matches_block_transpose_solve() {
        // For a block-diagonal A, M^{-T} y solves A^T y per block.
        let a = mat![
            [4.0, 1.0, 0.0, 0.0, 0.0],
            [2.0, 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 6.0, 1.0, 2.0],
            [0.0, 0.0, 3.0, 5.0, 1.0],
            [0.0, 0.0, 2.0, 1.0, 4.0_f64],
        ];
        let pc = BlockJacobiPrecond::try_new(a.as_ref(), &[0, 2, 5]).unwrap();

        let x = mat![[1.0], [2.0], [3.0], [-1.0], [0.5_f64],];
        let b = a.transpose() * &x;

        let mut out = Mat::<f64>::zeros(5, 1);
        with_stack(pc.transpose_apply_scratch(b.ncols(), Par::Seq), |stack| {
            pc.transpose_apply(out.as_mut(), b.as_ref(), Par::Seq, stack);
        });
        assert_close(out.as_ref(), x.as_ref(), 1e-12);
    }

    #[test]
    fn adjoint_equals_transpose_for_real() {
        let a = test_matrix();
        let pc = BlockJacobiPrecond::try_new(a.as_ref(), &[0, 2, 5]).unwrap();

        let rhs = mat![[1.0], [-2.0], [3.0], [4.0], [-1.0_f64],];

        let mut out_t = Mat::<f64>::zeros(5, 1);
        with_stack(pc.transpose_apply_scratch(rhs.ncols(), Par::Seq), |stack| {
            pc.transpose_apply(out_t.as_mut(), rhs.as_ref(), Par::Seq, stack);
        });
        let mut out_h = Mat::<f64>::zeros(5, 1);
        with_stack(pc.transpose_apply_scratch(rhs.ncols(), Par::Seq), |stack| {
            pc.adjoint_apply(out_h.as_mut(), rhs.as_ref(), Par::Seq, stack);
        });
        assert_close(out_t.as_ref(), out_h.as_ref(), 1e-12);
    }

    #[test]
    fn single_block_matches_full_lu() {
        // One big block == applying A^{-1} via partial-pivoted LU.
        let a = mat![[4.0, 1.0, 2.0], [3.0, 5.0, 1.0], [1.0, 2.0, 6.0_f64],];
        let pc = BlockJacobiPrecond::try_new(a.as_ref(), &[0, 3]).unwrap();

        let x = mat![[1.0], [2.0], [3.0_f64],];
        let b = &a * &x;

        let mut out = Mat::<f64>::zeros(3, 1);
        with_stack(pc.apply_scratch(b.ncols(), Par::Seq), |stack| {
            pc.apply(out.as_mut(), b.as_ref(), Par::Seq, stack);
        });
        assert_close(out.as_ref(), x.as_ref(), 1e-12);
    }
}
