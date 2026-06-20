//! Factorized sparse approximate inverse (FSAI) preconditioner.
//!
//! FSAI builds an *explicit* approximation to `A^{-1}` for a Hermitian
//! positive-definite `A`: a sparse lower-triangular factor `G ~= L^{-1}` (with
//! `A = L L^H`) such that `M^{-1} = G^H G`. Applying the preconditioner is then
//! two sparse matrix-vector products — no triangular solves — which, like the
//! [polynomial](crate::poly) preconditioner, parallelises far better than the
//! sequential sweeps of ILU/IC. Building `G` is also naturally parallel: each
//! row is an independent small dense SPD solve.
//!
//! # When to use it
//!
//! Reach for FSAI on SPD systems when you want an approximate inverse whose
//! *apply* is a matvec — many cores, accelerators, or repeated applies where the
//! triangular-solve dependency chain of [`crate::Ic0`] is the bottleneck. The
//! trade-off is a heavier build (one dense factorisation per row) and a quality
//! that hinges on the chosen pattern.
//!
//! # Pattern
//!
//! The accuracy/cost knob is the prescribed lower-triangular pattern of `G`:
//!
//! - [`FsaiPattern::LowerOfA`] — the lower triangle of `A` (cheapest).
//! - [`FsaiPattern::LowerOfPower`] — the lower triangle of `pattern(A^power)`,
//!   denser and more accurate for larger `power`.
//!
//! Adaptive pattern selection is left as future work.
//!
//! # Storage
//!
//! `G` is stored CSC. Apply reads it through a
//! [`faer::sparse::SparseColMatRef`] and its adjoint view, with
//! a single work column block drawn from the caller's [`MemStack`] — no heap
//! allocation.

use core::fmt::Debug;

use dyn_stack::{MemStack, StackReq};
use faer::matrix_free::{BiLinOp, BiPrecond, LinOp, Precond};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::{MatMut, MatRef, Par};
use faer_traits::{ComplexField, Index};

mod apply;
mod build;

/// Prescribed lower-triangular pattern for the FSAI factor `G`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FsaiPattern {
    /// Lower triangle of `A`.
    #[default]
    LowerOfA,
    /// Lower triangle of `pattern(A^power)` (`power >= 1`; `1` equals
    /// [`FsaiPattern::LowerOfA`]).
    LowerOfPower { power: usize },
}

/// Error returned by FSAI construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FsaiError {
    /// The source matrix was not square.
    NonSquareMatrix { nrows: usize, ncols: usize },
    /// A `LowerOfPower` power of zero was requested.
    InvalidPower,
    /// The local block at row `row` was not positive definite.
    NotPositiveDefinite { row: usize },
}

impl core::fmt::Display for FsaiError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonSquareMatrix { nrows, ncols } => {
                write!(f, "matrix must be square but is {nrows}x{ncols}")
            }
            Self::InvalidPower => f.write_str("FSAI pattern power must be at least 1"),
            Self::NotPositiveDefinite { row } => {
                write!(f, "local block at row {row} is not positive definite")
            }
        }
    }
}

impl core::error::Error for FsaiError {}

/// Factorized sparse approximate inverse: `M^{-1} = G^H G`.
///
/// Stores the lower-triangular factor `G` in CSC. Apply is two sparse matvecs
/// with no triangular solves and no heap allocation. See the
/// [module documentation](self) for guidance.
#[derive(Debug, Clone)]
pub struct Fsai<I, T> {
    pub(crate) dim: usize,
    pub(crate) g_col_ptr: Vec<I>,
    pub(crate) g_row_idx: Vec<I>,
    pub(crate) g_values: Vec<T>,
}

impl<I, T> Fsai<I, T> {
    /// Dimension `n` of the preconditioner.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl<I: Index, T: ComplexField> Fsai<I, T> {
    /// View over the lower-triangular factor `G`.
    #[inline]
    pub(crate) fn g_view(&self) -> SparseColMatRef<'_, I, T> {
        let symbolic = unsafe {
            SymbolicSparseColMatRef::<'_, I>::new_unchecked(
                self.dim,
                self.dim,
                &self.g_col_ptr,
                None,
                &self.g_row_idx,
            )
        };
        SparseColMatRef::new(symbolic, &self.g_values)
    }
}

impl<I, T> LinOp<T> for Fsai<I, T>
where
    I: Index,
    T: ComplexField + Debug + Sync,
{
    fn apply_scratch(&self, rhs_ncols: usize, _par: Par) -> StackReq {
        apply::scratch(self, rhs_ncols)
    }

    fn nrows(&self) -> usize {
        self.dim
    }

    fn ncols(&self) -> usize {
        self.dim
    }

    fn apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
        apply::apply_out(self, out, rhs, false, par, stack);
    }

    fn conj_apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
        apply::apply_out(self, out, rhs, true, par, stack);
    }
}

impl<I, T> Precond<T> for Fsai<I, T>
where
    I: Index,
    T: ComplexField + Debug + Sync,
{
    fn apply_in_place_scratch(&self, rhs_ncols: usize, _par: Par) -> StackReq {
        apply::scratch(self, rhs_ncols)
    }

    fn apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
        apply::apply_inplace(self, rhs, false, par, stack);
    }

    fn conj_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
        apply::apply_inplace(self, rhs, true, par, stack);
    }
}

impl<I, T> BiLinOp<T> for Fsai<I, T>
where
    I: Index,
    T: ComplexField + Debug + Sync,
{
    fn transpose_apply_scratch(&self, rhs_ncols: usize, _par: Par) -> StackReq {
        apply::scratch(self, rhs_ncols)
    }

    fn transpose_apply(
        &self,
        out: MatMut<'_, T>,
        rhs: MatRef<'_, T>,
        par: Par,
        stack: &mut MemStack,
    ) {
        // M^{-T} = conj(M^{-1}).
        apply::apply_out(self, out, rhs, true, par, stack);
    }

    fn adjoint_apply(
        &self,
        out: MatMut<'_, T>,
        rhs: MatRef<'_, T>,
        par: Par,
        stack: &mut MemStack,
    ) {
        // M^{-H} = M^{-1} for Hermitian M.
        apply::apply_out(self, out, rhs, false, par, stack);
    }
}

impl<I, T> BiPrecond<T> for Fsai<I, T>
where
    I: Index,
    T: ComplexField + Debug + Sync,
{
    fn transpose_apply_in_place_scratch(&self, rhs_ncols: usize, _par: Par) -> StackReq {
        apply::scratch(self, rhs_ncols)
    }

    fn transpose_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
        apply::apply_inplace(self, rhs, true, par, stack);
    }

    fn adjoint_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
        apply::apply_inplace(self, rhs, false, par, stack);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::MaybeUninit;
    use faer::sparse::{SparseColMat, Triplet};
    use faer::{Mat, MatRef, mat};

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

    fn to_dense(a: &SparseColMat<usize, f64>) -> Mat<f64> {
        let n = a.nrows();
        let mut out = Mat::<f64>::zeros(n, a.ncols());
        let a_ref = a.as_ref();
        for j in 0..a.ncols() {
            let rows = a_ref.symbolic().row_idx_of_col_raw(j);
            let vals = a_ref.val_of_col(j);
            for (r, v) in rows.iter().zip(vals.iter()) {
                *out.as_mut().get_mut(*r, j) = *v;
            }
        }
        out
    }

    fn tridiagonal(n: usize, diag: f64, off: f64) -> SparseColMat<usize, f64> {
        let mut triplets = Vec::new();
        for i in 0..n {
            triplets.push(Triplet::new(i, i, diag));
            if i > 0 {
                triplets.push(Triplet::new(i, i - 1, off));
                triplets.push(Triplet::new(i - 1, i, off));
            }
        }
        SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap()
    }

    fn laplacian_2d(grid: usize) -> SparseColMat<usize, f64> {
        let n = grid * grid;
        let mut triplets = Vec::new();
        for gy in 0..grid {
            for gx in 0..grid {
                let idx = gy * grid + gx;
                triplets.push(Triplet::new(idx, idx, 4.0));
                if gx > 0 {
                    triplets.push(Triplet::new(idx, idx - 1, -1.0));
                }
                if gx + 1 < grid {
                    triplets.push(Triplet::new(idx, idx + 1, -1.0));
                }
                if gy > 0 {
                    triplets.push(Triplet::new(idx, idx - grid, -1.0));
                }
                if gy + 1 < grid {
                    triplets.push(Triplet::new(idx, idx + grid, -1.0));
                }
            }
        }
        SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap()
    }

    fn apply_inplace(pc: &Fsai<usize, f64>, rhs: &mut Mat<f64>) {
        with_stack(pc.apply_in_place_scratch(rhs.ncols(), Par::Seq), |stack| {
            pc.apply_in_place(rhs.as_mut(), Par::Seq, stack);
        });
    }

    fn residual_ratio(a: &SparseColMat<usize, f64>, pc: &Fsai<usize, f64>, b: &Mat<f64>) -> f64 {
        let a_dense = to_dense(a);
        let mut x = b.clone();
        apply_inplace(pc, &mut x);
        let residual = &a_dense * &x - b;
        let b_norm: f64 = b.as_ref().col(0).iter().map(|v| v * v).sum::<f64>().sqrt();
        let r_norm: f64 = residual
            .as_ref()
            .col(0)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        r_norm / b_norm
    }

    #[test]
    fn diagonal_is_exact_inverse() {
        let mut triplets = Vec::new();
        for (i, &v) in [2.0, 4.0, 8.0].iter().enumerate() {
            triplets.push(Triplet::new(i, i, v));
        }
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(3, 3, &triplets).unwrap();
        let pc = Fsai::try_new(a.as_ref(), FsaiPattern::LowerOfA).unwrap();
        let mut x = mat![[2.0_f64], [8.0], [16.0]];
        apply_inplace(&pc, &mut x);
        let expected = mat![[1.0_f64], [2.0], [2.0]];
        assert_close(x.as_ref(), expected.as_ref(), 1e-12);
    }

    #[test]
    fn reduces_residual_on_laplacian() {
        let a = laplacian_2d(8);
        let n = a.nrows();
        let pc = Fsai::try_new(a.as_ref(), FsaiPattern::LowerOfA).unwrap();
        let b = Mat::<f64>::from_fn(n, 1, |i, _| (i % 7) as f64 - 3.0);
        let ratio = residual_ratio(&a, &pc, &b);
        assert!(ratio < 1.0, "FSAI should reduce the residual: {ratio}");
    }

    #[test]
    fn denser_pattern_improves_accuracy() {
        let a = laplacian_2d(8);
        let n = a.nrows();
        let b = Mat::<f64>::from_fn(n, 1, |i, _| (i % 7) as f64 - 3.0);
        let p1 = Fsai::try_new(a.as_ref(), FsaiPattern::LowerOfA).unwrap();
        let p2 = Fsai::try_new(a.as_ref(), FsaiPattern::LowerOfPower { power: 2 }).unwrap();
        let r1 = residual_ratio(&a, &p1, &b);
        let r2 = residual_ratio(&a, &p2, &b);
        assert!(r2 < r1, "denser FSAI pattern should help: {r2} !< {r1}");
    }

    #[test]
    fn symmetric_two_matvec_matches_dense() {
        let a = tridiagonal(6, 4.0, -1.0);
        let pc = Fsai::try_new(a.as_ref(), FsaiPattern::LowerOfA).unwrap();
        // Forward apply == transpose apply for the real symmetric case.
        let rhs = mat![[1.0_f64], [-2.0], [3.0], [0.5], [-1.0], [2.0]];
        let mut fwd = rhs.clone();
        apply_inplace(&pc, &mut fwd);
        let mut tr = rhs.clone();
        with_stack(pc.transpose_apply_in_place_scratch(1, Par::Seq), |stack| {
            pc.transpose_apply_in_place(tr.as_mut(), Par::Seq, stack);
        });
        assert_close(fwd.as_ref(), tr.as_ref(), 1e-12);
    }

    #[test]
    fn rejects_non_square() {
        let mut triplets = Vec::new();
        for i in 0..3 {
            triplets.push(Triplet::new(i, i, 1.0));
        }
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(3, 4, &triplets).unwrap();
        assert_eq!(
            Fsai::try_new(a.as_ref(), FsaiPattern::LowerOfA).unwrap_err(),
            FsaiError::NonSquareMatrix { nrows: 3, ncols: 4 }
        );
    }
}
