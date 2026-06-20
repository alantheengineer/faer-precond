//! Sparse approximate inverse (SPAI) preconditioner.
//!
//! SPAI builds an explicit sparse `M ~= A^{-1}` by minimising `||A M - I||_F`.
//! That objective separates by column, so `M` is assembled from independent
//! small least-squares problems — embarrassingly parallel to build. Applying the
//! preconditioner is then a single sparse matrix-vector product, with no
//! triangular solves; the transpose and adjoint are matvecs against the
//! transposed/adjoint views of `M`.
//!
//! Unlike [`crate::fsai::Fsai`], SPAI does **not** assume symmetry — it is the
//! nonsymmetric approximate-inverse, the matvec-only counterpart to
//! [`crate::Ilu0`] / [`crate::Ilutp`].
//!
//! # When to use it
//!
//! Reach for SPAI on nonsymmetric systems when matvecs are cheap and you want an
//! approximate inverse that applies as a matvec — parallel hardware, or settings
//! where the sequential triangular solve of an ILU does not scale. The build is
//! heavier than an ILU (a dense least-squares solve per column), so it pays off
//! when the same preconditioner is applied many times.
//!
//! # Pattern
//!
//! The accuracy/cost knob is the prescribed column pattern of `M`:
//!
//! - [`SpaiPattern::ColumnsOfA`] — each column of `M` takes the pattern of the
//!   matching column of `A` (cheapest).
//! - [`SpaiPattern::ColumnsOfPower`] — the pattern of `A^power`'s columns,
//!   denser and more accurate for larger `power`.
//!
//! Adaptive (dynamic) pattern growth — the Grote–Huckle algorithm — is left as
//! future work.
//!
//! # Storage
//!
//! `M` is stored CSC. Apply reads it through a
//! [`faer::sparse::SparseColMatRef`] (and its transpose/adjoint
//! views); the in-place path uses one work column block from the caller's
//! [`MemStack`], so no heap allocation occurs during apply.

use core::fmt::Debug;

use dyn_stack::{MemStack, StackReq};
use faer::matrix_free::{BiLinOp, BiPrecond, LinOp, Precond};
use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer::{MatMut, MatRef, Par};
use faer_traits::{ComplexField, Index};

mod apply;
mod build;

/// Prescribed column pattern for the SPAI factor `M`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpaiPattern {
    /// Each column of `M` takes the pattern of the matching column of `A`.
    #[default]
    ColumnsOfA,
    /// Columns take the pattern of `A^power` (`power >= 1`; `1` equals
    /// [`SpaiPattern::ColumnsOfA`]).
    ColumnsOfPower { power: usize },
}

/// Error returned by SPAI construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpaiError {
    /// The source matrix was not square.
    NonSquareMatrix { nrows: usize, ncols: usize },
    /// A `ColumnsOfPower` power of zero was requested.
    InvalidPower,
}

impl core::fmt::Display for SpaiError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonSquareMatrix { nrows, ncols } => {
                write!(f, "matrix must be square but is {nrows}x{ncols}")
            }
            Self::InvalidPower => f.write_str("SPAI pattern power must be at least 1"),
        }
    }
}

impl core::error::Error for SpaiError {}

/// Sparse approximate inverse: `M^{-1} = M ~= A^{-1}`.
///
/// Stores `M` in CSC. Apply is a single sparse matvec with no triangular solves.
/// See the [module documentation](self) for guidance.
#[derive(Debug, Clone)]
pub struct Spai<I, T> {
    pub(crate) dim: usize,
    pub(crate) m_col_ptr: Vec<I>,
    pub(crate) m_row_idx: Vec<I>,
    pub(crate) m_values: Vec<T>,
}

impl<I, T> Spai<I, T> {
    /// Dimension `n` of the preconditioner.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl<I: Index, T: ComplexField> Spai<I, T> {
    /// View over the approximate inverse `M`.
    #[inline]
    pub(crate) fn m_view(&self) -> SparseColMatRef<'_, I, T> {
        let symbolic = unsafe {
            SymbolicSparseColMatRef::<'_, I>::new_unchecked(
                self.dim,
                self.dim,
                &self.m_col_ptr,
                None,
                &self.m_row_idx,
            )
        };
        SparseColMatRef::new(symbolic, &self.m_values)
    }
}

impl<I, T> LinOp<T> for Spai<I, T>
where
    I: Index,
    T: ComplexField + Debug + Sync,
{
    fn apply_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::EMPTY
    }

    fn nrows(&self) -> usize {
        self.dim
    }

    fn ncols(&self) -> usize {
        self.dim
    }

    fn apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, _stack: &mut MemStack) {
        apply::apply_out(self, false, false, out, rhs, par);
    }

    fn conj_apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, _stack: &mut MemStack) {
        apply::apply_out(self, false, true, out, rhs, par);
    }
}

impl<I, T> Precond<T> for Spai<I, T>
where
    I: Index,
    T: ComplexField + Debug + Sync,
{
    fn apply_in_place_scratch(&self, rhs_ncols: usize, _par: Par) -> StackReq {
        apply::inplace_scratch(self, rhs_ncols)
    }

    fn apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
        apply::apply_inplace(self, false, false, rhs, par, stack);
    }

    fn conj_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
        apply::apply_inplace(self, false, true, rhs, par, stack);
    }
}

impl<I, T> BiLinOp<T> for Spai<I, T>
where
    I: Index,
    T: ComplexField + Debug + Sync,
{
    fn transpose_apply_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::EMPTY
    }

    fn transpose_apply(
        &self,
        out: MatMut<'_, T>,
        rhs: MatRef<'_, T>,
        par: Par,
        _stack: &mut MemStack,
    ) {
        apply::apply_out(self, true, false, out, rhs, par);
    }

    fn adjoint_apply(
        &self,
        out: MatMut<'_, T>,
        rhs: MatRef<'_, T>,
        par: Par,
        _stack: &mut MemStack,
    ) {
        apply::apply_out(self, true, true, out, rhs, par);
    }
}

impl<I, T> BiPrecond<T> for Spai<I, T>
where
    I: Index,
    T: ComplexField + Debug + Sync,
{
    fn transpose_apply_in_place_scratch(&self, rhs_ncols: usize, _par: Par) -> StackReq {
        apply::inplace_scratch(self, rhs_ncols)
    }

    fn transpose_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
        apply::apply_inplace(self, true, false, rhs, par, stack);
    }

    fn adjoint_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
        apply::apply_inplace(self, true, true, rhs, par, stack);
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

    fn tridiagonal(n: usize, diag: f64, sub: f64, sup: f64) -> SparseColMat<usize, f64> {
        let mut triplets = Vec::new();
        for i in 0..n {
            triplets.push(Triplet::new(i, i, diag));
            if i > 0 {
                triplets.push(Triplet::new(i, i - 1, sub));
                triplets.push(Triplet::new(i - 1, i, sup));
            }
        }
        SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap()
    }

    fn apply_inplace(pc: &Spai<usize, f64>, rhs: &mut Mat<f64>) {
        with_stack(pc.apply_in_place_scratch(rhs.ncols(), Par::Seq), |stack| {
            pc.apply_in_place(rhs.as_mut(), Par::Seq, stack);
        });
    }

    fn residual_ratio(a: &SparseColMat<usize, f64>, pc: &Spai<usize, f64>, b: &Mat<f64>) -> f64 {
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
        let pc = Spai::try_new(a.as_ref(), SpaiPattern::ColumnsOfA).unwrap();
        let mut x = mat![[2.0_f64], [8.0], [16.0]];
        apply_inplace(&pc, &mut x);
        let expected = mat![[1.0_f64], [2.0], [2.0]];
        assert_close(x.as_ref(), expected.as_ref(), 1e-12);
    }

    #[test]
    fn reduces_residual_on_nonsymmetric() {
        let a = tridiagonal(12, 4.0, -2.0, -1.0);
        let n = a.nrows();
        let pc = Spai::try_new(a.as_ref(), SpaiPattern::ColumnsOfPower { power: 2 }).unwrap();
        let b = Mat::<f64>::from_fn(n, 1, |i, _| (i % 7) as f64 - 3.0);
        let ratio = residual_ratio(&a, &pc, &b);
        assert!(ratio < 1.0, "SPAI should reduce the residual: {ratio}");
    }

    #[test]
    fn transpose_differs_from_forward_on_nonsymmetric() {
        let a = tridiagonal(8, 4.0, -2.0, -1.0);
        let pc = Spai::try_new(a.as_ref(), SpaiPattern::ColumnsOfA).unwrap();
        let rhs = Mat::<f64>::from_fn(8, 1, |i, _| (i % 5) as f64 - 2.0);

        let mut fwd = rhs.clone();
        apply_inplace(&pc, &mut fwd);
        let mut tr = rhs.clone();
        with_stack(pc.transpose_apply_in_place_scratch(1, Par::Seq), |stack| {
            pc.transpose_apply_in_place(tr.as_mut(), Par::Seq, stack);
        });
        // For a nonsymmetric operator the two must differ.
        let diff: f64 = (0..8)
            .map(|i| (fwd.as_ref().get(i, 0) - tr.as_ref().get(i, 0)).abs())
            .sum();
        assert!(diff > 1e-8, "transpose apply should differ from forward");
    }

    #[test]
    fn out_of_place_matches_in_place() {
        let a = tridiagonal(10, 4.0, -2.0, -1.0);
        let pc = Spai::try_new(a.as_ref(), SpaiPattern::ColumnsOfA).unwrap();
        let rhs = Mat::<f64>::from_fn(10, 2, |i, j| ((i + 3 * j) % 7) as f64 - 3.0);
        let mut out = Mat::<f64>::zeros(10, 2);
        pc.apply(out.as_mut(), rhs.as_ref(), Par::Seq, MemStack::new(&mut []));
        let mut inplace = rhs.clone();
        apply_inplace(&pc, &mut inplace);
        assert_close(out.as_ref(), inplace.as_ref(), 1e-12);
    }

    #[test]
    fn rejects_non_square() {
        let mut triplets = Vec::new();
        for i in 0..3 {
            triplets.push(Triplet::new(i, i, 1.0));
        }
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(3, 4, &triplets).unwrap();
        assert_eq!(
            Spai::try_new(a.as_ref(), SpaiPattern::ColumnsOfA).unwrap_err(),
            SpaiError::NonSquareMatrix { nrows: 3, ncols: 4 }
        );
    }
}
