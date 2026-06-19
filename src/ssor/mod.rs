//! SSOR / symmetric Gauss-Seidel preconditioner.
//!
//! SSOR is a *stationary* preconditioner: it asks for no factorisation and adds
//! no fill. It is built straight from `A`'s own splitting `A = D + L + U` (the
//! diagonal, strict-lower and strict-upper parts) and a relaxation factor
//! `w in (0, 2)`:
//!
//! ```text
//! M = 1 / (w (2 - w)) * (D + w L) D^{-1} (D + w U)
//! ```
//!
//! With `w = 1` this is the symmetric Gauss-Seidel (SGS) preconditioner
//! `(D + L) D^{-1} (D + U)`. Applying `M^{-1}` is one lower triangular solve,
//! a diagonal scaling, and one upper triangular solve — all against `A`'s own
//! entries, with no heap allocation.
//!
//! # When to use it
//!
//! SSOR is the natural step up from [`crate::JacobiPrecond`] when you want
//! something stronger than diagonal scaling but cheaper to build than an
//! incomplete factorisation: there is nothing to store beyond a scaled copy of
//! `A`'s triangles, and refactorisation against new values is allocation-free.
//! It is most effective on diagonally-dominant and symmetric positive-definite
//! systems, where `M` is itself SPD and pairs with conjugate gradient. A good
//! `w` is problem-dependent (often between 1.0 and 1.9); `w = 1` (SGS) is a
//! robust default.
//!
//! It is a weaker approximation than [`crate::Ic0`] / [`crate::Ilu0`] for the
//! same apply cost on most PDE operators, so reach for those when you can. SSOR
//! shines when you want a parameter-light, fill-free preconditioner or a smoother.
//!
//! # Storage
//!
//! The two factors `(D + w L)` and `(D + w U)` are stored CSC following faer's
//! triangular-solve conventions — `(D + w L)`'s diagonal *first* in each column,
//! `(D + w U)`'s diagonal *last*. The `w (2 - w)` scalar is folded into a
//! precomputed `scaled_diag` so apply needs no scratch.

use core::fmt::Debug;

use dyn_stack::{MemStack, StackReq};
use faer::matrix_free::{BiLinOp, BiPrecond, LinOp, Precond};
use faer::{MatMut, MatRef, Par};
use faer_traits::{ComplexField, Index};

mod apply;
mod build;

/// Tuning parameters for [`Ssor`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsorParams {
    /// Relaxation factor `w`, required to lie in `(0, 2)`. `w = 1` gives
    /// symmetric Gauss-Seidel. Default `1.0`.
    pub omega: f64,
}

impl Default for SsorParams {
    fn default() -> Self {
        Self { omega: 1.0 }
    }
}

/// Error returned by SSOR construction or refactorisation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SsorError {
    /// The source matrix was not square.
    NonSquareMatrix { nrows: usize, ncols: usize },
    /// Column `col` of the source matrix does not contain its diagonal entry.
    MissingDiagonal { col: usize },
    /// Row indices in column `col` are not sorted ascending.
    UnsortedRowIndices { col: usize },
    /// Diagonal entry `col` was zero, so the triangular factors are singular.
    ZeroDiagonal { col: usize },
    /// `omega` was not in the open interval `(0, 2)`.
    InvalidOmega,
    /// A refactorisation was attempted with a matrix whose pattern does not
    /// match the stored factor.
    PatternMismatch,
}

impl core::fmt::Display for SsorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonSquareMatrix { nrows, ncols } => {
                write!(f, "matrix must be square but is {nrows}x{ncols}")
            }
            Self::MissingDiagonal { col } => write!(f, "column {col} is missing its diagonal entry"),
            Self::UnsortedRowIndices { col } => write!(f, "column {col} has unsorted row indices"),
            Self::ZeroDiagonal { col } => write!(f, "diagonal entry {col} is zero"),
            Self::InvalidOmega => f.write_str("omega must lie in the open interval (0, 2)"),
            Self::PatternMismatch => f.write_str("refactorisation pattern does not match"),
        }
    }
}

impl core::error::Error for SsorError {}

/// SSOR / symmetric Gauss-Seidel preconditioner.
///
/// Stores the scaled triangular factors `(D + w L)` and `(D + w U)` of `A` plus
/// the folded diagonal scaling. Apply is two sparse triangular solves and one
/// diagonal multiply, allocating nothing. See the [module documentation](self)
/// for when this is the right choice.
#[derive(Debug, Clone)]
pub struct Ssor<I, T> {
    pub(crate) dim: usize,
    pub(crate) omega: f64,
    /// `w (2 - w) * A[i,i]`, the folded middle diagonal scaling.
    pub(crate) scaled_diag: Vec<T>,
    /// `(D + w L)` CSC: column pointers, row indices (diagonal first), values.
    pub(crate) l_col_ptr: Vec<I>,
    pub(crate) l_row_idx: Vec<I>,
    pub(crate) l_values: Vec<T>,
    /// `(D + w U)` CSC: column pointers, row indices (diagonal last), values.
    pub(crate) u_col_ptr: Vec<I>,
    pub(crate) u_row_idx: Vec<I>,
    pub(crate) u_values: Vec<T>,
    pub(crate) diag_pos: Vec<usize>,
}

impl<I, T> Ssor<I, T> {
    /// Dimension `n` of the preconditioner.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The relaxation factor `w` this preconditioner was built with.
    #[inline]
    pub fn omega(&self) -> f64 {
        self.omega
    }
}

impl<I, T> LinOp<T> for Ssor<I, T>
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

    fn apply(&self, mut out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, _stack: &mut MemStack) {
        out.copy_from(rhs);
        apply::solve_in_place(self, false, false, out, par);
    }

    fn conj_apply(
        &self,
        mut out: MatMut<'_, T>,
        rhs: MatRef<'_, T>,
        par: Par,
        _stack: &mut MemStack,
    ) {
        out.copy_from(rhs);
        apply::solve_in_place(self, false, true, out, par);
    }
}

impl<I, T> Precond<T> for Ssor<I, T>
where
    I: Index,
    T: ComplexField + Debug + Sync,
{
    fn apply_in_place_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::EMPTY
    }

    fn apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, _stack: &mut MemStack) {
        apply::solve_in_place(self, false, false, rhs, par);
    }

    fn conj_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, _stack: &mut MemStack) {
        apply::solve_in_place(self, false, true, rhs, par);
    }
}

impl<I, T> BiLinOp<T> for Ssor<I, T>
where
    I: Index,
    T: ComplexField + Debug + Sync,
{
    fn transpose_apply_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::EMPTY
    }

    fn transpose_apply(
        &self,
        mut out: MatMut<'_, T>,
        rhs: MatRef<'_, T>,
        par: Par,
        _stack: &mut MemStack,
    ) {
        out.copy_from(rhs);
        apply::solve_in_place(self, true, false, out, par);
    }

    fn adjoint_apply(
        &self,
        mut out: MatMut<'_, T>,
        rhs: MatRef<'_, T>,
        par: Par,
        _stack: &mut MemStack,
    ) {
        out.copy_from(rhs);
        apply::solve_in_place(self, true, true, out, par);
    }
}

impl<I, T> BiPrecond<T> for Ssor<I, T>
where
    I: Index,
    T: ComplexField + Debug + Sync,
{
    fn transpose_apply_in_place_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::EMPTY
    }

    fn transpose_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, _stack: &mut MemStack) {
        apply::solve_in_place(self, true, false, rhs, par);
    }

    fn adjoint_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, _stack: &mut MemStack) {
        apply::solve_in_place(self, true, true, rhs, par);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::sparse::{SparseColMat, Triplet};
    use faer::{Mat, MatRef, mat};

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

    fn diagonal(diag: &[f64]) -> SparseColMat<usize, f64> {
        let mut triplets = Vec::new();
        for (i, &v) in diag.iter().enumerate() {
            triplets.push(Triplet::new(i, i, v));
        }
        SparseColMat::try_new_from_triplets(diag.len(), diag.len(), &triplets).unwrap()
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

    fn residual_ratio(a: &SparseColMat<usize, f64>, pc: &Ssor<usize, f64>, b: &Mat<f64>) -> f64 {
        let a_dense = to_dense(a);
        let mut x = b.clone();
        pc.apply_in_place(x.as_mut(), Par::Seq, MemStack::new(&mut []));
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
    fn sgs_on_diagonal_is_exact_inverse() {
        // For a diagonal A and w = 1, SGS reduces to M^{-1} = D^{-1}.
        let a = diagonal(&[2.0, 4.0, 8.0]);
        let pc = Ssor::try_new(a.as_ref(), SsorParams::default()).unwrap();

        let mut x = mat![[2.0_f64], [8.0], [16.0]];
        pc.apply_in_place(x.as_mut(), Par::Seq, MemStack::new(&mut []));
        let expected = mat![[1.0_f64], [2.0], [2.0]];
        assert_close(x.as_ref(), expected.as_ref(), 1e-12);
    }

    #[test]
    fn symmetric_input_makes_transpose_equal_apply() {
        // For symmetric A and real w, M is symmetric, so M^{-T} == M^{-1}.
        let a = tridiagonal(6, 4.0, -1.0, -1.0);
        let pc = Ssor::try_new(a.as_ref(), SsorParams { omega: 1.3 }).unwrap();
        let rhs = mat![[1.0_f64], [-2.0], [3.0], [0.5], [-1.0], [2.0]];

        let mut fwd = rhs.clone();
        pc.apply_in_place(fwd.as_mut(), Par::Seq, MemStack::new(&mut []));
        let mut tr = rhs.clone();
        pc.transpose_apply_in_place(tr.as_mut(), Par::Seq, MemStack::new(&mut []));
        assert_close(fwd.as_ref(), tr.as_ref(), 1e-12);
    }

    #[test]
    fn sgs_reduces_residual_on_laplacian() {
        let a = laplacian_2d(8);
        let n = a.nrows();
        let pc = Ssor::try_new(a.as_ref(), SsorParams::default()).unwrap();
        let b = Mat::<f64>::from_fn(n, 1, |i, _| (i % 7) as f64 - 3.0);
        let ratio = residual_ratio(&a, &pc, &b);
        assert!(ratio < 0.7, "SGS residual ratio {ratio} too large");
    }

    #[test]
    fn out_of_place_matches_in_place() {
        let a = tridiagonal(7, 4.0, -2.0, -1.0);
        let pc = Ssor::try_new(a.as_ref(), SsorParams { omega: 1.2 }).unwrap();
        let rhs = Mat::<f64>::from_fn(7, 2, |i, j| ((i + 2 * j) % 5) as f64 - 2.0);

        let mut out = Mat::<f64>::zeros(7, 2);
        pc.apply(out.as_mut(), rhs.as_ref(), Par::Seq, MemStack::new(&mut []));
        let mut inplace = rhs.clone();
        pc.apply_in_place(inplace.as_mut(), Par::Seq, MemStack::new(&mut []));
        assert_close(out.as_ref(), inplace.as_ref(), 1e-12);
    }

    #[test]
    fn refactorize_matches_fresh_construction() {
        let a1 = tridiagonal(7, 4.0, -1.0, -1.0);
        let a2 = tridiagonal(7, 5.0, -2.0, -1.5);
        let params = SsorParams { omega: 1.4 };

        let fresh = Ssor::try_new(a2.as_ref(), params).unwrap();
        let mut reused = Ssor::try_new(a1.as_ref(), params).unwrap();
        reused.refactorize(a2.as_ref()).unwrap();

        assert_eq!(fresh.l_values.len(), reused.l_values.len());
        for (a, b) in fresh.l_values.iter().zip(reused.l_values.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
        for (a, b) in fresh.u_values.iter().zip(reused.u_values.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
        for (a, b) in fresh.scaled_diag.iter().zip(reused.scaled_diag.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
    }

    #[test]
    fn rejects_invalid_omega() {
        let a = tridiagonal(3, 4.0, -1.0, -1.0);
        assert_eq!(
            Ssor::try_new(a.as_ref(), SsorParams { omega: 0.0 }).unwrap_err(),
            SsorError::InvalidOmega
        );
        assert_eq!(
            Ssor::try_new(a.as_ref(), SsorParams { omega: 2.0 }).unwrap_err(),
            SsorError::InvalidOmega
        );
    }

    #[test]
    fn rejects_zero_diagonal() {
        let a = diagonal(&[1.0, 0.0, 1.0]);
        assert_eq!(
            Ssor::try_new(a.as_ref(), SsorParams::default()).unwrap_err(),
            SsorError::ZeroDiagonal { col: 1 }
        );
    }

    #[test]
    fn rejects_non_square() {
        let mut triplets = Vec::new();
        for i in 0..3 {
            triplets.push(Triplet::new(i, i, 1.0));
        }
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(3, 4, &triplets).unwrap();
        assert_eq!(
            Ssor::try_new(a.as_ref(), SsorParams::default()).unwrap_err(),
            SsorError::NonSquareMatrix { nrows: 3, ncols: 4 }
        );
    }
}
