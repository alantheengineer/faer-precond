//! Zero-fill incomplete Cholesky preconditioner.
//!
//! IC(0) is the standard preconditioner for symmetric positive-definite sparse
//! systems solved with conjugate gradient — Laplacians, diffusion, elasticity
//! and similar discretised PDE operators. It is the symmetric specialisation of
//! [`crate::Ilu0`]: a Cholesky factorisation constrained to `A`'s own sparsity
//! pattern, so it adds no fill-in, takes no more memory than the lower triangle
//! of `A`, and is cheap to rebuild.
//!
//! Concretely, [`Ic0::try_new`] computes a lower-triangular factor `L` such
//! that:
//!
//! - `pattern(L)` equals the *lower triangular* subset of `pattern(A)`
//!   (no fill-in),
//! - `L L^H` agrees with `A` at every entry in `pattern(A)` on or below the
//!   diagonal.
//!
//! Only the lower triangle of the input matrix is used — entries above the
//! diagonal are ignored, so a `SparseColMat` storing either the full Hermitian
//! matrix or just its lower triangle is accepted.
//!
//! # Repeated factorisation
//!
//! Build [`SymbolicIc0`] once for a given sparsity pattern, allocate an
//! [`Ic0`] via [`Ic0::new_with_symbolic`], and call [`Ic0::refactorize`] in
//! the inner loop — refactorisation performs zero heap allocations.
//!
//! # Example
//!
//! ```
//! use dyn_stack::MemStack;
//! use faer::sparse::{SparseColMat, Triplet};
//! use faer::{mat, Par};
//! use faer::matrix_free::Precond;
//! use faer_precond::Ic0;
//!
//! // 5x5 tridiagonal SPD: diag 4, off-diagonals -1.
//! let mut triplets = Vec::new();
//! for i in 0..5 {
//!     triplets.push(Triplet::new(i, i, 4.0_f64));
//!     if i > 0 {
//!         triplets.push(Triplet::new(i, i - 1, -1.0));
//!         triplets.push(Triplet::new(i - 1, i, -1.0));
//!     }
//! }
//! let a = SparseColMat::<usize, f64>::try_new_from_triplets(5, 5, &triplets).unwrap();
//!
//! // Only the lower triangle is read; full Hermitian storage is fine.
//! let pc = Ic0::try_new(a.as_ref()).expect("matrix is positive definite");
//!
//! let mut b = mat![[1.0_f64], [0.0], [0.0], [0.0], [0.0]];
//! pc.apply_in_place(b.as_mut(), Par::Seq, MemStack::new(&mut []));
//! ```
//!
//! # Storage
//!
//! `L` is stored CSC with its diagonal *first* in each column, matching
//! [`faer::sparse::linalg::triangular_solve`]. Apply uses faer's sparse
//! triangular solves directly and allocates no heap memory.

use core::fmt::Debug;

use dyn_stack::{MemStack, StackReq};
use faer::matrix_free::{BiLinOp, BiPrecond, LinOp, Precond};
use faer::{Conj, MatMut, MatRef, Par};
use faer_traits::{ComplexField, Index};

pub mod apply;
pub mod numeric;
pub mod symbolic;

pub use numeric::Ic0;
pub use symbolic::SymbolicIc0;

/// Error returned by IC(0) construction or refactorisation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ic0Error {
    /// The source matrix was not square.
    NonSquareMatrix { nrows: usize, ncols: usize },
    /// Column `col` of the source matrix does not contain its diagonal entry.
    MissingDiagonal { col: usize },
    /// Row indices in column `col` are not sorted ascending.
    UnsortedRowIndices { col: usize },
    /// A refactorisation was attempted with a matrix whose pattern does not
    /// match the symbolic factor.
    PatternMismatch,
    /// A non-positive pivot was encountered at column `col` — the matrix is
    /// not positive definite, or IC(0) has broken down on a non-H-matrix
    /// input.
    NotPositiveDefinite { col: usize },
}

impl core::fmt::Display for Ic0Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonSquareMatrix { nrows, ncols } => {
                write!(f, "matrix must be square but is {nrows}x{ncols}")
            }
            Self::MissingDiagonal { col } => {
                write!(f, "column {col} is missing its diagonal entry")
            }
            Self::UnsortedRowIndices { col } => {
                write!(f, "column {col} has unsorted row indices")
            }
            Self::PatternMismatch => f.write_str("refactorisation pattern does not match symbolic"),
            Self::NotPositiveDefinite { col } => {
                write!(f, "encountered a non-positive pivot at column {col}")
            }
        }
    }
}

impl core::error::Error for Ic0Error {}

impl<I, T> LinOp<T> for Ic0<I, T>
where
    I: Index,
    T: ComplexField + Debug + Sync,
{
    fn apply_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::EMPTY
    }

    fn nrows(&self) -> usize {
        self.dim()
    }

    fn ncols(&self) -> usize {
        self.dim()
    }

    fn apply(&self, mut out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, _stack: &mut MemStack) {
        out.copy_from(rhs);
        apply::solve_in_place(self, Conj::No, out, par);
    }

    fn conj_apply(
        &self,
        mut out: MatMut<'_, T>,
        rhs: MatRef<'_, T>,
        par: Par,
        _stack: &mut MemStack,
    ) {
        out.copy_from(rhs);
        apply::solve_in_place(self, Conj::Yes, out, par);
    }
}

impl<I, T> Precond<T> for Ic0<I, T>
where
    I: Index,
    T: ComplexField + Debug + Sync,
{
    fn apply_in_place_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::EMPTY
    }

    fn apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, _stack: &mut MemStack) {
        apply::solve_in_place(self, Conj::No, rhs, par);
    }

    fn conj_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, _stack: &mut MemStack) {
        apply::solve_in_place(self, Conj::Yes, rhs, par);
    }
}

impl<I, T> BiLinOp<T> for Ic0<I, T>
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
        // M^T = conj(M) for Hermitian M, so M^{-T} = conj(M)^{-1}.
        out.copy_from(rhs);
        apply::solve_in_place(self, Conj::Yes, out, par);
    }

    fn adjoint_apply(
        &self,
        mut out: MatMut<'_, T>,
        rhs: MatRef<'_, T>,
        par: Par,
        _stack: &mut MemStack,
    ) {
        // M^H = M for Hermitian M, so M^{-H} = M^{-1}.
        out.copy_from(rhs);
        apply::solve_in_place(self, Conj::No, out, par);
    }
}

impl<I, T> BiPrecond<T> for Ic0<I, T>
where
    I: Index,
    T: ComplexField + Debug + Sync,
{
    fn transpose_apply_in_place_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::EMPTY
    }

    fn transpose_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, _stack: &mut MemStack) {
        apply::solve_in_place(self, Conj::Yes, rhs, par);
    }

    fn adjoint_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, _stack: &mut MemStack) {
        apply::solve_in_place(self, Conj::No, rhs, par);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::sparse::{SparseColMat, SparseColMatRef, Triplet};
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

    fn sparse_view_to_dense(a: SparseColMatRef<'_, usize, f64>) -> Mat<f64> {
        let mut dense = Mat::<f64>::zeros(a.nrows(), a.ncols());
        for j in 0..a.ncols() {
            let rows = a.symbolic().row_idx_of_col_raw(j);
            let vals = a.val_of_col(j);
            for (r, v) in rows.iter().zip(vals.iter()) {
                *dense.as_mut().get_mut(*r, j) = *v;
            }
        }
        dense
    }

    fn to_dense(a: &SparseColMat<usize, f64>) -> Mat<f64> {
        sparse_view_to_dense(a.as_ref())
    }

    /// Tridiagonal SPD: 4 on the diagonal, -1 off — strictly diagonally dominant.
    /// Stored as a full Hermitian CSC (both triangles) to exercise the
    /// "ignore upper triangle" path.
    fn tridiagonal_spd_full(n: usize) -> SparseColMat<usize, f64> {
        let mut triplets = Vec::new();
        for i in 0..n {
            triplets.push(Triplet::new(i, i, 4.0));
            if i > 0 {
                triplets.push(Triplet::new(i, i - 1, -1.0));
                triplets.push(Triplet::new(i - 1, i, -1.0));
            }
        }
        SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap()
    }

    /// 5-point Laplacian on an `grid x grid` mesh, full Hermitian storage.
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

    #[test]
    fn ic0_tridiagonal_matches_exact_inverse() {
        // Tridiagonal => no fill-in => IC(0) is the exact Cholesky factor =>
        // M^{-1} A x = x.
        let a = tridiagonal_spd_full(5);
        let pc = Ic0::try_new(a.as_ref()).unwrap();

        let a_dense = to_dense(&a);
        let x_true = mat![[1.0], [-2.0], [3.0], [-1.0], [0.5_f64]];
        let mut rhs = (&a_dense * &x_true).to_owned();

        pc.apply_in_place(rhs.as_mut(), Par::Seq, MemStack::new(&mut []));
        assert_close(rhs.as_ref(), x_true.as_ref(), 1e-12);
    }

    #[test]
    fn ic0_factor_satisfies_pattern_equation_lower_triangle() {
        // For IC(0): L L^T == A at every position in lower-triangle pattern.
        let a = laplacian_2d(4);
        let pc = Ic0::try_new(a.as_ref()).unwrap();

        let l_dense = sparse_view_to_dense(pc.l_view());
        let llt_dense = &l_dense * l_dense.transpose();
        let a_dense = to_dense(&a);

        let a_ref = a.as_ref();
        for j in 0..a.ncols() {
            for r in a_ref.symbolic().row_idx_of_col_raw(j) {
                let i = *r;
                if i < j {
                    continue;
                }
                let diff = (*llt_dense.as_ref().get(i, j) - *a_dense.as_ref().get(i, j)).abs();
                assert!(diff <= 1e-12, "L*L^T disagrees with A at ({i},{j}): {diff}");
            }
        }
    }

    #[test]
    fn ic0_l_has_positive_diagonal() {
        let a = laplacian_2d(5);
        let pc = Ic0::try_new(a.as_ref()).unwrap();
        let l = pc.l_view();
        for j in 0..l.ncols() {
            let diag = *l.val_of_col(j).first().unwrap();
            assert!(diag > 0.0, "L[{j},{j}] = {diag} should be positive");
        }
    }

    #[test]
    fn ic0_reduces_residual_significantly() {
        let a = laplacian_2d(8);
        let n = a.nrows();
        let pc = Ic0::try_new(a.as_ref()).unwrap();
        let a_dense = to_dense(&a);

        let b = Mat::<f64>::from_fn(n, 1, |i, _| (i % 7) as f64 - 3.0);
        let mut x = b.clone();
        pc.apply_in_place(x.as_mut(), Par::Seq, MemStack::new(&mut []));

        let residual = &a_dense * &x - &b;
        let b_norm: f64 = b.as_ref().col(0).iter().map(|v| v * v).sum::<f64>().sqrt();
        let r_norm: f64 = residual
            .as_ref()
            .col(0)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        assert!(
            r_norm / b_norm < 0.5,
            "IC(0) residual ratio {r_norm}/{b_norm} too large"
        );
    }

    #[test]
    fn refactorize_matches_fresh_construction() {
        let a1 = tridiagonal_spd_full(7);
        let mut triplets2 = Vec::new();
        for i in 0..7 {
            triplets2.push(Triplet::new(i, i, 5.0));
            if i > 0 {
                triplets2.push(Triplet::new(i, i - 1, -2.0));
                triplets2.push(Triplet::new(i - 1, i, -2.0));
            }
        }
        let a2 = SparseColMat::<usize, f64>::try_new_from_triplets(7, 7, &triplets2).unwrap();

        let pc_fresh = Ic0::try_new(a2.as_ref()).unwrap();

        let mut pc_reused = Ic0::try_new(a1.as_ref()).unwrap();
        pc_reused.refactorize(a2.as_ref()).unwrap();

        assert_eq!(pc_fresh.l_values.len(), pc_reused.l_values.len());
        for (a, b) in pc_fresh.l_values.iter().zip(pc_reused.l_values.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
    }

    #[test]
    fn transpose_and_adjoint_match_apply_for_real_spd() {
        let a = tridiagonal_spd_full(6);
        let pc = Ic0::try_new(a.as_ref()).unwrap();

        let rhs = mat![[1.0], [2.0], [3.0], [-1.0], [0.5], [-2.0_f64]];

        let mut x = rhs.clone();
        pc.apply_in_place(x.as_mut(), Par::Seq, MemStack::new(&mut []));

        let mut xt = rhs.clone();
        pc.transpose_apply_in_place(xt.as_mut(), Par::Seq, MemStack::new(&mut []));

        let mut xh = rhs.clone();
        pc.adjoint_apply_in_place(xh.as_mut(), Par::Seq, MemStack::new(&mut []));

        assert_close(x.as_ref(), xt.as_ref(), 1e-12);
        assert_close(x.as_ref(), xh.as_ref(), 1e-12);
    }

    #[test]
    fn rejects_non_square() {
        let triplets = (0..3).map(|i| Triplet::new(i, i, 1.0)).collect::<Vec<_>>();
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(3, 4, &triplets).unwrap();
        let err = Ic0::try_new(a.as_ref()).unwrap_err();
        assert_eq!(err, Ic0Error::NonSquareMatrix { nrows: 3, ncols: 4 });
    }

    #[test]
    fn rejects_missing_diagonal() {
        let triplets = vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(2, 1, 3.0),
            Triplet::new(2, 2, 4.0_f64),
        ];
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(3, 3, &triplets).unwrap();
        let err = Ic0::try_new(a.as_ref()).unwrap_err();
        assert_eq!(err, Ic0Error::MissingDiagonal { col: 1 });
    }

    #[test]
    fn rejects_indefinite_matrix() {
        // A = diag(1, -1) is not positive definite.
        let triplets = vec![Triplet::new(0, 0, 1.0), Triplet::new(1, 1, -1.0_f64)];
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(2, 2, &triplets).unwrap();
        let err = Ic0::try_new(a.as_ref()).unwrap_err();
        assert_eq!(err, Ic0Error::NotPositiveDefinite { col: 1 });
    }

    #[test]
    fn rejects_pattern_mismatch_on_refactorize() {
        let a1 = tridiagonal_spd_full(5);
        let a2 = tridiagonal_spd_full(6);
        let mut pc = Ic0::try_new(a1.as_ref()).unwrap();
        let err = pc.refactorize(a2.as_ref()).unwrap_err();
        assert_eq!(err, Ic0Error::PatternMismatch);
    }
}
