//! Zero-fill incomplete LU preconditioner.
//!
//! Given a sparse CSC matrix `A`, [`Ilu0::try_new`] computes the unique
//! factors `L` (unit lower) and `U` (upper) such that:
//!
//! - `pattern(L) ∪ pattern(U) = pattern(A)` (no fill-in),
//! - `L * U` agrees with `A` at every entry in `pattern(A)`.
//!
//! The factors are stored in column-compressed (CSC) form following faer's
//! sparse triangular-solve conventions — `L`'s unit diagonal stored *first* in
//! each column and `U`'s diagonal stored *last*. Apply uses
//! [`faer::sparse::linalg::triangular_solve`] directly and allocates no heap
//! memory.
//!
//! # Repeated factorisation
//!
//! Krylov methods inside nonlinear solvers typically refactorise the
//! preconditioner whenever the operator's values change but the sparsity
//! pattern stays the same. For that use-case, build [`SymbolicIlu0`] once,
//! allocate an [`Ilu0`] via [`Ilu0::new_with_symbolic`], and call
//! [`Ilu0::refactorize`] in the hot loop — no allocation occurs.
//!
//! # Example
//!
//! ```ignore
//! use faer::sparse::SparseColMat;
//! use faer_precond::ilu0::Ilu0;
//! let a: SparseColMat<usize, f64> = /* ... */;
//! let pc = Ilu0::try_new(a.as_ref()).expect("non-singular pattern");
//! ```

use core::fmt::Debug;

use dyn_stack::{MemStack, StackReq};
use faer::matrix_free::{BiLinOp, BiPrecond, LinOp, Precond};
use faer::{Conj, MatMut, MatRef, Par};
use faer_traits::{ComplexField, Index};

pub mod apply;
pub mod numeric;
pub mod symbolic;

pub use numeric::Ilu0;
pub use symbolic::SymbolicIlu0;

/// Error returned by ILU(0) construction or refactorisation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ilu0Error {
    /// The source matrix was not square.
    NonSquareMatrix { nrows: usize, ncols: usize },
    /// Column `col` of the source matrix does not contain its diagonal entry.
    MissingDiagonal { col: usize },
    /// Row indices in column `col` are not sorted ascending.
    UnsortedRowIndices { col: usize },
    /// A refactorisation was attempted with a matrix whose pattern does not
    /// match the symbolic factor.
    PatternMismatch,
    /// A zero pivot was encountered while eliminating column `col`.
    ZeroPivot { col: usize },
}

impl core::fmt::Display for Ilu0Error {
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
            Self::ZeroPivot { col } => write!(f, "encountered a zero pivot at column {col}"),
        }
    }
}

impl core::error::Error for Ilu0Error {}

impl<I, T> LinOp<T> for Ilu0<I, T>
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

impl<I, T> Precond<T> for Ilu0<I, T>
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

impl<I, T> BiLinOp<T> for Ilu0<I, T>
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
        apply::solve_transpose_in_place(self, Conj::No, out, par);
    }

    fn adjoint_apply(
        &self,
        mut out: MatMut<'_, T>,
        rhs: MatRef<'_, T>,
        par: Par,
        _stack: &mut MemStack,
    ) {
        out.copy_from(rhs);
        apply::solve_transpose_in_place(self, Conj::Yes, out, par);
    }
}

impl<I, T> BiPrecond<T> for Ilu0<I, T>
where
    I: Index,
    T: ComplexField + Debug + Sync,
{
    fn transpose_apply_in_place_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::EMPTY
    }

    fn transpose_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, _stack: &mut MemStack) {
        apply::solve_transpose_in_place(self, Conj::No, rhs, par);
    }

    fn adjoint_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, _stack: &mut MemStack) {
        apply::solve_transpose_in_place(self, Conj::Yes, rhs, par);
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

    /// 5-point Laplacian stencil on a 4x4 grid (n=16). Symmetric, diagonally
    /// dominant, banded. ILU(0) is exact-modulo-fill on this stencil.
    fn laplacian_2d(grid: usize) -> SparseColMat<usize, f64> {
        let n = grid * grid;
        let mut triplets: Vec<Triplet<usize, usize, f64>> = Vec::new();
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

    /// Convert a sparse matrix to dense for verification.
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

    fn tridiagonal_csc(n: usize, diag: f64, off: f64) -> SparseColMat<usize, f64> {
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

    #[test]
    fn ilu0_tridiagonal_matches_exact_inverse() {
        // For a tridiagonal SPD matrix, ILU(0) coincides with the exact LU
        // factorisation (no fill is introduced anyway). Thus M^{-1} A = I.
        let a = tridiagonal_csc(5, 4.0, -1.0);
        let pc = Ilu0::try_new(a.as_ref()).unwrap();

        let a_dense = to_dense(&a);
        let x_true = mat![[1.0], [-2.0], [3.0], [-1.0], [0.5_f64]];
        let mut rhs = (&a_dense * &x_true).to_owned();

        pc.apply_in_place(rhs.as_mut(), Par::Seq, MemStack::new(&mut []));
        assert_close(rhs.as_ref(), x_true.as_ref(), 1e-12);
    }

    fn sparse_view_to_dense(a: faer::sparse::SparseColMatRef<'_, usize, f64>) -> Mat<f64> {
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

    #[test]
    fn ilu0_factor_satisfies_pattern_equation() {
        // For ILU(0) the relation L*U == A holds at every position in pattern(A).
        let a = laplacian_2d(4);
        let pc = Ilu0::try_new(a.as_ref()).unwrap();

        let l_dense = sparse_view_to_dense(pc.l_view());
        let u_dense = sparse_view_to_dense(pc.u_view());
        let lu_dense = &l_dense * &u_dense;
        let a_dense = to_dense(&a);

        let a_ref = a.as_ref();
        for j in 0..a.ncols() {
            for r in a_ref.symbolic().row_idx_of_col_raw(j) {
                let i = *r;
                let diff = (*lu_dense.as_ref().get(i, j) - *a_dense.as_ref().get(i, j)).abs();
                assert!(diff <= 1e-12, "L*U disagrees with A at ({i},{j}): {diff}");
            }
        }
    }

    #[test]
    fn ilu0_reduces_residual_significantly() {
        // For diagonally-dominant matrices, ILU(0) should at minimum produce a
        // very small residual ||A x - b|| for a single right-preconditioned solve.
        let a = laplacian_2d(8);
        let n = a.nrows();
        let pc = Ilu0::try_new(a.as_ref()).unwrap();
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
        // ILU(0) for the 5-point Laplacian gives a much smaller residual than 1.
        assert!(
            r_norm / b_norm < 0.5,
            "ILU(0) residual ratio {r_norm}/{b_norm} too large"
        );
    }

    #[test]
    fn refactorize_matches_fresh_construction() {
        let a1 = tridiagonal_csc(7, 4.0, -1.0);
        let a2 = tridiagonal_csc(7, 5.0, -2.0);

        let pc_fresh = Ilu0::try_new(a2.as_ref()).unwrap();

        let mut pc_reused = Ilu0::try_new(a1.as_ref()).unwrap();
        pc_reused.refactorize(a2.as_ref()).unwrap();

        assert_eq!(pc_fresh.l_values.len(), pc_reused.l_values.len());
        assert_eq!(pc_fresh.u_values.len(), pc_reused.u_values.len());
        for (a, b) in pc_fresh.l_values.iter().zip(pc_reused.l_values.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
        for (a, b) in pc_fresh.u_values.iter().zip(pc_reused.u_values.iter()) {
            assert!((a - b).abs() < 1e-14);
        }
    }

    #[test]
    fn rejects_non_square() {
        let mut triplets = Vec::new();
        for i in 0..3 {
            triplets.push(Triplet::new(i, i, 1.0));
        }
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(3, 4, &triplets).unwrap();
        let err = Ilu0::try_new(a.as_ref()).unwrap_err();
        assert_eq!(err, Ilu0Error::NonSquareMatrix { nrows: 3, ncols: 4 });
    }

    #[test]
    fn rejects_missing_diagonal() {
        // 3x3 with no diagonal in column 1.
        let triplets = vec![
            Triplet::new(0, 0, 1.0),
            Triplet::new(0, 1, 2.0),
            Triplet::new(2, 1, 3.0),
            Triplet::new(2, 2, 4.0_f64),
        ];
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(3, 3, &triplets).unwrap();
        let err = Ilu0::try_new(a.as_ref()).unwrap_err();
        assert_eq!(err, Ilu0Error::MissingDiagonal { col: 1 });
    }

    #[test]
    fn rejects_pattern_mismatch_on_refactorize() {
        let a1 = tridiagonal_csc(5, 4.0, -1.0);
        let a2 = tridiagonal_csc(6, 4.0, -1.0);
        let mut pc = Ilu0::try_new(a1.as_ref()).unwrap();
        let err = pc.refactorize(a2.as_ref()).unwrap_err();
        assert_eq!(err, Ilu0Error::PatternMismatch);
    }

    #[test]
    fn transpose_apply_inverts_transposed_system() {
        let a = tridiagonal_csc(6, 4.0, -1.0);
        let pc = Ilu0::try_new(a.as_ref()).unwrap();
        let a_dense = to_dense(&a);

        let x_true = mat![[1.0], [-2.0], [3.0], [-1.0], [0.5], [2.0_f64]];
        let rhs = a_dense.transpose() * &x_true;

        let mut out = rhs.clone();
        pc.transpose_apply_in_place(out.as_mut(), Par::Seq, MemStack::new(&mut []));
        // Tridiagonal => ILU(0) exact => M^{-T} A^T x = x.
        assert_close(out.as_ref(), x_true.as_ref(), 1e-12);
    }
}
