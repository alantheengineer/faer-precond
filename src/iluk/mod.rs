//! Level-of-fill incomplete LU preconditioner, ILU(k).
//!
//! ILU(k) sits between [`crate::Ilu0`] and [`crate::Ilutp`]. Where ILU(0) is
//! locked to `A`'s own sparsity pattern, ILU(k) admits *structural* fill up to a
//! chosen level `k`: level-0 entries are those of `A`, and each elimination step
//! that touches two level-`a`/`b` entries can create a level-`a+b+1` fill entry,
//! kept only while that level stays `<= k`. Raising `k` gives a more accurate
//! factor (fewer Krylov iterations) at the cost of more fill and work.
//!
//! Unlike [`crate::Ilutp`], the pattern depends only on the *structure* of `A`,
//! not its values — so the symbolic factor can be built once and reused, and
//! refactorisation against new values is allocation-free.
//!
//! # When to use it
//!
//! Reach for ILU(k) on nonsymmetric sparse systems where [`crate::Ilu0`] is too
//! weak but you would rather control fill *structurally* (a predictable, value-
//! independent pattern) than through the drop tolerance of [`crate::Ilutp`].
//! `k = 1` or `k = 2` is the usual range; `k = 0` reproduces [`crate::Ilu0`]
//! exactly.
//!
//! # Repeated factorisation
//!
//! Build [`SymbolicIluk`] once, allocate an [`Iluk`] with
//! [`Iluk::new_with_symbolic`], and call [`Iluk::refactorize`] in the hot loop —
//! no allocation occurs.
//!
//! # Storage
//!
//! Identical to [`crate::Ilu0`]: `L` (unit lower) and `U` (upper) are stored CSC
//! with `L`'s diagonal first and `U`'s diagonal last, and apply uses faer's
//! sparse triangular solves directly with no heap allocation.

use core::fmt::Debug;

use dyn_stack::{MemStack, StackReq};
use faer::matrix_free::{BiLinOp, BiPrecond, LinOp, Precond};
use faer::{Conj, MatMut, MatRef, Par};
use faer_traits::{ComplexField, Index};

pub mod apply;
pub mod numeric;
pub mod symbolic;

pub use numeric::Iluk;
pub use symbolic::SymbolicIluk;

/// Tuning parameters for [`Iluk`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IlukParams {
    /// Fill level `k`. `0` reproduces ILU(0); higher admits more fill.
    pub level: usize,
}

impl Default for IlukParams {
    fn default() -> Self {
        Self { level: 1 }
    }
}

/// Error returned by ILU(k) construction or refactorisation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IlukError {
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

impl core::fmt::Display for IlukError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonSquareMatrix { nrows, ncols } => {
                write!(f, "matrix must be square but is {nrows}x{ncols}")
            }
            Self::MissingDiagonal { col } => write!(f, "column {col} is missing its diagonal entry"),
            Self::UnsortedRowIndices { col } => write!(f, "column {col} has unsorted row indices"),
            Self::PatternMismatch => f.write_str("refactorisation pattern does not match symbolic"),
            Self::ZeroPivot { col } => write!(f, "encountered a zero pivot at column {col}"),
        }
    }
}

impl core::error::Error for IlukError {}

impl<I, T> LinOp<T> for Iluk<I, T>
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

impl<I, T> Precond<T> for Iluk<I, T>
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

impl<I, T> BiLinOp<T> for Iluk<I, T>
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

impl<I, T> BiPrecond<T> for Iluk<I, T>
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
    use crate::ilu0::SymbolicIlu0;
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

    #[test]
    fn level_zero_matches_ilu0_pattern() {
        let a = laplacian_2d(5);
        let sk = SymbolicIluk::<usize>::try_new(a.as_ref().symbolic(), 0).unwrap();
        let s0 = SymbolicIlu0::<usize>::try_new(a.as_ref().symbolic()).unwrap();
        assert_eq!(sk.l_col_ptr, s0.l_col_ptr);
        assert_eq!(sk.l_row_idx, s0.l_row_idx);
        assert_eq!(sk.u_col_ptr, s0.u_col_ptr);
        assert_eq!(sk.u_row_idx, s0.u_row_idx);
    }

    #[test]
    fn level_one_grows_the_pattern() {
        let a = laplacian_2d(5);
        let s0 = SymbolicIluk::<usize>::try_new(a.as_ref().symbolic(), 0).unwrap();
        let s1 = SymbolicIluk::<usize>::try_new(a.as_ref().symbolic(), 1).unwrap();
        assert!(
            s1.l_nnz() + s1.u_nnz() > s0.l_nnz() + s0.u_nnz(),
            "ILU(1) should introduce fill over ILU(0)"
        );
    }

    #[test]
    fn factor_matches_a_on_its_own_pattern() {
        // L*U must agree with A at every structural entry of A.
        let a = laplacian_2d(5);
        let pc = Iluk::try_new(a.as_ref(), 1).unwrap();
        let l = sparse_view_to_dense(pc.l_view());
        let u = sparse_view_to_dense(pc.u_view());
        let lu = &l * &u;
        let a_dense = to_dense(&a);
        let a_ref = a.as_ref();
        for j in 0..a.ncols() {
            for r in a_ref.symbolic().row_idx_of_col_raw(j) {
                let i = *r;
                let diff = (*lu.as_ref().get(i, j) - *a_dense.as_ref().get(i, j)).abs();
                assert!(diff <= 1e-10, "L*U disagrees with A at ({i},{j}): {diff}");
            }
        }
    }

    #[test]
    fn tridiagonal_is_exact() {
        // No fill is possible in a tridiagonal, so ILU(k) is the exact LU.
        let a = tridiagonal(6, 4.0, -1.0, -1.0);
        let pc = Iluk::try_new(a.as_ref(), 2).unwrap();
        let a_dense = to_dense(&a);
        let x_true = mat![[1.0], [-2.0], [3.0], [-1.0], [0.5], [2.0_f64]];
        let mut rhs = (&a_dense * &x_true).to_owned();
        pc.apply_in_place(rhs.as_mut(), Par::Seq, MemStack::new(&mut []));
        assert_close(rhs.as_ref(), x_true.as_ref(), 1e-12);
    }

    #[test]
    fn refactorize_matches_fresh() {
        let a1 = laplacian_2d(4);
        let a2 = {
            // Same pattern, scaled values.
            let mut t = Vec::new();
            let a1_ref = a1.as_ref();
            for j in 0..a1.ncols() {
                for (r, v) in a1_ref
                    .symbolic()
                    .row_idx_of_col_raw(j)
                    .iter()
                    .zip(a1_ref.val_of_col(j))
                {
                    t.push(Triplet::new(*r, j, v * 1.5 + if *r == j { 0.5 } else { 0.0 }));
                }
            }
            SparseColMat::try_new_from_triplets(a1.nrows(), a1.ncols(), &t).unwrap()
        };
        let fresh = Iluk::try_new(a2.as_ref(), 1).unwrap();
        let mut reused = Iluk::try_new(a1.as_ref(), 1).unwrap();
        reused.refactorize(a2.as_ref()).unwrap();
        for (a, b) in fresh.l_values.iter().zip(reused.l_values.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
        for (a, b) in fresh.u_values.iter().zip(reused.u_values.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn reduces_residual_on_laplacian() {
        let a = laplacian_2d(8);
        let n = a.nrows();
        let pc = Iluk::try_new(a.as_ref(), 1).unwrap();
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
        assert!(r_norm / b_norm < 0.5, "ILU(1) residual ratio too large");
    }

    #[test]
    fn rejects_non_square() {
        let mut triplets = Vec::new();
        for i in 0..3 {
            triplets.push(Triplet::new(i, i, 1.0));
        }
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(3, 4, &triplets).unwrap();
        assert_eq!(
            Iluk::try_new(a.as_ref(), 1).unwrap_err(),
            IlukError::NonSquareMatrix { nrows: 3, ncols: 4 }
        );
    }
}
