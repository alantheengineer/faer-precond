//! Threshold incomplete Cholesky preconditioner (ICT).
//!
//! ICT is to [`crate::Ic0`] what [`crate::Ilutp`] is to [`crate::Ilu0`]: the
//! symmetric-positive-definite factorisation made *adaptive*. Instead of being
//! locked to `A`'s sparsity pattern, it keeps the most significant fill each
//! column produces — governed by a relative drop tolerance and a per-column fill
//! budget — and discards the rest. That makes it markedly more effective than
//! IC(0) on ill-conditioned SPD systems where zero-fill is too weak, at the cost
//! of a value-dependent pattern.
//!
//! Tune it through [`IctParams`]:
//!
//! - `drop_tol` — relative threshold; an entry is dropped when its magnitude
//!   falls below `drop_tol * ||column||`.
//! - `fill` — how many off-diagonal entries to keep per column (see
//!   [`FillControl`]).
//!
//! # When to use it
//!
//! Reach for ICT on symmetric positive-definite problems where [`crate::Ic0`]
//! stalls — strongly anisotropic or high-contrast diffusion, ill-conditioned
//! elasticity — and you are solving with conjugate gradient. For well-behaved
//! SPD operators [`crate::Ic0`] is cheaper; for nonsymmetric systems use
//! [`crate::Ilutp`].
//!
//! # Value-dependent pattern
//!
//! Like [`crate::Ilutp`], ICT's fill pattern depends on the matrix values, so
//! there is no separate symbolic phase and [`Ict::refactorize`] rebuilds the
//! pattern (reusing buffer capacity but **not** allocation-free). Apply itself
//! allocates nothing.
//!
//! # Storage
//!
//! The factor `L` is stored CSC with its diagonal *first* in each column,
//! matching [`faer::sparse::linalg::triangular_solve`]. Apply is the same two
//! triangular solves as [`crate::Ic0`] (`M^{-1} = L^{-H} L^{-1}`).

use core::fmt::Debug;

use dyn_stack::{MemStack, StackReq};
use faer::matrix_free::{BiLinOp, BiPrecond, LinOp, Precond};
use faer::{Conj, MatMut, MatRef, Par};
use faer_traits::{ComplexField, Index};

pub mod apply;
pub mod numeric;

pub use numeric::Ict;

// ICT shares the fill-budget and norm knobs with ILUTP.
pub use crate::ilutp::{FillControl, RowNorm};

/// Tuning parameters for [`Ict`].
///
/// Construct with [`IctParams::default`] and override fields as needed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IctParams {
    /// Relative drop tolerance. An entry is dropped when its magnitude is below
    /// `drop_tol * ||column||`. Default `1e-3`.
    pub drop_tol: f64,
    /// Per-column fill budget. Default `FillControl::Factor(5.0)`.
    pub fill: FillControl,
    /// Norm used to scale `drop_tol`. Default `RowNorm::Two`.
    pub norm: RowNorm,
}

impl Default for IctParams {
    fn default() -> Self {
        Self {
            drop_tol: 1e-3,
            fill: FillControl::Factor(5.0),
            norm: RowNorm::Two,
        }
    }
}

impl IctParams {
    pub(crate) fn validate(&self) -> Result<(), IctError> {
        if !self.drop_tol.is_finite() || self.drop_tol < 0.0 {
            return Err(IctError::InvalidDropTol);
        }
        if let FillControl::Factor(f) = self.fill
            && (!f.is_finite() || f <= 0.0)
        {
            return Err(IctError::InvalidFillControl);
        }
        Ok(())
    }
}

/// Error returned by ICT construction or refactorisation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IctError {
    /// The source matrix was not square.
    NonSquareMatrix { nrows: usize, ncols: usize },
    /// A non-positive pivot was encountered at column `col` — the matrix is not
    /// positive definite, or ICT has broken down for these parameters.
    NotPositiveDefinite { col: usize },
    /// A refactorisation was attempted with a matrix of a different dimension.
    PatternMismatch,
    /// `drop_tol` was negative or NaN.
    InvalidDropTol,
    /// `FillControl::Factor` was non-positive or NaN.
    InvalidFillControl,
}

impl core::fmt::Display for IctError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonSquareMatrix { nrows, ncols } => {
                write!(f, "matrix must be square but is {nrows}x{ncols}")
            }
            Self::NotPositiveDefinite { col } => {
                write!(f, "encountered a non-positive pivot at column {col}")
            }
            Self::PatternMismatch => f.write_str("refactorisation dimension does not match"),
            Self::InvalidDropTol => f.write_str("drop_tol must be finite and non-negative"),
            Self::InvalidFillControl => f.write_str("fill factor must be finite and positive"),
        }
    }
}

impl core::error::Error for IctError {}

impl<I, T> LinOp<T> for Ict<I, T>
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

impl<I, T> Precond<T> for Ict<I, T>
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

impl<I, T> BiLinOp<T> for Ict<I, T>
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
        // M = L L^H is Hermitian, so M^{-T} = conj(M^{-1}).
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
        // M^{-H} = M^{-1} for Hermitian M.
        out.copy_from(rhs);
        apply::solve_in_place(self, Conj::No, out, par);
    }
}

impl<I, T> BiPrecond<T> for Ict<I, T>
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

    /// drop_tol = 0 and a large fill budget makes ICT an exact Cholesky, so
    /// M^{-1} A x = x.
    fn exact_params(n: usize) -> IctParams {
        IctParams {
            drop_tol: 0.0,
            fill: FillControl::PerRow(n),
            norm: RowNorm::Two,
        }
    }

    #[test]
    fn exact_keep_inverts_tridiagonal() {
        let a = tridiagonal(7, 4.0, -1.0);
        let pc = Ict::try_new_with_params(a.as_ref(), exact_params(7)).unwrap();
        let a_dense = to_dense(&a);
        let x_true = mat![[1.0], [-2.0], [3.0], [-1.0], [0.5], [2.0], [-1.5_f64]];
        let mut rhs = (&a_dense * &x_true).to_owned();
        pc.apply_in_place(rhs.as_mut(), Par::Seq, MemStack::new(&mut []));
        assert_close(rhs.as_ref(), x_true.as_ref(), 1e-10);
    }

    #[test]
    fn exact_keep_reconstructs_laplacian() {
        // With no dropping, L L^T == A exactly (full Cholesky of the lower part).
        let a = laplacian_2d(4);
        let n = a.nrows();
        let pc = Ict::try_new_with_params(a.as_ref(), exact_params(n)).unwrap();
        let a_dense = to_dense(&a);
        let x_true = Mat::<f64>::from_fn(n, 1, |i, _| (i % 5) as f64 - 2.0);
        let mut rhs = (&a_dense * &x_true).to_owned();
        pc.apply_in_place(rhs.as_mut(), Par::Seq, MemStack::new(&mut []));
        assert_close(rhs.as_ref(), x_true.as_ref(), 1e-8);
    }

    #[test]
    fn reduces_residual_on_laplacian() {
        let a = laplacian_2d(8);
        let n = a.nrows();
        let pc = Ict::try_new(a.as_ref()).unwrap();
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
        assert!(r_norm / b_norm < 0.5, "ICT residual ratio too large");
    }

    #[test]
    fn symmetric_transpose_equals_apply() {
        let a = tridiagonal(6, 4.0, -1.0);
        let pc = Ict::try_new(a.as_ref()).unwrap();
        let rhs = mat![[1.0_f64], [-2.0], [3.0], [0.5], [-1.0], [2.0]];
        let mut fwd = rhs.clone();
        pc.apply_in_place(fwd.as_mut(), Par::Seq, MemStack::new(&mut []));
        let mut tr = rhs.clone();
        pc.transpose_apply_in_place(tr.as_mut(), Par::Seq, MemStack::new(&mut []));
        assert_close(fwd.as_ref(), tr.as_ref(), 1e-12);
    }

    #[test]
    fn refactorize_matches_fresh() {
        let a1 = tridiagonal(7, 4.0, -1.0);
        let a2 = tridiagonal(7, 5.0, -1.5);
        let fresh = Ict::try_new(a2.as_ref()).unwrap();
        let mut reused = Ict::try_new(a1.as_ref()).unwrap();
        reused.refactorize(a2.as_ref()).unwrap();
        assert_eq!(fresh.l_values.len(), reused.l_values.len());
        for (a, b) in fresh.l_values.iter().zip(reused.l_values.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn rejects_non_positive_definite() {
        // Indefinite matrix: negative diagonal.
        let a = mat_to_sparse(&[&[1.0, 2.0], &[2.0, 1.0]]);
        assert_eq!(
            Ict::try_new(a.as_ref()).unwrap_err(),
            IctError::NotPositiveDefinite { col: 1 }
        );
    }

    #[test]
    fn rejects_invalid_params() {
        let a = tridiagonal(3, 4.0, -1.0);
        let bad = IctParams {
            drop_tol: -1.0,
            ..Default::default()
        };
        assert_eq!(
            Ict::try_new_with_params(a.as_ref(), bad).unwrap_err(),
            IctError::InvalidDropTol
        );
    }

    fn mat_to_sparse(rows: &[&[f64]]) -> SparseColMat<usize, f64> {
        let n = rows.len();
        let mut triplets = Vec::new();
        for (i, row) in rows.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                if v != 0.0 {
                    triplets.push(Triplet::new(i, j, v));
                }
            }
        }
        SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap()
    }
}
