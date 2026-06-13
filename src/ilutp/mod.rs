//! Threshold ILU with partial pivoting (ILUTP).
//!
//! ILUTP is the general-purpose workhorse for nonsymmetric sparse systems
//! solved with GMRES or BiCGSTAB. Where [`crate::Ilu0`] is locked to `A`'s
//! sparsity pattern, ILUTP builds its factors *adaptively*: it keeps the most
//! significant fill each row produces (governed by a drop tolerance and a fill
//! budget) and pivots columns for numerical stability. That makes it far more
//! broadly effective — including on matrices with small or absent diagonal
//! entries, which it handles by pivoting a larger entry into place.
//!
//! It is the dual-threshold ILUT of Saad with column partial pivoting (the
//! SPARSKIT `ilutp` algorithm). Tune it through [`IlutpParams`]:
//!
//! - `drop_tol` — relative threshold; an entry is dropped when its magnitude
//!   falls below `drop_tol * ||row||`.
//! - `fill` — how many entries to keep per row (see [`FillControl`]).
//! - `pivot_tol` — in `[0, 1]`; how eagerly to pivot. `0` disables pivoting and
//!   reduces the method to plain ILUT.
//!
//! # When to use it
//!
//! Reach for ILUTP on general nonsymmetric problems where [`crate::Ilu0`]
//! stalls — convection-dominated flows, badly-scaled or indefinite operators,
//! anything where the fixed-pattern factor is too weak. For symmetric
//! positive-definite systems prefer [`crate::Ic0`], and for cheap, robust
//! cases [`crate::Ilu0`] is lighter.
//!
//! # Value-dependent pattern
//!
//! ILUTP's fill pattern depends on the matrix *values*, not just its sparsity,
//! so (unlike ILU(0)/IC(0)) there is no separate symbolic phase and
//! [`Ilutp::refactorize`] reuses buffer capacity but is **not** allocation-free.
//! The hot `apply` path stays allocation-free in the usual sense: its scratch
//! flows through the caller's `MemStack` (a single length-`n` column, used for
//! the permutation — analogous to [`crate::BlockJacobiPrecond`]).
//!
//! # Example
//!
//! ```
//! use dyn_stack::{MemBuffer, MemStack};
//! use faer::sparse::{SparseColMat, Triplet};
//! use faer::matrix_free::{LinOp, Precond};
//! use faer::{mat, Par};
//! use faer_precond::{Ilutp, IlutpParams};
//!
//! // A small nonsymmetric tridiagonal: diag 4, sub -2, super -1.
//! let mut triplets = Vec::new();
//! for i in 0..6usize {
//!     triplets.push(Triplet::new(i, i, 4.0_f64));
//!     if i > 0 {
//!         triplets.push(Triplet::new(i, i - 1, -2.0));
//!         triplets.push(Triplet::new(i - 1, i, -1.0));
//!     }
//! }
//! let a = SparseColMat::<usize, f64>::try_new_from_triplets(6, 6, &triplets).unwrap();
//!
//! let pc = Ilutp::try_new_with_params(a.as_ref(), IlutpParams::default()).unwrap();
//!
//! let mut b = mat![[1.0_f64], [0.0], [0.0], [0.0], [0.0], [0.0]];
//! let mut buf = MemBuffer::new(pc.apply_in_place_scratch(1, Par::Seq));
//! pc.apply_in_place(b.as_mut(), Par::Seq, MemStack::new(&mut buf));
//! ```
//!
//! # Storage
//!
//! The factors are stored in column-compressed (CSC) form following faer's
//! triangular-solve conventions — `L`'s unit diagonal first in each column,
//! `U`'s diagonal last — plus the column permutation discovered during
//! pivoting. Factorisation runs row-by-row (CSR) internally and transposes once
//! into this layout.

use core::fmt::Debug;

use dyn_stack::{MemStack, StackReq};
use faer::matrix_free::{BiLinOp, BiPrecond, LinOp, Precond};
use faer::{Conj, MatMut, MatRef, Par};
use faer_traits::{ComplexField, Index};

pub mod apply;
pub mod numeric;

pub use numeric::Ilutp;

/// How the per-row fill budget is chosen for each triangular part.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FillControl {
    /// Keep at most `p` off-diagonal entries per row in `L` and `p` in `U`
    /// (SPARSKIT `lfil` semantics); the diagonal is always kept.
    PerRow(usize),
    /// Derive the per-row budget as `round(factor * nnz(A) / n)`, i.e. a
    /// multiple of `A`'s average number of nonzeros per row.
    Factor(f64),
}

/// Row norm used to scale the relative drop tolerance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowNorm {
    /// `||row||_1` — sum of magnitudes.
    One,
    /// `||row||_2` — Euclidean norm.
    Two,
}

/// Tuning parameters for [`Ilutp`].
///
/// Construct with [`IlutpParams::default`] and override fields as needed:
///
/// ```
/// use faer_precond::{FillControl, IlutpParams, RowNorm};
/// let params = IlutpParams {
///     drop_tol: 1e-2,
///     fill: FillControl::PerRow(20),
///     pivot_tol: 0.5,
///     ..Default::default()
/// };
/// # let _ = (params.norm, RowNorm::Two);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IlutpParams {
    /// Relative drop tolerance. An entry is dropped when its magnitude is below
    /// `drop_tol * ||row||`. Default `1e-3`.
    pub drop_tol: f64,
    /// Per-row fill budget. Default `FillControl::Factor(5.0)`.
    pub fill: FillControl,
    /// Column pivot tolerance in `[0, 1]`; `0` disables pivoting. Default `0.1`.
    pub pivot_tol: f64,
    /// Norm used to scale `drop_tol`. Default `RowNorm::Two`.
    pub norm: RowNorm,
}

impl Default for IlutpParams {
    fn default() -> Self {
        Self {
            drop_tol: 1e-3,
            fill: FillControl::Factor(5.0),
            pivot_tol: 0.1,
            norm: RowNorm::Two,
        }
    }
}

impl IlutpParams {
    pub(crate) fn validate(&self) -> Result<(), IlutpError> {
        if !self.drop_tol.is_finite() || self.drop_tol < 0.0 {
            return Err(IlutpError::InvalidDropTol);
        }
        if !self.pivot_tol.is_finite() || self.pivot_tol < 0.0 || self.pivot_tol > 1.0 {
            return Err(IlutpError::InvalidPivotTol);
        }
        if let FillControl::Factor(f) = self.fill
            && (!f.is_finite() || f <= 0.0)
        {
            return Err(IlutpError::InvalidFillControl);
        }
        Ok(())
    }
}

/// Error returned by ILUTP construction or refactorisation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IlutpError {
    /// The source matrix was not square.
    NonSquareMatrix { nrows: usize, ncols: usize },
    /// Row `row` reduced to a zero pivot even after pivoting — the matrix is
    /// numerically singular for these parameters.
    ZeroPivot { row: usize },
    /// `drop_tol` was negative or NaN.
    InvalidDropTol,
    /// `pivot_tol` was outside `[0, 1]` or NaN.
    InvalidPivotTol,
    /// `FillControl::Factor` was non-positive or NaN.
    InvalidFillControl,
}

impl core::fmt::Display for IlutpError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonSquareMatrix { nrows, ncols } => {
                write!(f, "matrix must be square but is {nrows}x{ncols}")
            }
            Self::ZeroPivot { row } => write!(f, "encountered a zero pivot at row {row}"),
            Self::InvalidDropTol => f.write_str("drop_tol must be finite and non-negative"),
            Self::InvalidPivotTol => f.write_str("pivot_tol must be finite and within [0, 1]"),
            Self::InvalidFillControl => f.write_str("fill factor must be finite and positive"),
        }
    }
}

impl core::error::Error for IlutpError {}

impl<I, T> LinOp<T> for Ilutp<I, T>
where
    I: Index,
    T: ComplexField + Debug + Sync,
{
    fn apply_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::new::<T>(self.dim())
    }

    fn nrows(&self) -> usize {
        self.dim()
    }

    fn ncols(&self) -> usize {
        self.dim()
    }

    fn apply(&self, mut out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
        assert_eq!(
            out.nrows(),
            self.dim(),
            "out row count must match dimension"
        );
        assert_eq!(
            rhs.nrows(),
            self.dim(),
            "rhs row count must match dimension"
        );
        assert_eq!(out.ncols(), rhs.ncols(), "out and rhs ncols must match");
        out.copy_from(rhs);
        apply::solve_in_place(self, Conj::No, out, par, stack);
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
            self.dim(),
            "out row count must match dimension"
        );
        assert_eq!(
            rhs.nrows(),
            self.dim(),
            "rhs row count must match dimension"
        );
        assert_eq!(out.ncols(), rhs.ncols(), "out and rhs ncols must match");
        out.copy_from(rhs);
        apply::solve_in_place(self, Conj::Yes, out, par, stack);
    }
}

impl<I, T> Precond<T> for Ilutp<I, T>
where
    I: Index,
    T: ComplexField + Debug + Sync,
{
    fn apply_in_place_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::new::<T>(self.dim())
    }

    fn apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
        apply::solve_in_place(self, Conj::No, rhs, par, stack);
    }

    fn conj_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
        apply::solve_in_place(self, Conj::Yes, rhs, par, stack);
    }
}

impl<I, T> BiLinOp<T> for Ilutp<I, T>
where
    I: Index,
    T: ComplexField + Debug + Sync,
{
    fn transpose_apply_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::new::<T>(self.dim())
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
            self.dim(),
            "out row count must match dimension"
        );
        assert_eq!(
            rhs.nrows(),
            self.dim(),
            "rhs row count must match dimension"
        );
        assert_eq!(out.ncols(), rhs.ncols(), "out and rhs ncols must match");
        out.copy_from(rhs);
        apply::solve_transpose_in_place(self, Conj::No, out, par, stack);
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
            self.dim(),
            "out row count must match dimension"
        );
        assert_eq!(
            rhs.nrows(),
            self.dim(),
            "rhs row count must match dimension"
        );
        assert_eq!(out.ncols(), rhs.ncols(), "out and rhs ncols must match");
        out.copy_from(rhs);
        apply::solve_transpose_in_place(self, Conj::Yes, out, par, stack);
    }
}

impl<I, T> BiPrecond<T> for Ilutp<I, T>
where
    I: Index,
    T: ComplexField + Debug + Sync,
{
    fn transpose_apply_in_place_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::new::<T>(self.dim())
    }

    fn transpose_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
        apply::solve_transpose_in_place(self, Conj::No, rhs, par, stack);
    }

    fn adjoint_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
        apply::solve_transpose_in_place(self, Conj::Yes, rhs, par, stack);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::MaybeUninit;
    use faer::sparse::{SparseColMat, SparseColMatRef, Triplet};
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

    /// Nonsymmetric tridiagonal: `diag` on the diagonal, `sub` below, `sup` above.
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

    /// 5-point Laplacian on a `grid x grid` mesh with an added asymmetric
    /// first-derivative ("advection") term, giving a nonsymmetric M-matrix.
    fn advection_diffusion_2d(grid: usize, beta: f64) -> SparseColMat<usize, f64> {
        let n = grid * grid;
        let mut triplets = Vec::new();
        for gy in 0..grid {
            for gx in 0..grid {
                let idx = gy * grid + gx;
                triplets.push(Triplet::new(idx, idx, 4.0));
                if gx > 0 {
                    triplets.push(Triplet::new(idx, idx - 1, -1.0 - beta));
                }
                if gx + 1 < grid {
                    triplets.push(Triplet::new(idx, idx + 1, -1.0 + beta));
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

    fn apply_inplace(pc: &Ilutp<usize, f64>, rhs: &mut Mat<f64>) {
        with_stack(pc.apply_in_place_scratch(rhs.ncols(), Par::Seq), |stack| {
            pc.apply_in_place(rhs.as_mut(), Par::Seq, stack);
        });
    }

    /// Parameters that keep everything: ILUTP reduces to a complete LU.
    fn exact_params(n: usize, pivot_tol: f64) -> IlutpParams {
        IlutpParams {
            drop_tol: 0.0,
            fill: FillControl::PerRow(n),
            pivot_tol,
            norm: RowNorm::Two,
        }
    }

    #[test]
    fn exact_keep_inverts_system() {
        // drop_tol = 0, full fill, no pivoting => exact LU => M^{-1} A x == x.
        let a = tridiagonal(8, 4.0, -2.0, -1.0);
        let pc = Ilutp::try_new_with_params(a.as_ref(), exact_params(8, 0.0)).unwrap();
        assert!(!pc.is_permuted());

        let a_dense = to_dense(&a);
        let x_true = mat![
            [1.0],
            [-2.0],
            [3.0],
            [-1.0],
            [0.5],
            [2.0],
            [-3.0],
            [1.5_f64]
        ];
        let mut rhs = (&a_dense * &x_true).to_owned();
        apply_inplace(&pc, &mut rhs);
        assert_close(rhs.as_ref(), x_true.as_ref(), 1e-10);
    }

    #[test]
    fn exact_keep_with_pivoting_still_inverts() {
        // Even when pivoting reorders columns, full-keep is an exact LU of A P,
        // so M^{-1} A == I still holds.
        let a = tridiagonal(8, 4.0, -2.0, -1.0);
        let pc = Ilutp::try_new_with_params(a.as_ref(), exact_params(8, 0.5)).unwrap();

        let a_dense = to_dense(&a);
        let x_true = mat![
            [1.0],
            [-2.0],
            [3.0],
            [-1.0],
            [0.5],
            [2.0],
            [-3.0],
            [1.5_f64]
        ];
        let mut rhs = (&a_dense * &x_true).to_owned();
        apply_inplace(&pc, &mut rhs);
        assert_close(rhs.as_ref(), x_true.as_ref(), 1e-10);
    }

    #[test]
    fn reconstruction_matches_permuted_a() {
        // With no dropping and no pivoting, L * U == A exactly.
        let a = advection_diffusion_2d(4, 0.3);
        let n = a.nrows();
        let pc = Ilutp::try_new_with_params(a.as_ref(), exact_params(n, 0.0)).unwrap();
        assert!(!pc.is_permuted());

        let l = sparse_view_to_dense(pc.l_view());
        let u = sparse_view_to_dense(pc.u_view());
        let lu = &l * &u;
        let a_dense = to_dense(&a);
        assert_close(lu.as_ref(), a_dense.as_ref(), 1e-9);
    }

    #[test]
    fn reconstruction_matches_permuted_a_with_pivoting() {
        // With pivoting, L * U == A * P where (A P)[:,k] = A[:, perm[k]].
        let a = advection_diffusion_2d(4, 0.4);
        let n = a.nrows();
        let pc = Ilutp::try_new_with_params(a.as_ref(), exact_params(n, 0.5)).unwrap();

        let l = sparse_view_to_dense(pc.l_view());
        let u = sparse_view_to_dense(pc.u_view());
        let lu = &l * &u;

        let a_dense = to_dense(&a);
        let perm = pc.perm();
        let mut ap = Mat::<f64>::zeros(n, n);
        for (k, &pk) in perm.iter().enumerate() {
            for r in 0..n {
                *ap.as_mut().get_mut(r, k) = *a_dense.as_ref().get(r, pk);
            }
        }
        assert_close(lu.as_ref(), ap.as_ref(), 1e-9);
    }

    #[test]
    fn pivot_tol_zero_is_pure_ilut() {
        // A row with a tiny diagonal would normally pivot; pivot_tol = 0 forbids it.
        let a = mat_to_sparse(&[&[1e-8, 1.0, 0.0], &[1.0, 1.0, 1.0], &[0.0, 1.0, 1.0]]);
        let params = IlutpParams {
            pivot_tol: 0.0,
            ..exact_params(3, 0.0)
        };
        let pc = Ilutp::try_new_with_params(a.as_ref(), params).unwrap();
        assert!(!pc.is_permuted());
    }

    #[test]
    fn pivoting_triggers_on_tiny_diagonal() {
        let a = mat_to_sparse(&[&[1e-8, 1.0, 0.0], &[1.0, 1.0, 1.0], &[0.0, 1.0, 1.0]]);
        let pc = Ilutp::try_new_with_params(a.as_ref(), exact_params(3, 0.5)).unwrap();
        assert!(pc.is_permuted(), "tiny diagonal should force a pivot");

        // The pivot magnitude should be O(1), not O(1e-8).
        let u = sparse_view_to_dense(pc.u_view());
        assert!(
            u.as_ref().get(0, 0).abs() > 1e-3,
            "pivoting should avoid the tiny pivot, got {}",
            u.as_ref().get(0, 0)
        );
    }

    #[test]
    fn reduces_residual_on_nonsymmetric_problem() {
        let a = advection_diffusion_2d(8, 0.5);
        let n = a.nrows();
        let pc = Ilutp::try_new(a.as_ref()).unwrap();
        let a_dense = to_dense(&a);

        let b = Mat::<f64>::from_fn(n, 1, |i, _| (i % 7) as f64 - 3.0);
        let mut x = b.clone();
        apply_inplace(&pc, &mut x);

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
            "ILUTP residual ratio {r_norm}/{b_norm} too large"
        );
    }

    #[test]
    fn transpose_apply_inverts_transposed_system() {
        let a = tridiagonal(6, 4.0, -2.0, -1.0);
        let pc = Ilutp::try_new_with_params(a.as_ref(), exact_params(6, 0.0)).unwrap();
        let a_dense = to_dense(&a);

        let x_true = mat![[1.0], [-2.0], [3.0], [-1.0], [0.5], [2.0_f64]];
        let rhs = a_dense.transpose() * &x_true;

        let mut out = rhs.clone();
        with_stack(pc.transpose_apply_in_place_scratch(1, Par::Seq), |stack| {
            pc.transpose_apply_in_place(out.as_mut(), Par::Seq, stack);
        });
        assert_close(out.as_ref(), x_true.as_ref(), 1e-10);
    }

    #[test]
    fn out_of_place_matches_in_place() {
        let a = advection_diffusion_2d(5, 0.3);
        let n = a.nrows();
        let pc = Ilutp::try_new(a.as_ref()).unwrap();

        let rhs = Mat::<f64>::from_fn(n, 2, |i, j| ((i + 3 * j) % 11) as f64 - 4.0);

        let mut out = Mat::<f64>::zeros(n, 2);
        with_stack(pc.apply_scratch(2, Par::Seq), |stack| {
            pc.apply(out.as_mut(), rhs.as_ref(), Par::Seq, stack);
        });

        let mut inplace = rhs.clone();
        apply_inplace(&pc, &mut inplace);

        assert_close(out.as_ref(), inplace.as_ref(), 1e-12);
    }

    #[test]
    fn refactorize_matches_fresh_construction() {
        // Deterministic pivoting (pivot_tol = 0) so both paths choose identically.
        let a1 = tridiagonal(7, 4.0, -2.0, -1.0);
        let a2 = tridiagonal(7, 5.0, -1.0, -2.0);
        let params = IlutpParams {
            pivot_tol: 0.0,
            ..IlutpParams::default()
        };

        let fresh = Ilutp::try_new_with_params(a2.as_ref(), params).unwrap();

        let mut reused = Ilutp::try_new_with_params(a1.as_ref(), params).unwrap();
        reused.refactorize(a2.as_ref()).unwrap();

        let lf = sparse_view_to_dense(fresh.l_view());
        let lr = sparse_view_to_dense(reused.l_view());
        let uf = sparse_view_to_dense(fresh.u_view());
        let ur = sparse_view_to_dense(reused.u_view());
        assert_close(lf.as_ref(), lr.as_ref(), 1e-14);
        assert_close(uf.as_ref(), ur.as_ref(), 1e-14);
    }

    #[test]
    fn rejects_non_square() {
        let mut triplets = Vec::new();
        for i in 0..3 {
            triplets.push(Triplet::new(i, i, 1.0));
        }
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(3, 4, &triplets).unwrap();
        let err = Ilutp::try_new(a.as_ref()).unwrap_err();
        assert_eq!(err, IlutpError::NonSquareMatrix { nrows: 3, ncols: 4 });
    }

    #[test]
    fn rejects_zero_pivot_without_pivoting() {
        // Structurally singular row 0 (zero diagonal, single off-diagonal) with
        // pivoting disabled must fail with ZeroPivot.
        let a = mat_to_sparse(&[&[0.0, 0.0, 0.0], &[1.0, 1.0, 0.0], &[0.0, 1.0, 1.0]]);
        let params = IlutpParams {
            pivot_tol: 0.0,
            ..exact_params(3, 0.0)
        };
        let err = Ilutp::try_new_with_params(a.as_ref(), params).unwrap_err();
        assert_eq!(err, IlutpError::ZeroPivot { row: 0 });
    }

    #[test]
    fn rejects_invalid_params() {
        let a = tridiagonal(3, 4.0, -1.0, -1.0);
        let bad_drop = IlutpParams {
            drop_tol: -1.0,
            ..Default::default()
        };
        assert_eq!(
            Ilutp::try_new_with_params(a.as_ref(), bad_drop).unwrap_err(),
            IlutpError::InvalidDropTol
        );
        let bad_pivot = IlutpParams {
            pivot_tol: 1.5,
            ..Default::default()
        };
        assert_eq!(
            Ilutp::try_new_with_params(a.as_ref(), bad_pivot).unwrap_err(),
            IlutpError::InvalidPivotTol
        );
        let bad_fill = IlutpParams {
            fill: FillControl::Factor(0.0),
            ..Default::default()
        };
        assert_eq!(
            Ilutp::try_new_with_params(a.as_ref(), bad_fill).unwrap_err(),
            IlutpError::InvalidFillControl
        );
    }

    /// Build a CSC matrix from a dense row-major description (test helper).
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
