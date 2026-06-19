//! Polynomial preconditioner (Neumann series and Chebyshev).
//!
//! A polynomial preconditioner approximates `A^{-1}` by a polynomial in `A`:
//! `M^{-1} = p(A)`. Applying it is nothing but a handful of sparse
//! matrix-vector products and vector updates — there are **no triangular
//! solves**. That makes it the odd one out in this crate: where ILU/IC apply a
//! sequential forward/back substitution, a polynomial preconditioner is built
//! entirely from `A * x`, which parallelises and vectorises freely.
//!
//! Two flavours are provided:
//!
//! - **Neumann series** — `M^{-1} = w * sum_{k=0}^{d} (I - w A)^k`. One real
//!   damping parameter `w`; converges when `0 < w * lambda < 2` across the
//!   spectrum.
//! - **Chebyshev** — the degree-`d` Chebyshev polynomial that minimises
//!   `max |1 - lambda p(lambda)|` over `[lambda_min, lambda_max]`. Sharper than
//!   Neumann for the same degree, but it needs an estimate of the spectral
//!   interval.
//!
//! # When to use it
//!
//! Reach for a polynomial preconditioner when matrix-vector products are cheap
//! and plentiful but triangular solves are a bottleneck — many cores, a GPU, or
//! a distributed operator where the sequential sweep of an ILU does not scale.
//! It is also the classic choice for a *smoother* inside multigrid. On a single
//! core it rarely beats [`crate::Ic0`] / [`crate::Ilu0`]; its appeal is
//! parallelism and the absence of any factorisation.
//!
//! Chebyshev assumes a Hermitian positive-definite operator and is only as good
//! as its `[lambda_min, lambda_max]` estimate: an over-estimated `lambda_min`
//! (or under-estimated `lambda_max`) degrades or even diverges the polynomial.
//! Pass [`BoundEstimate::Manual`] when you know the spectrum; otherwise
//! [`BoundEstimate::PowerIteration`] gives a tight `lambda_max` and a
//! conservative `lambda_min`.
//!
//! # Storage
//!
//! The operator `A` is stored as an owned CSC copy (apply reads it through a
//! [`faer::sparse::SparseColMatRef`]). The recurrence's
//! temporaries — one work column for Neumann, two for Chebyshev — come from the
//! caller's [`MemStack`], so apply allocates no heap memory.

use core::fmt::Debug;

use dyn_stack::{MemStack, StackReq};
use faer::matrix_free::{BiLinOp, BiPrecond, LinOp, Precond};
use faer::{MatMut, MatRef, Par};
use faer_traits::{ComplexField, Index};

mod apply;
mod build;

/// Which polynomial to use for `M^{-1} = p(A)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PolyKind {
    /// Neumann series with real damping `omega`. `M^{-1} = w sum (I - wA)^k`.
    Neumann { omega: f64 },
    /// Chebyshev polynomial over the spectral interval `[lambda_min, lambda_max]`.
    Chebyshev { lambda_min: f64, lambda_max: f64 },
}

/// How to obtain the spectral interval for [`Poly::try_new_auto`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundEstimate {
    /// Gershgorin discs: a closed-form bracket from the matrix structure.
    /// `lambda_max` is safe; `lambda_min` may be loose.
    Gershgorin,
    /// `iters` steps of power iteration for a tight `lambda_max`; `lambda_min`
    /// falls back to the (conservative) Gershgorin lower bound.
    PowerIteration { iters: usize },
    /// Caller-supplied bounds.
    Manual { lambda_min: f64, lambda_max: f64 },
}

/// Tuning parameters for [`Poly::try_new`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolyParams {
    /// Polynomial degree (number of recurrence steps / matvecs). Must be `>= 1`.
    pub degree: usize,
    /// Which polynomial to build.
    pub kind: PolyKind,
}

/// Error returned by polynomial-preconditioner construction.
#[derive(Debug, Clone, PartialEq)]
pub enum PolyError {
    /// The source matrix was not square.
    NonSquareMatrix { nrows: usize, ncols: usize },
    /// `degree` was zero.
    ZeroDegree,
    /// A Neumann `omega` was non-positive or non-finite.
    InvalidOmega,
    /// The Chebyshev interval was not a usable `0 < lambda_min < lambda_max`.
    InvalidBounds,
    /// A refactorisation was attempted with a mismatched sparsity pattern.
    PatternMismatch,
}

impl core::fmt::Display for PolyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonSquareMatrix { nrows, ncols } => {
                write!(f, "matrix must be square but is {nrows}x{ncols}")
            }
            Self::ZeroDegree => f.write_str("polynomial degree must be at least 1"),
            Self::InvalidOmega => f.write_str("neumann omega must be finite and positive"),
            Self::InvalidBounds => {
                f.write_str("chebyshev bounds must satisfy 0 < lambda_min < lambda_max")
            }
            Self::PatternMismatch => f.write_str("refactorisation pattern does not match"),
        }
    }
}

impl core::error::Error for PolyError {}

/// Resolved polynomial coefficients (real values stored as `T`).
#[derive(Debug, Clone)]
pub(crate) enum Coeffs<T> {
    Neumann { omega: T },
    Chebyshev { lambda_min: T, lambda_max: T },
}

/// Polynomial preconditioner `M^{-1} = p(A)`.
///
/// Stores an owned copy of `A` and the polynomial coefficients. Apply is a
/// sequence of sparse matvecs and vector updates with no triangular solves and
/// no heap allocation. See the [module documentation](self) for guidance.
#[derive(Debug, Clone)]
pub struct Poly<I, T> {
    pub(crate) dim: usize,
    pub(crate) degree: usize,
    pub(crate) a_col_ptr: Vec<I>,
    pub(crate) a_row_idx: Vec<I>,
    pub(crate) a_values: Vec<T>,
    pub(crate) coeffs: Coeffs<T>,
    pub(crate) recompute: Option<BoundEstimate>,
}

impl<I, T> Poly<I, T> {
    /// Dimension `n` of the preconditioner.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Polynomial degree (number of matvecs per apply).
    #[inline]
    pub fn degree(&self) -> usize {
        self.degree
    }
}

impl<I, T> LinOp<T> for Poly<I, T>
where
    I: Index,
    T: ComplexField + Debug + Sync,
{
    fn apply_scratch(&self, rhs_ncols: usize, _par: Par) -> StackReq {
        apply::run_scratch(self, rhs_ncols)
    }

    fn nrows(&self) -> usize {
        self.dim
    }

    fn ncols(&self) -> usize {
        self.dim
    }

    fn apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
        apply::apply_out(self, out, rhs, par, stack);
    }

    fn conj_apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, par: Par, stack: &mut MemStack) {
        // p(A) has real coefficients, so conj(p(A)) = p(conj(A)); we apply the
        // same recurrence and rely on the matvec's own conjugation being absent
        // here — for real operators conj_apply == apply.
        apply::apply_out(self, out, rhs, par, stack);
    }
}

impl<I, T> Precond<T> for Poly<I, T>
where
    I: Index,
    T: ComplexField + Debug + Sync,
{
    fn apply_in_place_scratch(&self, rhs_ncols: usize, _par: Par) -> StackReq {
        apply::inplace_scratch(self, rhs_ncols)
    }

    fn apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
        apply::apply_inplace(self, rhs, par, stack);
    }

    fn conj_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
        apply::apply_inplace(self, rhs, par, stack);
    }
}

impl<I, T> BiLinOp<T> for Poly<I, T>
where
    I: Index,
    T: ComplexField + Debug + Sync,
{
    fn transpose_apply_scratch(&self, rhs_ncols: usize, _par: Par) -> StackReq {
        apply::run_scratch(self, rhs_ncols)
    }

    fn transpose_apply(
        &self,
        out: MatMut<'_, T>,
        rhs: MatRef<'_, T>,
        par: Par,
        stack: &mut MemStack,
    ) {
        // For the Hermitian-PD operators these preconditioners target, p(A) is
        // symmetric, so the transpose coincides with the forward apply.
        apply::apply_out(self, out, rhs, par, stack);
    }

    fn adjoint_apply(
        &self,
        out: MatMut<'_, T>,
        rhs: MatRef<'_, T>,
        par: Par,
        stack: &mut MemStack,
    ) {
        apply::apply_out(self, out, rhs, par, stack);
    }
}

impl<I, T> BiPrecond<T> for Poly<I, T>
where
    I: Index,
    T: ComplexField + Debug + Sync,
{
    fn transpose_apply_in_place_scratch(&self, rhs_ncols: usize, _par: Par) -> StackReq {
        apply::inplace_scratch(self, rhs_ncols)
    }

    fn transpose_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
        apply::apply_inplace(self, rhs, par, stack);
    }

    fn adjoint_apply_in_place(&self, rhs: MatMut<'_, T>, par: Par, stack: &mut MemStack) {
        apply::apply_inplace(self, rhs, par, stack);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::MaybeUninit;
    use faer::sparse::{SparseColMat, Triplet};
    use faer::{Mat, MatRef};

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

    fn apply_inplace(pc: &Poly<usize, f64>, rhs: &mut Mat<f64>) {
        with_stack(pc.apply_in_place_scratch(rhs.ncols(), Par::Seq), |stack| {
            pc.apply_in_place(rhs.as_mut(), Par::Seq, stack);
        });
    }

    fn residual_ratio(a: &SparseColMat<usize, f64>, pc: &Poly<usize, f64>, b: &Mat<f64>) -> f64 {
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
    fn neumann_higher_degree_reduces_residual() {
        let a = tridiagonal(20, 2.5, -1.0);
        let b = Mat::<f64>::from_fn(20, 1, |i, _| (i % 5) as f64 - 2.0);
        // omega chosen below 2/lambda_max (lambda_max < 4.5 here).
        let mk = |deg| {
            Poly::try_new(
                a.as_ref(),
                PolyParams {
                    degree: deg,
                    kind: PolyKind::Neumann { omega: 0.4 },
                },
            )
            .unwrap()
        };
        let r_low = residual_ratio(&a, &mk(2), &b);
        let r_high = residual_ratio(&a, &mk(12), &b);
        assert!(
            r_high < r_low,
            "higher Neumann degree should reduce residual: {r_high} !< {r_low}"
        );
    }

    #[test]
    fn chebyshev_reduces_residual_with_exact_bounds() {
        // 5-point Laplacian eigenvalues: 4 - 2cos(p pi h) - 2cos(q pi h),
        // h = 1/(grid+1). Use the exact extremes as bounds.
        let grid = 8;
        let a = laplacian_2d(grid);
        let n = a.nrows();
        let h = std::f64::consts::PI / (grid as f64 + 1.0);
        let lam_min = 4.0 - 4.0 * (h).cos();
        let lam_max = 4.0 - 4.0 * (grid as f64 * h).cos();
        let pc = Poly::try_new(
            a.as_ref(),
            PolyParams {
                degree: 8,
                kind: PolyKind::Chebyshev {
                    lambda_min: lam_min,
                    lambda_max: lam_max,
                },
            },
        )
        .unwrap();
        let b = Mat::<f64>::from_fn(n, 1, |i, _| (i % 7) as f64 - 3.0);
        let ratio = residual_ratio(&a, &pc, &b);
        assert!(ratio < 0.5, "Chebyshev residual ratio {ratio} too large");
    }

    #[test]
    fn out_of_place_matches_in_place() {
        let a = tridiagonal(12, 3.0, -1.0);
        let pc = Poly::try_new(
            a.as_ref(),
            PolyParams {
                degree: 5,
                kind: PolyKind::Chebyshev {
                    lambda_min: 1.0,
                    lambda_max: 5.0,
                },
            },
        )
        .unwrap();
        let rhs = Mat::<f64>::from_fn(12, 2, |i, j| ((i + 3 * j) % 7) as f64 - 3.0);

        let mut out = Mat::<f64>::zeros(12, 2);
        with_stack(pc.apply_scratch(2, Par::Seq), |stack| {
            pc.apply(out.as_mut(), rhs.as_ref(), Par::Seq, stack);
        });
        let mut inplace = rhs.clone();
        apply_inplace(&pc, &mut inplace);
        assert_close(out.as_ref(), inplace.as_ref(), 1e-12);
    }

    #[test]
    fn auto_power_iteration_builds_and_helps() {
        let a = laplacian_2d(8);
        let n = a.nrows();
        let pc = Poly::try_new_auto(a.as_ref(), 6, BoundEstimate::PowerIteration { iters: 30 })
            .unwrap();
        let b = Mat::<f64>::from_fn(n, 1, |i, _| (i % 7) as f64 - 3.0);
        let ratio = residual_ratio(&a, &pc, &b);
        assert!(ratio < 1.0, "auto Chebyshev should not diverge: {ratio}");
    }

    #[test]
    fn refactorize_updates_values() {
        let a1 = tridiagonal(10, 3.0, -1.0);
        let a2 = tridiagonal(10, 4.0, -1.5);
        let params = PolyParams {
            degree: 4,
            kind: PolyKind::Neumann { omega: 0.3 },
        };
        let fresh = Poly::try_new(a2.as_ref(), params).unwrap();
        let mut reused = Poly::try_new(a1.as_ref(), params).unwrap();
        reused.refactorize(a2.as_ref()).unwrap();
        assert_eq!(fresh.a_values, reused.a_values);
    }

    #[test]
    fn rejects_bad_params() {
        let a = tridiagonal(4, 3.0, -1.0);
        assert_eq!(
            Poly::try_new(
                a.as_ref(),
                PolyParams {
                    degree: 0,
                    kind: PolyKind::Neumann { omega: 0.3 }
                }
            )
            .unwrap_err(),
            PolyError::ZeroDegree
        );
        assert_eq!(
            Poly::try_new(
                a.as_ref(),
                PolyParams {
                    degree: 3,
                    kind: PolyKind::Neumann { omega: -1.0 }
                }
            )
            .unwrap_err(),
            PolyError::InvalidOmega
        );
        assert_eq!(
            Poly::try_new(
                a.as_ref(),
                PolyParams {
                    degree: 3,
                    kind: PolyKind::Chebyshev {
                        lambda_min: 2.0,
                        lambda_max: 1.0
                    }
                }
            )
            .unwrap_err(),
            PolyError::InvalidBounds
        );
    }
}
