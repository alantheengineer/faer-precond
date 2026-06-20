//! Eigenvalue-bound estimation for Hermitian positive-definite operators.
//!
//! Chebyshev acceleration needs an interval `[lambda_min, lambda_max]` that
//! brackets the spectrum of the (assumed Hermitian PD) operator. Two cheap
//! estimators live here:
//!
//! - [`gershgorin_bounds`] — closed form from the matrix structure. `hi` is a
//!   guaranteed upper bound on the spectrum; `lo` is a guaranteed lower bound
//!   but is often loose (and can be non-positive for, e.g., a Laplacian).
//! - [`power_iteration_max`] — a handful of matvecs giving a tight estimate of
//!   the largest eigenvalue.
//!
//! Both run at build time, where allocation is permitted. Estimating
//! `lambda_min` accurately is the hard part (it needs inverse iteration or
//! Lanczos); callers that need precision should pass bounds manually.

use faer::sparse::SparseColMatRef;
use faer::sparse::linalg::matmul::sparse_dense_matmul;
use faer::{Accum, Mat, MatMut, MatRef, Par};
use faer_traits::math_utils::{abs, add, conj, from_f64, mul, mul_real, one, real, recip, sqrt, sub, zero};
use faer_traits::{ComplexField, Index};

/// Gershgorin spectral interval `(lo, hi)` for a Hermitian matrix `A`.
///
/// For a Hermitian matrix every eigenvalue lies in `[min_i(d_i - R_i),
/// max_i(d_i + R_i)]`, where `d_i = Re(A[i,i])` and `R_i` is the sum of
/// off-diagonal magnitudes in row `i`. `hi` is therefore a safe upper bound and
/// `lo` a safe lower bound (`lo` may be `<= 0`).
pub(crate) fn gershgorin_bounds<I: Index, T: ComplexField>(
    a: SparseColMatRef<'_, I, T>,
) -> (T::Real, T::Real) {
    let n = a.nrows();
    let mut diag = vec![zero::<T::Real>(); n];
    let mut radius = vec![zero::<T::Real>(); n];

    for j in 0..n {
        let rows = a.symbolic().row_idx_of_col_raw(j);
        let vals = a.val_of_col(j);
        for (raw, v) in rows.iter().zip(vals.iter()) {
            let i = raw.zx();
            if i == j {
                diag[i] = real(v);
            } else {
                // Off-diagonal A[i,j] contributes |A[i,j]| to row i's radius.
                radius[i] = add(&radius[i], &abs(v));
            }
        }
    }

    if n == 0 {
        return (zero::<T::Real>(), zero::<T::Real>());
    }
    let mut lo = sub(&diag[0], &radius[0]);
    let mut hi = add(&diag[0], &radius[0]);
    for i in 1..n {
        let disc_lo = sub(&diag[i], &radius[i]);
        let disc_hi = add(&diag[i], &radius[i]);
        if disc_lo < lo {
            lo = disc_lo;
        }
        if disc_hi > hi {
            hi = disc_hi;
        }
    }
    (lo, hi)
}

/// Largest eigenvalue (by magnitude) estimate via `iters` steps of power
/// iteration. For Hermitian PD `A` this converges to `lambda_max`.
pub(crate) fn power_iteration_max<I: Index, T: ComplexField>(
    a: SparseColMatRef<'_, I, T>,
    iters: usize,
) -> T::Real {
    let n = a.nrows();
    if n == 0 {
        return zero::<T::Real>();
    }

    // Deterministic non-uniform start so we are not orthogonal to the dominant
    // eigenvector by accident (a constant vector can be unlucky).
    let mut x = Mat::<T>::from_fn(n, 1, |i, _| from_f64::<T>(1.0 + (i % 7) as f64 * 0.5));
    normalize(x.as_mut());
    let mut y = Mat::<T>::zeros(n, 1);

    let mut lambda = zero::<T::Real>();
    for _ in 0..iters {
        sparse_dense_matmul(y.as_mut(), Accum::Replace, a, x.as_ref(), one::<T>(), Par::Seq);
        // Rayleigh quotient x^H (A x); x is unit-norm so the denominator is 1.
        lambda = real_dot(x.as_ref(), y.as_ref());
        let nrm = sqrt::<T::Real>(&real_dot(y.as_ref(), y.as_ref()));
        if nrm == zero::<T::Real>() {
            break;
        }
        let inv = recip(&nrm);
        for i in 0..n {
            let scaled = mul_real(y.as_ref().get(i, 0), &inv);
            *x.as_mut().get_mut(i, 0) = scaled;
        }
    }
    abs(&lambda)
}

/// `Re(x^H y)` for single-column matrices.
fn real_dot<T: ComplexField>(x: MatRef<'_, T>, y: MatRef<'_, T>) -> T::Real {
    let mut acc = zero::<T::Real>();
    for i in 0..x.nrows() {
        acc = add(&acc, &real(&mul(&conj(x.get(i, 0)), y.get(i, 0))));
    }
    acc
}

fn normalize<T: ComplexField>(mut x: MatMut<'_, T>) {
    let nrm = sqrt::<T::Real>(&real_dot(x.as_ref(), x.as_ref()));
    if nrm == zero::<T::Real>() {
        return;
    }
    let inv = recip(&nrm);
    for i in 0..x.nrows() {
        let scaled = mul_real(x.as_ref().get(i, 0), &inv);
        *x.as_mut().get_mut(i, 0) = scaled;
    }
}
