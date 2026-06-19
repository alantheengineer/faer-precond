//! Matvec-only application of the polynomial preconditioner.
//!
//! Both forms evaluate `y = p(A) b` using only sparse matrix-vector products
//! (`sparse_dense_matmul`) and element-wise vector updates. No triangular
//! solves, and every temporary comes from the caller's [`MemStack`].

use dyn_stack::{MemStack, StackReq};
use faer::sparse::linalg::matmul::sparse_dense_matmul;
use faer::{Accum, MatMut, MatRef, Par};
use faer_traits::math_utils::{add, copy, div, from_f64, mul, mul_real, one, real, recip, sub, zero};
use faer_traits::{ComplexField, Index};

use super::{Coeffs, Poly};

/// Number of length-`n*ncols` work buffers the in-recurrence step needs.
fn run_buffers<I, T>(poly: &Poly<I, T>) -> usize {
    match poly.coeffs {
        Coeffs::Neumann { .. } => 1, // A*acc
        Coeffs::Chebyshev { .. } => 2, // direction + residual
    }
}

/// Scratch for the out-of-place [`crate::poly::Poly`] apply (the accumulator is
/// the caller's `out`).
pub(crate) fn run_scratch<I, T: ComplexField>(poly: &Poly<I, T>, ncols: usize) -> StackReq {
    StackReq::new::<T>(run_buffers(poly) * poly.dim * ncols)
}

/// Scratch for the in-place apply: one extra buffer to hold a read-only copy of
/// the right-hand side while the result is written back into it.
pub(crate) fn inplace_scratch<I, T: ComplexField>(poly: &Poly<I, T>, ncols: usize) -> StackReq {
    StackReq::new::<T>(poly.dim * ncols).and(run_scratch(poly, ncols))
}

/// Out-of-place apply: `out = p(A) rhs`.
pub(crate) fn apply_out<I, T>(
    poly: &Poly<I, T>,
    out: MatMut<'_, T>,
    rhs: MatRef<'_, T>,
    par: Par,
    stack: &mut MemStack,
) where
    I: Index,
    T: ComplexField,
{
    run(poly, rhs, out, par, stack);
}

/// In-place apply: `rhs = p(A) rhs`.
pub(crate) fn apply_inplace<I, T>(
    poly: &Poly<I, T>,
    mut rhs: MatMut<'_, T>,
    par: Par,
    stack: &mut MemStack,
) where
    I: Index,
    T: ComplexField,
{
    let n = rhs.nrows();
    let ncols = rhs.ncols();
    let (mut b_buf, stack) = stack.make_with::<T>(n * ncols, |_| zero::<T>());
    let mut b = MatMut::from_column_major_slice_mut(&mut b_buf[..], n, ncols);
    b.copy_from(rhs.as_ref());
    run(poly, b.as_ref(), rhs.as_mut(), par, stack);
}

/// Evaluate `x = p(A) b`. `b` is read-only; `x` is the accumulator/output.
fn run<I, T>(poly: &Poly<I, T>, b: MatRef<'_, T>, x: MatMut<'_, T>, par: Par, stack: &mut MemStack)
where
    I: Index,
    T: ComplexField,
{
    match &poly.coeffs {
        Coeffs::Neumann { omega } => run_neumann(poly, b, x, omega, par, stack),
        Coeffs::Chebyshev {
            lambda_min,
            lambda_max,
        } => run_chebyshev(poly, b, x, lambda_min, lambda_max, par, stack),
    }
}

/// Neumann series `p(A) = w * sum_{k=0}^{deg} (I - w A)^k`, evaluated by Horner.
fn run_neumann<I, T>(
    poly: &Poly<I, T>,
    b: MatRef<'_, T>,
    mut x: MatMut<'_, T>,
    omega: &T,
    par: Par,
    stack: &mut MemStack,
) where
    I: Index,
    T: ComplexField,
{
    let n = x.nrows();
    let ncols = x.ncols();
    let (mut vbuf, _) = stack.make_with::<T>(n * ncols, |_| zero::<T>());
    let mut v = MatMut::from_column_major_slice_mut(&mut vbuf[..], n, ncols);
    let a = poly.a_view();

    // acc_0 = b
    x.copy_from(b);
    for _ in 0..poly.degree {
        // v = A * acc
        sparse_dense_matmul(v.as_mut(), Accum::Replace, a, x.as_ref(), one::<T>(), par);
        // acc = b + acc - w v
        for j in 0..ncols {
            for i in 0..n {
                let val = sub(
                    &add(b.get(i, j), x.as_ref().get(i, j)),
                    &mul(omega, v.as_ref().get(i, j)),
                );
                *x.as_mut().get_mut(i, j) = val;
            }
        }
    }
    // y = w * acc
    for j in 0..ncols {
        for i in 0..n {
            let val = mul(omega, x.as_ref().get(i, j));
            *x.as_mut().get_mut(i, j) = val;
        }
    }
}

/// Chebyshev iteration (x0 = 0) on `A x = b` over `[lambda_min, lambda_max]`.
/// Running `degree` steps yields the degree-`degree` Chebyshev polynomial in A.
fn run_chebyshev<I, T>(
    poly: &Poly<I, T>,
    b: MatRef<'_, T>,
    mut x: MatMut<'_, T>,
    lambda_min: &T,
    lambda_max: &T,
    par: Par,
    stack: &mut MemStack,
) where
    I: Index,
    T: ComplexField,
{
    let n = x.nrows();
    let ncols = x.ncols();
    let (mut buf, _) = stack.make_with::<T>(2 * n * ncols, |_| zero::<T>());
    let (dbuf, rbuf) = buf.split_at_mut(n * ncols);
    let mut d = MatMut::from_column_major_slice_mut(dbuf, n, ncols);
    let mut r = MatMut::from_column_major_slice_mut(rbuf, n, ncols);
    let a = poly.a_view();

    let half = from_f64::<T::Real>(0.5);
    let two = from_f64::<T::Real>(2.0);
    let lo = real(lambda_min);
    let hi = real(lambda_max);
    let theta = mul(&add(&lo, &hi), &half);
    let delta = mul(&sub(&hi, &lo), &half);
    let sigma1 = div(&theta, &delta);
    let inv_theta = recip(&theta);

    // First step (x0 = 0): x1 = (1/theta) b, and the first direction d = x1.
    for j in 0..ncols {
        for i in 0..n {
            let val = mul_real(b.get(i, j), &inv_theta);
            *d.as_mut().get_mut(i, j) = copy(&val);
            *x.as_mut().get_mut(i, j) = val;
        }
    }

    let mut rho_prev = recip(&sigma1);
    for _ in 1..poly.degree {
        // r = b - A x
        sparse_dense_matmul(r.as_mut(), Accum::Replace, a, x.as_ref(), one::<T>(), par);
        for j in 0..ncols {
            for i in 0..n {
                let val = sub(b.get(i, j), r.as_ref().get(i, j));
                *r.as_mut().get_mut(i, j) = val;
            }
        }

        let rho = recip(&sub(&mul(&two, &sigma1), &rho_prev));
        let c1 = mul(&rho, &rho_prev);
        let c2 = div(&mul(&two, &rho), &delta);

        // d = c1 d + c2 r ; x += d
        for j in 0..ncols {
            for i in 0..n {
                let dval = add(
                    &mul_real(d.as_ref().get(i, j), &c1),
                    &mul_real(r.as_ref().get(i, j), &c2),
                );
                *d.as_mut().get_mut(i, j) = copy(&dval);
                let xnew = add(x.as_ref().get(i, j), &dval);
                *x.as_mut().get_mut(i, j) = xnew;
            }
        }
        rho_prev = rho;
    }
}
