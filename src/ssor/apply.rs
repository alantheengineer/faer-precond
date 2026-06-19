//! Triangular-solve dispatch for SSOR.
//!
//! `M^{-1} = w(2-w) (D+wU)^{-1} D (D+wL)^{-1}` (forward) is applied as a lower
//! solve, a diagonal scaling, and an upper solve. The `w(2-w)` scalar is folded
//! into the diagonal scaling (`scaled_diag`), so apply is three in-place passes
//! with no scratch.

use faer::sparse::linalg::triangular_solve;
use faer::{Conj, MatMut, Par};
use faer_traits::math_utils::{conj, copy, mul};
use faer_traits::{ComplexField, Index};

use super::Ssor;

/// Apply the SSOR operator in place.
///
/// - `transpose = false` applies `M^{-1}`; `true` applies `M^{-T}`.
/// - `conjugate = true` conjugates every factor (used for `conj_apply` and the
///   adjoint variants).
pub(crate) fn solve_in_place<I, T>(
    ssor: &Ssor<I, T>,
    transpose: bool,
    conjugate: bool,
    rhs: MatMut<'_, T>,
    par: Par,
) where
    I: Index,
    T: ComplexField,
{
    let lf = ssor.l_view();
    let uf = ssor.u_view();
    let cj = if conjugate { Conj::Yes } else { Conj::No };
    let mut rhs = rhs;

    if !transpose {
        // M^{-1} = (D+wU)^{-1} (s D) (D+wL)^{-1}
        triangular_solve::solve_lower_triangular_in_place(lf, cj, rhs.as_mut(), par);
        scale_by_diag(ssor, conjugate, rhs.as_mut());
        triangular_solve::solve_upper_triangular_in_place(uf, cj, rhs.as_mut(), par);
    } else {
        // M^{-T} = (D+wL)^{-T} (s D) (D+wU)^{-T}
        triangular_solve::solve_upper_triangular_transpose_in_place(uf, cj, rhs.as_mut(), par);
        scale_by_diag(ssor, conjugate, rhs.as_mut());
        triangular_solve::solve_lower_triangular_transpose_in_place(lf, cj, rhs.as_mut(), par);
    }
}

/// Multiply each row `i` by `scaled_diag[i] = w(2-w) * A[i,i]` (conjugated when
/// `conjugate`).
fn scale_by_diag<I, T>(ssor: &Ssor<I, T>, conjugate: bool, mut rhs: MatMut<'_, T>)
where
    I: Index,
    T: ComplexField,
{
    let d = &ssor.scaled_diag;
    for j in 0..rhs.ncols() {
        for (i, dv) in d.iter().enumerate() {
            let scale = if conjugate { conj(dv) } else { copy(dv) };
            let val = mul(&scale, rhs.as_ref().get(i, j));
            *rhs.as_mut().get_mut(i, j) = val;
        }
    }
}
