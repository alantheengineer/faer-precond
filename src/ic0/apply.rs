//! Triangular solve dispatch for IC(0).

use faer::sparse::linalg::triangular_solve;
use faer::{Conj, MatMut, Par};
use faer_traits::{ComplexField, Index};

use super::numeric::Ic0;

/// Apply the IC(0) preconditioner in place.
///
/// `conj = Conj::No`  produces `out = M^{-1} rhs`.
/// `conj = Conj::Yes` produces `out = conj(M)^{-1} rhs = M^{-T} rhs`
///                    (these coincide for Hermitian `M`).
pub(crate) fn solve_in_place<I, T>(ic: &Ic0<I, T>, conj: Conj, rhs: MatMut<'_, T>, par: Par)
where
    I: Index,
    T: ComplexField,
{
    let l = ic.l_view();
    let mut rhs = rhs;
    match conj {
        Conj::No => {
            // M^{-1} = L^{-H} L^{-1}
            triangular_solve::solve_lower_triangular_in_place(l, Conj::No, rhs.as_mut(), par);
            triangular_solve::solve_lower_triangular_transpose_in_place(
                l,
                Conj::Yes,
                rhs.as_mut(),
                par,
            );
        }
        Conj::Yes => {
            // conj(M)^{-1} = L^{-T} conj(L)^{-1}
            triangular_solve::solve_lower_triangular_in_place(l, Conj::Yes, rhs.as_mut(), par);
            triangular_solve::solve_lower_triangular_transpose_in_place(
                l,
                Conj::No,
                rhs.as_mut(),
                par,
            );
        }
    }
}
