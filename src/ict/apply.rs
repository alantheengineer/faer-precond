//! Triangular solve dispatch for ICT (identical to IC(0)).

use faer::sparse::linalg::triangular_solve;
use faer::{Conj, MatMut, Par};
use faer_traits::{ComplexField, Index};

use super::numeric::Ict;

/// Apply the ICT preconditioner in place.
///
/// `conj = Conj::No` produces `out = M^{-1} rhs = L^{-H} L^{-1} rhs`.
pub(crate) fn solve_in_place<I, T>(ict: &Ict<I, T>, conj: Conj, rhs: MatMut<'_, T>, par: Par)
where
    I: Index,
    T: ComplexField,
{
    let l = ict.l_view();
    let mut rhs = rhs;
    match conj {
        Conj::No => {
            triangular_solve::solve_lower_triangular_in_place(l, Conj::No, rhs.as_mut(), par);
            triangular_solve::solve_lower_triangular_transpose_in_place(
                l,
                Conj::Yes,
                rhs.as_mut(),
                par,
            );
        }
        Conj::Yes => {
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
