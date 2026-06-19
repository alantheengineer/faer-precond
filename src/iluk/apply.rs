//! Triangular solve dispatch for ILU(k) (identical to ILU(0)).

use faer::sparse::linalg::triangular_solve;
use faer::{Conj, MatMut, Par};
use faer_traits::{ComplexField, Index};

use super::numeric::Iluk;

/// Apply `M^{-1} = U^{-1} L^{-1}` to `rhs` in place.
pub(crate) fn solve_in_place<I, T>(iluk: &Iluk<I, T>, conj: Conj, rhs: MatMut<'_, T>, par: Par)
where
    I: Index,
    T: ComplexField,
{
    let l = iluk.l_view();
    let u = iluk.u_view();
    let mut rhs = rhs;
    triangular_solve::solve_unit_lower_triangular_in_place(l, conj, rhs.as_mut(), par);
    triangular_solve::solve_upper_triangular_in_place(u, conj, rhs.as_mut(), par);
}

/// Apply `M^{-T} = L^{-T} U^{-T}` to `rhs` in place.
pub(crate) fn solve_transpose_in_place<I, T>(
    iluk: &Iluk<I, T>,
    conj: Conj,
    rhs: MatMut<'_, T>,
    par: Par,
) where
    I: Index,
    T: ComplexField,
{
    let l = iluk.l_view();
    let u = iluk.u_view();
    let mut rhs = rhs;
    triangular_solve::solve_upper_triangular_transpose_in_place(u, conj, rhs.as_mut(), par);
    triangular_solve::solve_unit_lower_triangular_transpose_in_place(l, conj, rhs.as_mut(), par);
}
