//! Triangular-solve dispatch for ILUTP, including the column permutation.
//!
//! The factorisation produces `L U ≈ A P`, where `P` is the column permutation
//! with `(A P)[:, k] = A[:, perm[k]]`. Equivalently `A ≈ L U P^{-1}`, so the
//! preconditioner applies
//!
//! ```text
//!     M^{-1} = P U^{-1} L^{-1}.
//! ```
//!
//! Both directions below are derived from that identity. `P` is a plain
//! permutation (real), so its adjoint equals its transpose and the `Conj` flag
//! only ever touches the triangular factors.

use dyn_stack::MemStack;
use faer::sparse::linalg::triangular_solve;
use faer::{Conj, MatMut, Par};
use faer_traits::math_utils::zero;
use faer_traits::{ComplexField, Index};

use super::numeric::Ilutp;

/// Apply `M^{-1} = P U^{-1} L^{-1}` to `rhs` in place.
pub(crate) fn solve_in_place<I, T>(
    pc: &Ilutp<I, T>,
    conj: Conj,
    rhs: MatMut<'_, T>,
    par: Par,
    stack: &mut MemStack,
) where
    I: Index,
    T: ComplexField,
{
    let l = pc.l_view();
    let u = pc.u_view();
    let mut rhs = rhs;
    triangular_solve::solve_unit_lower_triangular_in_place(l, conj, rhs.as_mut(), par);
    triangular_solve::solve_upper_triangular_in_place(u, conj, rhs.as_mut(), par);
    if pc.permuted {
        // z = P y : z[r] = y[perm_inv[r]].
        permute_rows(rhs.as_mut(), &pc.perm_inv, stack);
    }
}

/// Apply `M^{-T} = L^{-T} U^{-T} P^{T}` (or `M^{-H}` with `conj = Yes`) in place.
pub(crate) fn solve_transpose_in_place<I, T>(
    pc: &Ilutp<I, T>,
    conj: Conj,
    rhs: MatMut<'_, T>,
    par: Par,
    stack: &mut MemStack,
) where
    I: Index,
    T: ComplexField,
{
    let mut rhs = rhs;
    if pc.permuted {
        // P^T r : (P^T r)[r] = r[perm[r]].
        permute_rows(rhs.as_mut(), &pc.perm, stack);
    }
    let l = pc.l_view();
    let u = pc.u_view();
    triangular_solve::solve_upper_triangular_transpose_in_place(u, conj, rhs.as_mut(), par);
    triangular_solve::solve_unit_lower_triangular_transpose_in_place(l, conj, rhs.as_mut(), par);
}

/// Overwrite each column of `rhs` with its `map`-gathered permutation:
/// `out[r] = in[map[r]]`. Uses one length-`n` scratch column from `stack`.
fn permute_rows<T>(rhs: MatMut<'_, T>, map: &[usize], stack: &mut MemStack)
where
    T: ComplexField,
{
    let mut rhs = rhs;
    let n = rhs.nrows();
    let ncols = rhs.ncols();
    let (mut tmp, _) = stack.make_with::<T>(n, |_| zero::<T>());
    for j in 0..ncols {
        for r in 0..n {
            tmp[r] = rhs.as_ref().get(map[r], j).clone();
        }
        for r in 0..n {
            *rhs.as_mut().get_mut(r, j) = tmp[r].clone();
        }
    }
}
