//! Apply the FSAI preconditioner `M^{-1} = G^H G` as two sparse matvecs.
//!
//! `M = (G^H G)^{-1}` is Hermitian, so `M^{-1}` and its adjoint coincide, while
//! the transpose / conjugate variants apply `M^{-T} = G^T conj(G)`.

use dyn_stack::{MemStack, StackReq};
use faer::sparse::linalg::matmul::sparse_dense_matmul;
use faer::{Accum, MatMut, MatRef, Par};
use faer_traits::math_utils::{one, zero};
use faer_traits::{ComplexField, Index};

use super::Fsai;

/// Out-of-place apply. `conjugate = false` gives `out = G^H G rhs`;
/// `conjugate = true` gives `out = G^T conj(G) rhs` (`= M^{-T} rhs`).
pub(crate) fn apply_out<I, T>(
    fsai: &Fsai<I, T>,
    mut out: MatMut<'_, T>,
    rhs: MatRef<'_, T>,
    conjugate: bool,
    par: Par,
    stack: &mut MemStack,
) where
    I: Index,
    T: ComplexField,
{
    let n = fsai.dim;
    let ncols = rhs.ncols();
    let (mut ybuf, _) = stack.make_with::<T>(n * ncols, |_| zero::<T>());
    let mut y = MatMut::from_column_major_slice_mut(&mut ybuf[..], n, ncols);
    let g = fsai.g_view();

    if !conjugate {
        sparse_dense_matmul(y.as_mut(), Accum::Replace, g, rhs, one::<T>(), par);
        sparse_dense_matmul(out.as_mut(), Accum::Replace, g.adjoint(), y.as_ref(), one::<T>(), par);
    } else {
        sparse_dense_matmul(y.as_mut(), Accum::Replace, g.conjugate(), rhs, one::<T>(), par);
        sparse_dense_matmul(out.as_mut(), Accum::Replace, g.transpose(), y.as_ref(), one::<T>(), par);
    }
}

/// In-place apply (see [`apply_out`] for the `conjugate` flag).
pub(crate) fn apply_inplace<I, T>(
    fsai: &Fsai<I, T>,
    mut rhs: MatMut<'_, T>,
    conjugate: bool,
    par: Par,
    stack: &mut MemStack,
) where
    I: Index,
    T: ComplexField,
{
    let n = fsai.dim;
    let ncols = rhs.ncols();
    let (mut ybuf, _) = stack.make_with::<T>(n * ncols, |_| zero::<T>());
    let mut y = MatMut::from_column_major_slice_mut(&mut ybuf[..], n, ncols);
    let g = fsai.g_view();

    if !conjugate {
        sparse_dense_matmul(y.as_mut(), Accum::Replace, g, rhs.as_ref(), one::<T>(), par);
        sparse_dense_matmul(rhs.as_mut(), Accum::Replace, g.adjoint(), y.as_ref(), one::<T>(), par);
    } else {
        sparse_dense_matmul(y.as_mut(), Accum::Replace, g.conjugate(), rhs.as_ref(), one::<T>(), par);
        sparse_dense_matmul(rhs.as_mut(), Accum::Replace, g.transpose(), y.as_ref(), one::<T>(), par);
    }
}

/// Scratch for either apply path: one work column block.
pub(crate) fn scratch<I, T: ComplexField>(fsai: &Fsai<I, T>, ncols: usize) -> StackReq {
    StackReq::new::<T>(fsai.dim * ncols)
}
