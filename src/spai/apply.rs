//! Apply the SPAI preconditioner `M^{-1} = M` as a single sparse matvec.

use dyn_stack::{MemStack, StackReq};
use faer::sparse::linalg::matmul::sparse_dense_matmul;
use faer::{Accum, MatMut, MatRef, Par};
use faer_traits::math_utils::{one, zero};
use faer_traits::{ComplexField, Index};

use super::Spai;

/// `dst = op(M) src`, where `op` is selected by the `transpose`/`conjugate`
/// flags (forward, conjugate, transpose or adjoint).
fn matmul<I, T>(
    spai: &Spai<I, T>,
    transpose: bool,
    conjugate: bool,
    dst: MatMut<'_, T>,
    src: MatRef<'_, T>,
    par: Par,
) where
    I: Index,
    T: ComplexField,
{
    let m = spai.m_view();
    match (transpose, conjugate) {
        (false, false) => sparse_dense_matmul(dst, Accum::Replace, m, src, one::<T>(), par),
        (false, true) => {
            sparse_dense_matmul(dst, Accum::Replace, m.conjugate(), src, one::<T>(), par)
        }
        (true, false) => {
            sparse_dense_matmul(dst, Accum::Replace, m.transpose(), src, one::<T>(), par)
        }
        (true, true) => sparse_dense_matmul(dst, Accum::Replace, m.adjoint(), src, one::<T>(), par),
    }
}

/// Out-of-place apply: writes straight into `out` (no scratch needed).
pub(crate) fn apply_out<I, T>(
    spai: &Spai<I, T>,
    transpose: bool,
    conjugate: bool,
    out: MatMut<'_, T>,
    rhs: MatRef<'_, T>,
    par: Par,
) where
    I: Index,
    T: ComplexField,
{
    matmul(spai, transpose, conjugate, out, rhs, par);
}

/// In-place apply: matvecs into scratch then copies back (avoids aliasing).
pub(crate) fn apply_inplace<I, T>(
    spai: &Spai<I, T>,
    transpose: bool,
    conjugate: bool,
    mut rhs: MatMut<'_, T>,
    par: Par,
    stack: &mut MemStack,
) where
    I: Index,
    T: ComplexField,
{
    let n = spai.dim;
    let ncols = rhs.ncols();
    let (mut buf, _) = stack.make_with::<T>(n * ncols, |_| zero::<T>());
    let mut out = MatMut::from_column_major_slice_mut(&mut buf[..], n, ncols);
    matmul(spai, transpose, conjugate, out.as_mut(), rhs.as_ref(), par);
    rhs.copy_from(out.as_ref());
}

/// Scratch for the in-place apply: one work column block.
pub(crate) fn inplace_scratch<I, T: ComplexField>(spai: &Spai<I, T>, ncols: usize) -> StackReq {
    StackReq::new::<T>(spai.dim * ncols)
}
