use core::fmt::Debug;

use dyn_stack::{MemStack, StackReq};
use faer::{
    MatMut, MatRef, Par,
    matrix_free::{BiLinOp, BiPrecond, LinOp, Precond},
    prelude::ReborrowMut,
};
use faer_traits::ComplexField;
use faer_traits::math_utils::{abs2, conj, copy, mul, recip, zero};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JacobiError {
    NonSquareMatrix { nrows: usize, ncols: usize },
    ZeroDiagonalEntry { index: usize },
}

#[derive(Debug, Clone)]
pub struct JacobiPrecond<T> {
    inv_diag: Vec<T>,
}

impl<T> JacobiPrecond<T> {
    pub fn from_inverse_diagonal(inv_diag: Vec<T>) -> Self {
        Self { inv_diag }
    }

    pub fn inverse_diagonal(&self) -> &[T] {
        &self.inv_diag
    }

    pub fn dim(&self) -> usize {
        self.inv_diag.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inv_diag.is_empty()
    }
}

impl<T: ComplexField> JacobiPrecond<T> {
    pub fn try_from_diagonal(diag: &[T]) -> Result<Self, JacobiError> {
        let mut inv_diag = Vec::with_capacity(diag.len());

        for (index, value) in diag.iter().enumerate() {
            if abs2(value) == zero::<T::Real>() {
                return Err(JacobiError::ZeroDiagonalEntry { index });
            }
            inv_diag.push(recip(value));
        }

        Ok(Self { inv_diag })
    }

    pub fn try_from_matrix_diagonal(mat: MatRef<'_, T>) -> Result<Self, JacobiError> {
        if mat.nrows() != mat.ncols() {
            return Err(JacobiError::NonSquareMatrix {
                nrows: mat.nrows(),
                ncols: mat.ncols(),
            });
        }

        let mut diag = Vec::with_capacity(mat.nrows());
        for i in 0..mat.nrows() {
            diag.push(copy(mat.get(i, i)));
        }

        Self::try_from_diagonal(&diag)
    }

    #[inline]
    fn check_dims(&self, out_nrows: usize, rhs_nrows: usize, rhs_ncols: usize, out_ncols: usize) {
        assert_eq!(
            rhs_nrows,
            self.dim(),
            "rhs row count must match preconditioner dimension"
        );
        assert_eq!(
            out_nrows,
            self.dim(),
            "out row count must match preconditioner dimension"
        );
        assert_eq!(
            out_ncols, rhs_ncols,
            "out and rhs must have the same number of columns"
        );
    }

    #[inline]
    fn apply_scale_to_out(&self, mut out: MatMut<'_, T>, rhs: MatRef<'_, T>, conjugate_diag: bool) {
        self.check_dims(out.nrows(), rhs.nrows(), rhs.ncols(), out.ncols());

        for j in 0..rhs.ncols() {
            for i in 0..rhs.nrows() {
                let scale = if conjugate_diag {
                    conj(&self.inv_diag[i])
                } else {
                    copy(&self.inv_diag[i])
                };
                *out.rb_mut().get_mut(i, j) = mul(&scale, rhs.get(i, j));
            }
        }
    }

    #[inline]
    fn apply_scale_in_place(&self, mut rhs: MatMut<'_, T>, conjugate_diag: bool) {
        assert_eq!(
            rhs.nrows(),
            self.dim(),
            "rhs row count must match preconditioner dimension"
        );

        for j in 0..rhs.ncols() {
            for i in 0..rhs.nrows() {
                let scale = if conjugate_diag {
                    conj(&self.inv_diag[i])
                } else {
                    copy(&self.inv_diag[i])
                };
                let elem = rhs.rb_mut().get_mut(i, j);
                *elem = mul(&scale, elem);
            }
        }
    }
}

impl<T> LinOp<T> for JacobiPrecond<T>
where
    T: ComplexField + Debug + Sync,
{
    fn apply_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::EMPTY
    }

    fn nrows(&self) -> usize {
        self.dim()
    }

    fn ncols(&self) -> usize {
        self.dim()
    }

    fn apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, _par: Par, _stack: &mut MemStack) {
        self.apply_scale_to_out(out, rhs, false);
    }

    fn conj_apply(&self, out: MatMut<'_, T>, rhs: MatRef<'_, T>, _par: Par, _stack: &mut MemStack) {
        self.apply_scale_to_out(out, rhs, true);
    }
}

impl<T> Precond<T> for JacobiPrecond<T>
where
    T: ComplexField + Debug + Sync,
{
    fn apply_in_place_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::EMPTY
    }

    fn apply_in_place(&self, rhs: MatMut<'_, T>, _par: Par, _stack: &mut MemStack) {
        self.apply_scale_in_place(rhs, false);
    }

    fn conj_apply_in_place(&self, rhs: MatMut<'_, T>, _par: Par, _stack: &mut MemStack) {
        self.apply_scale_in_place(rhs, true);
    }
}

impl<T> BiLinOp<T> for JacobiPrecond<T>
where
    T: ComplexField + Debug + Sync,
{
    fn transpose_apply_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::EMPTY
    }

    fn transpose_apply(
        &self,
        out: MatMut<'_, T>,
        rhs: MatRef<'_, T>,
        _par: Par,
        _stack: &mut MemStack,
    ) {
        // diagonal operator: transpose is identical
        self.apply_scale_to_out(out, rhs, false);
    }

    fn adjoint_apply(
        &self,
        out: MatMut<'_, T>,
        rhs: MatRef<'_, T>,
        _par: Par,
        _stack: &mut MemStack,
    ) {
        // diagonal operator: adjoint conjugates the diagonal
        self.apply_scale_to_out(out, rhs, true);
    }
}

impl<T> BiPrecond<T> for JacobiPrecond<T>
where
    T: ComplexField + Debug + Sync,
{
    fn transpose_apply_in_place_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::EMPTY
    }

    fn transpose_apply_in_place(&self, rhs: MatMut<'_, T>, _par: Par, _stack: &mut MemStack) {
        self.apply_scale_in_place(rhs, false);
    }

    fn adjoint_apply_in_place(&self, rhs: MatMut<'_, T>, _par: Par, _stack: &mut MemStack) {
        self.apply_scale_in_place(rhs, true);
    }
}

#[cfg(test)]
mod tests {
    use core::mem::MaybeUninit;

    use super::*;
    use faer::{
        Mat, MatRef, mat,
        matrix_free::{BiLinOp, BiPrecond, LinOp, Precond},
    };

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

    #[test]
    fn builds_from_diagonal() {
        let pc = JacobiPrecond::try_from_diagonal(&[2.0, 4.0, 8.0]).unwrap();
        assert_eq!(pc.dim(), 3);
        assert_eq!(pc.inverse_diagonal(), &[0.5, 0.25, 0.125]);
    }

    #[test]
    fn rejects_zero_diagonal() {
        let err = JacobiPrecond::try_from_diagonal(&[2.0, 0.0, 8.0]).unwrap_err();
        assert_eq!(err, JacobiError::ZeroDiagonalEntry { index: 1 });
    }

    #[test]
    fn builds_from_matrix_diagonal() {
        let a = mat![[2.0, 9.0, 0.0], [1.0, 4.0, 5.0], [0.0, 7.0, 8.0f64],];

        let pc = JacobiPrecond::try_from_matrix_diagonal(a.as_ref()).unwrap();
        assert_eq!(pc.inverse_diagonal(), &[0.5, 0.25, 0.125]);
    }

    #[test]
    fn apply_matches_expected_multiple_rhs() {
        let pc = JacobiPrecond::try_from_diagonal(&[2.0, 4.0]).unwrap();
        let rhs = mat![[2.0, 8.0], [4.0, 12.0f64],];
        let mut out = Mat::<f64>::zeros(2, 2);

        let req = pc.apply_scratch(rhs.ncols(), Par::Seq);
        with_stack(req, |stack| {
            pc.apply(out.as_mut(), rhs.as_ref(), Par::Seq, stack);
        });

        let expected = mat![[1.0, 4.0], [1.0, 3.0f64],];
        assert_close(out.as_ref(), expected.as_ref(), 1e-12);
    }

    #[test]
    fn apply_in_place_matches_apply() {
        let pc = JacobiPrecond::try_from_diagonal(&[2.0, 4.0]).unwrap();
        let rhs = mat![[2.0, 8.0], [4.0, 12.0f64],];

        let mut out = Mat::<f64>::zeros(2, 2);
        with_stack(pc.apply_scratch(rhs.ncols(), Par::Seq), |stack| {
            pc.apply(out.as_mut(), rhs.as_ref(), Par::Seq, stack);
        });

        let mut inplace = rhs.to_owned();
        with_stack(
            pc.apply_in_place_scratch(inplace.ncols(), Par::Seq),
            |stack| {
                pc.apply_in_place(inplace.as_mut(), Par::Seq, stack);
            },
        );

        assert_close(out.as_ref(), inplace.as_ref(), 1e-12);
    }

    #[test]
    fn transpose_and_adjoint_are_usable() {
        let pc = JacobiPrecond::try_from_diagonal(&[2.0, 4.0]).unwrap();
        let rhs = mat![[2.0], [4.0f64],];

        let mut out_t = Mat::<f64>::zeros(2, 1);
        with_stack(pc.transpose_apply_scratch(rhs.ncols(), Par::Seq), |stack| {
            pc.transpose_apply(out_t.as_mut(), rhs.as_ref(), Par::Seq, stack);
        });

        let mut out_h = Mat::<f64>::zeros(2, 1);
        with_stack(pc.transpose_apply_scratch(rhs.ncols(), Par::Seq), |stack| {
            pc.adjoint_apply(out_h.as_mut(), rhs.as_ref(), Par::Seq, stack);
        });

        let expected = mat![[1.0], [1.0f64],];
        assert_close(out_t.as_ref(), expected.as_ref(), 1e-12);
        assert_close(out_h.as_ref(), expected.as_ref(), 1e-12);
    }

    #[test]
    fn transpose_and_adjoint_in_place_are_usable() {
        let pc = JacobiPrecond::try_from_diagonal(&[2.0, 4.0]).unwrap();

        let mut rhs_t = mat![[2.0], [4.0f64],];
        with_stack(
            pc.transpose_apply_in_place_scratch(rhs_t.ncols(), Par::Seq),
            |stack| {
                pc.transpose_apply_in_place(rhs_t.as_mut(), Par::Seq, stack);
            },
        );

        let mut rhs_h = mat![[2.0], [4.0f64],];
        with_stack(
            pc.transpose_apply_in_place_scratch(rhs_h.ncols(), Par::Seq),
            |stack| {
                pc.adjoint_apply_in_place(rhs_h.as_mut(), Par::Seq, stack);
            },
        );

        let expected = mat![[1.0], [1.0f64],];
        assert_close(rhs_t.as_ref(), expected.as_ref(), 1e-12);
        assert_close(rhs_h.as_ref(), expected.as_ref(), 1e-12);
    }
}
