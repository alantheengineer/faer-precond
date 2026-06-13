use core::fmt::Debug;

use dyn_stack::{MemStack, StackReq};
use faer::{
    linalg::solvers::{ShapeCore, SolveCore},
    matrix_free::{BiLinOp, BiPrecond, LinOp, Precond},
    Conj, MatMut, MatRef, Par,
};
use faer_traits::ComplexField;

#[derive(Debug, Clone)]
pub struct SolvePrecond<S> {
    solver: S,
}

impl<S> SolvePrecond<S> {
    pub fn new(solver: S) -> Self {
        Self { solver }
    }

    pub fn inner(&self) -> &S {
        &self.solver
    }

    pub fn into_inner(self) -> S {
        self.solver
    }
}

impl<T, S> LinOp<T> for SolvePrecond<S>
where
    T: ComplexField,
    S: ShapeCore + SolveCore<T> + Sync + Debug,
{
    fn apply_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::EMPTY
    }

    fn nrows(&self) -> usize {
        self.solver.nrows()
    }

    fn ncols(&self) -> usize {
        self.solver.ncols()
    }

    fn apply(
        &self,
        mut out: MatMut<'_, T>,
        rhs: MatRef<'_, T>,
        _par: Par,
        _stack: &mut MemStack,
    ) {
        assert_eq!(rhs.nrows(), self.ncols(), "rhs row count must match operator input dimension");
        assert_eq!(out.nrows(), self.nrows(), "out row count must match operator output dimension");
        assert_eq!(out.ncols(), rhs.ncols(), "out and rhs must have the same number of columns");

        out.copy_from(rhs);
        self.solver.solve_in_place_with_conj(Conj::No, out);
    }

    fn conj_apply(
        &self,
        mut out: MatMut<'_, T>,
        rhs: MatRef<'_, T>,
        _par: Par,
        _stack: &mut MemStack,
    ) {
        assert_eq!(rhs.nrows(), self.ncols(), "rhs row count must match operator input dimension");
        assert_eq!(out.nrows(), self.nrows(), "out row count must match operator output dimension");
        assert_eq!(out.ncols(), rhs.ncols(), "out and rhs must have the same number of columns");

        out.copy_from(rhs);
        self.solver.solve_in_place_with_conj(Conj::Yes, out);
    }
}

impl<T, S> Precond<T> for SolvePrecond<S>
where
    T: ComplexField,
    S: ShapeCore + SolveCore<T> + Sync + Debug,
{
    fn apply_in_place_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::EMPTY
    }

    fn apply_in_place(
        &self,
        rhs: MatMut<'_, T>,
        _par: Par,
        _stack: &mut MemStack,
    ) {
        self.solver.solve_in_place_with_conj(Conj::No, rhs);
    }

    fn conj_apply_in_place(
        &self,
        rhs: MatMut<'_, T>,
        _par: Par,
        _stack: &mut MemStack,
    ) {
        self.solver.solve_in_place_with_conj(Conj::Yes, rhs);
    }
}

impl<T, S> BiLinOp<T> for SolvePrecond<S>
where
    T: ComplexField,
    S: ShapeCore + SolveCore<T> + Sync + Debug,
{
    fn transpose_apply_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::EMPTY
    }

    fn transpose_apply(
        &self,
        mut out: MatMut<'_, T>,
        rhs: MatRef<'_, T>,
        _par: Par,
        _stack: &mut MemStack,
    ) {
        assert_eq!(rhs.nrows(), self.ncols(), "rhs row count must match operator input dimension");
        assert_eq!(out.nrows(), self.nrows(), "out row count must match operator output dimension");
        assert_eq!(out.ncols(), rhs.ncols(), "out and rhs must have the same number of columns");

        out.copy_from(rhs);
        self.solver.solve_transpose_in_place_with_conj(Conj::No, out);
    }

    fn adjoint_apply(
        &self,
        mut out: MatMut<'_, T>,
        rhs: MatRef<'_, T>,
        _par: Par,
        _stack: &mut MemStack,
    ) {
        assert_eq!(rhs.nrows(), self.ncols(), "rhs row count must match operator input dimension");
        assert_eq!(out.nrows(), self.nrows(), "out row count must match operator output dimension");
        assert_eq!(out.ncols(), rhs.ncols(), "out and rhs must have the same number of columns");

        out.copy_from(rhs);
        self.solver.solve_transpose_in_place_with_conj(Conj::Yes, out);
    }
}

impl<T, S> BiPrecond<T> for SolvePrecond<S>
where
    T: ComplexField,
    S: ShapeCore + SolveCore<T> + Sync + Debug,
{
    fn transpose_apply_in_place_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::EMPTY
    }

    fn transpose_apply_in_place(
        &self,
        rhs: MatMut<'_, T>,
        _par: Par,
        _stack: &mut MemStack,
    ) {
        self.solver.solve_transpose_in_place_with_conj(Conj::No, rhs);
    }

    fn adjoint_apply_in_place(
        &self,
        rhs: MatMut<'_, T>,
        _par: Par,
        _stack: &mut MemStack,
    ) {
        self.solver.solve_transpose_in_place_with_conj(Conj::Yes, rhs);
    }
}

#[cfg(test)]
mod tests {
    use core::mem::MaybeUninit;

    use super::*;
    use faer::{
        linalg::solvers::Llt,
        mat, Mat, MatRef, Side,
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

    fn test_solver() -> SolvePrecond<Llt<f64>> {
        let a = mat![
            [4.0, 1.0],
            [1.0, 3.0f64],
        ];
        let llt = Llt::new(a.as_ref(), Side::Lower).expect("matrix should be SPD");
        SolvePrecond::new(llt)
    }

    #[test]
    fn exposes_dimensions() {
        let pc = test_solver();
        assert_eq!(pc.nrows(), 2);
        assert_eq!(pc.ncols(), 2);
    }

    #[test]
    fn apply_solves_system() {
        let pc = test_solver();
        let rhs = mat![
            [1.0],
            [2.0f64],
        ];
        let mut out = Mat::<f64>::zeros(2, 1);

        with_stack(pc.apply_scratch(rhs.ncols(), Par::Seq), |stack| {
            pc.apply(out.as_mut(), rhs.as_ref(), Par::Seq, stack);
        });

        let expected = mat![
            [1.0 / 11.0],
            [7.0 / 11.0f64],
        ];
        assert_close(out.as_ref(), expected.as_ref(), 1e-12);
    }

    #[test]
    fn apply_in_place_matches_apply() {
        let pc = test_solver();
        let rhs = mat![
            [1.0],
            [2.0f64],
        ];

        let mut out = Mat::<f64>::zeros(2, 1);
        with_stack(pc.apply_scratch(rhs.ncols(), Par::Seq), |stack| {
            pc.apply(out.as_mut(), rhs.as_ref(), Par::Seq, stack);
        });

        let mut inplace = rhs.to_owned();
        with_stack(pc.apply_in_place_scratch(inplace.ncols(), Par::Seq), |stack| {
            pc.apply_in_place(inplace.as_mut(), Par::Seq, stack);
        });

        assert_close(out.as_ref(), inplace.as_ref(), 1e-12);
    }

    #[test]
    fn transpose_apply_is_usable() {
        let pc = test_solver();
        let rhs = mat![
            [1.0],
            [2.0f64],
        ];
        let mut out = Mat::<f64>::zeros(2, 1);

        with_stack(pc.transpose_apply_scratch(rhs.ncols(), Par::Seq), |stack| {
            pc.transpose_apply(out.as_mut(), rhs.as_ref(), Par::Seq, stack);
        });

        let expected = mat![
            [1.0 / 11.0],
            [7.0 / 11.0f64],
        ];
        assert_close(out.as_ref(), expected.as_ref(), 1e-12);
    }

    #[test]
    fn adjoint_apply_is_usable() {
        let pc = test_solver();
        let rhs = mat![
            [1.0],
            [2.0f64],
        ];
        let mut out = Mat::<f64>::zeros(2, 1);

        with_stack(pc.transpose_apply_scratch(rhs.ncols(), Par::Seq), |stack| {
            pc.adjoint_apply(out.as_mut(), rhs.as_ref(), Par::Seq, stack);
        });

        let expected = mat![
            [1.0 / 11.0],
            [7.0 / 11.0f64],
        ];
        assert_close(out.as_ref(), expected.as_ref(), 1e-12);
    }

    #[test]
    fn transpose_and_adjoint_in_place_are_usable() {
        let pc = test_solver();

        let mut rhs_t = mat![
            [1.0],
            [2.0f64],
        ];
        with_stack(pc.transpose_apply_in_place_scratch(rhs_t.ncols(), Par::Seq), |stack| {
            pc.transpose_apply_in_place(rhs_t.as_mut(), Par::Seq, stack);
        });

        let mut rhs_h = mat![
            [1.0],
            [2.0f64],
        ];
        with_stack(pc.transpose_apply_in_place_scratch(rhs_h.ncols(), Par::Seq), |stack| {
            pc.adjoint_apply_in_place(rhs_h.as_mut(), Par::Seq, stack);
        });

        let expected = mat![
            [1.0 / 11.0],
            [7.0 / 11.0f64],
        ];
        assert_close(rhs_t.as_ref(), expected.as_ref(), 1e-12);
        assert_close(rhs_h.as_ref(), expected.as_ref(), 1e-12);
    }

    #[test]
    fn inner_accessors_work() {
        let pc = test_solver();
        assert_eq!(pc.inner().nrows(), 2);

        let pc2 = test_solver();
        let solver = pc2.into_inner();
        assert_eq!(solver.nrows(), 2);
    }
}