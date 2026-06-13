//! Use an exact factorisation as a preconditioner.
//!
//! [`SolvePrecond`] wraps any faer factorisation implementing `SolveCore`
//! (`Llt`, `Ldlt`, `Lu`, `Qr`, ...) and exposes it through the preconditioner
//! traits. Each apply is a forward/back substitution against the stored
//! factors, so "apply the preconditioner" becomes "solve with this
//! factorisation".
//!
//! # When to use it
//!
//! This is an *adapter*, not a factorisation method — it does no incomplete or
//! approximate work of its own. The useful pattern is to factorise something
//! *cheaper than the full `A`* and precondition `A` with it:
//!
//! - a lower-fidelity discretisation of the same problem,
//! - a frozen- or averaged-coefficient version of `A`,
//! - a single dominant block of a larger system,
//! - or any factorisation you already have lying around for another reason.
//!
//! The Krylov method then only has to correct for whatever the approximation
//! left out. (Factorising the *full* `A` exactly and preconditioning `A` with
//! it converges in one step — but then you may as well have called the solver
//! directly.)

use core::fmt::Debug;

use dyn_stack::{MemStack, StackReq};
use faer::{
    Conj, MatMut, MatRef, Par,
    linalg::solvers::{ShapeCore, SolveCore},
    matrix_free::{BiLinOp, BiPrecond, LinOp, Precond},
};
use faer_traits::ComplexField;

/// Adapter exposing a faer factorisation `S` as a preconditioner.
///
/// See the [module documentation](self) for the intended use. `S` is any
/// factorisation implementing `SolveCore` (`Llt`, `Ldlt`, `Lu`, `Qr`, ...).
#[derive(Debug, Clone)]
pub struct SolvePrecond<S> {
    solver: S,
}

impl<S> SolvePrecond<S> {
    /// Wrap a factorisation as a preconditioner.
    pub fn new(solver: S) -> Self {
        Self { solver }
    }

    /// Borrow the wrapped factorisation.
    pub fn inner(&self) -> &S {
        &self.solver
    }

    /// Unwrap and return the factorisation.
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

    fn apply(&self, mut out: MatMut<'_, T>, rhs: MatRef<'_, T>, _par: Par, _stack: &mut MemStack) {
        assert_eq!(
            rhs.nrows(),
            self.ncols(),
            "rhs row count must match operator input dimension"
        );
        assert_eq!(
            out.nrows(),
            self.nrows(),
            "out row count must match operator output dimension"
        );
        assert_eq!(
            out.ncols(),
            rhs.ncols(),
            "out and rhs must have the same number of columns"
        );

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
        assert_eq!(
            rhs.nrows(),
            self.ncols(),
            "rhs row count must match operator input dimension"
        );
        assert_eq!(
            out.nrows(),
            self.nrows(),
            "out row count must match operator output dimension"
        );
        assert_eq!(
            out.ncols(),
            rhs.ncols(),
            "out and rhs must have the same number of columns"
        );

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

    fn apply_in_place(&self, rhs: MatMut<'_, T>, _par: Par, _stack: &mut MemStack) {
        self.solver.solve_in_place_with_conj(Conj::No, rhs);
    }

    fn conj_apply_in_place(&self, rhs: MatMut<'_, T>, _par: Par, _stack: &mut MemStack) {
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
        assert_eq!(
            rhs.nrows(),
            self.ncols(),
            "rhs row count must match operator input dimension"
        );
        assert_eq!(
            out.nrows(),
            self.nrows(),
            "out row count must match operator output dimension"
        );
        assert_eq!(
            out.ncols(),
            rhs.ncols(),
            "out and rhs must have the same number of columns"
        );

        out.copy_from(rhs);
        self.solver
            .solve_transpose_in_place_with_conj(Conj::No, out);
    }

    fn adjoint_apply(
        &self,
        mut out: MatMut<'_, T>,
        rhs: MatRef<'_, T>,
        _par: Par,
        _stack: &mut MemStack,
    ) {
        assert_eq!(
            rhs.nrows(),
            self.ncols(),
            "rhs row count must match operator input dimension"
        );
        assert_eq!(
            out.nrows(),
            self.nrows(),
            "out row count must match operator output dimension"
        );
        assert_eq!(
            out.ncols(),
            rhs.ncols(),
            "out and rhs must have the same number of columns"
        );

        out.copy_from(rhs);
        self.solver
            .solve_transpose_in_place_with_conj(Conj::Yes, out);
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

    fn transpose_apply_in_place(&self, rhs: MatMut<'_, T>, _par: Par, _stack: &mut MemStack) {
        self.solver
            .solve_transpose_in_place_with_conj(Conj::No, rhs);
    }

    fn adjoint_apply_in_place(&self, rhs: MatMut<'_, T>, _par: Par, _stack: &mut MemStack) {
        self.solver
            .solve_transpose_in_place_with_conj(Conj::Yes, rhs);
    }
}

#[cfg(test)]
mod tests {
    use core::mem::MaybeUninit;

    use super::*;
    use faer::{
        Mat, MatRef, Side,
        linalg::solvers::Llt,
        mat,
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
        let a = mat![[4.0, 1.0], [1.0, 3.0f64],];
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
        let rhs = mat![[1.0], [2.0f64],];
        let mut out = Mat::<f64>::zeros(2, 1);

        with_stack(pc.apply_scratch(rhs.ncols(), Par::Seq), |stack| {
            pc.apply(out.as_mut(), rhs.as_ref(), Par::Seq, stack);
        });

        let expected = mat![[1.0 / 11.0], [7.0 / 11.0f64],];
        assert_close(out.as_ref(), expected.as_ref(), 1e-12);
    }

    #[test]
    fn apply_in_place_matches_apply() {
        let pc = test_solver();
        let rhs = mat![[1.0], [2.0f64],];

        let mut out = Mat::<f64>::zeros(2, 1);
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
    fn transpose_apply_is_usable() {
        let pc = test_solver();
        let rhs = mat![[1.0], [2.0f64],];
        let mut out = Mat::<f64>::zeros(2, 1);

        with_stack(pc.transpose_apply_scratch(rhs.ncols(), Par::Seq), |stack| {
            pc.transpose_apply(out.as_mut(), rhs.as_ref(), Par::Seq, stack);
        });

        let expected = mat![[1.0 / 11.0], [7.0 / 11.0f64],];
        assert_close(out.as_ref(), expected.as_ref(), 1e-12);
    }

    #[test]
    fn adjoint_apply_is_usable() {
        let pc = test_solver();
        let rhs = mat![[1.0], [2.0f64],];
        let mut out = Mat::<f64>::zeros(2, 1);

        with_stack(pc.transpose_apply_scratch(rhs.ncols(), Par::Seq), |stack| {
            pc.adjoint_apply(out.as_mut(), rhs.as_ref(), Par::Seq, stack);
        });

        let expected = mat![[1.0 / 11.0], [7.0 / 11.0f64],];
        assert_close(out.as_ref(), expected.as_ref(), 1e-12);
    }

    #[test]
    fn transpose_and_adjoint_in_place_are_usable() {
        let pc = test_solver();

        let mut rhs_t = mat![[1.0], [2.0f64],];
        with_stack(
            pc.transpose_apply_in_place_scratch(rhs_t.ncols(), Par::Seq),
            |stack| {
                pc.transpose_apply_in_place(rhs_t.as_mut(), Par::Seq, stack);
            },
        );

        let mut rhs_h = mat![[1.0], [2.0f64],];
        with_stack(
            pc.transpose_apply_in_place_scratch(rhs_h.ncols(), Par::Seq),
            |stack| {
                pc.adjoint_apply_in_place(rhs_h.as_mut(), Par::Seq, stack);
            },
        );

        let expected = mat![[1.0 / 11.0], [7.0 / 11.0f64],];
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
