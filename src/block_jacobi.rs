use dyn_stack::{MemStack, StackReq};
use faer::{
    matrix_free::{LinOp, Precond},
    MatMut, MatRef, Par,
};
use faer_traits::ComplexField;

#[derive(Debug, Clone)]
pub struct BlockJacobiPrecond<T> {
    block_offsets: Vec<usize>,
    _marker: core::marker::PhantomData<T>,
}

impl<T> BlockJacobiPrecond<T> {
    pub fn new(block_offsets: Vec<usize>) -> Self {
        Self {
            block_offsets,
            _marker: core::marker::PhantomData,
        }
    }

    pub fn dim(&self) -> usize {
        self.block_offsets.last().copied().unwrap_or(0)
    }
}

impl<T: ComplexField> LinOp<T> for BlockJacobiPrecond<T> {
    fn apply_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::EMPTY
    }

    fn nrows(&self) -> usize {
        self.dim()
    }

    fn ncols(&self) -> usize {
        self.dim()
    }

    fn apply(
        &self,
        _out: MatMut<'_, T>,
        _rhs: MatRef<'_, T>,
        _par: Par,
        _stack: &mut MemStack,
    ) {
        todo!("apply block-Jacobi block solves here")
    }

    fn conj_apply(
        &self,
        _out: MatMut<'_, T>,
        _rhs: MatRef<'_, T>,
        _par: Par,
        _stack: &mut MemStack,
    ) {
        todo!("apply conjugated block-Jacobi block solves here")
    }
}

impl<T: ComplexField> Precond<T> for BlockJacobiPrecond<T> {
    fn apply_in_place_scratch(&self, _rhs_ncols: usize, _par: Par) -> StackReq {
        StackReq::EMPTY
    }

    fn apply_in_place(
        &self,
        _rhs: MatMut<'_, T>,
        _par: Par,
        _stack: &mut MemStack,
    ) {
        todo!("apply block-Jacobi in place here")
    }
}