#[derive(Debug, Clone)]
pub struct SymbolicIc0<I> {
    pub dim: usize,
    _marker: core::marker::PhantomData<I>,
}

impl<I> SymbolicIc0<I> {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            _marker: core::marker::PhantomData,
        }
    }
}