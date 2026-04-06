#[derive(Debug, Clone)]
pub struct SymbolicIlu0<I> {
    pub dim: usize,
    _marker: core::marker::PhantomData<I>,
}

impl<I> SymbolicIlu0<I> {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            _marker: core::marker::PhantomData,
        }
    }
}