use super::symbolic::SymbolicIc0;

#[derive(Debug, Clone)]
pub struct Ic0<I, T> {
    pub symbolic: SymbolicIc0<I>,
    pub values: Vec<T>,
}

impl<I, T> Ic0<I, T> {
    pub fn new(symbolic: SymbolicIc0<I>) -> Self {
        Self {
            symbolic,
            values: Vec::new(),
        }
    }

    pub fn dim(&self) -> usize {
        self.symbolic.dim
    }
}