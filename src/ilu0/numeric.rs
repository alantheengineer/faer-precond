use super::symbolic::SymbolicIlu0;

#[derive(Debug, Clone)]
pub struct Ilu0<I, T> {
    pub symbolic: SymbolicIlu0<I>,
    pub l_values: Vec<T>,
    pub u_values: Vec<T>,
}

impl<I, T> Ilu0<I, T> {
    pub fn new(symbolic: SymbolicIlu0<I>) -> Self {
        Self {
            symbolic,
            l_values: Vec::new(),
            u_values: Vec::new(),
        }
    }

    pub fn dim(&self) -> usize {
        self.symbolic.dim
    }
}