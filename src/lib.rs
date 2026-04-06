pub mod adapters;
pub mod jacobi;
pub mod block_jacobi;
pub mod ilu0;
pub mod ic0;

// Convenience re-exports.
pub use adapters::SolvePrecond;
pub use block_jacobi::BlockJacobiPrecond;
pub use jacobi::JacobiPrecond;