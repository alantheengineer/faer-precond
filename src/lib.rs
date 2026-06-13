//! Numerical preconditioners for iterative linear solvers, built on
//! [faer](https://crates.io/crates/faer).
//!
//! Every preconditioner in this crate implements faer's
//! `matrix_free::{LinOp, Precond, BiLinOp, BiPrecond}` traits, so they plug
//! directly into faer's Krylov solvers (CG, GMRES, BiCGSTAB, LSMR, ...).
//!
//! # Available preconditioners
//!
//! | Type | Source | Apply cost | Notes |
//! |---|---|---|---|
//! | [`JacobiPrecond`] | diagonal of `A` | `O(n)` | Diagonally-dominant problems. |
//! | [`BlockJacobiPrecond`] | dense diagonal blocks of `A` | `O(sum b_k²)` | Arbitrary block partition; LU per block. |
//! | [`Ilu0`] | CSC sparsity of `A` | `O(nnz(A))` | Zero-fill incomplete LU. |
//! | [`Ic0`] | CSC lower triangle of `A` | `O(nnz_L)` | Zero-fill incomplete Cholesky for HPD `A`. |
//! | [`SolvePrecond`] | any faer factorisation (`Llt`, `Lu`, `Qr`, ...) | factorisation-dependent | Adapter, not a factorisation. |
//!
//! # Design contract
//!
//! - **No heap allocation during `apply`.** Every preconditioner returns
//!   either [`StackReq::EMPTY`](dyn_stack::StackReq::EMPTY) or a precise
//!   pre-computed scratch size from `apply_scratch` / `apply_in_place_scratch`.
//!   All temporary memory flows through the [`MemStack`](dyn_stack::MemStack)
//!   faer's trait interface provides.
//! - **Refactorisation reuses storage.** [`Ilu0`] and [`Ic0`] expose a
//!   `refactorize(&mut self, a)` method for the steady-state case (nonlinear
//!   Krylov drivers where the sparsity pattern is fixed but the values
//!   change). It allocates nothing.
//! - **In-place semantics match faer.** `apply` performs `out = M^{-1} rhs`;
//!   `apply_in_place` overwrites `rhs` with `M^{-1} rhs`. Transpose and
//!   adjoint variants follow the same convention against `M^{-T}` and
//!   `M^{-H}`.
//!
//! # Example: point-Jacobi
//!
//! ```
//! use dyn_stack::MemStack;
//! use faer::{mat, Par};
//! use faer::matrix_free::Precond;
//! use faer_precond::JacobiPrecond;
//!
//! let pc = JacobiPrecond::try_from_diagonal(&[4.0_f64, 2.0, 8.0]).unwrap();
//!
//! let mut x = mat![[8.0_f64], [6.0], [16.0]];
//! pc.apply_in_place(x.as_mut(), Par::Seq, MemStack::new(&mut []));
//!
//! assert!((*x.as_ref().get(0, 0) - 2.0).abs() < 1e-12);
//! assert!((*x.as_ref().get(1, 0) - 3.0).abs() < 1e-12);
//! assert!((*x.as_ref().get(2, 0) - 2.0).abs() < 1e-12);
//! ```
//!
//! # Example: incomplete LU on a sparse matrix
//!
//! ```
//! use dyn_stack::MemStack;
//! use faer::sparse::{SparseColMat, Triplet};
//! use faer::{mat, Par};
//! use faer::matrix_free::Precond;
//! use faer_precond::Ilu0;
//!
//! // Build a 5x5 SPD tridiagonal: diag = 4, off = -1.
//! let mut triplets = Vec::new();
//! for i in 0..5 {
//!     triplets.push(Triplet::new(i, i, 4.0_f64));
//!     if i > 0 {
//!         triplets.push(Triplet::new(i, i - 1, -1.0));
//!         triplets.push(Triplet::new(i - 1, i, -1.0));
//!     }
//! }
//! let a = SparseColMat::<usize, f64>::try_new_from_triplets(5, 5, &triplets).unwrap();
//!
//! let pc = Ilu0::try_new(a.as_ref()).expect("non-singular pattern");
//!
//! // For a tridiagonal there is no fill, so ILU(0) is the exact LU.
//! let mut b = mat![[1.0_f64], [0.0], [0.0], [0.0], [0.0]];
//! pc.apply_in_place(b.as_mut(), Par::Seq, MemStack::new(&mut []));
//! ```
//!
//! # Cargo features
//!
//! There are no opt-in feature flags in `0.1`. The crate inherits faer's
//! default feature set, which includes sparse linear algebra (`sparse-linalg`)
//! required by [`Ilu0`] and [`Ic0`].
//!
//! # License
//!
//! Dual-licensed under `MIT OR Apache-2.0`.

#![doc(html_root_url = "https://docs.rs/faer-precond/0.1.0")]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod adapters;
pub mod block_jacobi;
pub mod ic0;
pub mod ilu0;
pub mod jacobi;

pub use adapters::SolvePrecond;
pub use block_jacobi::{BlockJacobiError, BlockJacobiPrecond};
pub use ic0::{Ic0, Ic0Error, SymbolicIc0};
pub use ilu0::{Ilu0, Ilu0Error, SymbolicIlu0};
pub use jacobi::{JacobiError, JacobiPrecond};
