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
//! | [`Ssor`] | `A`'s `D`/`L`/`U` split + relaxation | `O(nnz(A))` | Stationary; two triangular solves, no fill. |
//! | [`Ilu0`] | CSC sparsity of `A` | `O(nnz(A))` | Zero-fill incomplete LU. |
//! | [`Iluk`] | CSC of `A` + fill level `k` | `O(nnz_LU)` | Level-of-fill incomplete LU; between ILU(0) and ILUTP. |
//! | [`Ic0`] | CSC lower triangle of `A` | `O(nnz_L)` | Zero-fill incomplete Cholesky for HPD `A`. |
//! | [`Ict`] | CSC of `A` + threshold/fill params | `O(nnz_L)` | Threshold incomplete Cholesky; SPD analogue of ILUTP. |
//! | [`Ilutp`] | CSC of `A` + threshold/fill params | `O(nnz_LU)` | Threshold ILU with partial pivoting; general nonsymmetric workhorse. |
//! | [`Poly`] | `A` + polynomial degree/bounds | `O(degree · nnz(A))` | Matvec-only (Neumann/Chebyshev); no triangular solves. |
//! | [`Fsai`] | CSC of `A` + pattern | `O(nnz_G)` | Factorised approximate inverse for HPD `A`; matvec apply. |
//! | [`Spai`] | CSC of `A` + pattern | `O(nnz_M)` | Sparse approximate inverse (nonsymmetric); matvec apply. |
//! | [`SolvePrecond`] | any faer factorisation (`Llt`, `Lu`, `Qr`, ...) | factorisation-dependent | Adapter, not a factorisation. |
//!
//! # Choosing a preconditioner
//!
//! There is no single best preconditioner; the right one depends on the
//! structure of `A` and how much work you can afford per iteration.
//!
//! - **Start with [`JacobiPrecond`].** It is almost free to build and apply,
//!   and it helps whenever `A`'s rows differ in scale — diagonally dominant
//!   systems, variable-coefficient PDEs, badly-scaled unknowns. If `A` has a
//!   constant diagonal it does nothing, so move on.
//! - **[`Ic0`] for symmetric positive-definite `A`.** The standard choice for
//!   SPD problems from PDE discretisations (Laplacians, diffusion, elasticity)
//!   solved with conjugate gradient. It cuts iteration counts sharply; each
//!   apply costs two sparse triangular solves, so it always wins on iteration
//!   count and wins on wall-clock time once the problem is ill-conditioned
//!   enough to need many iterations.
//! - **[`Ilu0`] for general (nonsymmetric) sparse `A`.** The nonsymmetric
//!   counterpart to IC(0), paired with GMRES or BiCGSTAB. Same zero-fill idea:
//!   cheap to build and stores nothing beyond `A`'s sparsity pattern.
//! - **[`Ilutp`] when [`Ilu0`] is too weak.** Threshold ILU with partial
//!   pivoting: it adds fill where the factor needs it (tuned by a drop tolerance
//!   and a fill budget) and pivots for stability — the robust choice for hard
//!   nonsymmetric problems, badly-scaled operators, or matrices with small/zero
//!   diagonal entries. Costs more to build and apply than [`Ilu0`], and its
//!   pattern is value-dependent (no zero-allocation refactorisation).
//! - **[`BlockJacobiPrecond`] when unknowns cluster into small dense groups.**
//!   Several fields per mesh node, coupled species, or tightly-coupled
//!   sub-systems. Inverting those blocks exactly captures the strong local
//!   coupling that point-Jacobi misses.
//! - **[`Ssor`] for a cheap, fill-free stationary preconditioner.** Built
//!   straight from `A`'s own triangles with one relaxation knob; a step up from
//!   point-Jacobi when you do not want to store a factorisation. Symmetric for
//!   SPD `A`, so it pairs with CG.
//! - **[`Iluk`] when [`Ilu0`] is too weak but you want a fixed pattern.**
//!   Level-of-fill ILU: more accurate than ILU(0), but with a value-independent
//!   pattern (allocation-free refactorisation), unlike [`Ilutp`].
//! - **[`Ict`] for hard SPD problems.** The threshold incomplete Cholesky:
//!   IC(0)'s adaptive cousin, for ill-conditioned SPD systems where zero-fill
//!   stalls. The SPD counterpart to [`Ilutp`].
//! - **[`Poly`], [`Fsai`] or [`Spai`] when triangular solves are the
//!   bottleneck.** These apply through matrix-vector products only — no
//!   sequential forward/back substitution — so they parallelise far better on
//!   many cores or accelerators. [`Poly`] (Neumann/Chebyshev) and [`Fsai`] are
//!   for SPD `A`; [`Spai`] is the nonsymmetric approximate inverse. The trade-off
//!   is a weaker approximation per unit of single-core work; [`Poly`]'s
//!   Chebyshev form also needs a spectral-interval estimate.
//! - **[`SolvePrecond`] to reuse an exact factorisation.** Factorise a cheaper
//!   approximation of `A` once and let the Krylov method correct the rest.
//!
//! When `A`'s values change between iterations but its sparsity pattern does
//! not — the inner loop of a nonlinear solver — build the symbolic factor once
//! and call `refactorize` (see [`Ilu0`] and [`Ic0`]) to avoid reallocating.
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
pub mod fsai;
pub mod ic0;
pub mod ict;
pub mod ilu0;
pub mod iluk;
pub mod ilutp;
pub mod jacobi;
pub mod poly;
pub mod spai;
pub mod ssor;
mod util;

pub use adapters::SolvePrecond;
pub use block_jacobi::{BlockJacobiError, BlockJacobiPrecond};
pub use fsai::{Fsai, FsaiError, FsaiPattern};
pub use ic0::{Ic0, Ic0Error, SymbolicIc0};
pub use ict::{Ict, IctError, IctParams};
pub use ilu0::{Ilu0, Ilu0Error, SymbolicIlu0};
pub use iluk::{Iluk, IlukError, IlukParams, SymbolicIluk};
pub use ilutp::{FillControl, Ilutp, IlutpError, IlutpParams, RowNorm};
pub use jacobi::{JacobiError, JacobiPrecond};
pub use poly::{BoundEstimate, Poly, PolyError, PolyKind, PolyParams};
pub use spai::{Spai, SpaiError, SpaiPattern};
pub use ssor::{Ssor, SsorError, SsorParams};
