# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `ilutp::Ilutp<I, T>` — threshold ILU with column partial pivoting (Saad's
  dual-threshold ILUT + the "P", SPARSKIT `ilutp`). The general-purpose
  workhorse for hard nonsymmetric systems: it keeps the most significant fill
  per row (tuned by a relative drop tolerance and a fill budget) and pivots
  columns for stability, so it handles badly-scaled operators and matrices with
  small or absent diagonal entries. Implements the full
  `LinOp` / `Precond` / `BiLinOp` / `BiPrecond` hierarchy.
  - `IlutpParams` (`drop_tol`, `fill`, `pivot_tol`, `norm`) with sensible
    defaults, `FillControl::{PerRow, Factor}`, and `RowNorm::{One, Two}`.
  - `try_new` / `try_new_with_params` construction plus `refactorize` (reuses
    buffer capacity; not allocation-free, since the fill pattern is
    value-dependent), and `l_view` / `u_view` / `perm` / `perm_inv` /
    `is_permuted` accessors.
  - `IlutpError` (`NonSquareMatrix`, `ZeroPivot`, `InvalidDropTol`,
    `InvalidPivotTol`, `InvalidFillControl`) implementing `core::error::Error`.
- `examples/speedup.rs` — measures end-to-end wall-clock time (build + solve)
  with vs without preconditioning on a high-contrast variable-coefficient
  diffusion problem, the regime where preconditioning genuinely pays off.
- `examples/ilutp.rs` — ILUTP + BiCGSTAB on a nonsymmetric advection-diffusion
  problem, showing the fill-budget vs iteration-count tradeoff.
- `cg_full_solve` benchmark group timing the full CG solve (build + iterate)
  with vs without preconditioners, plus `ilutp_construct` / `ilutp_apply_in_place`
  benchmarks.

### Changed

- `Ilutp::apply` returns a non-empty `StackReq` (`StackReq::new::<T>(n)`) so the
  pivot permutation can be applied through the caller's `MemStack`; `apply`
  remains heap-allocation-free in the design-contract sense, matching the
  precedent set by `BlockJacobiPrecond`.

### Documentation

- Documented the previously undocumented `JacobiPrecond` and `SolvePrecond`,
  including when each is the right choice.
- Added a "Choosing a preconditioner" guide to the crate docs and README in
  plain English, with per-method recommendations.
- Reordered the ILU(0)/IC(0)/Block-Jacobi module docs to lead with purpose and
  "when to use it", moving CSC storage-layout detail to a `# Storage` section,
  and replaced the non-compiling `ignore` doc examples with runnable ones.

## [0.1.0] - 2026-06-13

### Added

- `JacobiPrecond<T>` — diagonal (point-Jacobi) preconditioner with
  `try_from_diagonal` / `try_from_matrix_diagonal` constructors and full
  `LinOp` / `Precond` / `BiLinOp` / `BiPrecond` trait impls.
- `BlockJacobiPrecond<T>` — block-diagonal preconditioner with arbitrary
  block sizes given by `block_offsets`. Each diagonal block is factored
  once via partial-pivoted LU at construction; apply is heap-allocation-free.
- `SolvePrecond<S>` adapter that wraps any faer `SolveCore` factorisation
  (LLT, LDLT, LU, QR, ...) as a preconditioner.
- `ilu0::Ilu0<I, T>` — zero-fill incomplete LU for CSC matrices, with both
  one-shot construction (`try_new`) and zero-allocation refactorisation
  (`refactorize`) for the same sparsity pattern.
- `ic0::Ic0<I, T>` — zero-fill incomplete Cholesky for Hermitian PD CSC
  matrices, with the same `try_new` / `refactorize` API. Only the lower
  triangle of the input is consumed.
- Typed error enums (`BlockJacobiError`, `Ilu0Error`, `Ic0Error`,
  `JacobiError`) implementing `core::error::Error`.

### Notes

- Every preconditioner's `apply` and `apply_in_place` paths perform zero
  heap allocation — all scratch flows through the `MemStack` passed via
  faer's trait interface.
- ILU(0) and IC(0) reuse `faer::sparse::linalg::triangular_solve` for the
  forward and back solves rather than re-implementing them.

[Unreleased]: https://github.com/alantheengineer/faer-precond/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/alantheengineer/faer-precond/releases/tag/v0.1.0
