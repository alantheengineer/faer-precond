# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
