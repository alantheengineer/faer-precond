# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-06-20

### Added

- `ssor::Ssor<I, T>` — SSOR / symmetric Gauss-Seidel preconditioner, a
  fill-free stationary method built from `A`'s own `D`/`L`/`U` split and one
  relaxation factor (`SsorParams { omega }`; `omega = 1` is SGS). Apply is one
  lower solve, a diagonal scaling, and one upper solve — zero scratch. Symmetric
  for SPD `A`. `try_new` / `refactorize` (allocation-free, pattern depends only
  on `A`'s structure); `SsorError` (`NonSquareMatrix`, `MissingDiagonal`,
  `UnsortedRowIndices`, `ZeroDiagonal`, `InvalidOmega`, `PatternMismatch`).
- `poly::Poly<I, T>` — polynomial preconditioner, `M^{-1} = p(A)` applied with
  matrix-vector products only (no triangular solves), so it parallelises freely.
  `PolyKind::{Neumann, Chebyshev}`, `PolyParams { degree, kind }`, and
  `try_new_auto` with `BoundEstimate::{Gershgorin, PowerIteration, Manual}` for
  the Chebyshev spectral interval. Chebyshev is only as good as its bounds —
  prefer `Manual` when the spectrum is known. `refactorize` updates the stored
  operator (and re-estimates bounds when built via `try_new_auto`). `PolyError`
  (`NonSquareMatrix`, `ZeroDegree`, `InvalidOmega`, `InvalidBounds`,
  `PatternMismatch`).
- `iluk::Iluk<I, T>` — level-of-fill incomplete LU, ILU(k). Generalises ILU(0)
  with structural fill up to level `k` (`IlukParams { level }`); `k = 0`
  reproduces ILU(0) exactly. Value-independent pattern, so it keeps ILU(0)'s
  symbolic/numeric split (`SymbolicIluk`, `new_with_symbolic`) and
  allocation-free `refactorize`. Reuses ILU(0)'s triangular-solve apply.
  `IlukError` mirrors `Ilu0Error`.
- `ict::Ict<I, T>` — threshold incomplete Cholesky, the SPD analogue of ILUTP.
  Adaptive fill via a relative drop tolerance and a per-column budget
  (`IctParams { drop_tol, fill, norm }`, reusing `FillControl` / `RowNorm`).
  Apply is the same two triangular solves as IC(0). `try_new` /
  `try_new_with_params` / `refactorize` (reuses capacity; not allocation-free,
  as the pattern is value-dependent). `IctError` (`NonSquareMatrix`,
  `NotPositiveDefinite`, `PatternMismatch`, `InvalidDropTol`,
  `InvalidFillControl`).
- `fsai::Fsai<I, T>` — factorised sparse approximate inverse for HPD `A`:
  `M^{-1} = G^H G` with `G ~= L^{-1}`, built from one small dense SPD solve per
  row. Apply is two sparse matvecs (no triangular solves). `FsaiPattern::{LowerOfA,
  LowerOfPower}`. `FsaiError` (`NonSquareMatrix`, `InvalidPower`,
  `NotPositiveDefinite`).
- `spai::Spai<I, T>` — sparse approximate inverse for general (nonsymmetric)
  `A`, minimising `||A M - I||_F` column-by-column via dense least squares.
  Apply is a single sparse matvec. `SpaiPattern::{ColumnsOfA, ColumnsOfPower}`.
  `SpaiError` (`NonSquareMatrix`, `InvalidPower`). Note: the build is heavier
  than an ILU (a least-squares solve per column), so it pays off when the
  preconditioner is applied many times; adaptive pattern growth is future work.
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

[Unreleased]: https://github.com/alantheengineer/faer-precond/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/alantheengineer/faer-precond/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/alantheengineer/faer-precond/releases/tag/v0.1.0
