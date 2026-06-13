# faer-precond

Numerical preconditioners for iterative linear solvers, built on top of the
[faer](https://crates.io/crates/faer) linear algebra crate.

Every preconditioner implements faer's `LinOp`, `Precond`, `BiLinOp`, and
`BiPrecond` traits from the `matrix_free` module, so they plug directly into
faer's Krylov solvers (CG, GMRES, BiCGSTAB, LSMR, ...).

## Preconditioners

| Preconditioner | Description | Status |
|---|---|---|
| `JacobiPrecond<T>` | Diagonal (point-Jacobi) scaling — `M = diag(A)` | ✅ |
| `BlockJacobiPrecond<T>` | Block-diagonal Jacobi — arbitrary block sizes, LU per block | ✅ |
| `Ilu0<I, T>` | Zero-fill incomplete LU on a CSC matrix | ✅ |
| `Ic0<I, T>` | Zero-fill incomplete Cholesky on a CSC Hermitian PD matrix | ✅ |
| `SolvePrecond<S>` | Adapter wrapping any faer `SolveCore` factorisation (`Llt`, `Lu`, `Qr`, ...) | ✅ |

## Design contract

- **No heap allocation during `apply`.** Every `apply_scratch` returns either
  `StackReq::EMPTY` or a precise pre-computed size; all temporary memory
  flows through the `MemStack` provided by faer's trait interface.
- **Refactorisation reuses storage.** `Ilu0::refactorize` and
  `Ic0::refactorize` mutate the existing value buffer for the next iteration
  of a nonlinear Krylov driver — no allocation occurs after the first
  factorisation.
- **In-place semantics match faer.** `apply` performs `out = M⁻¹ rhs`;
  `apply_in_place` overwrites `rhs`.

## Install

```toml
[dependencies]
faer-precond = "0.1"
faer = "0.24"
dyn-stack = "0.13"
```

## Quick start

### Point-Jacobi

```rust
use dyn_stack::MemStack;
use faer::{mat, Par};
use faer::matrix_free::Precond;
use faer_precond::JacobiPrecond;

let pc = JacobiPrecond::try_from_diagonal(&[4.0_f64, 2.0, 8.0]).unwrap();

let mut x = mat![[8.0_f64], [6.0], [16.0]];
pc.apply_in_place(x.as_mut(), Par::Seq, MemStack::new(&mut []));
// x is now [2.0, 3.0, 2.0]
```

### Block-Jacobi

```rust
use dyn_stack::MemStack;
use faer::{mat, Par};
use faer::matrix_free::Precond;
use faer_precond::BlockJacobiPrecond;

// 5x5 matrix; two diagonal blocks of size 2 and 3.
let a = mat![
    [4.0_f64, 1.0, 0.0, 0.0, 0.0],
    [2.0,     3.0, 0.0, 0.0, 0.0],
    [0.0,     0.0, 6.0, 1.0, 2.0],
    [0.0,     0.0, 3.0, 5.0, 1.0],
    [0.0,     0.0, 2.0, 1.0, 4.0],
];
let pc = BlockJacobiPrecond::try_new(a.as_ref(), &[0, 2, 5]).unwrap();

let mut x = mat![[1.0_f64], [2.0], [3.0], [-1.0], [0.5]];
pc.apply_in_place(x.as_mut(), Par::Seq, MemStack::new(&mut []));
```

### ILU(0)

```rust
use dyn_stack::MemStack;
use faer::sparse::{SparseColMat, Triplet};
use faer::{mat, Par};
use faer::matrix_free::Precond;
use faer_precond::Ilu0;

// 5x5 tridiagonal SPD: diag 4, off -1 (no fill — ILU(0) is exact).
let mut triplets = Vec::new();
for i in 0..5 {
    triplets.push(Triplet::new(i, i, 4.0_f64));
    if i > 0 {
        triplets.push(Triplet::new(i, i - 1, -1.0));
        triplets.push(Triplet::new(i - 1, i, -1.0));
    }
}
let a = SparseColMat::<usize, f64>::try_new_from_triplets(5, 5, &triplets).unwrap();

let pc = Ilu0::try_new(a.as_ref()).expect("non-singular pattern");

let mut b = mat![[1.0_f64], [0.0], [0.0], [0.0], [0.0]];
pc.apply_in_place(b.as_mut(), Par::Seq, MemStack::new(&mut []));
```

### IC(0)

```rust
use dyn_stack::MemStack;
use faer::sparse::{SparseColMat, Triplet};
use faer::{mat, Par};
use faer::matrix_free::Precond;
use faer_precond::Ic0;

let mut triplets = Vec::new();
for i in 0..5 {
    triplets.push(Triplet::new(i, i, 4.0_f64));
    if i > 0 {
        triplets.push(Triplet::new(i, i - 1, -1.0));
        triplets.push(Triplet::new(i - 1, i, -1.0));
    }
}
let a = SparseColMat::<usize, f64>::try_new_from_triplets(5, 5, &triplets).unwrap();

// Hermitian PD input — IC(0) silently ignores the strict upper triangle.
let pc = Ic0::try_new(a.as_ref()).expect("matrix is positive definite");

let mut b = mat![[1.0_f64], [0.0], [0.0], [0.0], [0.0]];
pc.apply_in_place(b.as_mut(), Par::Seq, MemStack::new(&mut []));
```

### Wrapping a faer factorisation as a preconditioner

```rust
use dyn_stack::MemStack;
use faer::{mat, Par, Side};
use faer::linalg::solvers::Llt;
use faer::matrix_free::Precond;
use faer_precond::SolvePrecond;

let a = mat![[4.0_f64, 1.0], [1.0, 3.0]];
let llt = Llt::new(a.as_ref(), Side::Lower).expect("matrix is SPD");
let pc = SolvePrecond::new(llt);

let mut x = mat![[1.0_f64], [2.0]];
pc.apply_in_place(x.as_mut(), Par::Seq, MemStack::new(&mut []));
// x is now [1/11, 7/11]
```

## Repeated factorisation (nonlinear Krylov)

When `A`'s sparsity pattern is fixed but its values change between Krylov
iterations, build the symbolic factor once and refactorise repeatedly with
zero allocation:

```rust
use faer::sparse::{SparseColMat, Triplet};
use faer_precond::{Ilu0, SymbolicIlu0};

# let triplets: Vec<Triplet<usize, usize, f64>> = (0..3).map(|i| Triplet::new(i, i, 1.0_f64)).collect();
# let a0 = SparseColMat::<usize, f64>::try_new_from_triplets(3, 3, &triplets).unwrap();
let symbolic = SymbolicIlu0::try_new(a0.as_ref().symbolic()).unwrap();
let mut pc = Ilu0::<usize, f64>::new_with_symbolic(symbolic);

// Hot loop — no allocation:
pc.refactorize(a0.as_ref()).unwrap();
// pc.refactorize(a1.as_ref()).unwrap();
// pc.refactorize(a2.as_ref()).unwrap();
```

## Trait coverage

Every preconditioner implements the full trait hierarchy:

- **`LinOp<T>`** — `apply`, `conj_apply`
- **`Precond<T>`** — `apply_in_place`, `conj_apply_in_place`
- **`BiLinOp<T>`** — `transpose_apply`, `adjoint_apply`
- **`BiPrecond<T>`** — `transpose_apply_in_place`, `adjoint_apply_in_place`

## Minimum supported Rust version

`rustc 1.88` (edition 2024 + let-chains).

## License

Dual-licensed under either of

- Apache License, Version 2.0 — [LICENSE-APACHE](LICENSE-APACHE)
- MIT license — [LICENSE-MIT](LICENSE-MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual-licensed as above, without any additional terms or
conditions.
