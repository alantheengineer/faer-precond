# faer-precond

Preconditioners for iterative solvers built on top of the [faer](https://crates.io/crates/faer) linear algebra crate.

All preconditioners implement faer's `LinOp`, `Precond`, `BiLinOp`, and `BiPrecond` traits from the `matrix_free` module, so they plug directly into faer's Krylov solvers (CG, GMRES, BiCGSTAB, LSMR, etc.).

## Preconditioners

| Preconditioner | Description | Status |
|---|---|---|
| `JacobiPrecond<T>` | Diagonal (Jacobi) scaling — `M = diag(A)` | Complete |
| `SolvePrecond<S>` | Adapter that wraps any faer `SolveCore` factorization (LLT, LU, QR, ...) as a preconditioner | Complete |
| `BlockJacobiPrecond<T>` | Block-diagonal Jacobi | Planned |
| `Ilu0<I, T>` | Zero-fill incomplete LU | Planned |
| `Ic0<I, T>` | Zero-fill incomplete Cholesky | Planned |

## Usage

Add the dependency to your `Cargo.toml`:

```toml
[dependencies]
faer-precond = { git = "https://github.com/your-org/faer-precond" }
faer = "0.24"
dyn-stack = "0.13"
```

### Jacobi preconditioner

The simplest preconditioner — scales each equation by the reciprocal of its diagonal entry.
Best suited as a lightweight smoother or when the matrix is diagonally dominant.

```rust
use core::mem::MaybeUninit;

use faer::{mat, Mat, Par};
use faer::matrix_free::{LinOp, Precond};
use faer_precond::JacobiPrecond;
use dyn_stack::{MemStack, StackReq};

fn main() {
    // Build from an explicit diagonal
    let pc = JacobiPrecond::try_from_diagonal(&[4.0, 2.0, 8.0])
        .expect("no zero diagonal entries");

    // Or extract the diagonal from a matrix
    let a = mat![
        [4.0, 1.0, 0.0],
        [1.0, 2.0, 0.5],
        [0.0, 0.5, 8.0f64],
    ];
    let pc = JacobiPrecond::try_from_matrix_diagonal(a.as_ref())
        .expect("no zero diagonal entries");

    // Apply: out = M^{-1} * rhs
    let rhs = mat![
        [8.0],
        [6.0],
        [16.0f64],
    ];
    let mut out = Mat::<f64>::zeros(3, 1);

    let req = pc.apply_scratch(rhs.ncols(), Par::Seq);
    let mut buf = vec![MaybeUninit::<u8>::uninit(); req.unaligned_bytes_required().max(1)];
    let mut stack = MemStack::new(&mut buf);
    pc.apply(out.as_mut(), rhs.as_ref(), Par::Seq, &mut stack);
    // out is now [2.0, 3.0, 2.0]

    // Or apply in-place
    let mut x = rhs.to_owned();
    let req = pc.apply_in_place_scratch(x.ncols(), Par::Seq);
    let mut buf = vec![MaybeUninit::<u8>::uninit(); req.unaligned_bytes_required().max(1)];
    let mut stack = MemStack::new(&mut buf);
    pc.apply_in_place(x.as_mut(), Par::Seq, &mut stack);
    // x is now [2.0, 3.0, 2.0]
}
```

### SolvePrecond — using a factorization as a preconditioner

Wraps any faer `SolveCore` factorization so it can be passed wherever a
`Precond` is expected. This is useful when you want to use a direct solver
(e.g. Cholesky on a simpler approximation of A) as a preconditioner for an
iterative method on the full system.

```rust
use core::mem::MaybeUninit;

use faer::{mat, Mat, Par, Side};
use faer::linalg::solvers::Llt;
use faer::matrix_free::{LinOp, Precond};
use faer_precond::SolvePrecond;
use dyn_stack::{MemStack, StackReq};

fn main() {
    // Factor a symmetric positive-definite matrix
    let a = mat![
        [4.0, 1.0],
        [1.0, 3.0f64],
    ];
    let llt = Llt::new(a.as_ref(), Side::Lower)
        .expect("matrix should be SPD");

    // Wrap the factorization as a preconditioner
    let pc = SolvePrecond::new(llt);

    // Apply: out = A^{-1} * rhs
    let rhs = mat![
        [1.0],
        [2.0f64],
    ];
    let mut out = Mat::<f64>::zeros(2, 1);

    let req = pc.apply_scratch(rhs.ncols(), Par::Seq);
    let mut buf = vec![MaybeUninit::<u8>::uninit(); req.unaligned_bytes_required().max(1)];
    let mut stack = MemStack::new(&mut buf);
    pc.apply(out.as_mut(), rhs.as_ref(), Par::Seq, &mut stack);
    // out is now [1/11, 7/11]

    // Or apply in-place
    let mut x = rhs.to_owned();
    let req = pc.apply_in_place_scratch(x.ncols(), Par::Seq);
    let mut buf = vec![MaybeUninit::<u8>::uninit(); req.unaligned_bytes_required().max(1)];
    let mut stack = MemStack::new(&mut buf);
    pc.apply_in_place(x.as_mut(), Par::Seq, &mut stack);
    // x is now [1/11, 7/11]
}
```

## Trait implementations

Every completed preconditioner implements the full trait hierarchy:

- **`LinOp<T>`** — `apply` and `conj_apply` (out-of-place)
- **`Precond<T>`** — `apply_in_place` and `conj_apply_in_place`
- **`BiLinOp<T>`** — `transpose_apply` and `adjoint_apply`
- **`BiPrecond<T>`** — `transpose_apply_in_place` and `adjoint_apply_in_place`

All scratch requirements are `StackReq::EMPTY` — no heap allocation occurs during application.