//! ILU(0)-preconditioned BiCGSTAB on a non-symmetric sparse system.
//!
//! Run with:
//! ```text
//! cargo run --example ilu0
//! ```

use dyn_stack::{MemBuffer, MemStack};
use faer::matrix_free::IdentityPrecond;
use faer::matrix_free::bicgstab::{BicgParams, bicgstab, bicgstab_scratch};
use faer::sparse::{SparseColMat, Triplet};
use faer::{Mat, Par};
use faer_precond::Ilu0;

/// Non-symmetric convection-diffusion-style tridiagonal: 4 on diag, -2 below, -1 above.
fn nonsymmetric_tridiagonal(n: usize) -> SparseColMat<usize, f64> {
    let mut triplets = Vec::new();
    for i in 0..n {
        triplets.push(Triplet::new(i, i, 4.0));
        if i > 0 {
            triplets.push(Triplet::new(i, i - 1, -2.0));
            triplets.push(Triplet::new(i - 1, i, -1.0));
        }
    }
    SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap()
}

fn solve_bicgstab<L, R>(
    a: &SparseColMat<usize, f64>,
    b: &Mat<f64>,
    left: L,
    right: R,
) -> (usize, f64)
where
    L: faer::matrix_free::Precond<f64>,
    R: faer::matrix_free::Precond<f64>,
{
    let n = a.nrows();
    let mut out = Mat::<f64>::zeros(n, 1);

    let params = BicgParams::<f64> {
        max_iters: 1000,
        rel_tolerance: 1e-10,
        ..Default::default()
    };

    let mut buf = MemBuffer::new(bicgstab_scratch(&left, &right, a.as_ref(), 1, Par::Seq));
    let info = bicgstab(
        out.as_mut(),
        left,
        right,
        a.as_ref(),
        b.as_ref(),
        params,
        |_| {},
        Par::Seq,
        MemStack::new(&mut buf),
    )
    .expect("BiCGSTAB should converge");
    (info.iter_count, info.rel_residual)
}

fn main() {
    let n = 256;
    let a = nonsymmetric_tridiagonal(n);
    let b = Mat::<f64>::from_fn(n, 1, |i, _| (i % 11) as f64 - 5.0);

    println!("Problem: non-symmetric tridiagonal, n = {n}");

    let identity = IdentityPrecond { dim: n };
    let (iters_none, _) = solve_bicgstab(&a, &b, identity, identity);
    println!("BiCGSTAB (no preconditioner):    {iters_none:>4} iterations");

    let pc = Ilu0::<usize, f64>::try_new(a.as_ref()).expect("LU pattern is non-singular");
    let (iters_ilu, _) = solve_bicgstab(&a, &b, &pc, identity);
    println!("BiCGSTAB + ILU(0) (left-precond): {iters_ilu:>4} iterations");

    if iters_ilu > 0 {
        let speedup = iters_none as f64 / iters_ilu.max(1) as f64;
        println!("Iteration reduction:             {speedup:>4.1}x");
    }
}
