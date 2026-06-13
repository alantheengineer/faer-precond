//! IC(0)-preconditioned CG on the 2-D Laplacian.
//!
//! Compares the iteration count required to converge with and without the
//! IC(0) preconditioner — IC(0) is a standard choice for SPD problems
//! arising from PDE discretisations.
//!
//! Run with:
//! ```text
//! cargo run --example ic0
//! ```

use dyn_stack::{MemBuffer, MemStack};
use faer::matrix_free::IdentityPrecond;
use faer::matrix_free::conjugate_gradient::{
    CgParams, conjugate_gradient, conjugate_gradient_scratch,
};
use faer::sparse::{SparseColMat, Triplet};
use faer::{Mat, Par};
use faer_precond::Ic0;

fn laplacian_2d(grid: usize) -> SparseColMat<usize, f64> {
    let n = grid * grid;
    let mut triplets = Vec::new();
    for gy in 0..grid {
        for gx in 0..grid {
            let idx = gy * grid + gx;
            triplets.push(Triplet::new(idx, idx, 4.0));
            if gx > 0 {
                triplets.push(Triplet::new(idx, idx - 1, -1.0));
            }
            if gx + 1 < grid {
                triplets.push(Triplet::new(idx, idx + 1, -1.0));
            }
            if gy > 0 {
                triplets.push(Triplet::new(idx, idx - grid, -1.0));
            }
            if gy + 1 < grid {
                triplets.push(Triplet::new(idx, idx + grid, -1.0));
            }
        }
    }
    SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap()
}

fn solve_cg<P: faer::matrix_free::Precond<f64>>(
    a: &SparseColMat<usize, f64>,
    b: &Mat<f64>,
    pc: P,
) -> (usize, f64) {
    let n = a.nrows();
    let mut out = Mat::<f64>::zeros(n, 1);

    let params = CgParams::<f64> {
        max_iters: 1000,
        rel_tolerance: 1e-10,
        ..Default::default()
    };

    let mut buf = MemBuffer::new(conjugate_gradient_scratch(&pc, a.as_ref(), 1, Par::Seq));
    let info = conjugate_gradient(
        out.as_mut(),
        pc,
        a.as_ref(),
        b.as_ref(),
        params,
        |_| {},
        Par::Seq,
        MemStack::new(&mut buf),
    )
    .expect("CG should converge");
    (info.iter_count, info.rel_residual)
}

fn main() {
    let grid = 32;
    let a = laplacian_2d(grid);
    let n = a.nrows();
    let b = Mat::<f64>::from_fn(n, 1, |i, _| (i % 7) as f64 - 3.0);

    println!("Problem: 2-D Laplacian on a {grid}x{grid} grid ({n} unknowns)");

    let (iters_none, _) = solve_cg(&a, &b, IdentityPrecond { dim: n });
    println!("CG (no preconditioner):    {iters_none:>4} iterations");

    let pc = Ic0::<usize, f64>::try_new(a.as_ref()).expect("Laplacian is SPD");
    let (iters_ic0, _) = solve_cg(&a, &b, &pc);
    println!("CG + IC(0):                {iters_ic0:>4} iterations");

    let speedup = iters_none as f64 / iters_ic0 as f64;
    println!("Iteration reduction:       {speedup:>4.1}x");
}
