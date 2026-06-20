//! SSOR / symmetric Gauss-Seidel-preconditioned CG on a 2-D Laplacian, and how
//! the relaxation factor `omega` trades off against the iteration count.
//!
//! SSOR is a stationary preconditioner: no factorisation, no fill — just two
//! triangular solves against `A`'s own entries. It is the natural step up from
//! point-Jacobi when you want something stronger but still cheap to build.
//!
//! Run with:
//! ```text
//! cargo run --release --example ssor
//! ```

use dyn_stack::{MemBuffer, MemStack};
use faer::matrix_free::IdentityPrecond;
use faer::matrix_free::conjugate_gradient::{
    CgParams, conjugate_gradient, conjugate_gradient_scratch,
};
use faer::sparse::{SparseColMat, Triplet};
use faer::{Mat, Par};
use faer_precond::{Ssor, SsorParams};

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

fn cg_iters<P: faer::matrix_free::Precond<f64>>(
    a: &SparseColMat<usize, f64>,
    b: &Mat<f64>,
    pc: P,
) -> usize {
    let n = a.nrows();
    let mut out = Mat::<f64>::zeros(n, 1);
    let params = CgParams::<f64> {
        max_iters: 2000,
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
    info.iter_count
}

fn main() {
    let grid = 48;
    let a = laplacian_2d(grid);
    let n = a.nrows();
    let b = Mat::<f64>::from_fn(n, 1, |i, _| (i % 11) as f64 - 5.0);

    println!("Problem: 2-D Laplacian, {grid}x{grid} grid ({n} unknowns)");
    println!("CG to rel-residual 1e-10.\n");

    let none = cg_iters(&a, &b, IdentityPrecond { dim: n });
    println!("no preconditioner : {none:>4} iterations");

    println!("\nSSOR, varying relaxation factor omega:");
    for &omega in &[0.8_f64, 1.0, 1.3, 1.6, 1.9] {
        let pc = Ssor::<usize, f64>::try_new(a.as_ref(), SsorParams { omega }).unwrap();
        let it = cg_iters(&a, &b, &pc);
        let tag = if (omega - 1.0).abs() < 1e-9 { " (SGS)" } else { "" };
        println!("  omega = {omega:>3}{tag:<6} ->  {it:>4} iterations");
    }
}
