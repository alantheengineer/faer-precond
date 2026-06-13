//! End-to-end demonstration that preconditioning *pays off* in wall-clock time.
//!
//! Iteration counts alone don't prove a preconditioner is worth using — each
//! iteration now costs an extra triangular solve, and the preconditioner has to
//! be built up front. This example measures the **total** time to solve
//!
//! ```text
//!     A x = b
//! ```
//!
//! across a range of grid sizes, with conjugate gradient. For each problem it
//! times three setups end-to-end (build + solve):
//!
//! - plain CG (no preconditioner),
//! - CG + point-Jacobi (cheapest possible preconditioner),
//! - CG + IC(0) (zero-fill incomplete Cholesky).
//!
//! The operator is a **high-contrast variable-coefficient diffusion** problem,
//! `-div(k grad u) = b`, where the conductivity `k` jumps by four orders of
//! magnitude between blocks. This is the realistic regime: the constant-
//! coefficient Laplacian is so well-behaved that IC(0)'s iteration savings are
//! eaten by its per-iteration cost, and Jacobi does nothing at all (constant
//! diagonal). With jumping coefficients the matrix is badly conditioned and its
//! diagonal varies wildly — so Jacobi finally helps, and IC(0) wins decisively
//! in *wall-clock* time, not just iteration count.
//!
//! Run with:
//! ```text
//! cargo run --release --example speedup
//! ```

use std::time::{Duration, Instant};

use dyn_stack::{MemBuffer, MemStack};
use faer::matrix_free::conjugate_gradient::{
    CgParams, conjugate_gradient, conjugate_gradient_scratch,
};
use faer::matrix_free::{IdentityPrecond, Precond};
use faer::sparse::{SparseColMat, Triplet};
use faer::{Mat, Par};
use faer_precond::{Ic0, JacobiPrecond};

/// Per-cell conductivity field with high-contrast block jumps.
///
/// Deterministic (no RNG): conductivity alternates between `1.0` and `1e4`
/// across `4x4`-cell blocks in a checkerboard pattern. These four-orders-of-
/// magnitude jumps are what make the resulting system badly conditioned and
/// give it a strongly varying diagonal.
fn conductivity(gx: usize, gy: usize) -> f64 {
    let block = (gx / 4 + gy / 4) % 2;
    if block == 0 { 1.0 } else { 1.0e4 }
}

/// 5-point finite-volume discretisation of `-div(k grad u) = b` on a
/// `grid x grid` mesh with Dirichlet boundaries. SPD and sparse, with `grid^2`
/// unknowns. Face conductivities are the harmonic mean of the two adjacent
/// cells (the standard choice for jumping coefficients).
fn variable_diffusion_2d(grid: usize) -> SparseColMat<usize, f64> {
    let n = grid * grid;
    let harmonic = |a: f64, b: f64| 2.0 * a * b / (a + b);

    let mut triplets = Vec::new();
    for gy in 0..grid {
        for gx in 0..grid {
            let idx = gy * grid + gx;
            let ki = conductivity(gx, gy);
            let mut diag = 0.0;

            let mut face = |ngx: usize, ngy: usize, nidx: usize, diag: &mut f64| {
                let t = harmonic(ki, conductivity(ngx, ngy));
                *diag += t;
                triplets.push(Triplet::new(idx, nidx, -t));
            };

            if gx > 0 {
                face(gx - 1, gy, idx - 1, &mut diag);
            } else {
                diag += ki; // Dirichlet boundary face.
            }
            if gx + 1 < grid {
                face(gx + 1, gy, idx + 1, &mut diag);
            } else {
                diag += ki;
            }
            if gy > 0 {
                face(gx, gy - 1, idx - grid, &mut diag);
            } else {
                diag += ki;
            }
            if gy + 1 < grid {
                face(gx, gy + 1, idx + grid, &mut diag);
            } else {
                diag += ki;
            }

            triplets.push(Triplet::new(idx, idx, diag));
        }
    }
    SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap()
}

struct SolveResult {
    iters: usize,
    rel_residual: f64,
    elapsed: Duration,
}

/// Run CG to convergence and return iteration count + total elapsed time.
///
/// `build` is invoked inside the timed region so the cost of constructing the
/// preconditioner is charged honestly against the solve.
fn timed_solve<P: Precond<f64>>(
    a: &SparseColMat<usize, f64>,
    b: &Mat<f64>,
    build: impl Fn() -> P,
) -> SolveResult {
    let n = a.nrows();
    let params = CgParams::<f64> {
        max_iters: 5000,
        rel_tolerance: 1e-10,
        ..Default::default()
    };

    let start = Instant::now();
    let pc = build();
    let mut out = Mat::<f64>::zeros(n, 1);
    let mut buf = MemBuffer::new(conjugate_gradient_scratch(&pc, a.as_ref(), 1, Par::Seq));
    let info = conjugate_gradient(
        out.as_mut(),
        &pc,
        a.as_ref(),
        b.as_ref(),
        params,
        |_| {},
        Par::Seq,
        MemStack::new(&mut buf),
    )
    .expect("CG should converge");
    let elapsed = start.elapsed();

    SolveResult {
        iters: info.iter_count,
        rel_residual: info.rel_residual,
        elapsed,
    }
}

fn main() {
    println!("High-contrast variable-coefficient diffusion (k jumps 1 <-> 1e4).");
    println!("Conjugate gradient to rel-residual 1e-10.");
    println!("Times include building the preconditioner (best of a few runs).\n");

    println!(
        "{:>5}  {:>8} | {:>20} | {:>20} | {:>20}",
        "grid", "unknowns", "plain CG", "CG + Jacobi", "CG + IC(0)"
    );
    println!(
        "{:->5}  {:->8} | {:->20} | {:->20} | {:->20}",
        "", "", "", "", ""
    );

    for &grid in &[16usize, 32, 64, 128, 192] {
        let a = variable_diffusion_2d(grid);
        let n = a.nrows();
        // A smooth-ish, non-trivial right-hand side.
        let b = Mat::<f64>::from_fn(n, 1, |i, _| ((i % 13) as f64 - 6.0) * 0.5);

        // Take the best (min) of a few runs to damp scheduler noise; small grids
        // are fast enough that a single run is dominated by timer jitter.
        let runs = if n < 5000 { 7 } else { 3 };

        let none = best_of(runs, || timed_solve(&a, &b, || IdentityPrecond { dim: n }));
        let jacobi = best_of(runs, || {
            timed_solve(&a, &b, || {
                let diag: Vec<f64> = (0..n).map(|i| *a.as_ref().get(i, i).unwrap()).collect();
                JacobiPrecond::try_from_diagonal(&diag).unwrap()
            })
        });
        let ic0 = best_of(runs, || {
            timed_solve(&a, &b, || {
                Ic0::<usize, f64>::try_new(a.as_ref()).expect("diffusion operator is SPD")
            })
        });

        // Sanity: every method must reach the same tolerance.
        for r in [&none, &jacobi, &ic0] {
            assert!(
                r.rel_residual <= 1e-9,
                "a solve failed to converge: rel_residual = {}",
                r.rel_residual
            );
        }

        println!(
            "{:>5}  {:>8} | {:>6} it {:>9.2?} | {:>6} it {:>9.2?} | {:>6} it {:>9.2?}",
            grid, n, none.iters, none.elapsed, jacobi.iters, jacobi.elapsed, ic0.iters, ic0.elapsed,
        );
    }

    println!("\nSpeed-up (plain CG time / preconditioned time), largest grid:");
    let grid = 192usize;
    let a = variable_diffusion_2d(grid);
    let n = a.nrows();
    let b = Mat::<f64>::from_fn(n, 1, |i, _| ((i % 13) as f64 - 6.0) * 0.5);

    let none = best_of(3, || timed_solve(&a, &b, || IdentityPrecond { dim: n }));
    let ic0 = best_of(3, || {
        timed_solve(&a, &b, || Ic0::<usize, f64>::try_new(a.as_ref()).unwrap())
    });

    let iter_reduction = none.iters as f64 / ic0.iters as f64;
    let time_speedup = none.elapsed.as_secs_f64() / ic0.elapsed.as_secs_f64();
    println!(
        "  IC(0): {iter_reduction:.1}x fewer iterations, {time_speedup:.1}x faster wall-clock."
    );
}

/// Run `f` `n` times and keep the result with the smallest elapsed time.
fn best_of(n: usize, f: impl Fn() -> SolveResult) -> SolveResult {
    let mut best = f();
    for _ in 1..n {
        let candidate = f();
        if candidate.elapsed < best.elapsed {
            best = candidate;
        }
    }
    best
}
