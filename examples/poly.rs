//! Polynomial-preconditioned CG on a 2-D Laplacian: Chebyshev vs Neumann, and
//! how the polynomial degree trades matvecs-per-apply against iteration count.
//!
//! A polynomial preconditioner applies `M^{-1} = p(A)` using only matrix-vector
//! products — no triangular solves — so each apply costs `degree` matvecs but
//! parallelises freely. Chebyshev needs a spectral interval; for the Laplacian
//! we know it in closed form and pass it explicitly.
//!
//! Run with:
//! ```text
//! cargo run --release --example poly
//! ```

use dyn_stack::{MemBuffer, MemStack};
use faer::matrix_free::IdentityPrecond;
use faer::matrix_free::conjugate_gradient::{
    CgParams, conjugate_gradient, conjugate_gradient_scratch,
};
use faer::sparse::{SparseColMat, Triplet};
use faer::{Mat, Par};
use faer_precond::{Poly, PolyKind, PolyParams};

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

    // Exact 5-point Laplacian spectral bounds.
    let h = std::f64::consts::PI / (grid as f64 + 1.0);
    let lambda_min = 4.0 - 4.0 * h.cos();
    let lambda_max = 4.0 - 4.0 * (grid as f64 * h).cos();

    println!("Problem: 2-D Laplacian, {grid}x{grid} grid ({n} unknowns)");
    println!("Spectrum in [{lambda_min:.4}, {lambda_max:.4}]. CG to rel-residual 1e-10.\n");

    let none = cg_iters(&a, &b, IdentityPrecond { dim: n });
    println!("no preconditioner          : {none:>4} CG iterations");

    println!("\nChebyshev polynomial, varying degree (each apply = `degree` matvecs):");
    for &degree in &[2usize, 4, 8, 16] {
        let pc = Poly::<usize, f64>::try_new(a.as_ref(), PolyParams {
            degree,
            kind: PolyKind::Chebyshev {
                lambda_min,
                lambda_max,
            },
        })
        .unwrap();
        let it = cg_iters(&a, &b, &pc);
        println!("  degree = {degree:>2}  ->  {it:>4} CG iterations   (~{} matvecs/apply)", degree);
    }

    println!("\nNeumann series (omega = 1 / lambda_max), varying degree:");
    for &degree in &[2usize, 4, 8, 16] {
        let pc = Poly::<usize, f64>::try_new(a.as_ref(), PolyParams {
            degree,
            kind: PolyKind::Neumann {
                omega: 1.0 / lambda_max,
            },
        })
        .unwrap();
        let it = cg_iters(&a, &b, &pc);
        println!("  degree = {degree:>2}  ->  {it:>4} CG iterations");
    }

    println!(
        "\nNote: a polynomial preconditioner rarely beats IC(0) on raw CG iterations,\n\
         but every apply is matvec-only — no sequential triangular solve — which is\n\
         what makes it attractive on parallel hardware and as a multigrid smoother."
    );
}
