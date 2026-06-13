//! ILUTP-preconditioned BiCGSTAB on a strongly non-symmetric system, and a look
//! at how the drop tolerance / fill / pivoting knobs trade factor cost against
//! convergence.
//!
//! The operator is a 2-D advection–diffusion stencil: the 5-point Laplacian
//! plus an asymmetric first-derivative term of strength `beta`. As `beta` grows
//! the matrix becomes strongly non-symmetric and ILU(0)'s fixed pattern weakens
//! — exactly where ILUTP's adaptive fill and pivoting pay off.
//!
//! Run with:
//! ```text
//! cargo run --release --example ilutp
//! ```

use dyn_stack::{MemBuffer, MemStack};
use faer::matrix_free::IdentityPrecond;
use faer::matrix_free::bicgstab::{BicgParams, bicgstab, bicgstab_scratch};
use faer::sparse::{SparseColMat, Triplet};
use faer::{Mat, Par};
use faer_precond::{FillControl, Ilu0, Ilutp, IlutpParams};

/// 2-D advection–diffusion on a `grid x grid` mesh with asymmetry `beta`.
fn advection_diffusion_2d(grid: usize, beta: f64) -> SparseColMat<usize, f64> {
    let n = grid * grid;
    let mut triplets = Vec::new();
    for gy in 0..grid {
        for gx in 0..grid {
            let idx = gy * grid + gx;
            triplets.push(Triplet::new(idx, idx, 4.0));
            if gx > 0 {
                triplets.push(Triplet::new(idx, idx - 1, -1.0 - beta));
            }
            if gx + 1 < grid {
                triplets.push(Triplet::new(idx, idx + 1, -1.0 + beta));
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

/// Left-preconditioned BiCGSTAB; returns the iteration count.
fn iters<P: faer::matrix_free::Precond<f64>>(
    a: &SparseColMat<usize, f64>,
    b: &Mat<f64>,
    left: P,
) -> usize {
    let n = a.nrows();
    let right = IdentityPrecond { dim: n };
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
    info.iter_count
}

fn main() {
    let grid = 24;
    let beta = 0.9; // strongly non-symmetric
    let a = advection_diffusion_2d(grid, beta);
    let n = a.nrows();
    let b = Mat::<f64>::from_fn(n, 1, |i, _| (i % 11) as f64 - 5.0);

    println!("Problem: 2-D advection-diffusion, {grid}x{grid} grid ({n} unknowns), beta = {beta}");
    println!("BiCGSTAB to rel-residual 1e-10, left-preconditioned.\n");

    let none = iters(&a, &b, IdentityPrecond { dim: n });
    println!("no preconditioner : {none:>4} iterations");

    let ilu0 = Ilu0::<usize, f64>::try_new(a.as_ref()).unwrap();
    println!("ILU(0)            : {:>4} iterations", iters(&a, &b, &ilu0));

    // Sweep ILUTP fill budgets: more kept fill => stronger factor, fewer iters.
    println!("\nILUTP (drop_tol = 1e-3, pivot_tol = 0.1), varying fill budget:");
    for &p in &[2usize, 5, 10, 20] {
        let params = IlutpParams {
            drop_tol: 1e-3,
            fill: FillControl::PerRow(p),
            pivot_tol: 0.1,
            ..Default::default()
        };
        let pc = Ilutp::<usize, f64>::try_new_with_params(a.as_ref(), params).unwrap();
        let it = iters(&a, &b, &pc);
        let nnz = pc.l_view().val().len() + pc.u_view().val().len();
        println!(
            "  fill/row = {p:>2}  ->  {it:>4} iterations   (factor nnz = {nnz}, permuted = {})",
            pc.is_permuted()
        );
    }

    let default = Ilutp::<usize, f64>::try_new(a.as_ref()).unwrap();
    let it = iters(&a, &b, &default);
    println!("\nILUTP (defaults)  : {it:>4} iterations");
    if it > 0 {
        println!(
            "Iteration reduction vs unpreconditioned: {:.1}x",
            none as f64 / it as f64
        );
    }
}
