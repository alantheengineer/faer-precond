//! Sparse approximate inverses applied as matvecs: FSAI on an SPD Laplacian
//! (with CG) and SPAI on a nonsymmetric advection-diffusion operator (with
//! BiCGSTAB). Both build an explicit `M ~= A^{-1}` whose apply is a sparse
//! matrix-vector product — no triangular solves — and a denser prescribed
//! pattern buys accuracy.
//!
//! Run with:
//! ```text
//! cargo run --release --example approx_inverse
//! ```

use dyn_stack::{MemBuffer, MemStack};
use faer::matrix_free::IdentityPrecond;
use faer::matrix_free::bicgstab::{BicgParams, bicgstab, bicgstab_scratch};
use faer::matrix_free::conjugate_gradient::{
    CgParams, conjugate_gradient, conjugate_gradient_scratch,
};
use faer::sparse::{SparseColMat, Triplet};
use faer::{Mat, Par};
use faer_precond::{Fsai, FsaiPattern, Spai, SpaiPattern};

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

fn bicg_iters<P: faer::matrix_free::Precond<f64>>(
    a: &SparseColMat<usize, f64>,
    b: &Mat<f64>,
    left: P,
) -> usize {
    let n = a.nrows();
    let right = IdentityPrecond { dim: n };
    let mut out = Mat::<f64>::zeros(n, 1);
    let params = BicgParams::<f64> {
        max_iters: 2000,
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
    // --- FSAI on an SPD Laplacian, driven by CG ---
    let grid = 40;
    let a = laplacian_2d(grid);
    let n = a.nrows();
    let b = Mat::<f64>::from_fn(n, 1, |i, _| (i % 11) as f64 - 5.0);

    println!("FSAI (factorised approximate inverse) + CG on a {grid}x{grid} Laplacian ({n} unknowns):");
    println!("  no preconditioner       : {:>4} CG iterations", cg_iters(&a, &b, IdentityPrecond { dim: n }));
    for power in [1usize, 2, 3] {
        let pc = Fsai::<usize, f64>::try_new(a.as_ref(), FsaiPattern::LowerOfPower { power }).unwrap();
        println!(
            "  pattern = lower(A^{power})   : {:>4} CG iterations",
            cg_iters(&a, &b, &pc)
        );
    }

    // --- SPAI on a nonsymmetric operator, driven by BiCGSTAB ---
    let beta = 0.7;
    let a = advection_diffusion_2d(grid, beta);
    let n = a.nrows();
    let b = Mat::<f64>::from_fn(n, 1, |i, _| (i % 11) as f64 - 5.0);

    println!("\nSPAI (sparse approximate inverse) + BiCGSTAB on a {grid}x{grid} advection-diffusion operator (beta = {beta}):");
    println!("  no preconditioner       : {:>4} BiCGSTAB iterations", bicg_iters(&a, &b, IdentityPrecond { dim: n }));
    for power in [1usize, 2, 3] {
        let pc = Spai::<usize, f64>::try_new(a.as_ref(), SpaiPattern::ColumnsOfPower { power }).unwrap();
        println!(
            "  pattern = cols(A^{power})     : {:>4} BiCGSTAB iterations",
            bicg_iters(&a, &b, &pc)
        );
    }

    println!(
        "\nNote: both apply as a sparse matvec (FSAI: two; SPAI: one) — no triangular\n\
         solve — and the build is heavier than an ILU, so they pay off when the\n\
         preconditioner is applied many times or on parallel hardware."
    );
}
