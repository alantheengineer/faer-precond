//! End-to-end integration tests.
//!
//! These tests drive faer's iterative solvers (`conjugate_gradient`,
//! `bicgstab`) using preconditioners from this crate, ensuring the trait
//! implementations actually compose with the wider faer ecosystem.

use dyn_stack::{MemBuffer, MemStack};
use faer::matrix_free::IdentityPrecond;
use faer::matrix_free::bicgstab::{BicgParams, bicgstab, bicgstab_scratch};
use faer::matrix_free::conjugate_gradient::{
    CgParams, conjugate_gradient, conjugate_gradient_scratch,
};
use faer::sparse::{SparseColMat, Triplet};
use faer::{Mat, Par, Side, mat};

use faer_precond::{BlockJacobiPrecond, Ic0, Ilu0, Ilutp, JacobiPrecond, SolvePrecond};

/// 5-point Laplacian on a `grid × grid` mesh (full Hermitian CSC storage).
/// SPD and diagonally dominant — used as the canonical SPD test problem.
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

/// Non-symmetric tridiagonal: 4 on diag, -2 below, -1 above. Used to exercise
/// ILU(0) + BiCGSTAB on a problem CG cannot solve.
fn tridiagonal_nonsymmetric(n: usize) -> SparseColMat<usize, f64> {
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

/// 2-D advection-diffusion: the 5-point Laplacian with an asymmetric
/// first-derivative term of strength `beta`. Strongly nonsymmetric for large
/// `beta` — the regime where ILUTP earns its keep over ILU(0).
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

fn cg_iter_count<P>(a: &SparseColMat<usize, f64>, precond: P, max_iters: usize) -> usize
where
    P: faer::matrix_free::Precond<f64>,
{
    let n = a.nrows();
    let b = Mat::<f64>::from_fn(n, 1, |i, _| (i % 7) as f64 - 3.0);
    let mut out = Mat::<f64>::zeros(n, 1);

    let params = CgParams::<f64> {
        max_iters,
        rel_tolerance: 1e-10,
        ..Default::default()
    };

    let mut buf = MemBuffer::new(conjugate_gradient_scratch(
        &precond,
        a.as_ref(),
        1,
        Par::Seq,
    ));
    let result = conjugate_gradient(
        out.as_mut(),
        precond,
        a.as_ref(),
        b.as_ref(),
        params,
        |_| {},
        Par::Seq,
        MemStack::new(&mut buf),
    );
    let info = result.expect("CG should converge");
    assert!(
        info.rel_residual <= params.rel_tolerance * 10.0,
        "CG did not actually converge (rel_residual = {})",
        info.rel_residual,
    );
    info.iter_count
}

#[test]
fn ic0_accelerates_cg_on_laplacian() {
    let a = laplacian_2d(12);

    let identity = IdentityPrecond { dim: a.nrows() };
    let baseline_iters = cg_iter_count(&a, identity, 500);

    let pc = Ic0::<usize, f64>::try_new(a.as_ref()).expect("Laplacian is SPD");
    let pc_iters = cg_iter_count(&a, &pc, 500);

    assert!(
        pc_iters < baseline_iters,
        "IC(0) should accelerate CG: baseline {baseline_iters} iters vs preconditioned {pc_iters}",
    );
}

#[test]
fn jacobi_accelerates_or_matches_cg_on_laplacian() {
    // For a Laplacian with constant diagonal, point-Jacobi is just a rescaling
    // and Krylov subspaces are invariant under uniform diagonal scaling.
    // The iteration count should be <= baseline (it certainly should not regress).
    let a = laplacian_2d(10);
    let n = a.nrows();

    let identity = IdentityPrecond { dim: n };
    let baseline_iters = cg_iter_count(&a, identity, 500);

    // Build a Jacobi from the diagonal.
    let diag: Vec<f64> = (0..n)
        .map(|i| {
            // Look up A[i,i] in the sparse pattern.
            let a_ref = a.as_ref();
            let rows = a_ref.symbolic().row_idx_of_col_raw(i);
            let vals = a_ref.val_of_col(i);
            *vals
                .iter()
                .zip(rows.iter())
                .find_map(|(v, r)| if *r == i { Some(v) } else { None })
                .expect("explicit diagonal")
        })
        .collect();
    let pc = JacobiPrecond::try_from_diagonal(&diag).unwrap();
    let pc_iters = cg_iter_count(&a, &pc, 500);

    assert!(
        pc_iters <= baseline_iters,
        "Jacobi should not regress CG iter count: baseline {baseline_iters} vs preconditioned {pc_iters}",
    );
}

#[test]
fn solve_precond_with_exact_llt_converges_in_one_step() {
    // CG preconditioned by an exact M = A converges in exactly one step
    // (modulo single-precision setup; this is f64 so we expect iter_count == 1).
    let a = laplacian_2d(6);
    let n = a.nrows();

    // Build dense Llt of A.
    let a_dense = sparse_to_dense(&a);
    let llt =
        faer::linalg::solvers::Llt::new(a_dense.as_ref(), Side::Lower).expect("Laplacian is SPD");
    let pc = SolvePrecond::new(llt);

    let b = Mat::<f64>::from_fn(n, 1, |i, _| (i as f64 - 3.0).sin());
    let mut out = Mat::<f64>::zeros(n, 1);

    let params = CgParams::<f64> {
        max_iters: 5,
        rel_tolerance: 1e-12,
        ..Default::default()
    };

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

    assert!(
        info.iter_count <= 1,
        "exact-preconditioned CG should converge in 1 iter, got {}",
        info.iter_count,
    );
}

#[test]
fn ilu0_accelerates_bicgstab_on_nonsymmetric_problem() {
    let a = tridiagonal_nonsymmetric(64);
    let n = a.nrows();

    let identity = IdentityPrecond { dim: n };
    let baseline_iters = bicgstab_iter_count(&a, identity, identity, 500);

    let pc = Ilu0::<usize, f64>::try_new(a.as_ref()).expect("LU pattern is non-singular");
    // Left-preconditioning only.
    let pc_iters = bicgstab_iter_count(&a, &pc, IdentityPrecond { dim: n }, 500);

    assert!(
        pc_iters < baseline_iters,
        "ILU(0) should accelerate BiCGSTAB: baseline {baseline_iters} iters vs preconditioned {pc_iters}",
    );
}

#[test]
fn ilutp_accelerates_bicgstab_on_nonsymmetric_problem() {
    let a = advection_diffusion_2d(12, 0.7);
    let n = a.nrows();

    let identity = IdentityPrecond { dim: n };
    let baseline_iters = bicgstab_iter_count(&a, identity, identity, 500);

    let pc = Ilutp::<usize, f64>::try_new(a.as_ref()).expect("ILUTP factorisation");
    let pc_iters = bicgstab_iter_count(&a, &pc, IdentityPrecond { dim: n }, 500);

    assert!(
        pc_iters < baseline_iters,
        "ILUTP should accelerate BiCGSTAB: baseline {baseline_iters} iters vs preconditioned {pc_iters}",
    );
}

#[test]
fn ilutp_outperforms_ilu0_on_strongly_nonsymmetric_problem() {
    // With a strong advection term ILU(0)'s fixed pattern is weak; ILUTP's
    // adaptive fill + pivoting should converge in no more iterations.
    let a = advection_diffusion_2d(12, 0.9);
    let n = a.nrows();

    let ilu0 = Ilu0::<usize, f64>::try_new(a.as_ref()).expect("ILU(0) factorisation");
    let ilu0_iters = bicgstab_iter_count(&a, &ilu0, IdentityPrecond { dim: n }, 500);

    let ilutp = Ilutp::<usize, f64>::try_new(a.as_ref()).expect("ILUTP factorisation");
    let ilutp_iters = bicgstab_iter_count(&a, &ilutp, IdentityPrecond { dim: n }, 500);

    assert!(
        ilutp_iters <= ilu0_iters,
        "ILUTP should not need more iterations than ILU(0): ILU(0) {ilu0_iters} vs ILUTP {ilutp_iters}",
    );
}

#[test]
fn block_jacobi_with_block_diagonal_a_converges_in_one_step() {
    // For a block-diagonal A, block-Jacobi with matching blocks is the exact
    // inverse and CG converges in 1 step.
    let a = mat![
        [4.0_f64, 1.0, 0.0, 0.0, 0.0],
        [2.0, 3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 6.0, 1.0, 2.0],
        [0.0, 0.0, 3.0, 5.0, 1.0],
        [0.0, 0.0, 2.0, 1.0, 4.0],
    ];
    // A is not symmetric but BiCGSTAB does not require symmetry.
    let n = a.nrows();
    let pc = BlockJacobiPrecond::try_new(a.as_ref(), &[0, 2, 5]).unwrap();

    let b = Mat::<f64>::from_fn(n, 1, |i, _| i as f64 + 1.0);
    let mut out = Mat::<f64>::zeros(n, 1);

    let params = BicgParams::<f64> {
        max_iters: 5,
        rel_tolerance: 1e-12,
        ..Default::default()
    };

    let identity = IdentityPrecond { dim: n };
    let mut buf = MemBuffer::new(bicgstab_scratch(&pc, identity, a.as_ref(), 1, Par::Seq));
    let info = bicgstab(
        out.as_mut(),
        &pc,
        identity,
        a.as_ref(),
        b.as_ref(),
        params,
        |_| {},
        Par::Seq,
        MemStack::new(&mut buf),
    )
    .expect("BiCGSTAB should converge");

    assert!(
        info.iter_count <= 1,
        "block-Jacobi on block-diagonal A should converge in 1 iter, got {}",
        info.iter_count,
    );
}

#[test]
fn ic0_refactorize_drives_cg_on_changed_values() {
    // Build IC(0) on A1, then refactorize against A2 (same pattern), drive CG.
    let a1 = laplacian_2d(8);
    let n = a1.nrows();

    // A2: stronger diagonal (6) with the same -1 off-diagonals — pattern is
    // identical to A1, still SPD (d/|o| = 6 > 4 = 2*dim for the 2-D Laplacian).
    let mut triplets = Vec::new();
    for gy in 0..8 {
        for gx in 0..8 {
            let idx = gy * 8 + gx;
            triplets.push(Triplet::new(idx, idx, 6.0));
            if gx > 0 {
                triplets.push(Triplet::new(idx, idx - 1, -1.0));
            }
            if gx + 1 < 8 {
                triplets.push(Triplet::new(idx, idx + 1, -1.0));
            }
            if gy > 0 {
                triplets.push(Triplet::new(idx, idx - 8, -1.0));
            }
            if gy + 1 < 8 {
                triplets.push(Triplet::new(idx, idx + 8, -1.0));
            }
        }
    }
    let a2 = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &triplets).unwrap();

    let mut pc = Ic0::<usize, f64>::try_new(a1.as_ref()).unwrap();
    pc.refactorize(a2.as_ref()).unwrap();

    let iters = cg_iter_count(&a2, &pc, 500);
    let baseline = cg_iter_count(&a2, IdentityPrecond { dim: n }, 500);
    assert!(
        iters < baseline,
        "IC(0) refactorize should accelerate CG: baseline {baseline} vs reused {iters}",
    );
}

// ---------------- helpers ----------------

fn sparse_to_dense(a: &SparseColMat<usize, f64>) -> Mat<f64> {
    let mut dense = Mat::<f64>::zeros(a.nrows(), a.ncols());
    let a_ref = a.as_ref();
    for j in 0..a.ncols() {
        let rows = a_ref.symbolic().row_idx_of_col_raw(j);
        let vals = a_ref.val_of_col(j);
        for (r, v) in rows.iter().zip(vals.iter()) {
            *dense.as_mut().get_mut(*r, j) = *v;
        }
    }
    dense
}

fn bicgstab_iter_count<P1, P2>(
    a: &SparseColMat<usize, f64>,
    left: P1,
    right: P2,
    max_iters: usize,
) -> usize
where
    P1: faer::matrix_free::Precond<f64>,
    P2: faer::matrix_free::Precond<f64>,
{
    let n = a.nrows();
    let b = Mat::<f64>::from_fn(n, 1, |i, _| (i % 7) as f64 - 3.0);
    let mut out = Mat::<f64>::zeros(n, 1);

    let params = BicgParams::<f64> {
        max_iters,
        rel_tolerance: 1e-10,
        ..Default::default()
    };

    let mut buf = MemBuffer::new(bicgstab_scratch(&left, &right, a.as_ref(), 1, Par::Seq));
    let result = bicgstab(
        out.as_mut(),
        left,
        right,
        a.as_ref(),
        b.as_ref(),
        params,
        |_| {},
        Par::Seq,
        MemStack::new(&mut buf),
    );
    let info = result.expect("BiCGSTAB should converge");
    assert!(
        info.rel_residual <= params.rel_tolerance * 10.0,
        "BiCGSTAB did not actually converge (rel_residual = {})",
        info.rel_residual,
    );
    info.iter_count
}
