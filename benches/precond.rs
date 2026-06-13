use core::mem::MaybeUninit;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use dyn_stack::{MemStack, StackReq};
use faer::sparse::{SparseColMat, Triplet};
use faer::{
    Mat, Par, Side,
    linalg::solvers::Llt,
    matrix_free::{LinOp, Precond},
};
use faer_precond::{BlockJacobiPrecond, Ic0, Ilu0, JacobiPrecond, SolvePrecond, SymbolicIlu0};

fn with_stack(req: StackReq, f: impl FnOnce(&mut MemStack)) {
    let nbytes = req.unaligned_bytes_required().max(1);
    let mut buf = vec![MaybeUninit::<u8>::uninit(); nbytes].into_boxed_slice();
    f(MemStack::new(&mut buf));
}

fn make_spd_tridiagonal(n: usize) -> Mat<f64> {
    Mat::from_fn(n, n, |i, j| {
        if i == j {
            4.0
        } else if i + 1 == j || j + 1 == i {
            -1.0
        } else {
            0.0
        }
    })
}

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

fn bench_jacobi(c: &mut Criterion) {
    let mut group = c.benchmark_group("jacobi_apply_in_place");

    for &(n, rhs_ncols) in &[(256usize, 1usize), (1024, 4), (4096, 8)] {
        let diag = vec![4.0; n];
        let pc = JacobiPrecond::try_from_diagonal(&diag).unwrap();
        let template = Mat::from_fn(n, rhs_ncols, |i, j| ((i + j) % 17) as f64 + 1.0);
        let mut rhs = template.clone();

        group.bench_with_input(
            BenchmarkId::new("f64", format!("{n}x{rhs_ncols}")),
            &(n, rhs_ncols),
            |b, _| {
                b.iter(|| {
                    rhs.as_mut().copy_from(template.as_ref());
                    with_stack(pc.apply_in_place_scratch(rhs_ncols, Par::Seq), |stack| {
                        pc.apply_in_place(black_box(rhs.as_mut()), Par::Seq, stack);
                    });
                    black_box(&rhs);
                });
            },
        );
    }

    group.finish();
}

fn bench_jacobi_out_of_place(c: &mut Criterion) {
    let mut group = c.benchmark_group("jacobi_apply");

    let n = 2048usize;
    let rhs_ncols = 8usize;
    let diag = vec![4.0; n];
    let pc = JacobiPrecond::try_from_diagonal(&diag).unwrap();
    let rhs = Mat::from_fn(n, rhs_ncols, |i, j| ((i + 5 * j) % 31) as f64 + 1.0);
    let mut out = Mat::<f64>::zeros(n, rhs_ncols);

    group.bench_function("f64/2048x8", |b| {
        b.iter(|| {
            with_stack(pc.apply_scratch(rhs_ncols, Par::Seq), |stack| {
                pc.apply(
                    black_box(out.as_mut()),
                    black_box(rhs.as_ref()),
                    Par::Seq,
                    stack,
                );
            });
            black_box(&out);
        });
    });

    group.finish();
}

fn bench_solve_precond(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve_precond_apply_in_place");

    for &(n, rhs_ncols) in &[(64usize, 1usize), (256, 4), (512, 8)] {
        let a = make_spd_tridiagonal(n);
        let llt = Llt::new(a.as_ref(), Side::Lower).unwrap();
        let pc = SolvePrecond::new(llt);

        let template = Mat::from_fn(n, rhs_ncols, |i, j| ((3 * i + j) % 23) as f64 + 1.0);
        let mut rhs = template.clone();

        group.bench_with_input(
            BenchmarkId::new("llt_f64", format!("{n}x{rhs_ncols}")),
            &(n, rhs_ncols),
            |b, _| {
                b.iter(|| {
                    rhs.as_mut().copy_from(template.as_ref());
                    with_stack(pc.apply_in_place_scratch(rhs_ncols, Par::Seq), |stack| {
                        pc.apply_in_place(black_box(rhs.as_mut()), Par::Seq, stack);
                    });
                    black_box(&rhs);
                });
            },
        );
    }

    group.finish();
}

fn bench_block_jacobi(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_jacobi_apply_in_place");

    // Two configurations: small blocks (more blocks) vs large blocks (fewer).
    for &(n, block_size, rhs_ncols) in &[(256usize, 8usize, 1usize), (1024, 16, 4), (4096, 32, 8)] {
        let a = Mat::from_fn(n, n, |i, j| {
            let bi = i / block_size;
            let bj = j / block_size;
            if bi == bj {
                if i == j { 4.0 } else { 0.5 }
            } else {
                0.0
            }
        });
        let block_offsets: Vec<usize> = (0..=(n / block_size)).map(|k| k * block_size).collect();
        let pc = BlockJacobiPrecond::try_new(a.as_ref(), &block_offsets).unwrap();

        let template = Mat::from_fn(n, rhs_ncols, |i, j| ((i + j) % 13) as f64 + 1.0);
        let mut rhs = template.clone();

        group.bench_with_input(
            BenchmarkId::new("f64", format!("{n}x{rhs_ncols}_bs{block_size}")),
            &(n, rhs_ncols),
            |b, _| {
                b.iter(|| {
                    rhs.as_mut().copy_from(template.as_ref());
                    with_stack(pc.apply_in_place_scratch(rhs_ncols, Par::Seq), |stack| {
                        pc.apply_in_place(black_box(rhs.as_mut()), Par::Seq, stack);
                    });
                    black_box(&rhs);
                });
            },
        );
    }

    group.finish();
}

fn bench_ilu0_apply(c: &mut Criterion) {
    let mut group = c.benchmark_group("ilu0_apply_in_place");

    for &(n, rhs_ncols) in &[(256usize, 1usize), (1024, 1), (4096, 1)] {
        let a = nonsymmetric_tridiagonal(n);
        let pc = Ilu0::<usize, f64>::try_new(a.as_ref()).unwrap();

        let template = Mat::from_fn(n, rhs_ncols, |i, j| ((i + j) % 13) as f64 + 1.0);
        let mut rhs = template.clone();

        group.bench_with_input(
            BenchmarkId::new("tridiag_f64", format!("{n}x{rhs_ncols}")),
            &(n, rhs_ncols),
            |b, _| {
                b.iter(|| {
                    rhs.as_mut().copy_from(template.as_ref());
                    with_stack(pc.apply_in_place_scratch(rhs_ncols, Par::Seq), |stack| {
                        pc.apply_in_place(black_box(rhs.as_mut()), Par::Seq, stack);
                    });
                    black_box(&rhs);
                });
            },
        );
    }

    group.finish();
}

fn bench_ilu0_refactorize(c: &mut Criterion) {
    let mut group = c.benchmark_group("ilu0_refactorize");

    // The hot path for nonlinear Krylov drivers: rebuild values on a fixed pattern.
    for &grid in &[16usize, 32, 64] {
        let a = laplacian_2d(grid);
        let n = a.nrows();
        let symbolic = SymbolicIlu0::<usize>::try_new(a.as_ref().symbolic()).unwrap();
        let mut pc = Ilu0::<usize, f64>::new_with_symbolic(symbolic);

        group.bench_with_input(
            BenchmarkId::new("laplacian2d_f64", format!("grid{grid}_n{n}")),
            &grid,
            |b, _| {
                b.iter(|| {
                    pc.refactorize(black_box(a.as_ref())).unwrap();
                    black_box(&pc);
                });
            },
        );
    }

    group.finish();
}

fn bench_ic0_apply(c: &mut Criterion) {
    let mut group = c.benchmark_group("ic0_apply_in_place");

    for &grid in &[16usize, 32, 64] {
        let a = laplacian_2d(grid);
        let n = a.nrows();
        let pc = Ic0::<usize, f64>::try_new(a.as_ref()).unwrap();

        let template = Mat::from_fn(n, 1, |i, _| ((i % 17) as f64) - 5.0);
        let mut rhs = template.clone();

        group.bench_with_input(
            BenchmarkId::new("laplacian2d_f64", format!("grid{grid}_n{n}")),
            &grid,
            |b, _| {
                b.iter(|| {
                    rhs.as_mut().copy_from(template.as_ref());
                    with_stack(pc.apply_in_place_scratch(1, Par::Seq), |stack| {
                        pc.apply_in_place(black_box(rhs.as_mut()), Par::Seq, stack);
                    });
                    black_box(&rhs);
                });
            },
        );
    }

    group.finish();
}

fn bench_ic0_refactorize(c: &mut Criterion) {
    let mut group = c.benchmark_group("ic0_refactorize");

    for &grid in &[16usize, 32, 64] {
        let a = laplacian_2d(grid);
        let n = a.nrows();
        let symbolic = faer_precond::SymbolicIc0::<usize>::try_new(a.as_ref().symbolic()).unwrap();
        let mut pc = Ic0::<usize, f64>::new_with_symbolic(symbolic);

        group.bench_with_input(
            BenchmarkId::new("laplacian2d_f64", format!("grid{grid}_n{n}")),
            &grid,
            |b, _| {
                b.iter(|| {
                    pc.refactorize(black_box(a.as_ref())).unwrap();
                    black_box(&pc);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_jacobi,
    bench_jacobi_out_of_place,
    bench_solve_precond,
    bench_block_jacobi,
    bench_ilu0_apply,
    bench_ilu0_refactorize,
    bench_ic0_apply,
    bench_ic0_refactorize,
);
criterion_main!(benches);
