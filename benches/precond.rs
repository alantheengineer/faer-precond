use core::mem::MaybeUninit;
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use dyn_stack::{MemStack, StackReq};
use faer::{
    linalg::solvers::Llt,
    matrix_free::{Precond, LinOp},
    Mat, Side, Par,
};
use faer_precond::{jacobi::JacobiPrecond, SolvePrecond};

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
                pc.apply(black_box(out.as_mut()), black_box(rhs.as_ref()), Par::Seq, stack);
            });
            black_box(&out);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_jacobi,
    bench_solve_precond,
    bench_jacobi_out_of_place
);
criterion_main!(benches);