//! Wrapping a faer factorisation as a preconditioner.
//!
//! `SolvePrecond` is an adapter: it takes any faer factorisation implementing
//! `SolveCore` (`Llt`, `Ldlt`, `Lu`, `Qr`, ...) and exposes it as a `Precond`.
//! The most common pattern is to factorise a *simplified* approximation of
//! `A` once, then drive a Krylov method on the full system.
//!
//! Run with:
//! ```text
//! cargo run --example solve_precond
//! ```

use dyn_stack::MemStack;
use faer::linalg::solvers::Llt;
use faer::matrix_free::Precond;
use faer::{Par, Side, mat};
use faer_precond::SolvePrecond;

fn main() {
    let a = mat![[4.0_f64, 1.0], [1.0, 3.0],];

    // Cholesky-factorise A (real SPD), then wrap as a preconditioner.
    let llt = Llt::new(a.as_ref(), Side::Lower).expect("matrix is SPD");
    let pc = SolvePrecond::new(llt);

    // Applying the preconditioner is now an exact direct solve.
    let mut x = mat![[1.0_f64], [2.0]];
    pc.apply_in_place(x.as_mut(), Par::Seq, MemStack::new(&mut []));

    println!("A^{{-1}} b = ");
    for i in 0..x.nrows() {
        println!("  {:>10.6}", *x.as_ref().get(i, 0));
    }
    println!();
    println!("(Exact solution: [1/11, 7/11] = [0.0909..., 0.6363...])");
}
