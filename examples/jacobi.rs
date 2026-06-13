//! Point-Jacobi preconditioner applied to a small diagonally-dominant system.
//!
//! Run with:
//! ```text
//! cargo run --example jacobi
//! ```

use dyn_stack::MemStack;
use faer::matrix_free::Precond;
use faer::{Par, mat};
use faer_precond::JacobiPrecond;

fn main() {
    // Dense diagonally-dominant matrix.
    let a = mat![[10.0_f64, -1.0, 2.0], [-1.0, 11.0, -1.0], [2.0, -1.0, 12.0],];

    // Build the Jacobi preconditioner from the diagonal of A.
    let pc = JacobiPrecond::try_from_matrix_diagonal(a.as_ref()).expect("no zero diagonals");

    // Apply M^{-1} to a right-hand side, in place.
    let mut x = mat![[6.0_f64], [25.0], [-11.0]];
    pc.apply_in_place(x.as_mut(), Par::Seq, MemStack::new(&mut []));

    println!("M^{{-1}} b = ");
    for i in 0..x.nrows() {
        println!("  {:>8.4}", *x.as_ref().get(i, 0));
    }
    println!();
    println!("(Jacobi rescales each row by 1 / A[i,i] = [0.6, 25/11, -11/12])");
}
