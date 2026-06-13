//! Block-Jacobi preconditioner applied to a block-diagonal-dominant matrix.
//!
//! Run with:
//! ```text
//! cargo run --example block_jacobi
//! ```

use dyn_stack::{MemBuffer, MemStack};
use faer::matrix_free::Precond;
use faer::{Par, mat};
use faer_precond::BlockJacobiPrecond;

fn main() {
    // 5x5 matrix with two diagonal blocks of size 2 and 3 (plus weak off-block
    // coupling that block-Jacobi ignores).
    let a = mat![
        [4.0_f64, 1.0, 0.1, 0.0, 0.0],
        [2.0, 3.0, 0.0, 0.1, 0.0],
        [0.1, 0.0, 6.0, 1.0, 2.0],
        [0.0, 0.1, 3.0, 5.0, 1.0],
        [0.0, 0.0, 2.0, 1.0, 4.0],
    ];

    // Two blocks: rows/cols [0..2) and [2..5).
    let pc = BlockJacobiPrecond::try_new(a.as_ref(), &[0, 2, 5]).expect("blocks are non-singular");

    println!("block count    = {}", pc.block_count());
    println!("max block size = {}", pc.max_block_size());

    let mut x = mat![[1.0_f64], [2.0], [3.0], [-1.0], [0.5]];
    let scratch = pc.apply_in_place_scratch(x.ncols(), Par::Seq);
    let mut buf = MemBuffer::new(scratch);
    pc.apply_in_place(x.as_mut(), Par::Seq, MemStack::new(&mut buf));

    println!("M^{{-1}} b = ");
    for i in 0..x.nrows() {
        println!("  {:>8.4}", *x.as_ref().get(i, 0));
    }
    println!();
    println!("(Each block is factored by partial-pivoted LU once at construction.)");
}
