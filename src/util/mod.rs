//! Internal helpers shared across preconditioner families.
//!
//! Nothing here is part of the public API — these are small utilities that more
//! than one preconditioner needs and that would otherwise be copy-pasted.

pub(crate) mod diag_split;
pub(crate) mod spd_bounds;
