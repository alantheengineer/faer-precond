//! Validated diagonal-position extraction for CSC sparsity patterns.
//!
//! ILU(0), IC(0) and ILUTP each re-derive the same three facts about an input
//! pattern: that it is square, that every column lists its row indices in
//! ascending order, and that the diagonal entry is present. This helper
//! centralises that single pass so SSOR and ILU(k) do not copy it again.

use faer::sparse::SymbolicSparseColMatRef;
use faer_traits::Index;

/// Why a CSC pattern could not be split around its diagonal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DiagError {
    /// The pattern was not square.
    NonSquare { nrows: usize, ncols: usize },
    /// Column `col` does not list its diagonal entry `(col, col)`.
    MissingDiagonal { col: usize },
    /// Column `col` has row indices that are not sorted ascending.
    UnsortedRowIndices { col: usize },
}

/// Validate `pattern` and return, for each column `j`, the index *within* that
/// column where row `j` (the diagonal) sits.
///
/// Mirrors the validation in [`crate::ilu0::symbolic`]: the pattern must be
/// square, every column's row indices must be ascending, and the diagonal must
/// be present in every column.
pub(crate) fn validated_diag_pos<I: Index>(
    pattern: SymbolicSparseColMatRef<'_, I>,
) -> Result<Vec<usize>, DiagError> {
    let nrows = pattern.nrows();
    let ncols = pattern.ncols();
    if nrows != ncols {
        return Err(DiagError::NonSquare { nrows, ncols });
    }
    let n = nrows;

    let mut diag_pos = Vec::with_capacity(n);
    for j in 0..n {
        let row_idx = pattern.row_idx_of_col_raw(j);
        let mut diag_idx: Option<usize> = None;
        let mut prev: Option<usize> = None;
        for (k, raw) in row_idx.iter().enumerate() {
            let r = raw.zx();
            if let Some(p) = prev
                && r <= p
            {
                return Err(DiagError::UnsortedRowIndices { col: j });
            }
            if r == j {
                diag_idx = Some(k);
            }
            prev = Some(r);
        }
        diag_pos.push(diag_idx.ok_or(DiagError::MissingDiagonal { col: j })?);
    }
    Ok(diag_pos)
}
