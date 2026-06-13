//! Symbolic structure for zero-fill incomplete Cholesky (IC(0)).
//!
//! IC(0) factorises a Hermitian positive-definite matrix `A` as
//! `A ≈ L L^H` where `pattern(L)` is the *lower triangular* subset of
//! `pattern(A)` — no fill-in is introduced. Only the lower triangle of `A`
//! is consumed; entries above the diagonal in the input matrix are ignored.
//!
//! The symbolic stage:
//! 1. Validates that `A` is square with sorted row indices and an explicit
//!    diagonal in every column.
//! 2. Records `diag_pos[j]`: the index *within* column `j` of `A` where row
//!    `j` sits — used to slice each column's lower triangle in O(1).
//! 3. Builds the CSC pattern of `L` (diagonal stored *first* in each column,
//!    matching the convention of [`faer::sparse::linalg::triangular_solve`]).
//! 4. Builds a row-pattern index — for each row `i`, the list of columns
//!    `k < i` where `L[i, k]` is in pattern, together with the position of
//!    each `(i, k)` entry in the `L` value array. The numeric stage uses
//!    this for the left-looking update without an O(n²) row search.

use faer::sparse::SymbolicSparseColMatRef;
use faer_traits::Index;

use super::Ic0Error;

/// Symbolic factor for IC(0).
///
/// Built once for a given sparsity pattern, then reused across any number of
/// numerical refactorisations.
#[derive(Debug, Clone)]
pub struct SymbolicIc0<I> {
    pub(crate) dim: usize,
    pub(crate) a_nnz: usize,
    /// Index within column `j` of `A` where row `j` (the diagonal) appears.
    pub(crate) diag_pos: Vec<usize>,
    /// `L`'s CSC column pointers. Length `n + 1`.
    pub(crate) l_col_ptr: Vec<I>,
    /// `L`'s CSC row indices. Diagonal is stored *first* in each column.
    pub(crate) l_row_idx: Vec<I>,
    /// Row-pattern column pointers (length `n + 1`). Row `i` owns entries
    /// `row_col_idx[row_col_ptr[i]..row_col_ptr[i+1]]`.
    pub(crate) row_col_ptr: Vec<usize>,
    /// For each row-pattern entry, the column `k` (with `k < i`) where row
    /// `i` appears below the diagonal in `L`.
    pub(crate) row_col_idx: Vec<I>,
    /// For each row-pattern entry, the index into the `L` value array of
    /// the `(i, k)` entry.
    pub(crate) row_lpos: Vec<usize>,
}

impl<I: Index> SymbolicIc0<I> {
    /// Build the symbolic IC(0) structure from a CSC sparsity pattern.
    ///
    /// Only the lower triangle of `pattern` is consumed — entries above the
    /// diagonal are skipped.
    ///
    /// # Requirements
    ///
    /// - `pattern` must be square.
    /// - Every column must contain its diagonal entry `(j, j)` explicitly.
    /// - Row indices within each column must be sorted ascending.
    ///
    /// # Errors
    ///
    /// Returns the corresponding [`Ic0Error`] variant when any of the above
    /// is violated.
    pub fn try_new(pattern: SymbolicSparseColMatRef<'_, I>) -> Result<Self, Ic0Error> {
        let nrows = pattern.nrows();
        let ncols = pattern.ncols();
        if nrows != ncols {
            return Err(Ic0Error::NonSquareMatrix { nrows, ncols });
        }
        let n = nrows;

        let mut diag_pos = Vec::with_capacity(n);
        let mut l_col_ptr: Vec<I> = Vec::with_capacity(n + 1);
        l_col_ptr.push(I::truncate(0));

        let mut a_nnz = 0usize;
        let mut l_running = 0usize;

        for j in 0..n {
            let row_idx = pattern.row_idx_of_col_raw(j);
            let col_len = row_idx.len();
            a_nnz += col_len;

            let mut diag_idx: Option<usize> = None;
            let mut prev: Option<usize> = None;
            for (k, raw) in row_idx.iter().enumerate() {
                let r = raw.zx();
                if let Some(p) = prev
                    && r <= p
                {
                    return Err(Ic0Error::UnsortedRowIndices { col: j });
                }
                if r == j {
                    diag_idx = Some(k);
                }
                prev = Some(r);
            }
            let d = diag_idx.ok_or(Ic0Error::MissingDiagonal { col: j })?;

            diag_pos.push(d);
            l_running += col_len - d;
            l_col_ptr.push(I::truncate(l_running));
        }

        let mut l_row_idx: Vec<I> = Vec::with_capacity(l_running);
        for (j, &d) in diag_pos.iter().enumerate() {
            let row_idx = pattern.row_idx_of_col_raw(j);
            // L receives row j (diagonal) first, then rows (d..n).
            l_row_idx.extend_from_slice(&row_idx[d..]);
        }
        debug_assert_eq!(l_row_idx.len(), l_running);

        // Build the row-pattern index for the strict lower triangle of L.
        // counts[i + 1] accumulates entries belonging to row i, then we
        // turn it into the column-pointer array via prefix sum.
        let mut row_col_ptr = vec![0usize; n + 1];
        for k in 0..n {
            let lk_start = l_col_ptr[k].zx() + 1;
            let lk_end = l_col_ptr[k + 1].zx();
            for raw_row in &l_row_idx[lk_start..lk_end] {
                let i = raw_row.zx();
                row_col_ptr[i + 1] += 1;
            }
        }
        for i in 0..n {
            row_col_ptr[i + 1] += row_col_ptr[i];
        }
        let total = row_col_ptr[n];

        let mut row_col_idx: Vec<I> = vec![I::truncate(0); total];
        let mut row_lpos: Vec<usize> = vec![0usize; total];
        let mut fill = vec![0usize; n];
        for k in 0..n {
            let lk_start = l_col_ptr[k].zx() + 1;
            let lk_end = l_col_ptr[k + 1].zx();
            for (offset, raw_row) in l_row_idx[lk_start..lk_end].iter().enumerate() {
                let pos = lk_start + offset;
                let i = raw_row.zx();
                let dst = row_col_ptr[i] + fill[i];
                row_col_idx[dst] = I::truncate(k);
                row_lpos[dst] = pos;
                fill[i] += 1;
            }
        }

        Ok(Self {
            dim: n,
            a_nnz,
            diag_pos,
            l_col_ptr,
            l_row_idx,
            row_col_ptr,
            row_col_idx,
            row_lpos,
        })
    }

    /// Dimension `n` of the original matrix.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of structural nonzeros in `A` (as supplied — including any
    /// strictly upper-triangle entries the algorithm ignored).
    #[inline]
    pub fn a_nnz(&self) -> usize {
        self.a_nnz
    }

    /// Number of structural nonzeros in `L` (includes the diagonal slots).
    #[inline]
    pub fn l_nnz(&self) -> usize {
        self.l_row_idx.len()
    }
}
