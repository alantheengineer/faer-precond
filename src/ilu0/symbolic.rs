//! Symbolic structure for zero-fill incomplete LU (ILU(0)).
//!
//! ILU(0) takes the sparsity pattern of `A` (CSC) and produces two factors
//! `L` (unit lower triangular) and `U` (upper triangular) whose patterns are
//! exactly the lower/upper triangular subsets of `pattern(A)`. No fill-in is
//! introduced.
//!
//! The symbolic stage:
//! 1. Validates that `A` is square, has explicit diagonal entries, and that
//!    row indices are sorted ascending in each column.
//! 2. Records `diag_pos[j]`: the index *within* column `j` of `A` where row `j`
//!    sits — used to split each column into U-prefix and L-suffix without an
//!    extra mapping array.
//! 3. Builds the CSC patterns of `L` and `U` in the diagonal-first / diagonal-last
//!    convention required by [`faer::sparse::linalg::triangular_solve`].

use faer::sparse::SymbolicSparseColMatRef;
use faer_traits::Index;

use super::Ilu0Error;

/// Symbolic factor for ILU(0).
///
/// Built once for a given sparsity pattern, then reused across any number of
/// numerical refactorisations.
#[derive(Debug, Clone)]
pub struct SymbolicIlu0<I> {
    pub(crate) dim: usize,
    pub(crate) a_nnz: usize,
    /// Index *within* column `j` of `A` where row `j` (the diagonal) appears.
    pub(crate) diag_pos: Vec<usize>,
    /// `L`'s CSC column pointers. Length `n + 1`.
    pub(crate) l_col_ptr: Vec<I>,
    /// `L`'s CSC row indices. Diagonal is stored *first* in each column.
    pub(crate) l_row_idx: Vec<I>,
    /// `U`'s CSC column pointers. Length `n + 1`.
    pub(crate) u_col_ptr: Vec<I>,
    /// `U`'s CSC row indices. Diagonal is stored *last* in each column.
    pub(crate) u_row_idx: Vec<I>,
}

impl<I: Index> SymbolicIlu0<I> {
    /// Build the symbolic ILU(0) structure from a CSC sparsity pattern.
    ///
    /// # Requirements
    ///
    /// - `pattern` must be square.
    /// - Every column must contain its diagonal entry `(j, j)` explicitly.
    /// - Row indices within each column must be sorted ascending.
    ///
    /// # Errors
    ///
    /// Returns the corresponding [`Ilu0Error`] variant when any of the above
    /// is violated.
    pub fn try_new(pattern: SymbolicSparseColMatRef<'_, I>) -> Result<Self, Ilu0Error> {
        let nrows = pattern.nrows();
        let ncols = pattern.ncols();
        if nrows != ncols {
            return Err(Ilu0Error::NonSquareMatrix { nrows, ncols });
        }
        let n = nrows;

        let mut diag_pos = Vec::with_capacity(n);
        let mut l_col_ptr: Vec<I> = Vec::with_capacity(n + 1);
        l_col_ptr.push(I::truncate(0));
        let mut u_col_ptr: Vec<I> = Vec::with_capacity(n + 1);
        u_col_ptr.push(I::truncate(0));

        let mut a_nnz = 0usize;
        let mut l_running = 0usize;
        let mut u_running = 0usize;

        for j in 0..n {
            let row_idx = pattern.row_idx_of_col_raw(j);
            let col_len = row_idx.len();
            a_nnz += col_len;

            // Locate diagonal and verify sorted ascending in one pass.
            let mut diag_idx: Option<usize> = None;
            let mut prev: Option<usize> = None;
            for (k, raw) in row_idx.iter().enumerate() {
                let r = raw.zx();
                if let Some(p) = prev
                    && r <= p
                {
                    return Err(Ilu0Error::UnsortedRowIndices { col: j });
                }
                if r == j {
                    diag_idx = Some(k);
                }
                prev = Some(r);
            }
            let d = diag_idx.ok_or(Ilu0Error::MissingDiagonal { col: j })?;

            diag_pos.push(d);
            u_running += d + 1;
            l_running += col_len - d;
            u_col_ptr.push(I::truncate(u_running));
            l_col_ptr.push(I::truncate(l_running));
        }

        let mut l_row_idx: Vec<I> = Vec::with_capacity(l_running);
        let mut u_row_idx: Vec<I> = Vec::with_capacity(u_running);

        for (j, &d) in diag_pos.iter().enumerate() {
            let row_idx = pattern.row_idx_of_col_raw(j);
            // U receives rows [0..=d] in their original order.
            u_row_idx.extend_from_slice(&row_idx[..=d]);
            // L receives the implicit diagonal slot first, then rows (d..n).
            l_row_idx.push(I::truncate(j));
            l_row_idx.extend_from_slice(&row_idx[d + 1..]);
        }

        debug_assert_eq!(l_row_idx.len(), l_running);
        debug_assert_eq!(u_row_idx.len(), u_running);

        Ok(Self {
            dim: n,
            a_nnz,
            diag_pos,
            l_col_ptr,
            l_row_idx,
            u_col_ptr,
            u_row_idx,
        })
    }

    /// Dimension `n` of the original matrix.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of structural nonzeros in `A`.
    #[inline]
    pub fn a_nnz(&self) -> usize {
        self.a_nnz
    }

    /// Number of structural nonzeros in `L` (includes its unit diagonal slots).
    #[inline]
    pub fn l_nnz(&self) -> usize {
        self.l_row_idx.len()
    }

    /// Number of structural nonzeros in `U` (includes its diagonal slots).
    #[inline]
    pub fn u_nnz(&self) -> usize {
        self.u_row_idx.len()
    }
}
