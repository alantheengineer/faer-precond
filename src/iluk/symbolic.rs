//! Level-of-fill symbolic factorisation for ILU(k).
//!
//! ILU(k) generalises [`crate::ilu0::SymbolicIlu0`]: instead of restricting the
//! factor to `pattern(A)`, it admits fill entries whose *level* is at most `k`.
//! A structural entry of `A` has level `0`; a fill entry created by eliminating
//! position `(i, m)` against `(m, j)` has level `lev(i,m) + lev(m,j) + 1`. Only
//! entries with `lev <= k` are kept.
//!
//! With `k = 0` no fill is admitted, so the resulting `L`/`U` patterns coincide
//! exactly with [`SymbolicIlu0`](crate::ilu0::SymbolicIlu0).
//!
//! The level computation is row-oriented (the classic IKJ algorithm); the
//! resulting CSR factor pattern is then transposed into the diagonal-first `L`
//! and diagonal-last `U` CSC layout faer's triangular solves expect.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use faer::sparse::SymbolicSparseColMatRef;
use faer_traits::Index;

use super::IlukError;
use crate::util::diag_split::{DiagError, validated_diag_pos};

const INF: u32 = u32::MAX;

fn map_diag_err(e: DiagError) -> IlukError {
    match e {
        DiagError::NonSquare { nrows, ncols } => IlukError::NonSquareMatrix { nrows, ncols },
        DiagError::MissingDiagonal { col } => IlukError::MissingDiagonal { col },
        DiagError::UnsortedRowIndices { col } => IlukError::UnsortedRowIndices { col },
    }
}

/// Symbolic ILU(k) factor: the level-of-fill `L`/`U` patterns.
///
/// Built once for a given pattern and fill level, then reused across any number
/// of numeric refactorisations against matrices with the same pattern.
#[derive(Debug, Clone)]
pub struct SymbolicIluk<I> {
    pub(crate) dim: usize,
    /// `L`'s CSC column pointers. Diagonal stored *first* in each column.
    pub(crate) l_col_ptr: Vec<I>,
    pub(crate) l_row_idx: Vec<I>,
    /// `U`'s CSC column pointers. Diagonal stored *last* in each column.
    pub(crate) u_col_ptr: Vec<I>,
    pub(crate) u_row_idx: Vec<I>,
}

impl<I: Index> SymbolicIluk<I> {
    /// Build the symbolic ILU(k) structure from a CSC pattern and fill level.
    ///
    /// # Requirements
    ///
    /// - `pattern` must be square, every column's row indices sorted ascending,
    ///   and every column must contain its diagonal entry.
    ///
    /// # Errors
    ///
    /// Returns the matching [`IlukError`] variant when any requirement fails.
    pub fn try_new(
        pattern: SymbolicSparseColMatRef<'_, I>,
        level: usize,
    ) -> Result<Self, IlukError> {
        validated_diag_pos(pattern).map_err(map_diag_err)?;
        let n = pattern.nrows();
        let maxlev = level.min(n).min(u32::MAX as usize) as u32;

        // Build A's CSR pattern (row_ptr, col_idx). Iterating columns ascending
        // and appending makes each row's column list ascending automatically.
        let mut row_cnt = vec![0usize; n];
        for j in 0..n {
            for raw in pattern.row_idx_of_col_raw(j) {
                row_cnt[raw.zx()] += 1;
            }
        }
        let mut row_ptr = vec![0usize; n + 1];
        for i in 0..n {
            row_ptr[i + 1] = row_ptr[i] + row_cnt[i];
        }
        let mut col_idx = vec![0usize; row_ptr[n]];
        let mut cursor = row_ptr.clone();
        for j in 0..n {
            for raw in pattern.row_idx_of_col_raw(j) {
                let i = raw.zx();
                col_idx[cursor[i]] = j;
                cursor[i] += 1;
            }
        }

        // Row-oriented level-of-fill. `factor_rows[i]` holds (column, level) for
        // every kept entry of row i, sorted ascending by column.
        let mut lev = vec![INF; n];
        let mut touched: Vec<usize> = Vec::new();
        let mut queued = vec![u32::MAX; n];
        let mut heap: BinaryHeap<Reverse<usize>> = BinaryHeap::new();
        let mut factor_rows: Vec<Vec<(usize, u32)>> = Vec::with_capacity(n);

        for i in 0..n {
            touched.clear();
            heap.clear();
            let tag = i as u32;
            for &j in &col_idx[row_ptr[i]..row_ptr[i + 1]] {
                lev[j] = 0;
                touched.push(j);
                if j < i && queued[j] != tag {
                    queued[j] = tag;
                    heap.push(Reverse(j));
                }
            }

            // Eliminate columns k < i in increasing order, expanding the row.
            while let Some(Reverse(k)) = heap.pop() {
                let lk = lev[k];
                if lk > maxlev {
                    continue;
                }
                for &(j, lkj) in &factor_rows[k] {
                    if j > k {
                        let nl = lk + lkj + 1;
                        if nl <= maxlev && nl < lev[j] {
                            if lev[j] == INF {
                                touched.push(j);
                            }
                            lev[j] = nl;
                            if j < i && queued[j] != tag {
                                queued[j] = tag;
                                heap.push(Reverse(j));
                            }
                        }
                    }
                }
            }

            let mut rowcols: Vec<(usize, u32)> = touched
                .iter()
                .filter_map(|&j| {
                    let l = lev[j];
                    (l <= maxlev).then_some((j, l))
                })
                .collect();
            rowcols.sort_by_key(|&(j, _)| j);
            for &j in &touched {
                lev[j] = INF;
            }
            factor_rows.push(rowcols);
        }

        // Transpose the CSR factor pattern into the CSC L/U layout.
        let mut u_cnt = vec![0usize; n];
        let mut l_cnt = vec![0usize; n];
        for (i, row) in factor_rows.iter().enumerate() {
            for &(j, _) in row {
                if i < j {
                    u_cnt[j] += 1;
                } else if i > j {
                    l_cnt[j] += 1;
                } else {
                    u_cnt[j] += 1;
                    l_cnt[j] += 1;
                }
            }
        }
        let mut u_col_ptr = vec![I::truncate(0); n + 1];
        let mut l_col_ptr = vec![I::truncate(0); n + 1];
        let mut u_run = 0usize;
        let mut l_run = 0usize;
        for j in 0..n {
            u_col_ptr[j] = I::truncate(u_run);
            l_col_ptr[j] = I::truncate(l_run);
            u_run += u_cnt[j];
            l_run += l_cnt[j];
        }
        u_col_ptr[n] = I::truncate(u_run);
        l_col_ptr[n] = I::truncate(l_run);

        let mut u_row_idx = vec![I::truncate(0); u_run];
        let mut l_row_idx = vec![I::truncate(0); l_run];
        let mut u_fill: Vec<usize> = (0..n).map(|j| u_col_ptr[j].zx()).collect();
        let mut l_fill: Vec<usize> = (0..n).map(|j| l_col_ptr[j].zx()).collect();
        for (i, row) in factor_rows.iter().enumerate() {
            for &(j, _) in row {
                if i <= j {
                    u_row_idx[u_fill[j]] = I::truncate(i);
                    u_fill[j] += 1;
                }
                if i >= j {
                    l_row_idx[l_fill[j]] = I::truncate(i);
                    l_fill[j] += 1;
                }
            }
        }

        Ok(Self {
            dim: n,
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
