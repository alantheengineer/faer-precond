//! Numeric threshold incomplete Cholesky (ICT).
//!
//! Left-looking column Cholesky with dual dropping (a drop tolerance and a
//! per-column fill budget), restricted to the lower triangle. Because the fill
//! pattern depends on the matrix *values*, there is no symbolic phase — the
//! factor is rebuilt in full each time (reusing buffer capacity).
//!
//! To find, when forming column `j`, every earlier column `k` that has a stored
//! entry in row `j`, we keep the standard linked-list of active columns: bucket
//! `head[j]` chains the columns whose current cursor points at row `j`.

use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer_traits::math_utils::{
    abs, abs2, add, conj, copy, from_f64, from_real, mul, neg, real, recip, sqrt, sub, zero,
};
use faer_traits::{ComplexField, Index};

use super::{IctError, IctParams};
use crate::ilutp::{FillControl, RowNorm};

/// Numeric ICT factor of a Hermitian positive-definite CSC matrix `A`.
///
/// Stores the lower-triangular factor `L` (diagonal first per column) such that
/// `L L^H ~= A`. Apply uses the same two triangular solves as [`crate::Ic0`].
#[derive(Debug, Clone)]
pub struct Ict<I, T> {
    pub(crate) dim: usize,
    pub(crate) params: IctParams,
    pub(crate) l_col_ptr: Vec<I>,
    pub(crate) l_row_idx: Vec<I>,
    pub(crate) l_values: Vec<T>,
}

impl<I: Index, T: ComplexField> Ict<I, T> {
    /// Build an ICT factor with the default parameters.
    pub fn try_new(a: SparseColMatRef<'_, I, T>) -> Result<Self, IctError> {
        Self::try_new_with_params(a, IctParams::default())
    }

    /// Build an ICT factor with explicit parameters.
    ///
    /// Only the lower triangle of `A` is read.
    ///
    /// # Errors
    ///
    /// - [`IctError::NonSquareMatrix`] if `A` is not square.
    /// - [`IctError::InvalidDropTol`] / [`IctError::InvalidFillControl`] for bad
    ///   parameters.
    /// - [`IctError::NotPositiveDefinite`] if a non-positive pivot appears.
    pub fn try_new_with_params(
        a: SparseColMatRef<'_, I, T>,
        params: IctParams,
    ) -> Result<Self, IctError> {
        params.validate()?;
        if a.nrows() != a.ncols() {
            return Err(IctError::NonSquareMatrix {
                nrows: a.nrows(),
                ncols: a.ncols(),
            });
        }
        let n = a.nrows();
        let mut me = Self {
            dim: n,
            params,
            l_col_ptr: Vec::with_capacity(n + 1),
            l_row_idx: Vec::new(),
            l_values: Vec::new(),
        };
        me.factor(a)?;
        Ok(me)
    }

    /// Refactorise against a new matrix with the same dimension.
    ///
    /// Reuses the existing buffer capacity but, because the pattern is value-
    /// dependent, recomputes it from scratch — this is **not** allocation-free
    /// in the strict sense (it may grow the buffers).
    ///
    /// # Errors
    ///
    /// As [`Ict::try_new_with_params`], plus [`IctError::PatternMismatch`] if
    /// `a`'s dimension differs.
    pub fn refactorize(&mut self, a: SparseColMatRef<'_, I, T>) -> Result<(), IctError> {
        if a.nrows() != self.dim || a.ncols() != self.dim {
            return Err(IctError::PatternMismatch);
        }
        self.factor(a)
    }

    /// Dimension `n` of the factored matrix.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    fn per_column_budget(&self, a: SparseColMatRef<'_, I, T>) -> usize {
        match self.params.fill {
            FillControl::PerRow(p) => p,
            FillControl::Factor(f) => {
                let nnz: usize = (0..self.dim)
                    .map(|j| a.symbolic().row_idx_of_col_raw(j).len())
                    .sum();
                let per = (f * nnz as f64 / self.dim.max(1) as f64).round() as usize;
                per.max(1)
            }
        }
    }

    fn factor(&mut self, a: SparseColMatRef<'_, I, T>) -> Result<(), IctError> {
        let n = self.dim;
        let budget = self.per_column_budget(a);
        let drop_tol = from_f64::<T::Real>(self.params.drop_tol);

        self.l_col_ptr.clear();
        self.l_row_idx.clear();
        self.l_values.clear();
        self.l_col_ptr.push(I::truncate(0));

        // Dense accumulator and pattern marker.
        let mut w = (0..n).map(|_| zero::<T>()).collect::<Vec<T>>();
        let mut marker = vec![usize::MAX; n];
        let mut touched: Vec<usize> = Vec::new();

        // Linked list of active columns: `head[r]` is the first column whose
        // cursor points at row r; `next_col` chains the rest. `cursor[k]` is the
        // position in l_row_idx of column k's next-to-use entry.
        let mut head = vec![NONE; n];
        let mut next_col = vec![NONE; n];
        let mut cursor = vec![0usize; n];

        for j in 0..n {
            touched.clear();
            // Diagonal slot is always part of the pattern.
            w[j] = zero::<T>();
            marker[j] = j;
            touched.push(j);

            // Gather A's lower column j (rows >= j) and accumulate its norm.
            let a_rows = a.symbolic().row_idx_of_col_raw(j);
            let a_vals = a.val_of_col(j);
            for (raw, val) in a_rows.iter().zip(a_vals.iter()) {
                let i = raw.zx();
                if i < j {
                    continue;
                }
                if marker[i] != j {
                    marker[i] = j;
                    touched.push(i);
                    w[i] = copy(val);
                } else {
                    w[i] = copy(val);
                }
            }

            // Left-looking updates from every column k with an entry in row j.
            let mut k = head[j];
            while k != NONE {
                let kc = k as usize;
                let nk = next_col[kc];
                let pos = cursor[kc];
                let l_jk = copy(&self.l_values[pos]);
                let conj_l_jk = conj(&l_jk);
                let k_end = self.l_col_ptr[kc + 1].zx();
                for p in pos..k_end {
                    let i = self.l_row_idx[p].zx();
                    let upd = mul(&self.l_values[p], &conj_l_jk);
                    if marker[i] == j {
                        w[i] = sub(&w[i], &upd);
                    } else {
                        marker[i] = j;
                        touched.push(i);
                        w[i] = neg(&upd);
                    }
                }
                // Advance column k's cursor and re-bucket it.
                let new_pos = pos + 1;
                cursor[kc] = new_pos;
                if new_pos < k_end {
                    let nr = self.l_row_idx[new_pos].zx();
                    next_col[kc] = head[nr];
                    head[nr] = kc as isize;
                }
                k = nk;
            }
            head[j] = NONE;

            // Pivot.
            let pivot_real = real(&w[j]);
            if pivot_real.partial_cmp(&zero::<T::Real>()) != Some(core::cmp::Ordering::Greater) {
                return Err(IctError::NotPositiveDefinite { col: j });
            }
            let pivot = from_real::<T>(&sqrt::<T::Real>(&pivot_real));
            let pivot_inv = recip(&pivot);

            // Candidate off-diagonal entries (i > j) and their relative-drop norm.
            let mut col_norm = zero::<T::Real>();
            let mut candidates: Vec<usize> = Vec::new();
            for &i in &touched {
                if i > j {
                    candidates.push(i);
                    col_norm = match self.params.norm {
                        RowNorm::One => add(&col_norm, &abs(&w[i])),
                        RowNorm::Two => add(&col_norm, &abs2(&w[i])),
                    };
                }
            }
            if matches!(self.params.norm, RowNorm::Two) {
                col_norm = sqrt::<T::Real>(&col_norm);
            }
            let tau = mul(&drop_tol, &col_norm);

            // Drop sub-threshold entries, then keep the `budget` largest.
            candidates.retain(|&i| abs(&w[i]) >= tau);
            if candidates.len() > budget {
                candidates
                    .sort_by(|&a_i, &b_i| abs(&w[b_i]).partial_cmp(&abs(&w[a_i])).unwrap_or(core::cmp::Ordering::Equal));
                candidates.truncate(budget);
            }
            candidates.sort_unstable();

            // Append column j: diagonal first, then kept entries scaled by 1/pivot.
            self.l_row_idx.push(I::truncate(j));
            self.l_values.push(copy(&pivot));
            for &i in &candidates {
                self.l_row_idx.push(I::truncate(i));
                self.l_values.push(mul(&w[i], &pivot_inv));
            }
            self.l_col_ptr.push(I::truncate(self.l_row_idx.len()));

            // Insert column j into the linked list at its first off-diagonal row.
            let col_start = self.l_col_ptr[j].zx();
            let col_end = self.l_col_ptr[j + 1].zx();
            if col_end > col_start + 1 {
                cursor[j] = col_start + 1;
                let nr = self.l_row_idx[col_start + 1].zx();
                next_col[j] = head[nr];
                head[nr] = j as isize;
            }
        }

        Ok(())
    }

    /// View over the lower-triangular factor `L` (diagonal first per column).
    #[inline]
    pub fn l_view(&self) -> SparseColMatRef<'_, I, T> {
        let symbolic = unsafe {
            SymbolicSparseColMatRef::<'_, I>::new_unchecked(
                self.dim,
                self.dim,
                &self.l_col_ptr,
                None,
                &self.l_row_idx,
            )
        };
        SparseColMatRef::new(symbolic, &self.l_values)
    }
}

const NONE: isize = -1;
