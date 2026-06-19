//! Construction and refactorisation of the SSOR factors `(D+wL)` and `(D+wU)`.

use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer_traits::math_utils::{abs2, copy, from_f64, mul, zero};
use faer_traits::{ComplexField, Index};

use super::{Ssor, SsorError, SsorParams};
use crate::util::diag_split::{DiagError, validated_diag_pos};

fn map_diag_err(e: DiagError) -> SsorError {
    match e {
        DiagError::NonSquare { nrows, ncols } => SsorError::NonSquareMatrix { nrows, ncols },
        DiagError::MissingDiagonal { col } => SsorError::MissingDiagonal { col },
        DiagError::UnsortedRowIndices { col } => SsorError::UnsortedRowIndices { col },
    }
}

impl<I: Index, T: ComplexField> Ssor<I, T> {
    /// Build an SSOR preconditioner from a CSC matrix `A` and a relaxation
    /// factor.
    ///
    /// # Errors
    ///
    /// - [`SsorError::InvalidOmega`] if `omega` is outside `(0, 2)`.
    /// - [`SsorError::NonSquareMatrix`], [`SsorError::MissingDiagonal`] or
    ///   [`SsorError::UnsortedRowIndices`] if `A`'s pattern is unusable.
    /// - [`SsorError::ZeroDiagonal`] if a diagonal entry is zero.
    pub fn try_new(a: SparseColMatRef<'_, I, T>, params: SsorParams) -> Result<Self, SsorError> {
        if !(params.omega > 0.0 && params.omega < 2.0) {
            return Err(SsorError::InvalidOmega);
        }
        let diag_pos = validated_diag_pos(a.symbolic()).map_err(map_diag_err)?;
        let n = a.nrows();

        // Build the CSC patterns of (D+wL) (diagonal first per column) and
        // (D+wU) (diagonal last per column).
        let mut l_col_ptr: Vec<I> = Vec::with_capacity(n + 1);
        let mut u_col_ptr: Vec<I> = Vec::with_capacity(n + 1);
        l_col_ptr.push(I::truncate(0));
        u_col_ptr.push(I::truncate(0));
        let mut l_row_idx: Vec<I> = Vec::new();
        let mut u_row_idx: Vec<I> = Vec::new();

        for (j, &d) in diag_pos.iter().enumerate() {
            let rows = a.symbolic().row_idx_of_col_raw(j);
            // L column j: unit diagonal slot (row j) first, then rows > j.
            l_row_idx.push(I::truncate(j));
            l_row_idx.extend_from_slice(&rows[d + 1..]);
            // U column j: rows < j, then the diagonal (row j) last.
            u_row_idx.extend_from_slice(&rows[..d]);
            u_row_idx.push(I::truncate(j));
            l_col_ptr.push(I::truncate(l_row_idx.len()));
            u_col_ptr.push(I::truncate(u_row_idx.len()));
        }

        let l_values = (0..l_row_idx.len()).map(|_| zero::<T>()).collect();
        let u_values = (0..u_row_idx.len()).map(|_| zero::<T>()).collect();
        let scaled_diag = (0..n).map(|_| zero::<T>()).collect();

        let mut me = Self {
            dim: n,
            omega: params.omega,
            scaled_diag,
            l_col_ptr,
            l_row_idx,
            l_values,
            u_col_ptr,
            u_row_idx,
            u_values,
            diag_pos,
        };
        me.fill_values(a)?;
        Ok(me)
    }

    /// Refactorise against a new matrix with the same sparsity pattern.
    ///
    /// Performs zero heap allocations — the factor patterns and value buffers
    /// are reused.
    ///
    /// # Errors
    ///
    /// - [`SsorError::PatternMismatch`] if `a`'s shape or column structure
    ///   disagrees with the stored pattern.
    /// - [`SsorError::ZeroDiagonal`] if a diagonal entry is zero.
    pub fn refactorize(&mut self, a: SparseColMatRef<'_, I, T>) -> Result<(), SsorError> {
        self.fill_values(a)
    }

    /// Fill `scaled_diag`, `l_values` and `u_values` from `A` against the
    /// already-built pattern.
    fn fill_values(&mut self, a: SparseColMatRef<'_, I, T>) -> Result<(), SsorError> {
        let n = self.dim;
        if a.nrows() != n || a.ncols() != n {
            return Err(SsorError::PatternMismatch);
        }
        let omega = from_f64::<T>(self.omega);
        let scale = from_f64::<T>(self.omega * (2.0 - self.omega));

        for j in 0..n {
            let rows = a.symbolic().row_idx_of_col_raw(j);
            let vals = a.val_of_col(j);
            let d = self.diag_pos[j];
            if d >= rows.len() || rows[d].zx() != j {
                return Err(SsorError::PatternMismatch);
            }

            let diag_val = copy(&vals[d]);
            if abs2(&diag_val) == zero::<T::Real>() {
                return Err(SsorError::ZeroDiagonal { col: j });
            }
            self.scaled_diag[j] = mul(&scale, &diag_val);

            let l_start = self.l_col_ptr[j].zx();
            let l_end = self.l_col_ptr[j + 1].zx();
            if l_end - l_start != rows.len() - d {
                return Err(SsorError::PatternMismatch);
            }
            self.l_values[l_start] = copy(&diag_val);
            for (k, v) in vals[d + 1..].iter().enumerate() {
                self.l_values[l_start + 1 + k] = mul(&omega, v);
            }

            let u_start = self.u_col_ptr[j].zx();
            let u_end = self.u_col_ptr[j + 1].zx();
            if u_end - u_start != d + 1 {
                return Err(SsorError::PatternMismatch);
            }
            for (k, v) in vals[..d].iter().enumerate() {
                self.u_values[u_start + k] = mul(&omega, v);
            }
            self.u_values[u_end - 1] = copy(&diag_val);
        }
        Ok(())
    }

    /// View of `(D+wL)`: lower triangular, diagonal stored first per column.
    #[inline]
    pub(crate) fn l_view(&self) -> SparseColMatRef<'_, I, T> {
        let sym = unsafe {
            SymbolicSparseColMatRef::<'_, I>::new_unchecked(
                self.dim,
                self.dim,
                &self.l_col_ptr,
                None,
                &self.l_row_idx,
            )
        };
        SparseColMatRef::new(sym, &self.l_values)
    }

    /// View of `(D+wU)`: upper triangular, diagonal stored last per column.
    #[inline]
    pub(crate) fn u_view(&self) -> SparseColMatRef<'_, I, T> {
        let sym = unsafe {
            SymbolicSparseColMatRef::<'_, I>::new_unchecked(
                self.dim,
                self.dim,
                &self.u_col_ptr,
                None,
                &self.u_row_idx,
            )
        };
        SparseColMatRef::new(sym, &self.u_values)
    }
}
