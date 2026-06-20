//! Numeric ILU(k) factorisation.
//!
//! The elimination is identical to ILU(0)'s left-looking sweep — incomplete LU
//! restricted to a prescribed pattern — only here the pattern is the larger
//! level-of-fill pattern from [`SymbolicIluk`]. `A`'s values are scattered into
//! the (wider) `L`/`U` slots, fill positions starting at zero.

use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer_traits::math_utils::{abs2, copy, mul, one, recip, sub, zero};
use faer_traits::{ComplexField, Index};

use super::IlukError;
use super::symbolic::SymbolicIluk;

/// Numeric ILU(k) factor of a CSC matrix `A`.
///
/// Holds the `L`/`U` value buffers, the symbolic structure and the dense
/// workspace used during refactorisation. [`Iluk::refactorize`] performs zero
/// heap allocations.
#[derive(Debug, Clone)]
pub struct Iluk<I, T> {
    pub(crate) symbolic: SymbolicIluk<I>,
    pub(crate) l_values: Vec<T>,
    pub(crate) u_values: Vec<T>,
    pub(crate) workspace_w: Vec<T>,
    pub(crate) workspace_marker: Vec<usize>,
}

impl<I: Index, T: ComplexField> Iluk<I, T> {
    /// Allocate the value buffers and dense workspace for a symbolic factor.
    pub fn new_with_symbolic(symbolic: SymbolicIluk<I>) -> Self {
        let n = symbolic.dim;
        let l_values = (0..symbolic.l_nnz()).map(|_| zero::<T>()).collect();
        let u_values = (0..symbolic.u_nnz()).map(|_| zero::<T>()).collect();
        let workspace_w = (0..n).map(|_| zero::<T>()).collect();
        let workspace_marker = vec![usize::MAX; n];
        Self {
            symbolic,
            l_values,
            u_values,
            workspace_w,
            workspace_marker,
        }
    }

    /// Build a fully populated ILU(k) factor from a CSC matrix `A`.
    pub fn try_new(a: SparseColMatRef<'_, I, T>, level: usize) -> Result<Self, IlukError> {
        let symbolic = SymbolicIluk::try_new(a.symbolic(), level)?;
        let mut me = Self::new_with_symbolic(symbolic);
        me.refactorize(a)?;
        Ok(me)
    }

    /// Dimension `n` of the factored matrix.
    #[inline]
    pub fn dim(&self) -> usize {
        self.symbolic.dim
    }

    /// Borrow the symbolic factor.
    #[inline]
    pub fn symbolic(&self) -> &SymbolicIluk<I> {
        &self.symbolic
    }

    /// Refactorise against a new matrix with the same sparsity pattern.
    ///
    /// Performs zero heap allocations.
    ///
    /// # Errors
    ///
    /// - [`IlukError::PatternMismatch`] if `a`'s shape disagrees with the
    ///   symbolic structure.
    /// - [`IlukError::ZeroPivot`] if a pivot of magnitude zero is encountered.
    pub fn refactorize(&mut self, a: SparseColMatRef<'_, I, T>) -> Result<(), IlukError> {
        let n = self.symbolic.dim;
        if a.nrows() != n || a.ncols() != n {
            return Err(IlukError::PatternMismatch);
        }

        let symbolic = &self.symbolic;
        let w = self.workspace_w.as_mut_slice();
        let marker = self.workspace_marker.as_mut_slice();
        let l_values = self.l_values.as_mut_slice();
        let u_values = self.u_values.as_mut_slice();

        for j in 0..n {
            let u_start = symbolic.u_col_ptr[j].zx();
            let u_end = symbolic.u_col_ptr[j + 1].zx();
            let l_start = symbolic.l_col_ptr[j].zx();
            let l_end = symbolic.l_col_ptr[j + 1].zx();

            // Mark every pattern position in column j and zero it.
            for raw in &symbolic.u_row_idx[u_start..u_end] {
                let i = raw.zx();
                marker[i] = j;
                w[i] = zero::<T>();
            }
            for raw in &symbolic.l_row_idx[l_start..l_end] {
                let i = raw.zx();
                marker[i] = j;
                w[i] = zero::<T>();
            }
            // Overlay A's column j (its pattern is a subset of the fill pattern).
            let a_rows = a.symbolic().row_idx_of_col_raw(j);
            let a_vals = a.val_of_col(j);
            for (raw, val) in a_rows.iter().zip(a_vals.iter()) {
                let i = raw.zx();
                if marker[i] == j {
                    w[i] = copy(val);
                }
            }

            // Left-looking elimination: for each p < j in U's column j, subtract
            // L[:,p] * U[p,j] from w at pattern positions.
            for raw_p in &symbolic.u_row_idx[u_start..u_end - 1] {
                let p = raw_p.zx();
                let u_pj = copy(&w[p]);
                let lp_start = symbolic.l_col_ptr[p].zx();
                let lp_end = symbolic.l_col_ptr[p + 1].zx();
                for (raw_row, lv) in symbolic.l_row_idx[lp_start + 1..lp_end]
                    .iter()
                    .zip(l_values[lp_start + 1..lp_end].iter())
                {
                    let i = raw_row.zx();
                    if marker[i] == j {
                        let upd = mul(lv, &u_pj);
                        w[i] = sub(&w[i], &upd);
                    }
                }
            }

            // Gather U column j.
            for (raw_row, dst) in symbolic.u_row_idx[u_start..u_end]
                .iter()
                .zip(u_values[u_start..u_end].iter_mut())
            {
                let i = raw_row.zx();
                *dst = copy(&w[i]);
            }

            let pivot = copy(&u_values[u_end - 1]);
            if abs2(&pivot) == zero::<T::Real>() {
                return Err(IlukError::ZeroPivot { col: j });
            }
            let pivot_inv = recip(&pivot);

            // Gather L column j (off-diagonal entries divided by the pivot).
            l_values[l_start] = one::<T>();
            for (raw_row, dst) in symbolic.l_row_idx[l_start + 1..l_end]
                .iter()
                .zip(l_values[l_start + 1..l_end].iter_mut())
            {
                let i = raw_row.zx();
                *dst = mul(&w[i], &pivot_inv);
            }
        }

        Ok(())
    }

    /// View over the `L` factor (unit lower triangular, diagonal first).
    #[inline]
    pub fn l_view(&self) -> SparseColMatRef<'_, I, T> {
        let symbolic = unsafe {
            SymbolicSparseColMatRef::<'_, I>::new_unchecked(
                self.symbolic.dim,
                self.symbolic.dim,
                &self.symbolic.l_col_ptr,
                None,
                &self.symbolic.l_row_idx,
            )
        };
        SparseColMatRef::new(symbolic, &self.l_values)
    }

    /// View over the `U` factor (upper triangular, diagonal last).
    #[inline]
    pub fn u_view(&self) -> SparseColMatRef<'_, I, T> {
        let symbolic = unsafe {
            SymbolicSparseColMatRef::<'_, I>::new_unchecked(
                self.symbolic.dim,
                self.symbolic.dim,
                &self.symbolic.u_col_ptr,
                None,
                &self.symbolic.u_row_idx,
            )
        };
        SparseColMatRef::new(symbolic, &self.u_values)
    }
}
