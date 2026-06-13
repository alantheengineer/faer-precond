//! Numeric IC(0) factorisation.

use faer::sparse::SparseColMatRef;
use faer_traits::math_utils::{conj, copy, from_real, mul, real, recip, sqrt, sub, zero};
use faer_traits::{ComplexField, Index};

use super::Ic0Error;
use super::symbolic::SymbolicIc0;

/// Numeric IC(0) factor of a Hermitian positive-definite CSC matrix `A`.
///
/// Holds the `L` value buffer together with the symbolic structure and the
/// dense workspace used during refactorisation. After construction the
/// factor can be refactored against any matrix with the same lower-triangle
/// pattern — [`Ic0::refactorize`] performs zero heap allocations.
#[derive(Debug, Clone)]
pub struct Ic0<I, T> {
    pub(crate) symbolic: SymbolicIc0<I>,
    pub(crate) l_values: Vec<T>,
    pub(crate) workspace_w: Vec<T>,
    pub(crate) workspace_marker: Vec<usize>,
}

impl<I: Index, T: ComplexField> Ic0<I, T> {
    /// Allocate the value buffer and dense workspace for a given symbolic factor.
    ///
    /// The returned factor is *not* yet populated — call [`Ic0::refactorize`]
    /// to fill it with values from a matrix matching `symbolic`'s pattern.
    pub fn new_with_symbolic(symbolic: SymbolicIc0<I>) -> Self {
        let n = symbolic.dim;
        let l_nnz = symbolic.l_nnz();
        let l_values = (0..l_nnz).map(|_| zero::<T>()).collect();
        let workspace_w = (0..n).map(|_| zero::<T>()).collect();
        let workspace_marker = vec![usize::MAX; n];
        Self {
            symbolic,
            l_values,
            workspace_w,
            workspace_marker,
        }
    }

    /// Build a fully populated IC(0) factor from a Hermitian PD CSC matrix `A`.
    ///
    /// Only the lower triangle of `A` is consumed — values above the diagonal
    /// are silently ignored. Allocates internally for the symbolic structure,
    /// the `L` value buffer, and the dense workspace.
    pub fn try_new(a: SparseColMatRef<'_, I, T>) -> Result<Self, Ic0Error> {
        let symbolic = SymbolicIc0::try_new(a.symbolic())?;
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
    pub fn symbolic(&self) -> &SymbolicIc0<I> {
        &self.symbolic
    }

    /// Refactorise against a new matrix with the same sparsity pattern.
    ///
    /// Performs zero heap allocations.
    ///
    /// # Errors
    ///
    /// - [`Ic0Error::PatternMismatch`] if `a`'s shape or column lengths
    ///   disagree with the symbolic structure, or if the diagonal entries
    ///   land at unexpected positions.
    /// - [`Ic0Error::NotPositiveDefinite`] if a non-positive pivot is
    ///   encountered (indicating the matrix is not positive definite, or that
    ///   the IC(0) algorithm has broken down on a non-H-matrix input).
    pub fn refactorize(&mut self, a: SparseColMatRef<'_, I, T>) -> Result<(), Ic0Error> {
        let n = self.symbolic.dim;
        if a.nrows() != n || a.ncols() != n {
            return Err(Ic0Error::PatternMismatch);
        }
        let a_sym = a.symbolic();

        // Stage 1: copy A's lower triangle into L slots.
        for j in 0..n {
            let a_row_idx = a_sym.row_idx_of_col_raw(j);
            let a_col_len = a_row_idx.len();
            let d = self.symbolic.diag_pos[j];
            if d >= a_col_len || a_row_idx[d].zx() != j {
                return Err(Ic0Error::PatternMismatch);
            }
            let l_start = self.symbolic.l_col_ptr[j].zx();
            let l_end = self.symbolic.l_col_ptr[j + 1].zx();
            if l_end - l_start != a_col_len - d {
                return Err(Ic0Error::PatternMismatch);
            }

            let a_val = a.val_of_col(j);
            for (dst, val) in self.l_values[l_start..l_end]
                .iter_mut()
                .zip(a_val[d..].iter())
            {
                *dst = copy(val);
            }
        }

        // Stage 2: IC(0) elimination — column-by-column, left-looking.
        let symbolic = &self.symbolic;
        let w = self.workspace_w.as_mut_slice();
        let marker = self.workspace_marker.as_mut_slice();
        let l_values = self.l_values.as_mut_slice();

        for j in 0..n {
            let l_start = symbolic.l_col_ptr[j].zx();
            let l_end = symbolic.l_col_ptr[j + 1].zx();

            // Scatter L col j (currently containing A's lower triangle) into w.
            for (raw_row, val) in symbolic.l_row_idx[l_start..l_end]
                .iter()
                .zip(l_values[l_start..l_end].iter())
            {
                let i = raw_row.zx();
                marker[i] = j;
                w[i] = copy(val);
            }

            // For each k < j with L[j, k] in pattern, subtract
            // L[i, k] * conj(L[j, k]) from w[i] at every i >= j with
            // (i, j) ∈ pattern(L col j).
            let rcp_start = symbolic.row_col_ptr[j];
            let rcp_end = symbolic.row_col_ptr[j + 1];
            for (raw_k, &l_jk_pos) in symbolic.row_col_idx[rcp_start..rcp_end]
                .iter()
                .zip(symbolic.row_lpos[rcp_start..rcp_end].iter())
            {
                let k = raw_k.zx();
                let lk_end = symbolic.l_col_ptr[k + 1].zx();
                let conj_l_jk = conj(&l_values[l_jk_pos]);
                // Range [l_jk_pos..lk_end] covers L col k rows >= j (including i = j,
                // which contributes -|L[j,k]|^2 to w[j]).
                for (raw_row, lv) in symbolic.l_row_idx[l_jk_pos..lk_end]
                    .iter()
                    .zip(l_values[l_jk_pos..lk_end].iter())
                {
                    let i = raw_row.zx();
                    if marker[i] == j {
                        let upd = mul(lv, &conj_l_jk);
                        w[i] = sub(&w[i], &upd);
                    }
                }
            }

            // Pivot: A[j,j] - sum_k |L[j,k]|^2 must be real and positive.
            // `partial_cmp` catches NaN as well as zero/negative.
            let pivot_real = real(&w[j]);
            if pivot_real.partial_cmp(&zero::<T::Real>()) != Some(core::cmp::Ordering::Greater) {
                return Err(Ic0Error::NotPositiveDefinite { col: j });
            }
            let pivot_root = sqrt::<T::Real>(&pivot_real);
            let pivot = from_real::<T>(&pivot_root);
            let pivot_inv = recip(&pivot);

            // Gather L col j: diagonal slot gets pivot, off-diagonal entries
            // get w[i] / pivot.
            l_values[l_start] = pivot;
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

    /// Construct a [`SparseColMatRef`] view over the `L` factor (lower
    /// triangular, diagonal stored first in each column with the positive
    /// real pivot value).
    #[inline]
    pub fn l_view(&self) -> SparseColMatRef<'_, I, T> {
        let symbolic = unsafe {
            faer::sparse::SymbolicSparseColMatRef::<'_, I>::new_unchecked(
                self.symbolic.dim,
                self.symbolic.dim,
                &self.symbolic.l_col_ptr,
                None,
                &self.symbolic.l_row_idx,
            )
        };
        SparseColMatRef::new(symbolic, &self.l_values)
    }
}
