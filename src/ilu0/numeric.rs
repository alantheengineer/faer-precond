//! Numeric ILU(0) factorisation.

use faer::sparse::SparseColMatRef;
use faer_traits::math_utils::{abs2, copy, mul, one, recip, sub, zero};
use faer_traits::{ComplexField, Index};

use super::symbolic::SymbolicIlu0;
use super::Ilu0Error;

/// Numeric ILU(0) factor of a CSC matrix `A`.
///
/// Holds the `L` and `U` value buffers together with the symbolic structure
/// and the dense workspace used during refactorisation. After construction the
/// factor can be refactored against any matrix with the same sparsity pattern
/// — [`Ilu0::refactorize`] performs zero heap allocations.
#[derive(Debug, Clone)]
pub struct Ilu0<I, T> {
    pub(crate) symbolic: SymbolicIlu0<I>,
    pub(crate) l_values: Vec<T>,
    pub(crate) u_values: Vec<T>,
    pub(crate) workspace_w: Vec<T>,
    pub(crate) workspace_marker: Vec<usize>,
}

impl<I: Index, T: ComplexField> Ilu0<I, T> {
    /// Allocate the value buffers and dense workspace for a given symbolic factor.
    ///
    /// The returned factor is *not* yet populated — call [`Ilu0::refactorize`]
    /// to fill it with values from a matrix matching `symbolic`'s pattern.
    pub fn new_with_symbolic(symbolic: SymbolicIlu0<I>) -> Self {
        let n = symbolic.dim;
        let l_nnz = symbolic.l_nnz();
        let u_nnz = symbolic.u_nnz();
        let l_values = (0..l_nnz).map(|_| zero::<T>()).collect();
        let u_values = (0..u_nnz).map(|_| zero::<T>()).collect();
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

    /// Build a fully populated ILU(0) factor from a CSC matrix `A` in one shot.
    ///
    /// Allocates internally for the symbolic structure, the `L`/`U` value
    /// buffers, and the dense workspace.
    pub fn try_new(a: SparseColMatRef<'_, I, T>) -> Result<Self, Ilu0Error> {
        let symbolic = SymbolicIlu0::try_new(a.symbolic())?;
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
    pub fn symbolic(&self) -> &SymbolicIlu0<I> {
        &self.symbolic
    }

    /// Refactorise against a new matrix with the same sparsity pattern.
    ///
    /// Performs zero heap allocations.
    ///
    /// # Errors
    ///
    /// - [`Ilu0Error::PatternMismatch`] if `a`'s shape or column lengths
    ///   disagree with the symbolic structure, or if the diagonal entries land
    ///   at unexpected positions.
    /// - [`Ilu0Error::ZeroPivot`] if a pivot of magnitude zero is encountered
    ///   during elimination.
    pub fn refactorize(&mut self, a: SparseColMatRef<'_, I, T>) -> Result<(), Ilu0Error> {
        let n = self.symbolic.dim;
        if a.nrows() != n || a.ncols() != n {
            return Err(Ilu0Error::PatternMismatch);
        }
        let a_sym = a.symbolic();

        // Stage 1: copy A's values into L and U slots.
        for j in 0..n {
            let a_row_idx = a_sym.row_idx_of_col_raw(j);
            let a_col_len = a_row_idx.len();
            let d = self.symbolic.diag_pos[j];
            if d >= a_col_len || a_row_idx[d].zx() != j {
                return Err(Ilu0Error::PatternMismatch);
            }
            let u_start = self.symbolic.u_col_ptr[j].zx();
            let u_end = self.symbolic.u_col_ptr[j + 1].zx();
            let l_start = self.symbolic.l_col_ptr[j].zx();
            let l_end = self.symbolic.l_col_ptr[j + 1].zx();
            if u_end - u_start != d + 1 || l_end - l_start != a_col_len - d {
                return Err(Ilu0Error::PatternMismatch);
            }

            let a_val = a.val_of_col(j);
            for (k, val) in a_val[..=d].iter().enumerate() {
                self.u_values[u_start + k] = copy(val);
            }
            self.l_values[l_start] = one::<T>();
            for (k, val) in a_val[d + 1..].iter().enumerate() {
                self.l_values[l_start + 1 + k] = copy(val);
            }
        }

        // Stage 2: ILU(0) elimination — column-by-column, left-looking.
        // Borrow the workspaces and value buffers separately to avoid aliasing.
        let symbolic = &self.symbolic;
        let w = self.workspace_w.as_mut_slice();
        let marker = self.workspace_marker.as_mut_slice();
        let l_values = self.l_values.as_mut_slice();
        let u_values = self.u_values.as_mut_slice();

        for j in 0..n {
            let l_start = symbolic.l_col_ptr[j].zx();
            let l_end = symbolic.l_col_ptr[j + 1].zx();
            let u_start = symbolic.u_col_ptr[j].zx();
            let u_end = symbolic.u_col_ptr[j + 1].zx();

            // Scatter U-column and L-column (skip L's unit diagonal slot)
            // into the dense working vector, marking the touched rows.
            for (raw_row, val) in symbolic.u_row_idx[u_start..u_end]
                .iter()
                .zip(u_values[u_start..u_end].iter())
            {
                let i = raw_row.zx();
                marker[i] = j;
                w[i] = copy(val);
            }
            for (raw_row, val) in symbolic.l_row_idx[l_start + 1..l_end]
                .iter()
                .zip(l_values[l_start + 1..l_end].iter())
            {
                let i = raw_row.zx();
                marker[i] = j;
                w[i] = copy(val);
            }

            // For each row p < j present in U's column j, subtract L[:,p] * U[p,j]
            // from w[:] — but only at positions present in pattern(A col j).
            // U's diagonal sits at position u_end - 1; everything before it is p < j.
            for raw_p in &symbolic.u_row_idx[u_start..u_end - 1] {
                let p = raw_p.zx();
                let u_pj = copy(&w[p]);
                let lp_start = symbolic.l_col_ptr[p].zx();
                let lp_end = symbolic.l_col_ptr[p + 1].zx();
                // Skip the unit diagonal slot at lp_start (row p).
                for (raw_row, lv) in symbolic.l_row_idx[lp_start + 1..lp_end]
                    .iter()
                    .zip(l_values[lp_start + 1..lp_end].iter())
                {
                    let i = raw_row.zx();
                    if marker[i] == j {
                        let l_ip = copy(lv);
                        let upd = mul(&l_ip, &u_pj);
                        w[i] = sub(&w[i], &upd);
                    }
                }
            }

            // Gather updated U-column.
            for (raw_row, dst) in symbolic.u_row_idx[u_start..u_end]
                .iter()
                .zip(u_values[u_start..u_end].iter_mut())
            {
                let i = raw_row.zx();
                *dst = copy(&w[i]);
            }

            // Pivot is U[j,j] which sits at the last U slot of column j.
            let pivot = copy(&u_values[u_end - 1]);
            if abs2(&pivot) == zero::<T::Real>() {
                return Err(Ilu0Error::ZeroPivot { col: j });
            }
            let pivot_inv = recip(&pivot);

            // Gather updated L-column (divided by pivot), keeping the unit
            // diagonal at l_start.
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

    /// Construct a [`SparseColMatRef`] view over the `L` factor (unit lower
    /// triangular, diagonal stored first in each column with value `1`).
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

    /// Construct a [`SparseColMatRef`] view over the `U` factor (upper
    /// triangular, diagonal stored last in each column).
    #[inline]
    pub fn u_view(&self) -> SparseColMatRef<'_, I, T> {
        let symbolic = unsafe {
            faer::sparse::SymbolicSparseColMatRef::<'_, I>::new_unchecked(
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
