//! Numeric ILUTP factorisation.
//!
//! Row-oriented IKJ Gaussian elimination with dual-threshold dropping and
//! column partial pivoting (Saad's ILUT + the "P", SPARSKIT `ilutp`). The
//! factor is produced one row at a time in CSR form, then transposed once into
//! the CSC layout faer's sparse triangular solves expect (`L` diagonal first,
//! `U` diagonal last).

use faer::sparse::SparseColMatRef;
use faer_traits::math_utils::{abs, abs2, add, from_f64, mul, one, recip, sqrt, sub, zero};
use faer_traits::{ComplexField, Index, RealField};

use super::{FillControl, IlutpError, IlutpParams, RowNorm};

/// Numeric ILUTP factor of a CSC matrix `A`.
///
/// Holds the `L`/`U` factors (CSC), the column permutation discovered during
/// pivoting, the tuning [`IlutpParams`], and the reusable workspaces the
/// factorisation runs on.
///
/// Unlike [`crate::Ilu0`], ILUTP's fill pattern is *value-dependent*, so there
/// is no separate symbolic phase and [`Ilutp::refactorize`] reuses buffer
/// capacity but is **not** allocation-free (it recomputes the pattern, which may
/// grow the buffers). The hot [`apply`](crate::Ilutp#impl-Precond<T>) path stays
/// allocation-free in the contract sense — its scratch flows through the
/// caller's `MemStack`.
#[derive(Debug, Clone)]
pub struct Ilutp<I: Index, T: ComplexField> {
    pub(crate) n: usize,

    // Final factors, CSC, faer diagonal convention.
    pub(crate) l_col_ptr: Vec<I>, // len n+1
    pub(crate) l_row_idx: Vec<I>, // unit diagonal stored first per column
    pub(crate) l_values: Vec<T>,
    pub(crate) u_col_ptr: Vec<I>, // len n+1
    pub(crate) u_row_idx: Vec<I>, // diagonal stored last per column
    pub(crate) u_values: Vec<T>,

    // Column permutation P, with (A P)[:, k] = A[:, perm[k]].
    pub(crate) perm: Vec<usize>,
    pub(crate) perm_inv: Vec<usize>,
    pub(crate) permuted: bool,

    pub(crate) params: IlutpParams,

    // --- Reusable factorisation scratch (capacity retained across refactorize) ---
    // A transposed to CSR so we can stream A's rows.
    at_row_ptr: Vec<usize>, // len n+1
    at_col_idx: Vec<usize>, // original column indices
    at_row_val: Vec<T>,
    // Dense accumulator + bookkeeping, indexed by PERMUTED column.
    w: Vec<T>,                  // len n
    marker: Vec<usize>,         // marker[pc] == i iff w[pc] is live for current row i; len n
    w_used: Vec<usize>,         // positions touched this row (cleared per row)
    heap: Vec<usize>,           // min-heap of live L-side (pc < i) columns for ordered IKJ
    u_diag: Vec<T>,             // cached U[k,k]; len n
    sel: Vec<(T::Real, usize)>, // select-the-p-largest scratch
    // Factor built row-wise (CSR) before the transpose to CSC.
    lr_ptr: Vec<usize>, // L off-diagonals per row (unit diagonal implicit)
    lr_idx: Vec<usize>,
    lr_val: Vec<T>,
    ur_ptr: Vec<usize>, // U per row, diagonal FIRST then off-diagonals (pc > i)
    ur_idx: Vec<usize>,
    ur_val: Vec<T>,
    cnt: Vec<usize>,   // per-column counts during CSC assembly; len n
    front: Vec<usize>, // per-column fill cursors during CSC assembly; len n
}

const SENTINEL: usize = usize::MAX;

impl<I: Index, T: ComplexField> Ilutp<I, T> {
    /// Build a fully populated ILUTP factor of `A` with default parameters.
    pub fn try_new(a: SparseColMatRef<'_, I, T>) -> Result<Self, IlutpError> {
        Self::try_new_with_params(a, IlutpParams::default())
    }

    /// Build a fully populated ILUTP factor of `A` with the given parameters.
    ///
    /// # Errors
    ///
    /// - [`IlutpError::NonSquareMatrix`] if `a` is not square.
    /// - [`IlutpError::InvalidDropTol`] / [`IlutpError::InvalidPivotTol`] /
    ///   [`IlutpError::InvalidFillControl`] for out-of-range parameters.
    /// - [`IlutpError::ZeroPivot`] if a row reduces to a zero pivot even after
    ///   pivoting (the matrix is numerically singular for these parameters).
    pub fn try_new_with_params(
        a: SparseColMatRef<'_, I, T>,
        params: IlutpParams,
    ) -> Result<Self, IlutpError> {
        params.validate()?;
        let nrows = a.nrows();
        let ncols = a.ncols();
        if nrows != ncols {
            return Err(IlutpError::NonSquareMatrix { nrows, ncols });
        }
        let n = nrows;
        let mut me = Self {
            n,
            l_col_ptr: Vec::new(),
            l_row_idx: Vec::new(),
            l_values: Vec::new(),
            u_col_ptr: Vec::new(),
            u_row_idx: Vec::new(),
            u_values: Vec::new(),
            perm: (0..n).collect(),
            perm_inv: (0..n).collect(),
            permuted: false,
            params,
            at_row_ptr: vec![0; n + 1],
            at_col_idx: Vec::new(),
            at_row_val: Vec::new(),
            w: (0..n).map(|_| zero::<T>()).collect(),
            marker: vec![SENTINEL; n],
            w_used: vec![0; n],
            heap: Vec::with_capacity(n),
            u_diag: (0..n).map(|_| zero::<T>()).collect(),
            sel: Vec::with_capacity(n),
            lr_ptr: Vec::with_capacity(n + 1),
            lr_idx: Vec::new(),
            lr_val: Vec::new(),
            ur_ptr: Vec::with_capacity(n + 1),
            ur_idx: Vec::new(),
            ur_val: Vec::new(),
            cnt: vec![0; n],
            front: vec![0; n],
        };
        me.factorize(a)?;
        Ok(me)
    }

    /// Refactorise against a new matrix of the same dimension.
    ///
    /// Reuses the existing buffer **capacity** but recomputes the
    /// value-dependent pattern and permutation — it may reallocate if `a`
    /// produces more fill than the current buffers hold. This is *not* the
    /// zero-allocation guarantee [`crate::Ilu0::refactorize`] gives; ILUTP's
    /// pattern cannot be fixed ahead of time.
    ///
    /// # Errors
    ///
    /// [`IlutpError::NonSquareMatrix`] if `a`'s dimension differs, or
    /// [`IlutpError::ZeroPivot`] on a numerically singular row.
    pub fn refactorize(&mut self, a: SparseColMatRef<'_, I, T>) -> Result<(), IlutpError> {
        if a.nrows() != self.n || a.ncols() != self.n {
            return Err(IlutpError::NonSquareMatrix {
                nrows: a.nrows(),
                ncols: a.ncols(),
            });
        }
        self.factorize(a)
    }

    /// Dimension `n` of the factored matrix.
    #[inline]
    pub fn dim(&self) -> usize {
        self.n
    }

    /// The tuning parameters this factor was built with.
    #[inline]
    pub fn params(&self) -> &IlutpParams {
        &self.params
    }

    /// The column permutation `perm`, where `(A P)[:, k] = A[:, perm[k]]`.
    #[inline]
    pub fn perm(&self) -> &[usize] {
        &self.perm
    }

    /// The inverse column permutation: `perm_inv[perm[k]] == k`.
    #[inline]
    pub fn perm_inv(&self) -> &[usize] {
        &self.perm_inv
    }

    /// Whether pivoting actually permuted any columns. When `false` the factor
    /// is a plain ILUT and [`apply`](crate::Ilutp) skips the permutation step.
    #[inline]
    pub fn is_permuted(&self) -> bool {
        self.permuted
    }

    /// View over the `L` factor (unit lower triangular, diagonal stored first
    /// in each column with value `1`).
    #[inline]
    pub fn l_view(&self) -> SparseColMatRef<'_, I, T> {
        let symbolic = unsafe {
            faer::sparse::SymbolicSparseColMatRef::<'_, I>::new_unchecked(
                self.n,
                self.n,
                &self.l_col_ptr,
                None,
                &self.l_row_idx,
            )
        };
        SparseColMatRef::new(symbolic, &self.l_values)
    }

    /// View over the `U` factor (upper triangular, diagonal stored last in each
    /// column).
    #[inline]
    pub fn u_view(&self) -> SparseColMatRef<'_, I, T> {
        let symbolic = unsafe {
            faer::sparse::SymbolicSparseColMatRef::<'_, I>::new_unchecked(
                self.n,
                self.n,
                &self.u_col_ptr,
                None,
                &self.u_row_idx,
            )
        };
        SparseColMatRef::new(symbolic, &self.u_values)
    }

    /// The per-row fill budget `p` for one triangular part.
    fn fill_budget(&self, a_nnz: usize) -> usize {
        match self.params.fill {
            FillControl::PerRow(p) => p,
            FillControl::Factor(f) => {
                let avg = a_nnz as f64 / self.n as f64;
                ((f * avg).round() as usize).max(1)
            }
        }
    }

    /// Build the CSR transpose of `A` into `at_*`, reusing capacity.
    fn build_csr(&mut self, a: SparseColMatRef<'_, I, T>) {
        let n = self.n;
        let a_sym = a.symbolic();

        for c in self.at_row_ptr.iter_mut() {
            *c = 0;
        }
        // Count entries per row.
        for j in 0..n {
            for r in a_sym.row_idx_of_col_raw(j) {
                self.at_row_ptr[r.zx() + 1] += 1;
            }
        }
        for i in 0..n {
            self.at_row_ptr[i + 1] += self.at_row_ptr[i];
        }
        let nnz = self.at_row_ptr[n];

        self.at_col_idx.clear();
        self.at_col_idx.resize(nnz, 0);
        self.at_row_val.clear();
        self.at_row_val.resize(nnz, zero::<T>());

        // Scatter into row-major order (rows come out column-sorted because we
        // visit columns ascending). `front` doubles as the per-row cursor.
        self.front[..n].copy_from_slice(&self.at_row_ptr[..n]);
        for j in 0..n {
            let rows = a_sym.row_idx_of_col_raw(j);
            let vals = a.val_of_col(j);
            for (r, v) in rows.iter().zip(vals.iter()) {
                let r = r.zx();
                let pos = self.front[r];
                self.front[r] += 1;
                self.at_col_idx[pos] = j;
                self.at_row_val[pos] = v.clone();
            }
        }
    }

    /// The full row-oriented IKJ factorisation.
    fn factorize(&mut self, a: SparseColMatRef<'_, I, T>) -> Result<(), IlutpError> {
        let n = self.n;
        self.build_csr(a);
        let p = self.fill_budget(self.at_row_ptr[n]);

        // Reset permutation, markers, and the CSR factor buffers.
        for k in 0..n {
            self.perm[k] = k;
            self.perm_inv[k] = k;
            self.marker[k] = SENTINEL;
        }
        self.permuted = false;
        self.lr_ptr.clear();
        self.lr_idx.clear();
        self.lr_val.clear();
        self.ur_ptr.clear();
        self.ur_idx.clear();
        self.ur_val.clear();
        self.lr_ptr.push(0);
        self.ur_ptr.push(0);

        let drop_tol = from_f64::<T::Real>(self.params.drop_tol);
        let pivot_tol = from_f64::<T::Real>(self.params.pivot_tol);
        let do_pivot = self.params.pivot_tol > 0.0;

        for i in 0..n {
            // (1) Scatter A's row i into w by permuted column; accumulate norm.
            let mut used_len = 0usize;
            let mut row_norm = zero::<T::Real>();
            for t in self.at_row_ptr[i]..self.at_row_ptr[i + 1] {
                let c = self.at_col_idx[t];
                let pc = self.perm_inv[c];
                let v = self.at_row_val[t].clone();
                row_norm = match self.params.norm {
                    RowNorm::One => add(&row_norm, &abs(&v)),
                    RowNorm::Two => add(&row_norm, &abs2(&v)),
                };
                self.marker[pc] = i;
                self.w[pc] = v;
                self.w_used[used_len] = pc;
                used_len += 1;
                if pc < i {
                    heap_push(&mut self.heap, pc);
                }
            }
            let row_norm = match self.params.norm {
                RowNorm::One => row_norm,
                RowNorm::Two => sqrt::<T::Real>(&row_norm),
            };
            let tau = mul(&drop_tol, &row_norm);

            // (2) IKJ elimination over live L-side columns k < i, ascending.
            while let Some(k) = heap_pop_min(&mut self.heap) {
                if self.marker[k] != i {
                    continue; // stale (dropped) entry
                }
                let ukk_inv = recip(&self.u_diag[k]);
                let wk = mul(&self.w[k], &ukk_inv);
                if abs(&wk) < tau {
                    self.marker[k] = SENTINEL; // first dropping rule on the multiplier
                    continue;
                }
                self.w[k] = wk.clone();
                // w[j] -= wk * U[k,j] for the off-diagonals j > k of U row k.
                let ks = self.ur_ptr[k];
                let ke = self.ur_ptr[k + 1];
                for t in (ks + 1)..ke {
                    let j = self.ur_idx[t];
                    let upd = mul(&wk, &self.ur_val[t]);
                    if self.marker[j] == i {
                        self.w[j] = sub(&self.w[j], &upd);
                    } else {
                        self.marker[j] = i;
                        self.w[j] = sub(&zero::<T>(), &upd);
                        self.w_used[used_len] = j;
                        used_len += 1;
                        if j < i {
                            heap_push(&mut self.heap, j);
                        }
                    }
                }
            }

            // (3) Column partial pivoting over live U-side candidates (pc >= i).
            let diag_live = self.marker[i] == i;
            let diag_mag = if diag_live {
                abs(&self.w[i])
            } else {
                zero::<T::Real>()
            };
            if do_pivot {
                let mut best = i;
                let mut best_mag = diag_mag.clone();
                for t in 0..used_len {
                    let pos = self.w_used[t];
                    if pos > i && self.marker[pos] == i {
                        let m = abs(&self.w[pos]);
                        if m > best_mag {
                            best_mag = m;
                            best = pos;
                        }
                    }
                }
                if best != i && best_mag > mul(&pivot_tol, &diag_mag) {
                    let oi = self.perm[i];
                    let ob = self.perm[best];
                    self.perm[i] = ob;
                    self.perm[best] = oi;
                    self.perm_inv[ob] = i;
                    self.perm_inv[oi] = best;
                    self.w.swap(i, best);
                    self.marker.swap(i, best);
                    self.permuted = true;
                    debug_assert!(best > i, "pivot only swaps columns >= i");
                    debug_assert_eq!(self.marker[i], i, "pivot column must be live at i");
                }
            }

            // (4) Pivot check.
            let pivot = if self.marker[i] == i {
                self.w[i].clone()
            } else {
                zero::<T>()
            };
            let pivot_mag = abs(&pivot);
            if pivot_mag.partial_cmp(&zero::<T::Real>()) != Some(core::cmp::Ordering::Greater) {
                return Err(IlutpError::ZeroPivot { row: i });
            }
            self.u_diag[i] = pivot.clone();

            // (5) Second dropping rule + emit row i.
            // U row: diagonal first, then the p largest off-diagonals (pc > i).
            self.ur_idx.push(i);
            self.ur_val.push(pivot);
            self.sel.clear();
            for t in 0..used_len {
                let pos = self.w_used[t];
                if pos > i && self.marker[pos] == i {
                    let m = abs(&self.w[pos]);
                    if m >= tau {
                        self.sel.push((m, pos));
                    }
                }
            }
            select_topk(&mut self.sel, p);
            for &(_, pos) in self.sel.iter() {
                self.ur_idx.push(pos);
                self.ur_val.push(self.w[pos].clone());
            }
            self.ur_ptr.push(self.ur_idx.len());

            // L row: the p largest multipliers (pc < i); unit diagonal is implicit.
            self.sel.clear();
            for t in 0..used_len {
                let pos = self.w_used[t];
                if pos < i && self.marker[pos] == i {
                    self.sel.push((abs(&self.w[pos]), pos));
                }
            }
            select_topk(&mut self.sel, p);
            for &(_, pos) in self.sel.iter() {
                self.lr_idx.push(pos);
                self.lr_val.push(self.w[pos].clone());
            }
            self.lr_ptr.push(self.lr_idx.len());

            // (6) Clear markers for next row.
            for t in 0..used_len {
                self.marker[self.w_used[t]] = SENTINEL;
            }
        }

        self.assemble_csc();
        Ok(())
    }

    /// Transpose the CSR factor buffers into the CSC layout the triangular
    /// solves require: `L` diagonal first (unit), `U` diagonal last.
    fn assemble_csc(&mut self) {
        let n = self.n;

        // ---- U: CSC, diagonal last ----
        for c in self.cnt[..n].iter_mut() {
            *c = 0;
        }
        for r in 0..n {
            for t in self.ur_ptr[r]..self.ur_ptr[r + 1] {
                self.cnt[self.ur_idx[t]] += 1;
            }
        }
        self.u_col_ptr.clear();
        self.u_col_ptr.resize(n + 1, I::truncate(0));
        let mut acc = 0usize;
        for c in 0..n {
            self.u_col_ptr[c] = I::truncate(acc);
            acc += self.cnt[c];
        }
        self.u_col_ptr[n] = I::truncate(acc);
        self.u_row_idx.clear();
        self.u_row_idx.resize(acc, I::truncate(0));
        self.u_values.clear();
        self.u_values.resize(acc, zero::<T>());
        for c in 0..n {
            self.front[c] = self.u_col_ptr[c].zx();
        }
        for r in 0..n {
            for t in self.ur_ptr[r]..self.ur_ptr[r + 1] {
                let c = self.ur_idx[t];
                if c == r {
                    // diagonal goes to the last slot of column c
                    let pos = self.u_col_ptr[c + 1].zx() - 1;
                    self.u_row_idx[pos] = I::truncate(r);
                    self.u_values[pos] = self.ur_val[t].clone();
                } else {
                    let pos = self.front[c];
                    self.front[c] += 1;
                    self.u_row_idx[pos] = I::truncate(r);
                    self.u_values[pos] = self.ur_val[t].clone();
                }
            }
        }

        // ---- L: CSC, diagonal first (unit) ----
        for c in 0..n {
            self.cnt[c] = 1; // implicit unit diagonal
        }
        for r in 0..n {
            for t in self.lr_ptr[r]..self.lr_ptr[r + 1] {
                self.cnt[self.lr_idx[t]] += 1;
            }
        }
        self.l_col_ptr.clear();
        self.l_col_ptr.resize(n + 1, I::truncate(0));
        let mut acc = 0usize;
        for c in 0..n {
            self.l_col_ptr[c] = I::truncate(acc);
            acc += self.cnt[c];
        }
        self.l_col_ptr[n] = I::truncate(acc);
        self.l_row_idx.clear();
        self.l_row_idx.resize(acc, I::truncate(0));
        self.l_values.clear();
        self.l_values.resize(acc, zero::<T>());
        for c in 0..n {
            let pos = self.l_col_ptr[c].zx();
            self.l_row_idx[pos] = I::truncate(c);
            self.l_values[pos] = one::<T>();
            self.front[c] = pos + 1;
        }
        for r in 0..n {
            for t in self.lr_ptr[r]..self.lr_ptr[r + 1] {
                let c = self.lr_idx[t];
                let pos = self.front[c];
                self.front[c] += 1;
                self.l_row_idx[pos] = I::truncate(r);
                self.l_values[pos] = self.lr_val[t].clone();
            }
        }

        if cfg!(debug_assertions) {
            for c in 0..n {
                let ls = self.l_col_ptr[c].zx();
                debug_assert_eq!(self.l_row_idx[ls].zx(), c, "L diagonal must be first");
                let ue = self.u_col_ptr[c + 1].zx() - 1;
                debug_assert_eq!(self.u_row_idx[ue].zx(), c, "U diagonal must be last");
            }
        }
    }
}

/// Push `x` onto the binary min-heap `h`.
fn heap_push(h: &mut Vec<usize>, x: usize) {
    h.push(x);
    let mut c = h.len() - 1;
    while c > 0 {
        let parent = (c - 1) / 2;
        if h[parent] <= h[c] {
            break;
        }
        h.swap(parent, c);
        c = parent;
    }
}

/// Pop the minimum element of the binary min-heap `h`.
fn heap_pop_min(h: &mut Vec<usize>) -> Option<usize> {
    let len = h.len();
    if len == 0 {
        return None;
    }
    h.swap(0, len - 1);
    let min = h.pop().unwrap();
    let len = h.len();
    let mut c = 0;
    loop {
        let l = 2 * c + 1;
        let r = 2 * c + 2;
        let mut m = c;
        if l < len && h[l] < h[m] {
            m = l;
        }
        if r < len && h[r] < h[m] {
            m = r;
        }
        if m == c {
            break;
        }
        h.swap(c, m);
        c = m;
    }
    Some(min)
}

/// Keep the `p` largest-magnitude entries of `sel`, then sort them by position
/// ascending (so the emitted row is column-ordered).
fn select_topk<R: RealField>(sel: &mut Vec<(R, usize)>, p: usize) {
    if sel.len() > p {
        sel.select_nth_unstable_by(p, |a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(core::cmp::Ordering::Equal)
        });
        sel.truncate(p);
    }
    sel.sort_unstable_by_key(|&(_, pos)| pos);
}
