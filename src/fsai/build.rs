//! Construction of the FSAI factor `G`.
//!
//! For a prescribed lower-triangular pattern, FSAI computes `G ~= L^{-1}` (with
//! `A = L L^H` the Cholesky factor) *without* forming `L`, by solving one small
//! dense SPD system per row. Row `i` with pattern set `J_i = {j <= i : (i,j) in
//! pattern}` solves `A[J_i, J_i] g = e_i` and scales by `1 / sqrt(g_i)`; the
//! resulting `M^{-1} = G^H G` has unit diagonal in `G A G^H`.

use faer::sparse::SparseColMatRef;
use faer::linalg::solvers::Solve;
use faer::linalg::solvers::Llt;
use faer::{Mat, Side};
use faer_traits::math_utils::{conj, copy, from_real, mul, one, real, recip, sqrt, zero};
use faer_traits::{ComplexField, Index};

use super::{Fsai, FsaiError, FsaiPattern};

impl<I: Index, T: ComplexField> Fsai<I, T> {
    /// Build an FSAI preconditioner for the Hermitian positive-definite `A`.
    ///
    /// # Errors
    ///
    /// - [`FsaiError::NonSquareMatrix`] if `A` is not square.
    /// - [`FsaiError::InvalidPower`] if a `LowerOfPower` power is zero.
    /// - [`FsaiError::NotPositiveDefinite`] if a local block is not SPD.
    pub fn try_new(a: SparseColMatRef<'_, I, T>, pattern: FsaiPattern) -> Result<Self, FsaiError> {
        if a.nrows() != a.ncols() {
            return Err(FsaiError::NonSquareMatrix {
                nrows: a.nrows(),
                ncols: a.ncols(),
            });
        }
        let power = match pattern {
            FsaiPattern::LowerOfA => 1,
            FsaiPattern::LowerOfPower { power } => power,
        };
        if power == 0 {
            return Err(FsaiError::InvalidPower);
        }
        let n = a.nrows();
        let row_sets = lower_row_patterns(a, power);

        // Accumulate G column-by-column (rows appended in ascending order).
        let mut col_rows: Vec<Vec<I>> = (0..n).map(|_| Vec::new()).collect();
        let mut col_vals: Vec<Vec<T>> = (0..n).map(|_| Vec::new()).collect();

        for (i, ji) in row_sets.iter().enumerate() {
            let m = ji.len();
            let pos_i = ji.iter().position(|&j| j == i).expect("diagonal in pattern");

            // Dense local SPD block A[J_i, J_i] and unit RHS e_i (at pos_i).
            let mut a_sub = Mat::<T>::zeros(m, m);
            for (cc, &c) in ji.iter().enumerate() {
                for (rr, &r) in ji.iter().enumerate() {
                    *a_sub.as_mut().get_mut(rr, cc) = get_hermitian(a, r, c);
                }
            }
            let mut e = Mat::<T>::zeros(m, 1);
            *e.as_mut().get_mut(pos_i, 0) = one::<T>();

            let llt = Llt::new(a_sub.as_ref(), Side::Lower)
                .map_err(|_| FsaiError::NotPositiveDefinite { row: i })?;
            let g = llt.solve(&e);

            // Scale row i by 1 / sqrt(g_i); g_i = (A_sub^{-1})_{ii} is real > 0.
            let g_ii = real(g.as_ref().get(pos_i, 0));
            if g_ii.partial_cmp(&zero::<T::Real>()) != Some(core::cmp::Ordering::Greater) {
                return Err(FsaiError::NotPositiveDefinite { row: i });
            }
            let d = recip(&from_real::<T>(&sqrt::<T::Real>(&g_ii)));

            for (rr, &c) in ji.iter().enumerate() {
                let val = mul(&d, g.as_ref().get(rr, 0));
                col_rows[c].push(I::truncate(i));
                col_vals[c].push(val);
            }
        }

        // Assemble CSC G.
        let mut g_col_ptr: Vec<I> = Vec::with_capacity(n + 1);
        g_col_ptr.push(I::truncate(0));
        let mut g_row_idx: Vec<I> = Vec::new();
        let mut g_values: Vec<T> = Vec::new();
        for j in 0..n {
            g_row_idx.append(&mut col_rows[j]);
            g_values.append(&mut col_vals[j]);
            g_col_ptr.push(I::truncate(g_row_idx.len()));
        }

        Ok(Self {
            dim: n,
            g_col_ptr,
            g_row_idx,
            g_values,
        })
    }
}

/// `A[r, c]` for a Hermitian matrix that may store only its lower triangle.
fn get_hermitian<I: Index, T: ComplexField>(a: SparseColMatRef<'_, I, T>, r: usize, c: usize) -> T {
    let (lo, hi) = if r <= c { (r, c) } else { (c, r) };
    let rows = a.symbolic().row_idx_of_col_raw(lo);
    let vals = a.val_of_col(lo);
    let mut found = zero::<T>();
    for (raw, v) in rows.iter().zip(vals.iter()) {
        if raw.zx() == hi {
            found = copy(v);
            break;
        }
    }
    if r >= c { found } else { conj(&found) }
}

/// Per-row lower-triangular pattern of `pattern(A^power)` (each list sorted,
/// columns `<= i`, diagonal included).
fn lower_row_patterns<I: Index, T: ComplexField>(
    a: SparseColMatRef<'_, I, T>,
    power: usize,
) -> Vec<Vec<usize>> {
    let n = a.nrows();

    // CSR pattern of A (ascending columns per row).
    let mut row_cnt = vec![0usize; n];
    for j in 0..n {
        for raw in a.symbolic().row_idx_of_col_raw(j) {
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
        for raw in a.symbolic().row_idx_of_col_raw(j) {
            let i = raw.zx();
            col_idx[cursor[i]] = j;
            cursor[i] += 1;
        }
    }

    let mut p: Vec<Vec<usize>> = (0..n)
        .map(|i| col_idx[row_ptr[i]..row_ptr[i + 1]].to_vec())
        .collect();

    // Symbolic powers: P <- pattern(P * A).
    let mut marker = vec![usize::MAX; n];
    for _ in 1..power {
        let mut np: Vec<Vec<usize>> = Vec::with_capacity(n);
        for (i, row) in p.iter().enumerate() {
            let mut cols = Vec::new();
            for &m in row {
                for &c in &col_idx[row_ptr[m]..row_ptr[m + 1]] {
                    if marker[c] != i {
                        marker[c] = i;
                        cols.push(c);
                    }
                }
            }
            cols.sort_unstable();
            np.push(cols);
        }
        p = np;
    }

    // Lower triangle, diagonal guaranteed present.
    for (i, row) in p.iter_mut().enumerate() {
        row.retain(|&c| c <= i);
        if !row.contains(&i) {
            row.push(i);
            row.sort_unstable();
        }
    }
    p
}
