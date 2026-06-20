//! Construction of the SPAI approximate inverse `M`.
//!
//! SPAI minimises `||A M - I||_F`, which decouples into independent least-
//! squares problems, one per column: `min ||A m_k - e_k||`. For a prescribed
//! column pattern `J_k`, only the rows `I_k = union of pattern(A[:,j]) for j in
//! J_k` are involved, so each solve is a small dense overdetermined system
//! `A[I_k, J_k] m_k = e_k[I_k]`, handled by a QR least-squares solve.

use faer::linalg::solvers::{Qr, SolveLstsq};
use faer::sparse::SparseColMatRef;
use faer::Mat;
use faer_traits::math_utils::{copy, one};
use faer_traits::{ComplexField, Index};

use super::{Spai, SpaiError, SpaiPattern};

impl<I: Index, T: ComplexField> Spai<I, T> {
    /// Build a SPAI preconditioner for `A`.
    ///
    /// # Errors
    ///
    /// - [`SpaiError::NonSquareMatrix`] if `A` is not square.
    /// - [`SpaiError::InvalidPower`] if a `ColumnsOfPower` power is zero.
    pub fn try_new(a: SparseColMatRef<'_, I, T>, pattern: SpaiPattern) -> Result<Self, SpaiError> {
        if a.nrows() != a.ncols() {
            return Err(SpaiError::NonSquareMatrix {
                nrows: a.nrows(),
                ncols: a.ncols(),
            });
        }
        let power = match pattern {
            SpaiPattern::ColumnsOfA => 1,
            SpaiPattern::ColumnsOfPower { power } => power,
        };
        if power == 0 {
            return Err(SpaiError::InvalidPower);
        }
        let n = a.nrows();
        let col_pats = column_patterns(a, power);

        let mut m_col_ptr: Vec<I> = Vec::with_capacity(n + 1);
        m_col_ptr.push(I::truncate(0));
        let mut m_row_idx: Vec<I> = Vec::new();
        let mut m_values: Vec<T> = Vec::new();

        let mut marker = vec![usize::MAX; n];
        let mut local = vec![0usize; n];

        for k in 0..n {
            let jk = &col_pats[k];

            // Row set I_k = union of the patterns of A's columns in J_k.
            let mut ik: Vec<usize> = Vec::new();
            for &j in jk {
                for raw in a.symbolic().row_idx_of_col_raw(j) {
                    let r = raw.zx();
                    if marker[r] != k {
                        marker[r] = k;
                        ik.push(r);
                    }
                }
            }
            ik.sort_unstable();
            for (rr, &r) in ik.iter().enumerate() {
                local[r] = rr;
            }

            let m_rows = ik.len();
            let n_cols = jk.len();
            let mut a_sub = Mat::<T>::zeros(m_rows, n_cols);
            for (cc, &j) in jk.iter().enumerate() {
                for (raw, val) in a
                    .symbolic()
                    .row_idx_of_col_raw(j)
                    .iter()
                    .zip(a.val_of_col(j).iter())
                {
                    let r = raw.zx();
                    // every such r is in I_k by construction
                    *a_sub.as_mut().get_mut(local[r], cc) = copy(val);
                }
            }

            let mut e = Mat::<T>::zeros(m_rows, 1);
            if marker[k] == k {
                *e.as_mut().get_mut(local[k], 0) = one::<T>();
            }

            let m_k = Qr::new(a_sub.as_ref()).solve_lstsq(&e);

            for (cc, &j) in jk.iter().enumerate() {
                m_row_idx.push(I::truncate(j));
                m_values.push(copy(m_k.as_ref().get(cc, 0)));
            }
            m_col_ptr.push(I::truncate(m_row_idx.len()));
        }

        Ok(Self {
            dim: n,
            m_col_ptr,
            m_row_idx,
            m_values,
        })
    }
}

/// Pattern of each column of `A^power` (each list sorted).
fn column_patterns<I: Index, T: ComplexField>(
    a: SparseColMatRef<'_, I, T>,
    power: usize,
) -> Vec<Vec<usize>> {
    let n = a.ncols();
    let mut p: Vec<Vec<usize>> = (0..n)
        .map(|k| {
            let mut v: Vec<usize> = a
                .symbolic()
                .row_idx_of_col_raw(k)
                .iter()
                .map(|r| r.zx())
                .collect();
            v.sort_unstable();
            v
        })
        .collect();

    let mut marker = vec![usize::MAX; n];
    for _ in 1..power {
        let mut np: Vec<Vec<usize>> = Vec::with_capacity(n);
        for (k, col) in p.iter().enumerate() {
            let mut rows = Vec::new();
            for &m in col {
                for raw in a.symbolic().row_idx_of_col_raw(m) {
                    let r = raw.zx();
                    if marker[r] != k {
                        marker[r] = k;
                        rows.push(r);
                    }
                }
            }
            rows.sort_unstable();
            np.push(rows);
        }
        p = np;
    }
    p
}
