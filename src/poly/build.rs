//! Construction, bound estimation and refactorisation for [`Poly`].

use faer::sparse::{SparseColMatRef, SymbolicSparseColMatRef};
use faer_traits::math_utils::{copy, from_f64, from_real, max, mul, zero};
use faer_traits::{ComplexField, Index};

use super::{BoundEstimate, Coeffs, Poly, PolyError, PolyKind, PolyParams};
use crate::util::spd_bounds::{gershgorin_bounds, power_iteration_max};

/// Heuristic floor for `lambda_min` (as a fraction of `lambda_max`) when the
/// Gershgorin lower bound is non-positive. Deliberately tiny so the interval is
/// only ever *widened* (a safe error for Chebyshev) rather than narrowed.
const LAMBDA_MIN_FLOOR: f64 = 1e-4;

fn check_square<I: Index, T: ComplexField>(
    a: SparseColMatRef<'_, I, T>,
) -> Result<(), PolyError> {
    if a.nrows() != a.ncols() {
        return Err(PolyError::NonSquareMatrix {
            nrows: a.nrows(),
            ncols: a.ncols(),
        });
    }
    Ok(())
}

fn resolve_kind<T: ComplexField>(kind: &PolyKind) -> Result<Coeffs<T>, PolyError> {
    match *kind {
        PolyKind::Neumann { omega } => {
            if !omega.is_finite() || omega <= 0.0 {
                return Err(PolyError::InvalidOmega);
            }
            Ok(Coeffs::Neumann {
                omega: from_f64::<T>(omega),
            })
        }
        PolyKind::Chebyshev {
            lambda_min,
            lambda_max,
        } => {
            if !lambda_min.is_finite()
                || !lambda_max.is_finite()
                || lambda_min <= 0.0
                || lambda_max <= lambda_min
            {
                return Err(PolyError::InvalidBounds);
            }
            Ok(Coeffs::Chebyshev {
                lambda_min: from_f64::<T>(lambda_min),
                lambda_max: from_f64::<T>(lambda_max),
            })
        }
    }
}

/// Estimate `(lambda_min, lambda_max)` for the (assumed Hermitian PD) operator.
fn resolve_bounds<I: Index, T: ComplexField>(
    a: SparseColMatRef<'_, I, T>,
    estimate: &BoundEstimate,
) -> Result<(T::Real, T::Real), PolyError> {
    match *estimate {
        BoundEstimate::Manual {
            lambda_min,
            lambda_max,
        } => {
            if !lambda_min.is_finite()
                || !lambda_max.is_finite()
                || lambda_min <= 0.0
                || lambda_max <= lambda_min
            {
                return Err(PolyError::InvalidBounds);
            }
            Ok((from_f64::<T::Real>(lambda_min), from_f64::<T::Real>(lambda_max)))
        }
        BoundEstimate::Gershgorin => {
            let (glo, ghi) = gershgorin_bounds(a);
            finalize_bounds::<T>(glo, ghi)
        }
        BoundEstimate::PowerIteration { iters } => {
            let hi = power_iteration_max(a, iters.max(1));
            let (glo, _) = gershgorin_bounds(a);
            finalize_bounds::<T>(glo, hi)
        }
    }
}

/// Clamp a raw `(lo, hi)` estimate to a usable Chebyshev interval: `hi` must be
/// positive, and `lo` is floored to a small positive value when the structural
/// lower bound is non-positive.
fn finalize_bounds<T: ComplexField>(glo: T::Real, hi: T::Real) -> Result<(T::Real, T::Real), PolyError> {
    if hi.partial_cmp(&zero::<T::Real>()) != Some(core::cmp::Ordering::Greater) {
        return Err(PolyError::InvalidBounds);
    }
    let floor = mul(&hi, &from_f64::<T::Real>(LAMBDA_MIN_FLOOR));
    let lo = max(&glo, &floor);
    Ok((lo, hi))
}

impl<I: Index, T: ComplexField> Poly<I, T> {
    /// Build a polynomial preconditioner with an explicit [`PolyKind`].
    ///
    /// # Errors
    ///
    /// - [`PolyError::NonSquareMatrix`] if `a` is not square.
    /// - [`PolyError::ZeroDegree`] if `degree` is zero.
    /// - [`PolyError::InvalidOmega`] / [`PolyError::InvalidBounds`] if the kind
    ///   parameters are out of range.
    pub fn try_new(a: SparseColMatRef<'_, I, T>, params: PolyParams) -> Result<Self, PolyError> {
        check_square(a)?;
        if params.degree == 0 {
            return Err(PolyError::ZeroDegree);
        }
        let coeffs = resolve_kind::<T>(&params.kind)?;
        Ok(Self::assemble(a, params.degree, coeffs, None))
    }

    /// Build a Chebyshev preconditioner, estimating the spectral interval with
    /// `estimate`.
    ///
    /// Chebyshev acceleration is only as good as its bounds; see
    /// [`BoundEstimate`]. For precise control pass [`BoundEstimate::Manual`] or
    /// use [`Poly::try_new`] with [`PolyKind::Chebyshev`].
    ///
    /// # Errors
    ///
    /// As [`Poly::try_new`], plus [`PolyError::InvalidBounds`] if the estimate
    /// does not yield a usable `0 < lambda_min < lambda_max`.
    pub fn try_new_auto(
        a: SparseColMatRef<'_, I, T>,
        degree: usize,
        estimate: BoundEstimate,
    ) -> Result<Self, PolyError> {
        check_square(a)?;
        if degree == 0 {
            return Err(PolyError::ZeroDegree);
        }
        let (lo, hi) = resolve_bounds::<I, T>(a, &estimate)?;
        let coeffs = Coeffs::Chebyshev {
            lambda_min: from_real::<T>(&lo),
            lambda_max: from_real::<T>(&hi),
        };
        Ok(Self::assemble(a, degree, coeffs, Some(estimate)))
    }

    fn assemble(
        a: SparseColMatRef<'_, I, T>,
        degree: usize,
        coeffs: Coeffs<T>,
        recompute: Option<BoundEstimate>,
    ) -> Self {
        let n = a.ncols();
        let mut a_col_ptr: Vec<I> = Vec::with_capacity(n + 1);
        a_col_ptr.push(I::truncate(0));
        let mut a_row_idx: Vec<I> = Vec::new();
        let mut a_values: Vec<T> = Vec::new();
        for j in 0..n {
            let rows = a.symbolic().row_idx_of_col_raw(j);
            let vals = a.val_of_col(j);
            a_row_idx.extend_from_slice(rows);
            for v in vals {
                a_values.push(copy(v));
            }
            a_col_ptr.push(I::truncate(a_row_idx.len()));
        }
        Self {
            dim: n,
            degree,
            a_col_ptr,
            a_row_idx,
            a_values,
            coeffs,
            recompute,
        }
    }

    /// Refactorise against a new matrix with the same sparsity pattern.
    ///
    /// Updates the stored operator values in place. If the preconditioner was
    /// built via [`Poly::try_new_auto`], the spectral bounds are re-estimated
    /// from the new values; otherwise the original [`PolyKind`] is retained.
    ///
    /// # Errors
    ///
    /// [`PolyError::PatternMismatch`] if `a`'s shape or column lengths disagree
    /// with the stored pattern.
    pub fn refactorize(&mut self, a: SparseColMatRef<'_, I, T>) -> Result<(), PolyError> {
        let n = self.dim;
        if a.nrows() != n || a.ncols() != n {
            return Err(PolyError::PatternMismatch);
        }
        for j in 0..n {
            let vals = a.val_of_col(j);
            let start = self.a_col_ptr[j].zx();
            let end = self.a_col_ptr[j + 1].zx();
            if end - start != vals.len() {
                return Err(PolyError::PatternMismatch);
            }
            for (k, v) in vals.iter().enumerate() {
                self.a_values[start + k] = copy(v);
            }
        }
        if let Some(est) = self.recompute {
            let (lo, hi) = resolve_bounds::<I, T>(a, &est)?;
            self.coeffs = Coeffs::Chebyshev {
                lambda_min: from_real::<T>(&lo),
                lambda_max: from_real::<T>(&hi),
            };
        }
        Ok(())
    }

    /// View of the stored operator `A`.
    #[inline]
    pub(crate) fn a_view(&self) -> SparseColMatRef<'_, I, T> {
        let sym = unsafe {
            SymbolicSparseColMatRef::<'_, I>::new_unchecked(
                self.dim,
                self.dim,
                &self.a_col_ptr,
                None,
                &self.a_row_idx,
            )
        };
        SparseColMatRef::new(sym, &self.a_values)
    }
}
