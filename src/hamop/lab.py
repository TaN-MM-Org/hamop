"""Fit a tight-binding model to measured band energies -- and plan
the measurement first.

hamop deliberately ships no material constants; the numbers of YOUR
model come from your calculation or your measurement. This module
closes the measurement route: give it a builder function that turns a
parameter vector into a `TightBindingModel`, plus measured band
energies at known k-points (an ARPES map, a set of extracted band
positions), and it fits the parameters with error bars -- or, before
the beam time, predicts those error bars and picks the most
informative k-points.

The three tools, in the order a lab works:

1. `band_information`: would these k-points and bands determine the
   parameters, and how well? Computed before any data exist.
2. `design_kpoints`: pick the most informative subset of candidate
   k-points (D-optimal design; F. Pukelsheim, Optimal Design of
   Experiments, SIAM (2006)).
3. `fit_bands`: the weighted least-squares fit, with the standard
   (J^T W J)^-1 covariance (the Fisher information of independent
   Gaussian measurements; any statistics text, under "Cramer-Rao
   bound") and a chi-squared when measurement errors are supplied.

A design that cannot tell the parameters apart -- a single k-point
for a hopping and an on-site energy, say -- is refused with an
explanation, never silently pseudo-inverted. Units are the model's
own: energies in whatever unit your builder uses (eV throughout the
shipped lattices), k Cartesian in 1/Angstrom.

Two further tools (new in 0.12.0) carry a fit and a model to the
numbers a lab reports:

4. `propagate_uncertainty`: the error bar of anything computed from
   the fitted parameters (a band energy at another k, a band width, a
   gap), by linear propagation through the fit covariance, with a
   check that the quantity is linear enough over one standard
   deviation for that to be trusted, or by Monte Carlo sampling of the
   covariance. `BandFit.p_value` says whether the measured energies
   are consistent with the model at all.
5. `sheet_optics` and `sheet_conductivity_from_transmittance`: the
   normal-incidence transmittance, reflectance and absorptance of a 2D
   sheet whose sheet conductivity the Kubo functions give, and the
   inverse, Re sigma from a measured transmittance relative to the
   bare substrate.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares
from scipy.stats import chi2 as _chi2_dist

from .model import ALPHA
from .spectrum import bands

__all__ = ["BandFit", "fit_bands", "band_information",
           "design_kpoints", "propagate_uncertainty", "sheet_optics",
           "sheet_conductivity_from_transmittance"]

_COND_MAX = 1e10
_SINGULAR_MSG = ("these (k-point, band) observations cannot tell the "
                 "parameters apart (singular or near-singular "
                 "information matrix); measure bands or k-points that "
                 "respond differently to each parameter -- "
                 "`band_information` shows which designs work")


def _check(build, theta0, kpts, band_index, sigmas, values=None):
    if not callable(build):
        raise ValueError("build must be a callable theta -> "
                         "TightBindingModel")
    x0 = np.asarray(theta0, dtype=float).ravel()
    if x0.size < 1 or not np.all(np.isfinite(x0)):
        raise ValueError("theta0 must be a finite parameter vector")
    k = np.atleast_2d(np.asarray(kpts, dtype=float))
    if not np.all(np.isfinite(k)):
        raise ValueError("k-points must be finite")
    n = k.shape[0]
    bidx = np.broadcast_to(np.asarray(band_index, dtype=int),
                           (n,)).copy()
    if np.any(bidx < 0):
        raise ValueError("band indices must be >= 0 (ascending-sorted "
                         "eigenvalues)")
    sig = None
    if sigmas is not None:
        sig = np.broadcast_to(np.asarray(sigmas, dtype=float),
                              (n,)).copy()
        if np.any(sig <= 0.0) or not np.all(np.isfinite(sig)):
            raise ValueError("sigmas must be finite and positive")
    if values is not None:
        values = np.asarray(values, dtype=float).ravel()
        if values.shape != (n,) or not np.all(np.isfinite(values)):
            raise ValueError("need one finite measured energy per "
                             "(k-point, band) observation")
    return x0, k, bidx, sig, values


def _model(build, x, k, bidx):
    e = bands(build(x), k)
    if np.any(bidx >= e.shape[1]):
        raise ValueError(f"band index out of range: the model has "
                         f"{e.shape[1]} bands")
    return e[np.arange(k.shape[0]), bidx]


def _jacobian(build, x, k, bidx):
    """Central-difference d(energy)/d(parameter)."""
    jac = np.empty((k.shape[0], x.size))
    for j in range(x.size):
        h = 1e-6 * max(abs(x[j]), 1e-3)
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        jac[:, j] = (_model(build, xp, k, bidx)
                     - _model(build, xm, k, bidx)) / (2.0 * h)
    return jac


def _invert_information(fisher):
    """Scale-invariant inversion (parameters can mix energy and
    dimensionless units): identifiability is judged on the
    correlation-scaled matrix, so exact functional degeneracies
    survive the scaling and unit mismatches do not."""
    d = np.sqrt(np.diag(fisher))
    if np.any(d <= 0.0) or not np.all(np.isfinite(d)):
        return False, np.inf, None
    fs = fisher / np.outer(d, d)
    sv = np.linalg.svd(fs, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
    if not (np.isfinite(cond) and cond <= _COND_MAX):
        return False, cond, None
    cov = np.linalg.inv(fs) / np.outer(d, d)
    return True, cond, cov


@dataclasses.dataclass
class BandFit:
    """Result of `fit_bands`.

    theta : fitted parameter vector, in the order of `theta0`.
    sigma : per-parameter 1-sigma uncertainties, same order.
    cov : full parameter covariance.
    chi2, chi2_dof : goodness of fit (chi2 is None when no measurement
        errors were given; the error bars are then scaled from the
        residual scatter instead).
    condition_number : of the unit-free information matrix; large
        means barely identifiable.
    p_value : (property, new in 0.12.0) probability of a chi-squared
        at least this large if the model is right and the stated
        errors are Gaussian; None without measurement errors.
    """

    theta: np.ndarray
    sigma: np.ndarray
    cov: np.ndarray
    chi2: Optional[float]
    chi2_dof: Optional[int]
    n_points: int
    condition_number: float

    @property
    def p_value(self) -> Optional[float]:
        """Goodness of fit: P(chi2' >= chi2) for chi2' drawn from the
        chi-squared distribution with ``chi2_dof`` degrees of freedom.

        A very small value (say below 0.01) means the measured energies
        scatter about the fitted bands more than the stated errors
        allow: the errors are underestimated, or the model (its form,
        not just its parameters) does not describe the data.  A value
        near 1 can mean overestimated errors.  None when the fit had no
        measurement errors (then there is no absolute chi-squared) or
        no spare degrees of freedom."""
        if self.chi2 is None or not self.chi2_dof:
            return None
        return float(_chi2_dist.sf(self.chi2, self.chi2_dof))


def fit_bands(build: Callable[[np.ndarray], Any], theta0: ArrayLike,
              kpts: ArrayLike, energies: ArrayLike,
              band_index: Union[int, ArrayLike] = 0,
              sigmas: Optional[ArrayLike] = None,
              **ls_kwargs: Any) -> BandFit:
    """Fit model parameters to measured band energies.

    build : callable theta -> `TightBindingModel` (your
        parameterization: hoppings, on-site energies, SOC strengths --
        anything the builder exposes).
    theta0 : starting parameter vector.
    kpts : (n, dim) Cartesian k-points of the measurements (1/Angstrom).
    energies : (n,) measured band energies (the model's energy unit).
    band_index : which ascending-sorted band each measurement belongs
        to (scalar or (n,)).
    sigmas : optional measurement errors (scalar or (n,)). With them
        the covariance is exact for the stated errors and a
        chi-squared is reported; without them the error bars are
        scaled from the residual scatter, which needs at least one
        spare observation.

    Returns a `BandFit`. Refuses non-identifiable designs and too few
    points, with an explanation.
    """
    x0, k, bidx, sig, y = _check(build, theta0, kpts, band_index,
                                 sigmas, energies)
    n, p = k.shape[0], x0.size
    if sig is None and n < p + 1:
        raise ValueError(f"{n} observations cannot determine {p} "
                         "parameters and an error scale; add points "
                         "or supply sigmas")
    if n < p:
        raise ValueError(f"{n} observations cannot determine {p} "
                         "parameters")
    w = np.ones(n) if sig is None else 1.0 / sig

    def resid(x):
        return w * (_model(build, x, k, bidx) - y)

    res = least_squares(resid, x0, jac="3-point", xtol=1e-14,
                        ftol=1e-14, gtol=1e-14, **ls_kwargs)
    if not res.success:
        raise RuntimeError(f"band fit failed: {res.message}")
    jac = _jacobian(build, res.x, k, bidx) * w[:, None]
    fisher = jac.T @ jac
    ok, cond, cov = _invert_information(fisher)
    if not ok:
        raise ValueError(_SINGULAR_MSG)
    rss = float(2.0 * res.cost)
    if sig is None:
        cov = cov * (rss / (n - p))
        chi2 = chi2_dof = None
    else:
        chi2, chi2_dof = rss, n - p
    return BandFit(theta=res.x.copy(), sigma=np.sqrt(np.diag(cov)),
                   cov=cov, chi2=chi2, chi2_dof=chi2_dof, n_points=n,
                   condition_number=cond)


def band_information(build: Callable[[np.ndarray], Any], theta: ArrayLike,
                     kpts: ArrayLike, band_index: Union[int, ArrayLike] = 0,
                     sigmas: Optional[ArrayLike] = None) -> Dict[str, Any]:
    """Would this measurement determine the parameters? Ask first.

    The information matrix of the planned (k-point, band)
    observations around `theta`, and -- when it is invertible -- the
    error bars the fit would deliver. Without `sigmas` the answer is
    per unit measurement error, and real error bars scale linearly
    with your sigma.

    Returns dict(fisher, identifiable, condition_number, sigma).
    """
    x0, k, bidx, sig, _ = _check(build, theta, kpts, band_index,
                                 sigmas)
    w = np.ones(k.shape[0]) if sig is None else 1.0 / sig
    jac = _jacobian(build, x0, k, bidx) * w[:, None]
    fisher = jac.T @ jac
    ok, cond, cov = _invert_information(fisher)
    identifiable = bool(ok and k.shape[0] >= x0.size)
    sigma = np.sqrt(np.diag(cov)) if identifiable else None
    return {"fisher": fisher, "identifiable": identifiable,
            "condition_number": cond, "sigma": sigma}


def design_kpoints(build: Callable[[np.ndarray], Any], theta: ArrayLike,
                   candidates: ArrayLike, n_pick: int,
                   band_index: Union[int, ArrayLike] = 0,
                   sigmas: Optional[ArrayLike] = None) -> Dict[str, Any]:
    """Pick the most informative k-points to measure.

    Greedy D-optimal selection from the candidate k-points: each pick
    most increases the determinant of the information matrix.
    Transparent and monotone, but a good-practice heuristic, not a
    proof of the globally best subset.

    Returns dict(indices, fisher, condition_number, sigma) with the
    chosen candidate indices in pick order. Refuses when even the full
    candidate list cannot identify the parameters.
    """
    x0, k, bidx, sig, _ = _check(build, theta, candidates, band_index,
                                 sigmas)
    n, p = k.shape[0], x0.size
    n_pick = int(n_pick)
    if not p <= n_pick <= n:
        raise ValueError(f"n_pick must be between {p} (the number of "
                         f"parameters) and {n} (the number of "
                         "candidates)")
    full = band_information(build, x0, k, bidx, sig)
    if not full["identifiable"]:
        raise ValueError("even the full candidate list is not "
                         "identifiable: " + _SINGULAR_MSG)
    w = np.ones(n) if sig is None else 1.0 / sig
    rows = _jacobian(build, x0, k, bidx) * w[:, None]
    # column-scaled (unit-free) greedy; identical scaling leaves the
    # choices unchanged but keeps the start-up regularizer meaningful
    scale = np.sqrt(np.mean(rows * rows, axis=0))
    scale[scale == 0.0] = 1.0
    rs = rows / scale
    eps = 1e-12 * float(np.max(np.sum(rs * rs, axis=1)))
    fs = eps * np.eye(p)
    chosen = []
    for _ in range(n_pick):
        best_j, best_det = -1, -np.inf
        for j in range(n):
            if j in chosen:
                continue
            det = float(np.linalg.slogdet(fs + np.outer(rs[j],
                                                        rs[j]))[1])
            if det > best_det:
                best_j, best_det = j, det
        fs = fs + np.outer(rs[best_j], rs[best_j])
        chosen.append(best_j)
    fisher = (fs - eps * np.eye(p)) * np.outer(scale, scale)
    ok, cond, cov = _invert_information(fisher)
    sigma = np.sqrt(np.diag(cov)) if ok else None
    return {"indices": list(chosen), "fisher": fisher,
            "condition_number": cond, "sigma": sigma}


# ---------------- error bars of derived quantities (new in 0.12.0) ----

def _eval_quantity(quantity, theta):
    q = np.asarray(quantity(np.array(theta, dtype=float)), dtype=float)
    if not np.all(np.isfinite(q)):
        raise ValueError("quantity(theta) returned a non-finite value at "
                         f"theta = {np.asarray(theta).tolist()}")
    return q


def propagate_uncertainty(fit: BandFit, quantity: Callable[[np.ndarray],
                                                           Any],
                          method: str = "linear", n_samples: int = 4000,
                          seed: int = 0, nonlinearity_tol: float = 0.1
                          ) -> Dict[str, Any]:
    """Error bar of a quantity computed from fitted parameters.

    fit : the `BandFit` returned by `fit_bands` (its ``theta`` and
        ``cov`` are used).
    quantity : callable theta -> float or array, e.g. a band energy at
        a k-point you did not measure, a band width, a gap or an
        optical peak position computed from ``build(theta)``.
    method : "linear" (default) or "monte_carlo".

    "linear" is first-order (delta-method) propagation,
    cov_q = J cov J^T with J = dq/dtheta from central differences
    along the principal axes of ``cov``.  It is exact for a quantity
    linear in the parameters.  It is only trusted when it is close to
    exact: q is evaluated one standard deviation out along every
    principal axis of ``cov``, and if it deviates from its linear
    prediction there by more than ``nonlinearity_tol`` times the
    propagated error bar (plus a 1e-9 relative floor for round-off)
    the call is refused, because the linear error bar would then be
    wrong (a quantity at an extremum, or one crossing a band
    degeneracy where it has a kink).  Use method="monte_carlo" there.

    "monte_carlo" draws ``n_samples`` (at least 100) parameter vectors
    from the Gaussian N(theta, cov), with a seeded generator, and
    reports the spread of q over them, together with its 15.87th and
    84.13th percentiles (the +/- 1 sigma points of a Gaussian), which
    stay meaningful when q is not linear.  Its own statistical
    precision is about 1/sqrt(2 n_samples) relative on the error bar.

    Both methods assume, like the fit's covariance itself, Gaussian
    measurement errors and a model that describes the data
    (check ``fit.p_value``).

    Returns dict(value, sigma, cov, method) and, for "linear", the
    jacobian (n_q, n_par); for "monte_carlo", mean and interval
    (lower, upper).  ``value`` is q at the best fit.  value, sigma and
    mean are floats for a scalar quantity and arrays otherwise; cov is
    always a 2D array.
    """
    if not isinstance(fit, BandFit):
        raise ValueError("fit must be the BandFit returned by fit_bands")
    if not callable(quantity):
        raise ValueError("quantity must be a callable theta -> number "
                         "or array")
    theta = np.asarray(fit.theta, dtype=float)
    cov = np.asarray(fit.cov, dtype=float)
    p = theta.size
    if cov.shape != (p, p) or not np.all(np.isfinite(cov)):
        raise ValueError("fit.cov must be a finite (p, p) matrix")
    lam, V = np.linalg.eigh(0.5 * (cov + cov.T))
    if lam.min() < -1e-12 * max(lam.max(), 0.0):
        raise ValueError("fit.cov is not positive semidefinite")
    U = V * np.sqrt(np.clip(lam, 0.0, None))      # 1-sigma principal axes
    q0 = _eval_quantity(quantity, theta)
    scalar = q0.ndim == 0
    q0f = q0.ravel()

    def shape(x):
        return float(x[0]) if scalar else x.reshape(q0.shape)

    if method == "linear":
        eps = 1e-3
        JU = np.empty((q0f.size, p))               # dq/d(axis), per sigma
        for i in range(p):
            qp = _eval_quantity(quantity, theta + eps * U[:, i]).ravel()
            qm = _eval_quantity(quantity, theta - eps * U[:, i]).ravel()
            JU[:, i] = (qp - qm) / (2.0 * eps)
        cov_q = JU @ JU.T
        sig_q = np.sqrt(np.clip(np.diag(cov_q), 0.0, None))
        floor = 1e-9 * np.maximum(np.abs(q0f), 1.0)
        for i in range(p):
            for sgn in (1.0, -1.0):
                q1 = _eval_quantity(quantity, theta + sgn * U[:, i]).ravel()
                dev = np.abs(q1 - (q0f + sgn * JU[:, i]))
                bad = dev > nonlinearity_tol * sig_q + floor
                if np.any(bad):
                    c = int(np.argmax(bad))
                    raise ValueError(
                        "the quantity is not linear enough over one "
                        "standard deviation of the parameters: along a "
                        "principal axis of fit.cov it moves "
                        f"{dev[c]:.3g} away from its linear prediction, "
                        f"against a propagated error bar of {sig_q[c]:.3g}"
                        f" (nonlinearity_tol = {nonlinearity_tol:g}); "
                        "the linear error bar would be wrong here, use "
                        'method="monte_carlo"')
        safe = np.where(lam > 0, np.sqrt(np.clip(lam, 0.0, None)), np.inf)
        jac = (JU / safe) @ V.T                    # dq/dtheta
        return {"value": shape(q0f), "sigma": shape(sig_q), "cov": cov_q,
                "jacobian": jac, "method": "linear"}
    if method == "monte_carlo":
        n_samples = int(n_samples)
        if n_samples < 100:
            raise ValueError("n_samples must be at least 100 for a "
                             "meaningful spread")
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n_samples, p))
        qs = np.array([_eval_quantity(quantity, theta + U @ zi).ravel()
                       for zi in z])
        cov_q = np.atleast_2d(np.cov(qs, rowvar=False, ddof=1))
        sig_q = np.sqrt(np.diag(cov_q))
        lo, hi = np.percentile(qs, [15.865525393, 84.134474607], axis=0)
        return {"value": shape(q0f), "sigma": shape(sig_q), "cov": cov_q,
                "mean": shape(qs.mean(axis=0)),
                "interval": (shape(lo), shape(hi)),
                "method": "monte_carlo"}
    raise ValueError('method must be "linear" or "monte_carlo"')


# ---------------- a 2D sheet in an optical measurement (new in 0.12.0) --

def _check_index(n, name):
    if np.iscomplexobj(n) or np.ndim(n) != 0:
        raise ValueError(f"{name} must be one real refractive index (a "
                         "transparent medium); absorbing media are not "
                         "handled")
    n = float(n)
    if not (np.isfinite(n) and n > 0.0):
        raise ValueError(f"{name} must be a finite refractive index > 0")
    return n


def _unwrap(x, scalar):
    return float(x) if scalar else x


def sheet_optics(sigma: ArrayLike, n_in: float = 1.0, n_out: float = 1.0
                 ) -> Dict[str, Any]:
    """Transmittance, reflectance and absorptance of a conducting 2D
    sheet at normal incidence.

    sigma : sheet conductivity in units of e^2/(4 hbar), the unit of
        :func:`hamop.sigma_optical` and :func:`hamop.sigma_tensor` for a
        model periodic in two directions (real or complex; the
        imaginary part enters at second order, see below).
    n_in, n_out : real refractive indices of the transparent media on
        the incident side and behind the sheet (1 for vacuum or air; the
        substrate index for a sheet on a thick substrate, light incident
        from the air side).

    The sheet carries the surface current sigma E, so the tangential
    magnetic field jumps across it.  With the reduced conductivity
    s = Z0 sigma = pi alpha sigma[e^2/(4 hbar)] (Z0 the impedance of free
    space, alpha the fine-structure constant; e^2/(4 hbar) times Z0 is
    exactly pi alpha), the amplitude coefficients are
    t = 2 n_in / (n_in + n_out + s) and r = t - 1, which give

        T = 4 n_in n_out / |n_in + n_out + s|^2,
        R = |n_in - n_out - s|^2 / |n_in + n_out + s|^2,
        A = 4 n_in Re(s) / |n_in + n_out + s|^2   (Joule heating),

    with T + R + A = 1, and the transmittance relative to the same
    interface without the sheet, T_rel = (n_in + n_out)^2 /
    |n_in + n_out + s|^2, which is what a "sample over bare substrate"
    ratio measures.  For free-standing graphene on its plateau
    (sigma = 1) this gives T = (1 + pi alpha / 2)^-2 = 0.9774, the
    absorption of about pi alpha = 2.3 % observed by R. R. Nair et al.,
    Science 320, 1308 (2008).  The tests check the formulas against an
    independent thin-film (Airy) calculation in the limit of zero film
    thickness.

    Passing the real part only (``sigma_optical``) neglects Im sigma;
    the relative error in T is then about (pi alpha Im sigma)^2 /
    (n_in + n_out)^2, below 1.4e-4 for |Im sigma| <= 1 when n_in and
    n_out are at least 1.  A complex
    ``sigma_tensor(..., directions=(0, 0))`` includes it.

    Limits: normal incidence and a sheet much thinner than the
    wavelength; the back face of a substrate and multiple reflections
    inside it are not included (they largely cancel in T_rel); no
    absorbing media; Re sigma < 0 (gain) is refused.

    Returns dict(T, R, A, T_rel), floats for a scalar sigma, arrays
    otherwise.
    """
    n1 = _check_index(n_in, "n_in")
    n2 = _check_index(n_out, "n_out")
    scalar = np.ndim(sigma) == 0
    s = np.asarray(sigma, dtype=complex)
    if not np.all(np.isfinite(s)):
        raise ValueError("sigma must be finite")
    if np.any(s.real < 0.0):
        raise ValueError("Re sigma < 0 describes a sheet with gain; a "
                         "passive sheet has Re sigma >= 0")
    st = np.pi * ALPHA * s
    den2 = np.abs(n1 + n2 + st) ** 2
    out = {"T": 4.0 * n1 * n2 / den2,
           "R": np.abs(n1 - n2 - st) ** 2 / den2,
           "A": 4.0 * n1 * st.real / den2,
           "T_rel": (n1 + n2) ** 2 / den2}
    return {key: _unwrap(val, scalar) for key, val in out.items()}


def sheet_conductivity_from_transmittance(
        T_rel: ArrayLike, n_in: float = 1.0, n_out: float = 1.0,
        T_rel_err: Optional[ArrayLike] = None, sigma_imag: ArrayLike = 0.0
        ) -> Dict[str, Any]:
    """Re sigma of a 2D sheet from its measured relative transmittance.

    T_rel : transmittance of the sample divided by that of the same
        substrate (or of free space) without the sheet, at normal
        incidence: the ratio that :func:`sheet_optics` calls T_rel.
    n_in, n_out : as in :func:`sheet_optics`.
    T_rel_err : optional 1-sigma error of T_rel (same shape or scalar),
        propagated to an error bar on sigma.
    sigma_imag : assumed Im sigma in units of e^2/(4 hbar) (default 0);
        a transmittance alone cannot separate the real and imaginary
        parts, and Im sigma enters only at second order.

    Inverts T_rel = N^2 / ((N + pi alpha Re sigma)^2
    + (pi alpha Im sigma)^2), N = n_in + n_out:

        Re sigma = (sqrt(N^2 / T_rel - (pi alpha Im sigma)^2) - N)
                   / (pi alpha),

    in units of e^2/(4 hbar), directly comparable with
    :func:`hamop.sigma_optical` of a 2D model.  The error bar is
    d(Re sigma)/d(T_rel) times T_rel_err (first order; the tests check
    it by Monte Carlo).

    A T_rel above the largest value a passive sheet allows,
    N^2 / (N^2 + (pi alpha Im sigma)^2) (1 for Im sigma = 0), would
    mean Re sigma < 0.  It is refused, unless T_rel_err is given and
    T_rel lies within 3 T_rel_err of that bound: then the (negative)
    estimate is returned, as measurement noise can produce it.

    Returns dict(sigma, sigma_err) (sigma_err None without T_rel_err),
    floats for scalar input, arrays otherwise.
    """
    n1 = _check_index(n_in, "n_in")
    n2 = _check_index(n_out, "n_out")
    scalar = np.ndim(T_rel) == 0
    tr = np.asarray(T_rel, dtype=float)
    if not (np.all(np.isfinite(tr)) and np.all(tr > 0.0)):
        raise ValueError("T_rel must be finite and > 0")
    si = np.pi * ALPHA * np.asarray(sigma_imag, dtype=float)
    if not np.all(np.isfinite(si)):
        raise ValueError("sigma_imag must be finite")
    N = n1 + n2
    bound = N ** 2 / (N ** 2 + si ** 2)
    err = None
    if T_rel_err is not None:
        err = np.asarray(T_rel_err, dtype=float)
        if not (np.all(np.isfinite(err)) and np.all(err >= 0.0)):
            raise ValueError("T_rel_err must be finite and >= 0")
    allowed = bound if err is None else bound + 3.0 * err
    if np.any(tr > allowed):
        raise ValueError(
            "T_rel exceeds the largest transmittance a passive sheet can "
            "have (1 for Im sigma = 0), "
            + ("and no T_rel_err was given to attribute it to noise"
               if err is None else "by more than 3 T_rel_err")
            + "; check the reference (bare substrate) measurement")
    x = N ** 2 / tr - si ** 2
    if np.any(x <= 0.0):
        raise ValueError("no real conductivity reproduces this T_rel "
                         "with the given sigma_imag")
    root = np.sqrt(x)
    sig = (root - N) / (np.pi * ALPHA)
    sig_err = None
    if err is not None:
        sig_err = (N ** 2 / tr ** 2) / (2.0 * root) * err / (np.pi * ALPHA)
        sig_err = _unwrap(sig_err, scalar and np.ndim(T_rel_err) == 0)
    return {"sigma": _unwrap(sig, scalar and np.ndim(sigma_imag) == 0),
            "sigma_err": sig_err}
