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
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy.optimize import least_squares

from .spectrum import bands

__all__ = ["BandFit", "fit_bands", "band_information",
           "design_kpoints"]

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
    """

    theta: np.ndarray
    sigma: np.ndarray
    cov: np.ndarray
    chi2: float
    chi2_dof: int
    n_points: int
    condition_number: float


def fit_bands(build, theta0, kpts, energies, band_index=0, sigmas=None,
              **ls_kwargs):
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


def band_information(build, theta, kpts, band_index=0, sigmas=None):
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


def design_kpoints(build, theta, candidates, n_pick, band_index=0,
                   sigmas=None):
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
