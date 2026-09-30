"""Band structures, densities of states and band fillings.

Everything here diagonalizes the same Bloch matrices the Kubo module
uses, through the same canonically orthogonalized solver, so spectral
and optical statements about one model can never disagree about what
the eigenvalues are.
"""
from __future__ import annotations

import numpy as np

from .eigsolve import gen_eigh
from .model import KB, check_kgrid, fermi_dirac

__all__ = ["bands", "dos", "fermi_level", "band_edges", "k_path"]


def _eigs(model, kpts, thresh):
    out = []
    for k in kpts:
        H, S = model.bloch(k)
        out.append(gen_eigh(H, S, thresh=thresh))
    return np.array(out)


def bands(model, kpts, thresh=1e-10):
    """Eigenvalues along a list of Cartesian k-points; shape (nk, nao)."""
    return _eigs(model, kpts, thresh)


def dos(model, energies, mesh=None, kpts=None, weights=None, eta=0.05,
        thresh=1e-10):
    """Gaussian-broadened density of states per unit cell (states / eV).

    Either ``mesh`` (a Monkhorst-Pack grid) or explicit ``kpts`` with
    ``weights`` summing to one.  Spin degeneracy is *not* included; the
    integral over all energies equals the number of orbitals kept per
    cell.
    """
    kpts, weights = _grid(model, mesh, kpts, weights)
    rho = np.zeros_like(np.asarray(energies, dtype=float))
    for k, w in zip(kpts, weights):
        H, S = model.bloch(k)
        e = gen_eigh(H, S, thresh=thresh)
        for ei in e:
            rho += w * np.exp(-0.5 * ((energies - ei) / eta) ** 2) \
                / (eta * np.sqrt(2.0 * np.pi))
    return rho


def fermi_level(model, filling, mesh=None, kpts=None, weights=None,
                T=300.0, thresh=1e-10, tol=1e-10):
    """Chemical potential at which the mean band occupation per cell
    equals ``filling`` (states per cell, spin not included), by
    bisection on the Fermi-Dirac-weighted eigenvalue count."""
    if T < 0:
        raise ValueError("T must be >= 0 K")
    kpts, weights = _grid(model, mesh, kpts, weights)
    eigs = _eigs(model, kpts, thresh)

    def count(mu):
        f = fermi_dirac(eigs, mu, T)
        return float((f * np.asarray(weights)[:, None]).sum())

    lo, hi = eigs.min() - 5.0, eigs.max() + 5.0
    if not (count(lo) <= filling <= count(hi)):
        raise ValueError("filling outside the reachable range")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if count(mid) < filling:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def band_edges(model, mu, mesh=None, kpts=None, weights=None, thresh=1e-10):
    """(valence-band maximum, conduction-band minimum, gap) about mu."""
    # weights play no role in the band edges, so they are not checked
    kpts, _ = _grid(model, mesh, kpts, None)
    eigs = _eigs(model, kpts, thresh)
    below = eigs[eigs <= mu]
    above = eigs[eigs > mu]
    if below.size == 0 or above.size == 0:
        raise ValueError("mu lies outside the spectrum")
    vbm, cbm = float(below.max()), float(above.min())
    return vbm, cbm, cbm - vbm


def _grid(model, mesh, kpts, weights):
    if mesh is not None:
        return model.monkhorst_pack(mesh)
    if kpts is None:
        if model.cell is not None:
            raise ValueError("periodic model: give mesh or kpts")
        return [None], [1.0]
    return check_kgrid(kpts, weights)


def k_path(vertices, n_per_segment=30):
    """Piecewise-linear path through Cartesian k-space vertices.

    Returns (kpts, distances, tick_distances): the interpolated points
    (each vertex included once, endpoints inclusive), the cumulative
    path length per point, and the path length at each vertex -- the
    usual ingredients of a band-structure plot.
    """
    vertices = [np.atleast_1d(np.asarray(v, dtype=float)) for v in vertices]
    kpts = [vertices[0]]
    for a, b in zip(vertices[:-1], vertices[1:]):
        for s in range(1, int(n_per_segment) + 1):
            kpts.append(a + (b - a) * s / float(n_per_segment))
    kpts = np.array(kpts)
    seg = np.linalg.norm(np.diff(kpts, axis=0), axis=1)
    dists = np.concatenate([[0.0], np.cumsum(seg)])
    ticks = dists[::int(n_per_segment)]
    return kpts, dists, ticks
