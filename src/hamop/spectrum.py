"""Band structures, densities of states and band fillings.

Everything here diagonalizes the same Bloch matrices the Kubo module
uses, through the same canonically orthogonalized solver, so spectral
and optical statements about one model can never disagree about what
the eigenvalues are.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from .eigsolve import _eig_chunks
from .model import KB, check_kgrid, fermi_dirac

if TYPE_CHECKING:
    from .model import TightBindingModel

__all__ = ["bands", "dos", "fermi_level", "band_edges", "k_path"]


def _eig_rows(model, kpts, thresh):
    """Eigenvalues at every k-point, one 1D array per k-point.  Since
    0.13.0 the Bloch matrices are assembled and diagonalized a block of
    k-points at a time (`hamop.eigsolve._eig_chunks`), which gives the
    same eigenvalues as the former one-k-point loop to rounding (1e-12 in
    the tests) and is many times faster for small cells.  Where canonical
    orthogonalization drops overlap directions the rows can differ in
    length."""
    if not isinstance(kpts, np.ndarray):
        kpts = list(kpts)
    if len(kpts) == 0:
        return []
    out = []
    for _, e, _, _ in _eig_chunks(model, kpts, np.ones(len(kpts)),
                                  thresh=thresh):
        out.extend(e)
    return out


_RAGGED = ("the overlap matrix is nearly singular at some k-points, where "
           "canonical orthogonalization drops states (thresh = {:g}), so "
           "the number of bands changes with k and the eigenvalues do not "
           "form an (nk, nbands) array; take the k-points one at a time "
           "with gen_eigh, or use dos, fermi_level, band_edges or "
           "carrier_count, which handle a changing band count")


def _is_ragged(rows):
    return len({len(r) for r in rows}) > 1


def _eigs(model, kpts, thresh):
    """Eigenvalues at every k-point, shape (nk, nao); refused with an
    explanation when the band count changes with k (before 0.14.0 NumPy
    raised "inhomogeneous shape" here, or, with NumPy < 1.24, built an
    object array that failed later)."""
    rows = _eig_rows(model, kpts, thresh)
    if len(rows) == 0:
        return np.array([])
    if _is_ragged(rows):
        raise ValueError(_RAGGED.format(thresh))
    return np.array(rows)


def _flat_states(rows, weights):
    """All kept states as one array, with the weight of the k-point each
    belongs to."""
    w = np.asarray(weights, dtype=float).ravel()
    ev = np.concatenate([np.asarray(r, dtype=float) for r in rows])
    wv = np.repeat(w, [len(r) for r in rows])
    return ev, wv


def bands(model: TightBindingModel, kpts: Iterable,
          thresh: float = 1e-10) -> np.ndarray:
    """Eigenvalues along a list of Cartesian k-points; shape (nk, nao).

    With overlap, canonical orthogonalization can drop states where S(k)
    is nearly singular (see `gen_eigh`); if that changes the number of
    bands along the list, the bands cannot form one array and the call
    is refused with an explanation (since 0.14.0; before, NumPy raised
    an "inhomogeneous shape" error)."""
    return _eigs(model, kpts, thresh)


def dos(model: TightBindingModel, energies: ArrayLike,
        mesh: Optional[Union[int, Sequence[int]]] = None,
        kpts: Optional[ArrayLike] = None,
        weights: Optional[ArrayLike] = None, eta: float = 0.05,
        thresh: float = 1e-10) -> np.ndarray:
    """Gaussian-broadened density of states per unit cell (states / eV).

    Either ``mesh`` (a Monkhorst-Pack grid) or explicit ``kpts`` with
    ``weights`` summing to one.  Spin degeneracy is *not* included; the
    integral over all energies equals the number of orbitals kept per
    cell.
    """
    if not eta > 0:
        raise ValueError("eta (the Gaussian width, eV) must be > 0")
    kpts, weights = _grid(model, mesh, kpts, weights)
    E = np.asarray(energies, dtype=float)
    Ef = E.reshape(-1)
    rho = np.zeros(Ef.shape)
    piece = max(1, (1 << 22) // max(1, Ef.size))
    norm = 1.0 / (eta * np.sqrt(2.0 * np.pi))
    for w, e, _, _ in _eig_chunks(model, kpts, weights, thresh=thresh):
        ev = e.ravel()
        wv = np.repeat(w, e.shape[1])
        for s in range(0, ev.size, piece):
            x = (Ef[:, None] - ev[None, s:s + piece]) / eta
            rho += np.exp(-0.5 * x * x) @ wv[s:s + piece]
    return (rho * norm).reshape(E.shape)


def fermi_level(model: TightBindingModel, filling: float,
                mesh: Optional[Union[int, Sequence[int]]] = None,
                kpts: Optional[ArrayLike] = None,
                weights: Optional[ArrayLike] = None, T: float = 300.0,
                thresh: float = 1e-10, tol: float = 1e-10) -> float:
    """Chemical potential at which the mean band occupation per cell
    equals ``filling`` (states per cell, spin not included), by
    bisection on the Fermi-Dirac-weighted eigenvalue count."""
    if T < 0:
        raise ValueError("T must be >= 0 K")
    kpts, weights = _grid(model, mesh, kpts, weights)
    rows = _eig_rows(model, kpts, thresh)
    if not _is_ragged(rows):
        eigs = np.array(rows)

        def count(mu):
            f = fermi_dirac(eigs, mu, T)
            return float((f * np.asarray(weights)[:, None]).sum())
    else:
        # the band count changes with k (states dropped by canonical
        # orthogonalization); before 0.14.0 this raised a NumPy error
        eigs, wv = _flat_states(rows, weights)

        def count(mu):
            return float((fermi_dirac(eigs, mu, T) * wv).sum())

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


def band_edges(model: TightBindingModel, mu: float,
               mesh: Optional[Union[int, Sequence[int]]] = None,
               kpts: Optional[ArrayLike] = None,
               weights: Optional[ArrayLike] = None,
               thresh: float = 1e-10) -> Tuple[float, float, float]:
    """(valence-band maximum, conduction-band minimum, gap) about mu."""
    # weights play no role in the band edges, so they are not checked
    kpts, _ = _grid(model, mesh, kpts, None)
    rows = _eig_rows(model, kpts, thresh)
    eigs = (np.array(rows) if not _is_ragged(rows)
            else np.concatenate(rows))      # band count changes with k
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


def k_path(vertices: Sequence[ArrayLike], n_per_segment: int = 30
           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Piecewise-linear path through Cartesian k-space vertices.

    Returns (kpts, distances, tick_distances): the interpolated points
    (each vertex included once, endpoints inclusive), the cumulative
    path length per point, and the path length at each vertex -- the
    usual ingredients of a band-structure plot.
    """
    vertices = [np.atleast_1d(np.asarray(v, dtype=float)) for v in vertices]
    if len(vertices) < 2 or int(n_per_segment) < 1:
        raise ValueError("k_path needs at least two vertices and "
                         "n_per_segment >= 1")
    kpts = [vertices[0]]
    for a, b in zip(vertices[:-1], vertices[1:]):
        for s in range(1, int(n_per_segment) + 1):
            kpts.append(a + (b - a) * s / float(n_per_segment))
    kpts = np.array(kpts)
    seg = np.linalg.norm(np.diff(kpts, axis=0), axis=1)
    dists = np.concatenate([[0.0], np.cumsum(seg)])
    ticks = dists[::int(n_per_segment)]
    return kpts, dists, ticks
