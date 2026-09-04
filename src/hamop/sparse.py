"""Sparse assembly and large-system solvers.

Everything dense in the package has a small-matrix bias; this module is
the large-system path.  ``bloch_sparse`` assembles exactly the same
Bloch matrices as :meth:`TightBindingModel.bloch`, as CSR sparse
matrices (the test suite asserts element-for-element agreement with the
dense assembly).  ``lowest_bands`` diagonalizes only the low-energy
window with the Lanczos solver ``scipy.sparse.linalg.eigsh``,
generalized eigenproblem included.  ``kpm_dos`` is the kernel
polynomial method for the density of states -- Chebyshev moments with
the Jackson kernel, following A. Weisse, G. Wellein, A. Alvermann and
H. Fehske, Rev. Mod. Phys. 78, 275 (2006) -- with either an exact
(deterministic) trace or the standard stochastic estimator.

Deliberate scope: sparse spectra and DOS only.  The optics, topology
and transport modules remain dense; a sparse Kubo or NEGF path is not
implemented, and saying so plainly beats pretending otherwise.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

__all__ = ["bloch_sparse", "lowest_bands", "kpm_dos"]


def bloch_sparse(model, k=None):
    """H(k), S(k) as CSR sparse matrices; S is None for an orthogonal
    model (identity implied).

    Same assembly, phases and Hermitization as the dense
    :meth:`TightBindingModel.bloch`; agreement is asserted in the
    tests to machine precision.
    """
    k = model._kvec(k)
    n = model.nao
    overlap = model.has_overlap()
    rows, cols, hv, sv = [], [], [], []
    for oi, oj, d, Hb, Sb in model._terms():
        ph = np.exp(1j * float(k @ d))
        ni, nj = Hb.shape
        r, c = np.meshgrid(np.arange(oi, oi + ni),
                           np.arange(oj, oj + nj), indexing="ij")
        rows.append(r.ravel())
        cols.append(c.ravel())
        hv.append((ph * Hb).ravel())
        sv.append((ph * Sb).ravel())
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    H = coo_matrix((np.concatenate(hv), (rows, cols)),
                   shape=(n, n)).tocsr()
    H = 0.5 * (H + H.conj().T)
    if not overlap:
        return H, None
    S = coo_matrix((np.concatenate(sv), (rows, cols)),
                   shape=(n, n)).tocsr()
    S = 0.5 * (S + S.conj().T)
    return H, S


def lowest_bands(model, kpts, n_bands, sigma=None, tol=0.0):
    """The ``n_bands`` lowest eigenvalues at each k, by sparse Lanczos.

    kpts: list of Cartesian k-points ([None] for a finite system).
    sigma: optional shift for shift-invert mode (fastest when the
    states of interest sit near a known energy, e.g. a band edge);
    without it the smallest-algebraic mode is used.  Nonorthogonal
    models solve the generalized problem H c = e S c directly.
    Returns an (nk, n_bands) array, each row sorted ascending.

    This is the large-system counterpart of :func:`hamop.bands`; the
    tests pin it to the dense solver and to the closed-form spectrum
    2 t cos(pi j / (N + 1)) of the open chain.
    """
    n_bands = int(n_bands)
    if not 0 < n_bands < model.nao:
        raise ValueError("need 0 < n_bands < number of orbitals "
                         "(Lanczos cannot return the full spectrum)")
    out = np.empty((len(kpts), n_bands))
    for ik, k in enumerate(kpts):
        H, S = bloch_sparse(model, k)
        if sigma is None:
            e = eigsh(H, k=n_bands, M=S, which="SA", tol=tol,
                      return_eigenvectors=False)
        else:
            e = eigsh(H, k=n_bands, M=S, sigma=sigma, which="LM", tol=tol,
                      return_eigenvectors=False)
        out[ik] = np.sort(np.real(e))
    return out


def _jackson(M):
    """Jackson kernel damping factors g_0 .. g_{M-1} (Weisse et al.,
    Rev. Mod. Phys. 78, 275 (2006), Eq. (71))."""
    m = np.arange(M)
    q = np.pi / (M + 1.0)
    return ((M - m + 1.0) * np.cos(q * m)
            + np.sin(q * m) / np.tan(q)) / (M + 1.0)


def kpm_dos(model, energies, n_moments=200, n_random=None, seed=0,
            bounds=None, margin=0.01):
    """Kernel-polynomial density of states of a large *finite* model
    (states / eV, integrating to the total orbital count).

    Chebyshev moments of the spectral density with Jackson-kernel
    damping (Weisse et al., Rev. Mod. Phys. 78, 275 (2006)).
    n_random=None computes the exact trace by running the recurrence on
    the full identity block (deterministic, affordable up to ~10^4
    orbitals); an integer uses that many random +/-1 vectors -- the
    standard stochastic trace estimator, whose relative error shrinks
    as 1/sqrt(n_random * nao).

    bounds: optional (E_min, E_max) enclosing the whole spectrum; by
    default the extreme eigenvalues are found with two Lanczos runs and
    widened by ``margin`` of the span (moments diverge if the true
    spectrum leaks outside the scaling window).

    The energy resolution is roughly pi * span / (2 n_moments); finer
    structure than that is smoothed, exactly like a broadened
    :func:`hamop.dos`.  Orthogonal bases and finite models only, both
    refused explicitly otherwise; for a periodic model, build the
    corresponding supercell.
    """
    if model.cell is not None:
        raise ValueError("kpm_dos works on finite models; build a "
                        "supercell of a periodic model first")
    if model.has_overlap():
        raise ValueError("kpm_dos supports orthogonal bases only "
                        "(overlap would need S^-1 applications per "
                        "moment; not implemented)")
    energies = np.asarray(energies, dtype=float)
    H, _ = bloch_sparse(model, None)
    H = H.real.tocsr() if np.abs(H.imag).max() == 0.0 else H
    n = model.nao
    M = int(n_moments)
    if bounds is None:
        e_lo = eigsh(H, k=1, which="SA", return_eigenvectors=False,
                     tol=1e-6)[0]
        e_hi = eigsh(H, k=1, which="LA", return_eigenvectors=False,
                     tol=1e-6)[0]
    else:
        e_lo, e_hi = bounds
    span = float(e_hi - e_lo)
    a = 0.5 * span * (1.0 + margin)
    b = 0.5 * float(e_hi + e_lo)
    if n_random is None:
        V0 = np.eye(n, dtype=H.dtype)
        norm = 1.0
    else:
        rng = np.random.default_rng(seed)
        V0 = rng.choice([-1.0, 1.0], size=(n, int(n_random)))
        V0 = V0.astype(H.dtype)
        norm = 1.0 / int(n_random)
    mu = np.zeros(M)
    v_prev = V0
    v_cur = (H @ V0 - b * V0) / a
    mu[0] = norm * float(np.real(np.sum(V0.conj() * v_prev)))
    if M > 1:
        mu[1] = norm * float(np.real(np.sum(V0.conj() * v_cur)))
    for m in range(2, M):
        v_next = 2.0 * ((H @ v_cur) - b * v_cur) / a - v_prev
        mu[m] = norm * float(np.real(np.sum(V0.conj() * v_next)))
        v_prev, v_cur = v_cur, v_next
    g = _jackson(M)
    x = (energies - b) / a
    inside = np.abs(x) < 1.0
    rho = np.zeros_like(energies)
    xc = x[inside]
    theta = np.arccos(xc)
    series = g[0] * mu[0] * np.ones_like(xc)
    for m in range(1, M):
        series += 2.0 * g[m] * mu[m] * np.cos(m * theta)
    rho[inside] = series / (np.pi * np.sqrt(1.0 - xc ** 2) * a)
    return rho
