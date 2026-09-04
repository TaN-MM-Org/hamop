"""Band topology: Berry phases, Berry curvature and Chern numbers.

Discretized Wilson-loop formulas, in the gauge-invariant lattice
formulation of Fukui, Hatsugai and Suzuki, J. Phys. Soc. Jpn. 74, 1674
(2005): link variables are determinants of occupied-band overlap
matrices, plaquette fluxes are their arguments, and the total flux over
the Brillouin zone is the Chern number.  The formulation is exactly
gauge invariant, so the computed Chern number is an *exact integer* for
any grid dense enough to keep every plaquette flux inside (-pi, pi] --
which is precisely what the test suite asserts, to 1e-12, across the
known phase diagram of the Haldane model (Haldane, Phys. Rev. Lett. 61,
2015 (1988)).

Internally the Hamiltonian is rebuilt in the *periodic* gauge (Bloch
phases carry only the lattice vector, not the intra-cell positions), so
that H(k + G) = H(k) exactly and the Brillouin-zone torus closes without
boundary matching.  This choice affects Berry phases in the usual
convention-dependent way; the Zak phase of an inversion-symmetric chain
is quantized to 0 or pi (Zak, Phys. Rev. Lett. 62, 2747 (1989)), but
which of the two a given cell realizes depends on the choice of unit
cell origin.  The test suite therefore asserts the convention-free
statements: quantization itself, and the pi difference between the two
dimerizations of the SSH chain.

Nonorthogonal (LCAO-style) bases are supported through the smooth
Loewdin frame: with c the S-normalized eigenvectors (c^dag S c = 1),
the vectors d = S(k)^{1/2} c are orthonormal, and S(k)^{1/2} in the
periodic gauge is a smooth, periodic, invertible map on the Brillouin
zone -- a bundle isomorphism, under which the Chern number of the
occupied bands is invariant.  Wilson-loop links are therefore taken
between the d vectors.  (Berry *phase* values inherit this frame
convention on top of the usual origin convention; the quantized
statements -- integer Chern numbers, Zak phases of 0 or pi under
inversion, the pi difference between SSH dimerizations -- do not.)
For an orthogonal model S = 1 and d = c, so nothing changes there.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from .eigsolve import gen_eigh

__all__ = ["berry_phase", "chern_number", "berry_curvature"]


def _bloch_periodic(model, k):
    """H(k), S(k) in the periodic gauge: phase exp(i k . R) with R the
    lattice vector only, so H(k + G) = H(k) exactly."""
    k = model._kvec(k)
    n = model.nao
    H = np.zeros((n, n), dtype=complex)
    S = np.zeros((n, n), dtype=complex)
    overlap = model.has_overlap()
    for i, j, image, Hb, Sb in model._hops:
        oi, oj = model.offsets[i], model.offsets[j]
        R = np.asarray(image, dtype=float) @ model.cell
        ph = np.exp(1j * float(k @ R))
        ni, nj = Hb.shape
        onsite = (i == j) and not any(image)
        if Sb is None:
            Sb = (np.eye(ni, dtype=complex) if (onsite and overlap)
                  else np.zeros_like(Hb))
        H[oi:oi + ni, oj:oj + nj] += ph * Hb
        S[oi:oi + ni, oj:oj + nj] += ph * Sb
        if not onsite:
            H[oj:oj + nj, oi:oi + ni] += np.conj(ph) * Hb.conj().T
            S[oj:oj + nj, oi:oi + ni] += np.conj(ph) * Sb.conj().T
    if overlap:
        given = {i for i, j, im, _, _ in model._hops
                 if i == j and not any(im)}
        for i in range(len(model.norb)):
            if i not in given:
                oi = model.offsets[i]
                S[oi:oi + model.norb[i], oi:oi + model.norb[i]] += \
                    np.eye(model.norb[i], dtype=complex)
        S = 0.5 * (S + S.conj().T)
    else:
        S = np.eye(n, dtype=complex)
    return 0.5 * (H + H.conj().T), S


def _occ_states(model, k, n_occ):
    """Loewdin-orthonormal frame of the lowest n_occ bands at k."""
    H, S = _bloch_periodic(model, k)
    if not model.has_overlap():
        _, c = eigh(H)
        return c[:, :n_occ]
    w, V = eigh(S)
    if w.min() <= 1e-10:
        raise ValueError(
            "berry: S(k) is not positive definite at this k "
            f"(min overlap eigenvalue {w.min():.3e}); the Loewdin frame "
            "is undefined for an (over)complete basis here")
    _, c = gen_eigh(H, S, eigvals_only=False)
    d = (V * np.sqrt(w)) @ (V.conj().T @ c[:, :n_occ])
    return d


def berry_phase(model, kpts, n_occ=1):
    """Berry (Wilson-loop) phase of the lowest ``n_occ`` bands along a
    closed loop of k-points.

    kpts: sequence of Cartesian k-points tracing the loop *without*
    repeating the start point; closure is applied automatically in the
    periodic gauge.  Returns the phase in (-pi, pi].  For a
    one-dimensional model, a loop of evenly spaced points spanning one
    reciprocal period gives the Zak phase (origin-convention dependent;
    see the module docstring).
    """
    states = [_occ_states(model, k, n_occ) for k in kpts]
    states.append(states[0])
    prod = 1.0 + 0.0j
    for a, b in zip(states[:-1], states[1:]):
        d = np.linalg.det(a.conj().T @ b)
        if abs(d) < 1e-12:
            raise ValueError("vanishing overlap between neighboring "
                             "k-points; refine the loop")
        prod *= d / abs(d)
    return float(np.angle(prod))


def berry_curvature(model, mesh, n_occ=1):
    """Lattice Berry curvature (plaquette fluxes) of the lowest
    ``n_occ`` bands on a mesh x mesh grid of the 2D Brillouin zone.

    Returns an (mesh, mesh) array of fluxes in (-pi, pi]; their sum is
    2 pi times the Chern number.
    """
    if model.cell is None or model.cell.shape != (2, 2):
        raise ValueError("berry_curvature needs a 2D periodic model")
    N = int(mesh)
    recip = 2.0 * np.pi * np.linalg.inv(model.cell).T
    b1, b2 = recip[0], recip[1]
    # periodic gauge: u(k + b) is identified with u(k), so compute the
    # grid once and index modulo N
    states = [[_occ_states(model, (i / N) * b1 + (j / N) * b2, n_occ)
               for j in range(N)] for i in range(N)]

    def link(s_from, s_to):
        d = np.linalg.det(s_from.conj().T @ s_to)
        if abs(d) < 1e-12:
            raise ValueError("vanishing link overlap; refine the mesh")
        return d / abs(d)

    F = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            u1 = link(states[i][j], states[(i + 1) % N][j])
            u2 = link(states[(i + 1) % N][j], states[(i + 1) % N][(j + 1) % N])
            u3 = link(states[(i + 1) % N][(j + 1) % N], states[i][(j + 1) % N])
            u4 = link(states[i][(j + 1) % N], states[i][j])
            F[i, j] = float(np.angle(u1 * u2 * u3 * u4))
    return F


def chern_number(model, mesh=24, n_occ=1):
    """Chern number of the lowest ``n_occ`` bands (exact integer of the
    lattice field strength; float returned is integer to numerical
    rounding, which the tests pin to 1e-12)."""
    F = berry_curvature(model, mesh, n_occ)
    return float(F.sum() / (2.0 * np.pi))
