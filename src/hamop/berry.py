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

Two frames are available.  frame="lowdin" (default) is the periodic
gauge with the Loewdin map described above.  frame="atomic" instead
takes everything from the same atomic-gauge assembly as the rest of
the package (``model.bloch``), with inter-k links
c(k1)^dag S((k1+k2)/2) c(k2) -- the discrete inner product implied by
the site-diagonal position operator, i.e. exactly the approximation
the velocity operator already uses.  Berry-phase *values* differ
between frames by the usual convention shifts; the topological
statements -- integer Chern numbers, and which integer -- must agree,
and the tests assert that they do.  In the atomic gauge H(k+G) is not
H(k), so a loop that spans the zone must say so via ``closure``.

Large unit cells with few occupied bands can use solver="sparse"
(orthogonal models, Lanczos for the occupied states); it must return
the same integers as the dense path, and the tests assert that too.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from .eigsolve import gen_eigh

__all__ = ["berry_phase", "chern_number", "berry_curvature",
           "chern_marker"]


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


def _lowest_dense(H, n_occ):
    _, c = eigh(H)
    return c[:, :n_occ]


def _lowest_sparse(H, n_occ):
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    n = H.shape[0]
    if not n_occ < n - 1:
        raise ValueError(
            "solver='sparse' (Lanczos) needs n_occ < nao - 1; use the "
            "dense solver for this small a matrix")
    _, c = eigsh(csr_matrix(H), k=n_occ, which="SA")
    return c


def _occ_states(model, k, n_occ, solver="dense"):
    """Loewdin-orthonormal frame of the lowest n_occ bands at k
    (periodic gauge)."""
    H, S = _bloch_periodic(model, k)
    if not model.has_overlap():
        if solver == "sparse":
            return _lowest_sparse(H, n_occ)
        return _lowest_dense(H, n_occ)
    if solver == "sparse":
        raise ValueError("solver='sparse' supports orthogonal models "
                         "only (the Loewdin map is dense)")
    w, V = eigh(S)
    if w.min() <= 1e-10:
        raise ValueError(
            "berry: S(k) is not positive definite at this k "
            f"(min overlap eigenvalue {w.min():.3e}); the Loewdin frame "
            "is undefined for an (over)complete basis here")
    _, c = gen_eigh(H, S, eigvals_only=False)
    d = (V * np.sqrt(w)) @ (V.conj().T @ c[:, :n_occ])
    return d


def _occ_states_atomic(model, k, n_occ, solver="dense"):
    """Occupied eigenvectors at k in the atomic gauge, S-normalized --
    the same assembly every other observable uses."""
    H, S = model.bloch(k)
    if not model.has_overlap():
        if solver == "sparse":
            return _lowest_sparse(H, n_occ)
        return _lowest_dense(H, n_occ)
    if solver == "sparse":
        raise ValueError("solver='sparse' supports orthogonal models "
                         "only")
    _, c = gen_eigh(H, S, eigvals_only=False)
    return c[:, :n_occ]


def _get_states(model, k, n_occ, frame, solver):
    if frame == "lowdin":
        return _occ_states(model, k, n_occ, solver)
    if frame == "atomic":
        return _occ_states_atomic(model, k, n_occ, solver)
    raise ValueError("frame must be 'lowdin' or 'atomic'")


def _link(model, frame, k_from, s_from, k_to, s_to):
    """Normalized determinant link between occupied frames.  In the
    atomic frame with overlap, the inner product carries S at the bond
    midpoint -- the discrete metric implied by the site-diagonal
    position operator."""
    if frame == "atomic" and model.has_overlap():
        km = 0.5 * (np.asarray(k_from, dtype=float)
                    + np.asarray(k_to, dtype=float))
        _, Smid = model.bloch(km)
        d = np.linalg.det(s_from.conj().T @ Smid @ s_to)
    else:
        d = np.linalg.det(s_from.conj().T @ s_to)
    if abs(d) < 1e-12:
        raise ValueError("vanishing link overlap; refine the grid")
    return d / abs(d)


def berry_phase(model, kpts, n_occ=1, frame="lowdin", closure=None,
                solver="dense"):
    """Berry (Wilson-loop) phase of the lowest ``n_occ`` bands along a
    closed loop of k-points.

    kpts: sequence of Cartesian k-points tracing the loop *without*
    repeating the start point.  closure: the reciprocal vector by which
    the loop wraps the zone (e.g. b1 for a Zak loop); None means the
    loop is literally closed.  In the periodic Loewdin frame a wrap by
    a reciprocal vector is the identity, so closure may be omitted
    there; in the atomic frame it must be given, because
    H(k + G) != H(k).  Returns the phase in (-pi, pi].  For a
    one-dimensional model, a loop of evenly spaced points spanning one
    reciprocal period gives the Zak phase (origin- and
    frame-convention dependent; see the module docstring).
    """
    kpts = [np.atleast_1d(np.asarray(k, dtype=float)) for k in kpts]
    states = [_get_states(model, k, n_occ, frame, solver) for k in kpts]
    if closure is None:
        k_end = kpts[0]
        states.append(states[0])
    else:
        k_end = kpts[0] + np.atleast_1d(np.asarray(closure, dtype=float))
        states.append(_get_states(model, k_end, n_occ, frame, solver))
    klist = kpts + [k_end]
    prod = 1.0 + 0.0j
    for i in range(len(states) - 1):
        prod *= _link(model, frame, klist[i], states[i],
                      klist[i + 1], states[i + 1])
    return float(np.angle(prod))


def berry_curvature(model, mesh, n_occ=1, frame="lowdin", solver="dense"):
    """Lattice Berry curvature (plaquette fluxes) of the lowest
    ``n_occ`` bands on a mesh x mesh grid of the 2D Brillouin zone.

    Returns an (mesh, mesh) array of fluxes in (-pi, pi]; their sum is
    2 pi times the Chern number.  frame and solver as in
    :func:`berry_phase`; in the atomic frame the wrapped boundary
    points are evaluated explicitly rather than identified modulo the
    zone.
    """
    if model.cell is None or model.cell.shape != (2, 2):
        raise ValueError("berry_curvature needs a 2D periodic model")
    N = int(mesh)
    recip = 2.0 * np.pi * np.linalg.inv(model.cell).T
    b1, b2 = recip[0], recip[1]
    if frame == "lowdin":
        # periodic gauge: u(k + b) is identified with u(k); index mod N
        kg = [[(i / N) * b1 + (j / N) * b2 for j in range(N)]
              for i in range(N)]
        st = [[_get_states(model, kg[i][j], n_occ, frame, solver)
               for j in range(N)] for i in range(N)]

        def K(i, j):
            return kg[i % N][j % N]

        def S(i, j):
            return st[i % N][j % N]
    elif frame == "atomic":
        kg = [[(i / N) * b1 + (j / N) * b2 for j in range(N + 1)]
              for i in range(N + 1)]
        st = [[_get_states(model, kg[i][j], n_occ, frame, solver)
               for j in range(N + 1)] for i in range(N + 1)]

        def K(i, j):
            return kg[i][j]

        def S(i, j):
            return st[i][j]
    else:
        raise ValueError("frame must be 'lowdin' or 'atomic'")

    F = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            u1 = _link(model, frame, K(i, j), S(i, j),
                       K(i + 1, j), S(i + 1, j))
            u2 = _link(model, frame, K(i + 1, j), S(i + 1, j),
                       K(i + 1, j + 1), S(i + 1, j + 1))
            u3 = _link(model, frame, K(i + 1, j + 1), S(i + 1, j + 1),
                       K(i, j + 1), S(i, j + 1))
            u4 = _link(model, frame, K(i, j + 1), S(i, j + 1),
                       K(i, j), S(i, j))
            F[i, j] = float(np.angle(u1 * u2 * u3 * u4))
    return F


def chern_number(model, mesh=24, n_occ=1, frame="lowdin", solver="dense"):
    """Chern number of the lowest ``n_occ`` bands (exact integer of the
    lattice field strength; float returned is integer to numerical
    rounding, which the tests pin to 1e-12).  The tests also assert
    that the two frames and the two solvers return the same integer."""
    F = berry_curvature(model, mesh, n_occ, frame=frame, solver=solver)
    return float(F.sum() / (2.0 * np.pi))


def chern_marker(model, mu, thresh=1e-10):
    """Local (real-space) Chern marker of a *finite* 2D model: the
    per-site array m_i = 4 pi Im [P x Q y P]_ii (Angstrom^2), following
    R. Bianco and R. Resta, Phys. Rev. B 84, 241106(R) (2011), with P
    the projector on states below mu, Q = 1 - P, and x, y the
    site-diagonal position operators.

    Summed over a bulk region and divided by that region's area, the
    marker estimates the Chern number of the underlying periodic
    system; summed over the *whole* finite system it vanishes exactly
    (Im Tr[P x Q y] = 0 identically -- the same identity that makes a
    bounded system's DC Hall response zero), the bulk value being
    compensated at the edges.  Both statements are asserted in the
    tests against the package's chern_number on the periodic Haldane
    model, and the sign convention here is fixed by that agreement
    (which in turn is pinned to sigma_xy = C e^2/h by the TKNN test).

    Orthogonal bases only (the projector construction assumes an
    orthonormal site basis); overlap models are refused.
    """
    if model.cell is not None:
        raise ValueError("chern_marker works on finite models; use "
                         "chern_number for periodic ones")
    if model.has_overlap():
        raise ValueError("chern_marker supports orthogonal bases only")
    if model.positions.shape[1] < 2:
        raise ValueError("chern_marker needs two dimensions")
    H, _ = model.bloch(None)
    e, c = eigh(H)
    occ = c[:, e < mu]
    P = occ @ occ.conj().T
    Q = np.eye(len(H), dtype=complex) - P
    # orbital positions (site position repeated per orbital)
    xpos = np.repeat(model.positions[:, 0], model.norb)
    ypos = np.repeat(model.positions[:, 1], model.norb)
    M = P @ (xpos[:, None] * (Q @ (ypos[:, None] * P)))
    return 4.0 * np.pi * np.imag(np.diag(M))
