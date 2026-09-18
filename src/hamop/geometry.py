"""Quantum geometry: the metric that lives next to the Berry
curvature (new in v0.10).

The Bloch states of a band carry more geometry than their Berry
curvature. The full object is the quantum geometric tensor of the
occupied subspace,

    Q_ab(k) = Tr[ dP_a (1 - P) dP_b P ],    dP_a = d P / d k_a,

with P(k) the projector onto the occupied states: its real symmetric
part is the quantum metric g_ab (how fast the occupied subspace
turns as k moves; J. P. Provost and G. Vallee, Commun. Math. Phys.
76, 289 (1980)), and its imaginary antisymmetric part is the Berry
curvature. Sign conventions for Omega differ across the literature
by the orientation of the k-space loop; THIS package fixes it so
that the QGT curvature integrates to the same Chern number as its
own plaquette `berry_curvature` (asserted in the tests as an
independent-path identity), which lands on Omega = +2 Im Q_xy here.

Because Q is a Gram matrix of
projected derivative states it is positive semidefinite, which makes
the chain

    tr g  >=  2 sqrt(det g)  >=  |Omega|

an exact pointwise identity (the first step is the arithmetic-
geometric mean, the second is det Q >= 0) -- the inequality behind
flat-band superfluidity bounds (S. Peotta and P. Torma, Nat. Commun.
6, 8944 (2015)) and the recent bounds tying topology to optical
weight (Y. Onishi and L. Fu, Phys. Rev. X 14, 011052 (2024)). The
integrated metric is exposed as the quantum-weight tensor in the
convention K_ab = 2 pi * integral d^2k/(2 pi)^2 g_ab, for which the
pointwise chain integrates to tr K >= |C| -- asserted in the tests,
never assumed.

Everything is computed gauge-invariantly: the projector P(k) is
gauge invariant, so central finite differences of P need no phase
fixing. Two independent code paths anchor the implementation: on any
two-band model the closed forms g_ab = (1/4) dn.dn and
|Omega| = (1/2)|n . (dn x dn)| of the Bloch-sphere unit vector n(k)
must agree with the projector computation, and the Chern number
integrated from the QGT's Omega must equal the package's own
plaquette `chern_number` -- a third, independent path.

Conventions: k Cartesian in 1/Angstrom, so g is in Angstrom^2 and
Omega in Angstrom^2; models with orbital overlaps are refused (the
geometry of a nonorthogonal basis needs S-weighted inner products
this module does not implement -- designed out, not half-shipped);
and a band crossing through the occupied/empty boundary makes the
projector jump, so a gap smaller than `gap_min` at any evaluation
point is refused with the gap named.
"""
from __future__ import annotations

import numpy as np

__all__ = ["quantum_geometric_tensor", "quantum_metric",
           "quantum_weight"]


def _projector(model, k, n_occ, gap_min):
    H, S = model.bloch(np.asarray(k, dtype=float))
    if model.has_overlap():
        raise ValueError(
            "quantum geometry with orbital overlaps is not "
            "implemented (nonorthogonal bases need S-weighted inner "
            "products); orthogonalize the model first")
    e, v = np.linalg.eigh(H)
    if n_occ < 1 or n_occ >= e.size:
        raise ValueError(f"n_occ must lie in [1, {e.size - 1}]")
    gap = float(e[n_occ] - e[n_occ - 1])
    if gap < gap_min:
        raise ValueError(
            f"the gap between occupied and empty states is {gap:.3g} "
            f"(< gap_min = {gap_min:.3g}) at k = {list(k)}: the "
            "occupied projector is not smooth across a band "
            "crossing, so the quantum geometry there is undefined")
    vo = v[:, :n_occ]
    return vo @ vo.conj().T


def quantum_geometric_tensor(model, k, n_occ=1, delta=1e-5,
                             gap_min=1e-8):
    """The occupied-subspace quantum geometric tensor at one k.

    Returns the (dim, dim) complex matrix Q_ab; its real part is the
    quantum metric, and Omega = 2 Im Q_xy is the Berry curvature
    density (Angstrom^2) in this package's plaquette convention.
    `delta` is the finite-difference step in 1/Angstrom.
    """
    k = np.asarray(k, dtype=float).ravel()
    dim = k.size
    dP = []
    for a in range(dim):
        kp, km = k.copy(), k.copy()
        kp[a] += delta
        km[a] -= delta
        dP.append((_projector(model, kp, n_occ, gap_min)
                   - _projector(model, km, n_occ, gap_min))
                  / (2.0 * delta))
    P = _projector(model, k, n_occ, gap_min)
    one = np.eye(P.shape[0])
    Q = np.empty((dim, dim), dtype=complex)
    for a in range(dim):
        for b in range(dim):
            Q[a, b] = np.trace(dP[a] @ (one - P) @ dP[b] @ P)
    return Q


def quantum_metric(model, k, n_occ=1, delta=1e-5, gap_min=1e-8):
    """The quantum metric g_ab = Re Q_ab (Angstrom^2) at one k."""
    return np.real(quantum_geometric_tensor(model, k, n_occ, delta,
                                            gap_min))


def quantum_weight(model, mesh=24, n_occ=1, delta=1e-5, gap_min=1e-8):
    """The quantum-weight tensor and the QGT-integrated Chern number.

    Integrates over a Monkhorst-Pack mesh of the 2D Brillouin zone:

        K_ab = 2 pi * integral d^2k / (2 pi)^2  g_ab(k),
        C    = (1 / 2 pi) * integral d^2k  Omega(k),

    returning dict(K, chern, tr_K). The exact pointwise chain
    tr g >= |Omega| integrates to tr K >= |C| (cf. Onishi and Fu,
    PRX 14, 011052 (2024)); the tests assert it, and assert that
    `chern` matches the package's plaquette `chern_number` -- an
    independent code path.
    """
    if model.cell is None or model.cell.shape != (2, 2):
        raise ValueError("quantum_weight integrates a 2D Brillouin "
                         "zone; the model must be periodic in 2D")
    kpts, w = model.monkhorst_pack(mesh)
    a_bz = (2.0 * np.pi) ** 2 / model.cell_volume
    K = np.zeros((2, 2))
    c = 0.0
    for kp, wi in zip(kpts, w):
        Q = quantum_geometric_tensor(model, kp, n_occ, delta, gap_min)
        K += wi * np.real(Q)
        c += wi * (2.0 * np.imag(Q[0, 1]))
    K *= a_bz / (2.0 * np.pi)
    chern = c * a_bz / (2.0 * np.pi)
    return {"K": K, "chern": float(chern),
            "tr_K": float(np.trace(K))}
