"""Quantum-geometry anchors: three independent code paths agree (the
projector QGT, the two-band Bloch-sphere closed forms, and the
package's own plaquette Chern number); the positive-semidefiniteness
chain tr g >= 2 sqrt(det g) >= |Omega| holds pointwise as an exact
identity of the Gram construction; the integrated bound tr K >= |C|
holds in the topological AND the trivial phase of the Haldane model;
the metric is symmetric and positive semidefinite; and band
crossings and overlap models are refused."""
import numpy as np
import pytest

from hamop import chern_number, haldane
from hamop.geometry import (quantum_geometric_tensor, quantum_metric,
                            quantum_weight)

TOPO = haldane(t1=-1.0, t2=0.1, phi=0.5 * np.pi, m_ab=0.0)
TRIV = haldane(t1=-1.0, t2=0.1, phi=0.5 * np.pi, m_ab=1.2)
KPTS = [np.array([0.3, 0.1]), np.array([1.1, -0.7]),
        np.array([-0.4, 1.9]), np.array([2.0, 2.0])]


def _dvec(model, k):
    """Bloch-sphere decomposition H = eps I + d . sigma of a 2-band
    Bloch Hamiltonian -- the independent closed-form path."""
    H, _ = model.bloch(k)
    dx = float(np.real(H[0, 1]))
    dy = float(-np.imag(H[0, 1]))
    dz = float(np.real(H[0, 0] - H[1, 1])) / 2.0
    return np.array([dx, dy, dz])


def _closed_form(model, k, h=1e-5):
    """Lower-band metric and curvature from the unit vector n(k):
    g_ab = (1/4) dn_a . dn_b, |Omega| = (1/2) |n . (dn_x x dn_y)|."""
    def nhat(kk):
        d = _dvec(model, kk)
        return d / np.linalg.norm(d)
    k = np.asarray(k, dtype=float)
    dn = []
    for a in range(2):
        kp, km = k.copy(), k.copy()
        kp[a] += h
        km[a] -= h
        dn.append((nhat(kp) - nhat(km)) / (2.0 * h))
    g = 0.25 * np.array([[dn[a] @ dn[b] for b in range(2)]
                         for a in range(2)])
    omega_abs = 0.5 * abs(nhat(k) @ np.cross(dn[0], dn[1]))
    return g, omega_abs


def test_two_band_closed_form_agrees_with_projector_qgt():
    for m in (TOPO, TRIV):
        for k in KPTS:
            Q = quantum_geometric_tensor(m, k)
            g = np.real(Q)
            omega = 2.0 * float(np.imag(Q[0, 1]))
            g_cf, omega_abs_cf = _closed_form(m, k)
            assert np.allclose(g, g_cf, rtol=1e-4, atol=1e-8)
            assert abs(abs(omega) - omega_abs_cf) \
                < 1e-4 * max(omega_abs_cf, 1e-8) + 1e-8


def test_psd_chain_exact_pointwise():
    """tr g >= 2 sqrt(det g) >= |Omega|: arithmetic-geometric mean
    plus det Q >= 0 of the Gram construction -- checked across the
    zone in both phases."""
    rng = np.random.default_rng(4)
    for m in (TOPO, TRIV):
        for _ in range(25):
            k = rng.uniform(-2.5, 2.5, 2)
            Q = quantum_geometric_tensor(m, k)
            g = np.real(Q)
            omega = 2.0 * float(np.imag(Q[0, 1]))
            tr = float(np.trace(g))
            det = float(np.linalg.det(g))
            assert det > -1e-12
            root = 2.0 * np.sqrt(max(det, 0.0))
            assert tr >= root - 1e-9 * (1.0 + tr)
            assert root >= abs(omega) - 1e-6 * (1.0 + abs(omega))
            # the metric is symmetric PSD
            assert abs(g[0, 1] - g[1, 0]) < 1e-9 * (1.0 + tr)
            assert np.all(np.linalg.eigvalsh(g) > -1e-10)


def test_chern_from_qgt_matches_plaquette_path():
    """Third independent path: the QGT's integrated curvature must
    hit the package's own plaquette Chern number, +-1 and 0 in the
    two Haldane phases."""
    out_t = quantum_weight(TOPO, mesh=18)
    out_v = quantum_weight(TRIV, mesh=18)
    c_t = chern_number(TOPO, mesh=24)
    c_v = chern_number(TRIV, mesh=24)
    assert round(out_t["chern"]) == round(c_t) and abs(round(c_t)) == 1
    assert abs(out_t["chern"] - c_t) < 0.02
    assert round(out_v["chern"]) == round(c_v) == 0
    assert abs(out_v["chern"]) < 0.02 and abs(c_v) < 1e-9


def test_integrated_bound_tr_k_geq_chern():
    for m in (TOPO, TRIV):
        out = quantum_weight(m, mesh=18)
        assert out["tr_K"] >= abs(out["chern"]) - 1e-6
        assert out["tr_K"] > 0.0
        assert np.allclose(out["K"], out["K"].T, atol=1e-9)


def test_refusals():
    from hamop import graphene
    g = graphene()
    # the Dirac point closes the gap: the projector is not smooth
    # there and the geometry is refused with the gap named
    K_dirac = np.array([4.0 * np.pi / (3.0 * 2.46), 0.0])
    with pytest.raises(ValueError, match="gap"):
        quantum_geometric_tensor(g, K_dirac)
    with pytest.raises(ValueError, match="n_occ"):
        quantum_metric(TOPO, KPTS[0], n_occ=5)
    from hamop import linear_chain
    m = linear_chain(t=-1.0, s=0.2)          # overlap model
    if m.has_overlap():
        with pytest.raises(ValueError, match="overlap"):
            quantum_metric(m, np.array([0.3]))
