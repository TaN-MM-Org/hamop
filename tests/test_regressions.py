"""Regression tests for the bugs fixed in v0.10.1.  Each one failed on
v0.10.0 and passes after the fix."""
import numpy as np
import pytest
from scipy.linalg import eigh

from hamop import (PAULI, TightBindingModel, bands, carrier_count,
                   chern_number, device_ldos, drude_weight, fermi_level,
                   haldane, kpm_sigma, linear_chain, sigma_optical,
                   sigma_tensor, two_site, with_spin)
from hamop.berry import _bloch_periodic


def test_onsite_blocks_in_two_calls_do_not_double_the_overlap():
    """In an overlap model, an on-site block added without S_block must
    not add a second identity to S.  Zeeman term on a spinful
    nonorthogonal chain: E = (2 t cos ka -/+ B) / (1 + 2 s cos ka)."""
    t, s, B, k = -1.0, 0.2, 0.25, 0.7
    m = with_spin(linear_chain(t=t, e0=0.0, s=s))
    m.add_hop(0, 0, (0,), B * PAULI["z"])
    c = np.cos(k)
    exact = np.array([2 * t * c - B, 2 * t * c + B]) / (1 + 2 * s * c)
    assert np.abs(bands(m, [[k]])[0] - exact).max() < 1e-12
    # spinless: an on-site energy added in a second call
    m2 = linear_chain(t=t, e0=0.0, s=s)
    m2.add_hop(0, 0, (0,), [[0.1]])
    e = bands(m2, [[k]])[0][0]
    assert abs(e - (0.1 + 2 * t * c) / (1 + 2 * s * c)) < 1e-12
    # the periodic-gauge assembly used by the topology code agrees
    h = haldane(t1=-1.0, t2=0.1, s=0.1)
    h.add_hop(0, 0, (0, 0), [[0.0]])
    _, S_atomic = h.bloch(None)
    _, S_periodic = _bloch_periodic(h, np.zeros(2))
    assert np.abs(S_atomic - S_periodic).max() < 1e-12
    assert abs(S_periodic[0, 0] - 1.0) < 1e-12
    assert abs(chern_number(h, mesh=18) - 1.0) < 1e-12


def test_with_spin_keeps_the_dipole_blocks():
    """The spinful copy must carry the intra-atomic dipole: with spin
    explicit (spin=1) it gives the same conductivity as the spinless
    model with spin=2."""
    m = TightBindingModel([[0.0]], norb=2, cell=None)
    m.add_hop(0, 0, (0,), [[0.0, 0.0], [0.0, 1.6]])
    X = np.zeros((2, 2, 1), dtype=complex)
    X[0, 1, 0] = X[1, 0, 0] = 0.7
    m.set_dipole(0, X)
    om = np.linspace(1.5, 1.7, 21)
    s_spinless = sigma_optical(m, om, 0.8, eta=0.05, T=10.0, spin=2)
    ms = with_spin(m)
    assert ms.has_dipoles()
    s_spinful = sigma_optical(ms, om, 0.8, eta=0.05, T=10.0, spin=1)
    assert s_spinless.max() > 1.0
    assert np.abs(s_spinful - s_spinless).max() < 1e-10 * s_spinless.max()


def test_device_ldos_uses_the_inter_layer_overlap():
    """Mulliken LDOS -Im (G S)_ii / pi needs the whole device S,
    inter-layer blocks included.  Closed nonorthogonal chain: compare
    with the generalized eigenpairs directly; the total is the sum of
    Lorentzians."""
    s, N, eta, E = 0.2, 6, 0.05, 0.37
    H00, H01 = np.array([[0.0]]), np.array([[-1.0]])
    S00, S01 = np.eye(1), np.array([[s]])
    ld = device_ldos([E], [H00] * N, [H01] * (N - 1), H00, H01,
                     [S00] * N, [S01] * (N - 1), S00, S01, eta=eta,
                     attach_leads=False)[0]
    off = np.diag(np.ones(N - 1), 1)
    Hf = -(off + off.T)
    Sf = np.eye(N) + s * (off + off.T)
    w, v = eigh(Hf, Sf)
    G = (v / (E + 1j * eta - w)) @ v.T
    mulliken = -np.imag(np.diag(G @ Sf)) / np.pi
    assert np.abs(ld - mulliken).max() < 1e-12
    lor = (eta / np.pi / ((E - w) ** 2 + eta ** 2)).sum()
    assert abs(ld.sum() - lor) < 1e-12


def test_carrier_count_refuses_a_periodic_model_without_a_grid():
    """Without mesh or kpts a periodic model used to be summed at
    k = 0 only (2.0 instead of 1.0 for the half-filled chain)."""
    m = linear_chain()
    with pytest.raises(ValueError, match="mesh or kpts"):
        carrier_count(m, 0.0)
    assert abs(carrier_count(m, 0.0, mesh=400) - 1.0) < 1e-9
    assert abs(carrier_count(two_site(), 0.0) - 2.0) < 1e-9   # finite


def test_temperatures_the_fermi_factors_cannot_use_are_refused():
    """drude_weight returned NaN at T = 0 and a negative weight at
    T < 0; the other Fermi-weighted routines accepted T < 0."""
    m = linear_chain()
    with pytest.raises(ValueError, match="T must be > 0"):
        drude_weight(m, 0.0, mesh=100, T=0.0)
    with pytest.raises(ValueError, match="T must be"):
        drude_weight(m, 0.0, mesh=100, T=-10.0)
    om = np.array([1.0])
    with pytest.raises(ValueError, match="T must be"):
        sigma_optical(m, om, 0.0, mesh=100, T=-10.0)
    with pytest.raises(ValueError, match="T must be"):
        sigma_tensor(m, om, 0.0, directions=(0, 0), mesh=100, T=-10.0)
    with pytest.raises(ValueError, match="T must be"):
        carrier_count(m, 0.0, mesh=100, T=-10.0)
    with pytest.raises(ValueError, match="T must be"):
        fermi_level(m, 0.5, mesh=100, T=-10.0)
    with pytest.raises(ValueError, match="T must be"):
        kpm_sigma(two_site(), np.array([2.0]), 0.0, T=-10.0)


def test_negf_with_overlap_builds_z_s_minus_h_at_finite_eta():
    """The reverse coupling block of z S - H is z S^dag - H^dag, not the
    conjugate transpose of z S - H (z is complex).  Lead: the surface
    Green function must match its closed form; closed device: G must
    equal (z S - H)^-1; open device: the identity
    i (G - G^dag) = G (GamL + GamR + 2 eta S) G^dag must hold with
    overlap too; and the recursive sweep must still equal direct
    inversion at a large eta."""
    from hamop import (device_greens, transmission, transmission_direct,
                       transmission_sparse)
    s, N, E = 0.2, 4, 0.37
    H00, H01 = np.array([[0.0]]), np.array([[-1.0]])
    S00, S01 = np.eye(1), np.array([[s]])
    args = ([H00] * N, [H01] * (N - 1), H00, H01,
            [S00] * N, [S01] * (N - 1), S00, S01)
    off = np.diag(np.ones(N - 1), 1)
    Hf = -(off + off.T)
    Sf = np.eye(N) + s * (off + off.T)
    eta = 0.05
    # lead surface Green function: g = 1 / (z - a^2 g) with the same
    # coupling a = z s - t both ways; the decaying root has |a g| < 1
    from hamop import sancho_rubio
    z = E + 1j * eta
    a = z * s - H01[0, 0]
    g_cf = min(np.roots([a * a, -z, 1.0]), key=lambda r: abs(a * r))
    g_sr = sancho_rubio(E, H00, H01, S00, S01, eta)[0, 0]
    assert abs(g_sr - g_cf) < 1e-12
    G, _, _, _ = device_greens(E, *args, eta=eta, attach_leads=False)
    assert np.abs(G - np.linalg.inv((E + 1j * eta) * Sf - Hf)).max() < 1e-12
    for eta in (1e-3, 1e-7):
        G, _, GamL, GamR = device_greens(E, *args, eta=eta)
        lhs = 1j * (G - G.conj().T)
        rhs = G @ (GamL + GamR + 2 * eta * Sf) @ G.conj().T
        assert np.abs(lhs - rhs).max() < 1e-12
    Es = np.array([-0.8, 0.37, 1.1])
    T1 = transmission(Es, *args, eta=0.05)
    T2 = transmission_direct(Es, *args, eta=0.05)
    T3 = transmission_sparse(Es, *args, eta=0.05)
    assert np.abs(T1 - T2).max() < 1e-12
    assert np.abs(T3 - T2).max() < 1e-12
