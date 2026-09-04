"""Spin as a first-class convention, anchored to exact statements:
spin doubling is an exact degeneracy, a Zeeman term splits by exactly
2B, and the Kane-Mele model (Kane and Mele, PRL 95, 226801 (2005))
with S_z conserved is exactly two Haldane copies -- gap 6 sqrt(3)
lambda_so at K, total Chern number zero, opposite unit Chern numbers
in the two spin sectors."""
import numpy as np

from hamop import (PAULI, bands, chern_number, haldane, kane_mele,
                   linear_chain, with_spin)


def test_with_spin_gives_exact_doublets():
    m = with_spin(linear_chain(t=-1.0, e0=0.3, s=0.2))
    for k in ([0.0], [0.7], [2.1]):
        e = bands(m, [k])[0]
        assert abs(e[1] - e[0]) < 1e-12
        # and the doublet sits exactly at the spinless eigenvalue
        e0 = bands(linear_chain(t=-1.0, e0=0.3, s=0.2), [k])[0][0]
        assert abs(e[0] - e0) < 1e-12


def test_zeeman_term_splits_by_exactly_2B():
    B = 0.25
    m = with_spin(linear_chain(t=-1.0, e0=0.0))
    m.add_hop(0, 0, (0,), B * PAULI["z"])
    k = [0.7]
    e = bands(m, [k])[0]
    e0 = 2.0 * (-1.0) * np.cos(0.7)
    assert np.abs(e - np.array([e0 - B, e0 + B])).max() < 1e-12


def test_kane_mele_is_two_haldane_copies_exactly():
    """With S_z conserved the model block-diagonalizes into
    haldane(t2=lam, phi=+pi/2) and its conjugate copy; the spectra must
    agree to machine precision at arbitrary k."""
    lam = 0.06
    km = kane_mele(t1=-1.0, lam_so=lam)
    hu = haldane(t1=-1.0, t2=lam, phi=+np.pi / 2, m_ab=0.0)
    hd = haldane(t1=-1.0, t2=lam, phi=-np.pi / 2, m_ab=0.0)
    rng = np.random.default_rng(3)
    for _ in range(4):
        k = rng.uniform(-3.0, 3.0, 2)
        e_km = bands(km, [k])[0]
        e_h = np.sort(np.concatenate([bands(hu, [k])[0],
                                      bands(hd, [k])[0]]))
        assert np.abs(e_km - e_h).max() < 1e-12


def test_kane_mele_gap_at_K_is_6_sqrt3_lambda():
    lam = 0.06
    km = kane_mele(t1=-1.0, lam_so=lam)
    recip = 2.0 * np.pi * np.linalg.inv(km.cell).T
    K = (2.0 * recip[0] + recip[1]) / 3.0
    e = bands(km, [K])[0]
    assert abs((e[2] - e[1]) - 6.0 * np.sqrt(3.0) * lam) < 1e-9


def test_kane_mele_total_chern_zero_and_spin_sectors_opposite():
    lam = 0.06
    km = kane_mele(t1=-1.0, lam_so=lam)
    assert abs(chern_number(km, mesh=18, n_occ=2)) < 1e-12
    C_up = chern_number(haldane(t1=-1.0, t2=lam, phi=+np.pi / 2), mesh=18)
    C_dn = chern_number(haldane(t1=-1.0, t2=lam, phi=-np.pi / 2), mesh=18)
    assert abs(C_up + C_dn) < 1e-12
    assert abs(abs(0.5 * (C_up - C_dn)) - 1.0) < 1e-12   # spin Chern +/-1
