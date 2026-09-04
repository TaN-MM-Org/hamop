"""The intra-atomic dipole velocity term against hand-derived closed
forms: the site-diagonal position approximation leaves an on-site
s -> p transition exactly dark, the dipole block makes it bright with
the exact peak height, a dipole that commutes with H changes nothing,
and the overlap case is refused."""
import numpy as np
import pytest

from hamop import (TightBindingModel, linear_chain, sigma_optical,
                   sigma_tensor)

DELTA, DIP, ETA = 1.6, 0.7, 0.05


def _atom(delta=DELTA, d=DIP, with_dipole=True):
    m = TightBindingModel([[0.0]], norb=2, cell=None)
    m.add_hop(0, 0, (0,), [[0.0, 0.0], [0.0, delta]])
    if with_dipole:
        X = np.zeros((2, 2, 1), dtype=complex)
        X[0, 1, 0] = d
        X[1, 0, 0] = d
        m.set_dipole(0, X)
    return m


def test_onsite_transition_is_dark_without_the_dipole():
    om = np.linspace(1.55, 1.65, 51)
    sig = sigma_optical(_atom(with_dipole=False), om, mu=0.5 * DELTA,
                        eta=ETA, T=10.0)
    assert np.abs(sig).max() == 0.0


def test_dipole_makes_the_transition_bright_with_the_exact_peak():
    """|M| = |i (E_s - E_p) d| = Delta d by hand, so the Gaussian peak
    is spin 4 pi (Delta d)^2 / (eta sqrt(2 pi) Delta)."""
    om = np.linspace(1.55, 1.65, 201)
    sig = sigma_optical(_atom(), om, mu=0.5 * DELTA, eta=ETA, T=10.0)
    expected = 2.0 * 4.0 * np.pi * (DELTA * DIP) ** 2 \
        / (ETA * np.sqrt(2.0 * np.pi)) / DELTA
    ipk = int(np.argmax(sig))
    assert abs(om[ipk] - DELTA) < 1e-3
    assert abs(sig[ipk] - expected) / expected < 1e-3


def test_commuting_dipole_changes_nothing():
    """[H, X] = 0 for a diagonal X in the eigenbasis: the dipole term
    must vanish identically, not approximately."""
    m = TightBindingModel([[0.0]], norb=2, cell=None)
    m.add_hop(0, 0, (0,), [[0.0, 0.0], [0.0, DELTA]])
    X = np.zeros((2, 2, 1), dtype=complex)
    X[0, 0, 0], X[1, 1, 0] = 0.3, -1.1
    m.set_dipole(0, X)
    om = np.linspace(1.55, 1.65, 51)
    assert np.abs(sigma_optical(m, om, 0.5 * DELTA, eta=ETA,
                                T=10.0)).max() == 0.0


def test_tensor_route_agrees_with_the_optical_route():
    om = np.linspace(1.55, 1.65, 101)
    sL = sigma_optical(_atom(), om, 0.5 * DELTA, eta=ETA, T=10.0,
                       lineshape="lorentzian")
    sT = np.real(sigma_tensor(_atom(), om, 0.5 * DELTA, directions=(0, 0),
                              eta=ETA, T=10.0))
    assert np.abs(sT - sL).max() / sL.max() < 1e-3


def test_dipoles_with_overlap_are_refused():
    m = linear_chain(t=-1.0, s=0.2)
    X = np.zeros((1, 1, 1), dtype=complex)
    m.set_dipole(0, X)
    with pytest.raises(ValueError):
        sigma_optical(m, np.array([1.0]), mu=0.0, mesh=20)


def test_non_hermitian_dipole_block_is_refused():
    m = TightBindingModel([[0.0]], norb=2, cell=None)
    m.add_hop(0, 0, (0,), [[0.0, 0.0], [0.0, DELTA]])
    X = np.zeros((2, 2, 1), dtype=complex)
    X[0, 1, 0] = 1.0j
    X[1, 0, 0] = 1.0j          # not Hermitian
    with pytest.raises(ValueError):
        m.set_dipole(0, X)
