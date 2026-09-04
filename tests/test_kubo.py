"""Optics against its anchors: the graphene universal conductivity, the
two-site absorption line with a hand-derived matrix element, exact
invariance under a shift of the energy zero in a nonorthogonal basis,
and the carrier count."""
import numpy as np

from hamop import (carrier_count, drude_weight, graphene, linear_chain,
                   sigma_optical, two_site)


def test_graphene_universal_sheet_conductivity():
    """The interband plateau equals e^2/(4 hbar) -- i.e. 1.0 in the
    package units (Kuzmenko et al., PRL 100, 117401 (2008)).  Finite
    broadening and k-mesh keep this a few-percent statement."""
    g = graphene(t=-2.7, a=2.46)
    sig = sigma_optical(g, np.array([1.0, 1.3]), mu=0.0, mesh=120,
                        eta=0.12, T=10.0)
    assert np.all(np.abs(sig - 1.0) < 0.05)


def test_two_site_molecule_line_position_and_strength():
    """One transition at 2|t|.  The velocity matrix element between the
    bonding and antibonding states is |M| = |a t| by hand, so the peak
    of the Gaussian-broadened line is
    4 pi |a t|^2 / (eta sqrt(2 pi) * 2|t|) * spin (area = 1 for a
    finite system)."""
    t, a, eta = -0.8, 1.3, 0.05
    m = two_site(t=t, a=a)
    om = np.linspace(0.5, 3.0, 2501)
    sig = sigma_optical(m, om, mu=0.0, eta=eta, T=10.0, spin=2)
    ipk = int(np.argmax(sig))
    assert abs(om[ipk] - 2.0 * abs(t)) < 2e-3
    expected_peak = 2 * 4.0 * np.pi * (a * t) ** 2 \
        / (eta * np.sqrt(2.0 * np.pi)) / (2.0 * abs(t))
    assert abs(sig[ipk] - expected_peak) / expected_peak < 1e-3


def test_energy_zero_gauge_invariance_with_overlap():
    """H -> H + c S with mu -> mu + c must leave sigma exactly
    unchanged; this pins the -(e_n+e_m)/2 dS/dk velocity term."""
    om = np.linspace(0.5, 3.5, 7)
    m1 = linear_chain(t=-1.0, e0=0.0, s=0.2)
    s1 = sigma_optical(m1, om, mu=0.0, mesh=200, eta=0.1)
    c = 5.0
    m2 = linear_chain(t=-1.0 + c * 0.2, e0=c, s=0.2)   # H + c S
    s2 = sigma_optical(m2, om, mu=c, mesh=200, eta=0.1)
    assert np.abs(s1 - s2).max() < 1e-10


def test_no_absorption_without_occupation_contrast():
    """A completely filled (or empty) band absorbs nothing."""
    m = linear_chain(t=-1.0, e0=0.0)
    om = np.linspace(0.5, 3.0, 6)
    sig = sigma_optical(m, om, mu=10.0, mesh=200, eta=0.1, T=10.0)
    assert np.abs(sig).max() < 1e-12


def test_carrier_count_saturates_at_spin_times_orbitals():
    m = linear_chain(t=-1.0, e0=0.0)
    assert abs(carrier_count(m, mu=10.0, mesh=200) - 2.0) < 1e-9
    assert carrier_count(m, mu=-10.0, mesh=200) < 1e-9


def test_drude_weight_of_the_half_filled_chain_closed_form():
    """D = 8 spin |t| a: the exact delta-function integral over the
    Fermi points of E(k) = 2 t cos ka, against the k-sum with thermal
    smearing."""
    m = linear_chain(t=-1.0, e0=0.0, a=1.0)
    D = drude_weight(m, mu=0.0, mesh=4001, T=40.0, spin=2)
    assert abs(D - 16.0) / 16.0 < 1e-3


def test_drude_weight_vanishes_for_empty_and_filled_bands():
    m = linear_chain(t=-1.0, e0=0.0)
    assert drude_weight(m, mu=-10.0, mesh=400) < 1e-12
    assert drude_weight(m, mu=+10.0, mesh=400) < 1e-12


def test_drude_weight_energy_zero_gauge_invariance_with_overlap():
    c = 5.0
    m1 = linear_chain(t=-1.0, e0=0.0, s=0.2)
    m2 = linear_chain(t=-1.0 + c * 0.2, e0=c, s=0.2)   # H + c S
    D1 = drude_weight(m1, mu=0.0, mesh=800, T=100.0)
    D2 = drude_weight(m2, mu=c, mesh=800, T=100.0)
    assert abs(D1 - D2) < 1e-10 * max(abs(D1), 1.0)


def test_lorentzian_lineshape_molecule_peak():
    """Same transition, Lorentzian broadening: peak value
    spin * 4 pi (a t)^2 / (pi eta * 2|t|), exactly the g(0) = 1/(pi eta)
    replacement of the Gaussian peak."""
    t, a, eta = -0.8, 1.3, 0.05
    m = two_site(t=t, a=a)
    om = np.linspace(1.55, 1.65, 201)
    sig = sigma_optical(m, om, mu=0.0, eta=eta, T=10.0,
                        lineshape="lorentzian")
    expected = 2 * 4.0 * np.pi * (a * t) ** 2 / (np.pi * eta) / (2 * abs(t))
    assert abs(sig.max() - expected) / expected < 1e-3


def test_unknown_lineshape_is_refused():
    import pytest
    m = two_site()
    with pytest.raises(ValueError):
        sigma_optical(m, np.array([1.0]), mu=0.0, lineshape="voigt")
