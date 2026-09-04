"""The conductivity tensor against its anchors: TKNN quantization of
sigma_xy(0) with the package's own Chern number (sign included), zero
Hall response in the trivial phase, exact antisymmetry, agreement of
the longitudinal component with sigma_optical, and exact invariance
under a shift of the energy zero in a nonorthogonal basis."""
import numpy as np

from hamop import (chern_number, haldane, linear_chain, sigma_optical,
                   sigma_tensor, two_site)


def test_tknn_sigma_xy_equals_chern_number_sign_included():
    """sigma_xy(0) = C e^2/h = (2 C / pi) in units of e^2/(4 hbar),
    with C the package's chern_number of the occupied band (Thouless,
    Kohmoto, Nightingale and den Nijs, PRL 49, 405 (1982)).  Asserted
    for both flux directions, so the sign relation is pinned too."""
    for phi in (+np.pi / 2, -np.pi / 2):
        h = haldane(t1=-1.0, t2=0.1, phi=phi, m_ab=0.0)
        C = chern_number(h, mesh=18)
        sxy = sigma_tensor(h, [0.0], mu=0.0, directions=(0, 1), mesh=48,
                           T=10.0, eta=1e-4, spin=1)[0]
        assert abs(sxy.real * np.pi / 2.0 - C) < 1e-6
        assert abs(sxy.imag) < 1e-8


def test_sigma_xy_vanishes_in_the_trivial_phase():
    h = haldane(t1=-1.0, t2=0.1, phi=np.pi / 2, m_ab=0.9)
    sxy = sigma_tensor(h, [0.0], mu=0.0, directions=(0, 1), mesh=48,
                       T=10.0, eta=1e-4, spin=1)[0]
    assert abs(sxy) < 1e-6


def test_hall_tensor_is_antisymmetric():
    h = haldane()
    kw = dict(mu=0.0, mesh=24, T=10.0, eta=1e-3, spin=1)
    sxy = sigma_tensor(h, [0.4], directions=(0, 1), **kw)[0]
    syx = sigma_tensor(h, [0.4], directions=(1, 0), **kw)[0]
    assert abs(sxy + syx) < 1e-12


def test_longitudinal_component_matches_sigma_optical():
    """Re sigma_xx from the tensor equals the Lorentzian sigma_optical
    up to the antiresonant terms the latter drops, O(eta/omega)."""
    t, a, eta = -0.8, 1.3, 0.05
    m = two_site(t=t, a=a)
    om = np.linspace(1.4, 1.8, 21)
    ref = sigma_optical(m, om, 0.0, eta=eta, T=10.0,
                        lineshape="lorentzian")
    gen = np.real(sigma_tensor(m, om, 0.0, directions=(0, 0), eta=eta,
                               T=10.0))
    assert np.abs(gen - ref).max() / ref.max() < 1e-3


def test_tensor_energy_zero_gauge_invariance_with_overlap():
    c = 5.0
    m1 = linear_chain(t=-1.0, e0=0.0, s=0.2)
    m2 = linear_chain(t=-1.0 + c * 0.2, e0=c, s=0.2)   # H + c S
    s1 = sigma_tensor(m1, [1.1], 0.0, directions=(0, 0), mesh=150, eta=0.1)
    s2 = sigma_tensor(m2, [1.1], c, directions=(0, 0), mesh=150, eta=0.1)
    assert np.abs(s1 - s2).max() < 1e-10
