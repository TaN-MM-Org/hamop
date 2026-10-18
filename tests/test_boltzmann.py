"""Boltzmann transport in the relaxation-time approximation (new in
0.14.0; formulas of G. K. H. Madsen and D. J. Singh, Comput. Phys.
Commun. 175, 67 (2006), Eqs. 9 to 18).  Checked against an adaptive
quadrature of the chain's closed-form dispersion, the Sommerfeld closed
forms of a Dirac cone, the Drude weight, and the rotation of an
anisotropic model."""
import numpy as np
import pytest
from scipy import constants as sc
from scipy.integrate import quad

from hamop import (TightBindingModel, boltzmann_transport, drude_weight,
                   graphene, linear_chain, two_site)
from hamop.model import H_PLANCK, HBAR_EVS, KB

LORENZ0 = np.pi ** 2 / 3 * KB ** 2          # (pi^2/3)(k_B/e)^2, V^2/K^2


def test_planck_constant_is_the_exact_si_value():
    assert H_PLANCK == 6.62607015e-34
    assert H_PLANCK == sc.h
    assert abs(HBAR_EVS / (sc.hbar / sc.e) - 1.0) < 1e-15


def _chain_moments(t, a, mu, T, tau):
    """L_j = integral dk/(2 pi) tau(E) (dE/dk)^2 (E - mu)^j (-df/dE) for
    E = 2t cos(ka), dE/dk = -2ta sin(ka), by adaptive quadrature."""
    kT = KB * T

    def integrand(k, j):
        E = 2 * t * np.cos(k * a)
        x = (E - mu) / kT
        return (tau(E) * (2 * t * a * np.sin(k * a)) ** 2 * (E - mu) ** j
                / (4 * kT * np.cosh(0.5 * x) ** 2))
    return [quad(integrand, -np.pi / a, np.pi / a, args=(j,), limit=400,
                 epsabs=0.0, epsrel=1e-11)[0] / (2 * np.pi)
            for j in range(3)]


@pytest.mark.parametrize("tau_kind", ["constant", "linear"])
def test_chain_matches_adaptive_quadrature(tau_kind):
    """sigma, S and the Lorenz ratio of the chain against an adaptive
    quadrature of the closed-form dispersion (an independent route: no
    eigenvectors, no k-mesh).  The mesh sum of a smooth periodic
    integrand converges faster than any power of the mesh (measured
    agreement 1e-13 at mesh 4000); 1e-9 is allowed.  sigma in S m:
    spin e^2 tau L_0 / hbar^2 with L_0 converted to SI."""
    t, a, T = -1.0, 2.0, 300.0
    if tau_kind == "constant":
        def tau(E):
            return np.full_like(np.asarray(E, dtype=float), 1e-14)
    else:
        def tau(E):
            return 1e-14 * (1.0 + 0.3 * np.asarray(E, dtype=float))
    mus = np.array([-1.2, 0.5, 1.7])
    r = boltzmann_transport(linear_chain(t=t, a=a), mus, T, mesh=4000,
                            tau=tau)
    for i, mu in enumerate(mus):
        L0, L1, L2 = _chain_moments(t, a, mu, T, tau)
        sig = 2 * sc.e ** 2 * L0 * sc.e * 1e-20 / sc.hbar ** 2 / 1e-10
        S = -L1 / (L0 * T)
        lor = (L2 - L1 ** 2 / L0) / (L0 * T * T)
        assert abs(r["sigma"][i][0, 0] / sig - 1) < 1e-9
        assert abs(r["seebeck"][i][0, 0] / S - 1) < 1e-9
        assert abs(r["lorenz"][i][0] / lor - 1) < 1e-9
        assert abs(r["kappa"][i][0, 0]
                   / (lor * T * r["sigma"][i][0, 0]) - 1) < 1e-12
    assert r["units"]["sigma"] == "S m"


def test_graphene_follows_the_dirac_cone_closed_forms():
    """At mu = +/-0.3 eV and 300 K (Dirac point 11.6 kT away) the
    Sommerfeld integrals of a cone are exact: with constant tau,
    sigma(E) ~ |E| gives S = -(pi^2/3)(k_B/e) kT/mu and a Lorenz ratio
    (pi^2/3)(k_B/e)^2 [1 - (pi^2/3)(kT/mu)^2]; with tau ~ |E|,
    sigma(E) ~ E^2 gives S = -(1/T) 2 mu (pi^2/3)(kT)^2 /
    (mu^2 + (pi^2/3)(kT)^2); and sigma = e^2 tau v_F^2 N(mu) / 2 with
    N = 2 |mu| / (pi (hbar v_F)^2) (spin included).  The lattice deviates
    from the cone and the 300 x 300 mesh samples the Fermi circle; the
    measured deviations are below 1e-3 (S and sigma) and 2.5e-3
    (Lorenz), and 1 % is allowed."""
    g = graphene(t=-2.7, a=2.46)
    T, kT = 300.0, KB * 300.0
    mus = np.array([0.3, -0.3])
    r = boltzmann_transport(g, mus, T, mesh=300, tau=1e-14)
    rp = boltzmann_transport(g, mus, T, mesh=300,
                             tau=lambda E: 1e-14 * np.abs(E) / 0.3)
    hv = 1.5 * 2.7 * 2.46 / np.sqrt(3.0)             # hbar v_F, eV Angstrom
    c = np.pi ** 2 / 3
    for i, mu in enumerate(mus):
        S_cone = -c * KB * kT / mu
        lor_cone = LORENZ0 * (1 - c * (kT / mu) ** 2)
        S_p1 = -(2 * mu * c * kT ** 2) / (mu ** 2 + c * kT ** 2) / T
        dos = 2 * abs(mu) / (np.pi * hv ** 2) * 1e20 / sc.e   # 1/(J m^2)
        v = hv * 1e-10 * sc.e / sc.hbar                       # m/s
        sig_cone = sc.e ** 2 * 1e-14 * v ** 2 * dos / 2
        for d in (0, 1):
            assert abs(r["seebeck"][i][d, d] / S_cone - 1) < 1e-2
            assert abs(r["lorenz"][i][d] / lor_cone - 1) < 1e-2
            assert abs(rp["seebeck"][i][d, d] / S_p1 - 1) < 1e-2
            assert abs(r["sigma"][i][d, d] / sig_cone - 1) < 1e-2
        assert abs(r["sigma"][i][0, 1]) < 1e-12 * r["sigma"][i][0, 0]
    assert r["units"]["sigma"] == "S"


def test_constant_tau_conductivity_equals_the_drude_weight():
    """With a constant tau the Boltzmann conductivity is the Drude
    weight spread over a Lorentzian of width hbar/tau, evaluated at
    zero frequency: sigma = D tau / (pi hbar) in units of e^2/(4 hbar).
    Both sum over degenerate pairs; graphene on a mesh through K and K'
    (mesh 30) tests that, and the hexagonal symmetry makes sigma_xx =
    sigma_yy.  Same velocities, different code: 1e-10."""
    g = graphene(t=-2.7, a=2.46)
    tau = 2e-14
    for mu in (0.0, 0.4):
        r = boltzmann_transport(g, mu, 300.0, mesh=30, tau=tau,
                                max_share=1.0)
        for d in (0, 1):
            D = drude_weight(g, mu, mesh=30, T=300.0, direction=d)
            ref = D * tau / (np.pi * HBAR_EVS) * sc.e ** 2 / (4 * sc.hbar)
            assert abs(r["sigma"][d, d] / ref - 1) < 1e-10
        assert abs(r["sigma"][0, 0] - r["sigma"][1, 1]) \
            < 1e-10 * r["sigma"][0, 0]


def _strained(angle):
    """Graphene with one stronger bond (anisotropic), rotated by angle."""
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    g = graphene(t=-2.7, a=2.46)
    m = TightBindingModel(positions=g.positions @ R.T, norb=1,
                          cell=g.cell @ R.T)
    m.add_hop(0, 1, (0, 0), [[-3.3]])
    m.add_hop(0, 1, (-1, 0), [[-2.7]])
    m.add_hop(0, 1, (0, -1), [[-2.7]])
    return m, R


def test_tensors_rotate_with_the_crystal():
    """Rotating an anisotropic crystal rotates its tensors:
    sigma' = R sigma R^T, and the same for S and kappa_e (the mesh is
    the same in fractional coordinates, so the k-points rotate too);
    rounding only, 1e-9 of the largest element."""
    mu, T = 0.6, 600.0
    r0 = boltzmann_transport(_strained(0.0)[0], mu, T, mesh=90)
    m1, R = _strained(0.65)
    r1 = boltzmann_transport(m1, mu, T, mesh=90)
    for key in ("sigma", "seebeck", "kappa"):
        want = R @ r0[key] @ R.T
        assert np.abs(r1[key] - want).max() < 1e-9 * np.abs(want).max()
    assert abs(r0["sigma"][0, 0] / r0["sigma"][1, 1] - 1) > 0.05


def test_constant_tau_drops_out_of_seebeck_and_lorenz():
    """sigma and kappa_e scale with a constant tau, S and the Lorenz
    ratio do not (1e-12 of the largest element: rounding only)."""
    g = graphene(t=-2.7, a=2.46)
    a = boltzmann_transport(g, 0.5, 300.0, mesh=120, tau=1e-14)
    b = boltzmann_transport(g, 0.5, 300.0, mesh=120, tau=3e-14)
    for key, factor in (("sigma", 3.0), ("kappa", 3.0), ("seebeck", 1.0),
                        ("lorenz", 1.0)):
        want = factor * a[key]
        assert np.abs(b[key] - want).max() < 1e-12 * np.abs(want).max()


def test_refusals():
    c = linear_chain(t=-1.0, a=2.0)
    with pytest.raises(ValueError, match="periodic"):
        boltzmann_transport(two_site(), 0.0, 300.0)
    with pytest.raises(ValueError, match="T must be > 0"):
        boltzmann_transport(c, 0.5, 0.0, mesh=100)
    with pytest.raises(ValueError, match="tau"):
        boltzmann_transport(c, 0.5, 300.0, mesh=100, tau=0.0)
    with pytest.raises(ValueError, match="tau"):
        boltzmann_transport(c, 0.5, 300.0, mesh=100,
                            tau=lambda E: 1e-14 * E)
    with pytest.raises(ValueError, match="60 kT"):
        boltzmann_transport(c, 5.0, 300.0, mesh=400)
    with pytest.raises(ValueError, match="too few k-point states"):
        boltzmann_transport(c, 0.5, 30.0, mesh=40)
    flat = TightBindingModel(positions=[[0.0]], norb=1, cell=[[1.0]])
    flat.add_hop(0, 0, (0,), [[0.2]])
    with pytest.raises(ValueError, match="nothing conducts"):
        boltzmann_transport(flat, 0.2, 300.0, mesh=10)
