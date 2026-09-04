"""Spectral observables against closed forms: the chain DOS, DOS
normalization, half-filling chemical potential, Dirac-point closure."""
import numpy as np

from hamop import (bands, dos, fermi_level, band_edges, graphene,
                   k_path, linear_chain)


def test_chain_dos_matches_analytic_band_center():
    """rho(E) = 1 / (pi sqrt(4 t^2 - E^2)); at the band center that is
    1 / (2 pi |t|)."""
    m = linear_chain(t=-1.0, e0=0.0)
    rho = dos(m, np.array([0.0]), mesh=4001, eta=0.02)
    assert abs(rho[0] - 1.0 / (2.0 * np.pi)) < 5e-4


def test_dos_integrates_to_orbital_count():
    m = linear_chain(t=-1.0, e0=0.0)
    E = np.linspace(-4.0, 4.0, 1601)
    rho = dos(m, E, mesh=301, eta=0.05)
    integrate = getattr(np, "trapezoid", getattr(np, "trapz", None))
    assert abs(integrate(rho, E) - 1.0) < 1e-6


def test_half_filled_chain_fermi_level_is_band_center():
    """Particle-hole symmetry pins mu at e0 for half filling."""
    m = linear_chain(t=-1.0, e0=0.25)
    mu = fermi_level(m, filling=0.5, mesh=800, T=300.0)
    assert abs(mu - 0.25) < 1e-6


def test_graphene_dirac_point_and_bandwidth():
    """E(K) = 0 exactly; E(Gamma) = -/+ 3 |t| exactly."""
    t, a = -2.7, 2.46
    g = graphene(t=t, a=a)
    recip = 2.0 * np.pi * np.linalg.inv(g.cell).T
    K = (2.0 * recip[0] + recip[1]) / 3.0
    eK = bands(g, [K])[0]
    assert abs(eK[1] - eK[0]) < 1e-9
    eG = bands(g, [np.zeros(2)])[0]
    assert np.allclose(eG, [-3.0 * abs(t), 3.0 * abs(t)], atol=1e-12)


def test_band_edges_report_the_ssh_gap():
    from hamop import TightBindingModel
    t1, t2, a = -1.0, -0.6, 2.0
    m = TightBindingModel(positions=[[0.0], [0.7]], norb=1, cell=[[a]])
    m.add_hop(0, 1, (0,), [[t1]])
    m.add_hop(1, 0, (1,), [[t2]])
    vbm, cbm, gap = band_edges(m, mu=0.0, mesh=2001)
    assert abs(gap - 2.0 * abs(abs(t1) - abs(t2))) < 1e-5


def test_k_path_geometry():
    kpts, dists, ticks = k_path([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                                n_per_segment=10)
    assert kpts.shape == (21, 2)
    assert np.allclose(kpts[0], [0.0, 0.0])
    assert np.allclose(kpts[10], [1.0, 0.0])
    assert np.allclose(kpts[-1], [1.0, 1.0])
    assert abs(dists[-1] - 2.0) < 1e-12
    assert np.allclose(ticks, [0.0, 1.0, 2.0])
