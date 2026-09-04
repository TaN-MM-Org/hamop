"""Peierls substitution against exact statements: the closed-form
spectrum of a flux-threaded ring, machine-precision gauge invariance
(the midpoint rule is exact for linear gauges), exact plaquette flux,
exact periodicity in the flux quantum, and the refusal of periodic
models."""
import numpy as np
import pytest

from hamop import TightBindingModel, bands, linear_chain, with_peierls


def _ring(N=12, R=3.0, t=-1.0):
    ang = 2.0 * np.pi * np.arange(N) / N
    pos = np.stack([R * np.cos(ang), R * np.sin(ang)], axis=1)
    m = TightBindingModel(pos, 1, None)
    for i in range(N):
        m.add_hop(i, i, (0, 0), [[0.0]])
    for i in range(N):
        m.add_hop(i, (i + 1) % N, (0, 0), [[t]])
    return m


def _square(nx, ny, t=-1.0):
    pos = [[float(i), float(j)] for i in range(nx) for j in range(ny)]
    m = TightBindingModel(pos, 1, None)
    idx = lambda i, j: i * ny + j
    for i in range(nx):
        for j in range(ny):
            m.add_hop(idx(i, j), idx(i, j), (0, 0), [[0.0]])
            if i + 1 < nx:
                m.add_hop(idx(i, j), idx(i + 1, j), (0, 0), [[t]])
            if j + 1 < ny:
                m.add_hop(idx(i, j), idx(i, j + 1), (0, 0), [[t]])
    return m


def test_flux_threaded_ring_reproduces_the_closed_form():
    """E_j = 2 t cos((2 pi j + Theta) / N) with Theta = 2 pi phi times
    the polygon area -- exact, because the midpoint rule integrates a
    linear gauge exactly."""
    N, R, t, phi = 12, 3.0, -1.0, 0.013
    ring = with_peierls(_ring(N, R, t), phi, gauge="symmetric")
    e = bands(ring, [None])[0]
    area = 0.5 * N * R ** 2 * np.sin(2.0 * np.pi / N)
    theta = 2.0 * np.pi * phi * area
    exact = np.sort(2.0 * t * np.cos((2.0 * np.pi * np.arange(N) + theta)
                                     / N))
    assert np.abs(e - exact).max() < 1e-12


def test_gauge_invariance_to_machine_precision():
    m = _square(5, 4)
    phi = 0.021
    e1 = bands(with_peierls(m, phi, gauge="landau"), [None])[0]
    e2 = bands(with_peierls(m, phi, gauge="symmetric"), [None])[0]
    assert np.abs(e1 - e2).max() < 1e-12


def test_plaquette_flux_is_exact():
    """The product of the four bond phases around a unit plaquette is
    exactly exp(2 pi i phi) -- the defining property."""
    phi = 0.07
    m = with_peierls(_square(2, 2), phi, gauge="landau")
    hop = {}
    for i, j, image, Hb, Sb in m._hops:
        if i != j:
            hop[(i, j)] = Hb[0, 0]
            hop[(j, i)] = np.conj(Hb[0, 0])
    # sites: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3; loop 0->2->3->1->0
    prod = hop[(0, 2)] * hop[(2, 3)] * hop[(3, 1)] * hop[(1, 0)]
    prod /= abs(prod)
    assert abs(prod - np.exp(2j * np.pi * phi)) < 1e-12


def test_spectrum_is_periodic_in_the_flux_quantum():
    N, R = 10, 2.0
    ring = _ring(N, R)
    area = 0.5 * N * R ** 2 * np.sin(2.0 * np.pi / N)
    phi = 0.004
    e1 = np.sort(bands(with_peierls(ring, phi), [None])[0])
    e2 = np.sort(bands(with_peierls(ring, phi + 1.0 / area), [None])[0])
    assert np.abs(e1 - e2).max() < 1e-12


def test_periodic_models_are_refused():
    with pytest.raises(ValueError):
        with_peierls(linear_chain(), 0.01)


def test_lowest_landau_level_of_the_square_lattice():
    """Physics anchor: near the band bottom the square lattice has
    effective mass hbar^2/(2|t|a^2), so the lowest Landau level sits at
    -4|t| + hbar omega_c / 2 with hbar omega_c = 4 pi |t| phi a^2, and
    it is macroscopically degenerate (phi per plaquette times the
    number of plaquettes).  Lattice and edge corrections are small at
    this phi; asserted to 3%, with the degeneracy of the first six
    states resolved to well below the level spacing."""
    from hamop import lowest_bands
    phi = 0.01
    m = with_peierls(_square(40, 40), phi, gauge="landau")
    e = lowest_bands(m, [None], 6)[0]
    hwc = 4.0 * np.pi * phi
    assert abs((e[0] + 4.0) - 0.5 * hwc) / (0.5 * hwc) < 0.03
    assert e[5] - e[0] < 0.1 * hwc            # first LL degeneracy
