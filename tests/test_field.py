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


# ----------------------------------------------------------------------
# Hofstadter magnetic supercells (periodic systems at rational flux)

def _square_lattice(t=-1.0, a=1.0):
    m = TightBindingModel([[0.0, 0.0]], 1, cell=[[a, 0.0], [0.0, a]])
    m.add_hop(0, 0, (0, 0), [[0.0]])
    m.add_hop(0, 0, (1, 0), [[t]])
    m.add_hop(0, 0, (0, 1), [[t]])
    return m


def test_zero_flux_supercell_is_exact_band_folding():
    from hamop import magnetic_supercell
    sq = _square_lattice()
    sc = magnetic_supercell(sq, 0, 3)
    k = np.array([0.31, -0.7])
    recip = 2.0 * np.pi * np.linalg.inv(sq.cell).T
    e_sc = np.sort(bands(sc, [k])[0])
    folded = np.sort(np.concatenate(
        [bands(sq, [k + j * recip[0] / 3.0])[0] for j in range(3)]))
    assert np.abs(e_sc - folded).max() < 1e-12


def test_pi_flux_square_lattice_closed_form():
    """At flux 1/2 the magnetic bands are E = +-2|t| sqrt(cos^2 kx +
    cos^2 ky) -- derivable by hand from the two-site magnetic cell."""
    from hamop import magnetic_supercell
    sc = magnetic_supercell(_square_lattice(), 1, 2)
    for kx, ky in [(0.3, 0.4), (-0.9, 1.1), (0.0, 0.0)]:
        e = np.sort(bands(sc, [[kx, ky]])[0])
        exact = 2.0 * np.sqrt(np.cos(kx) ** 2 + np.cos(ky) ** 2)
        assert np.abs(e - np.array([-exact, exact])).max() < 1e-12


def test_hofstadter_band_count_and_tknn_consistency():
    """phi = 1/3: three magnetic bands, the lowest carrying a unit
    Chern number that must agree -- sign included -- with sigma_xy(0)
    computed on the same magnetic cell with mu in the first gap: the
    TKNN circle closed entirely inside the package."""
    from hamop import chern_number, magnetic_supercell, sigma_tensor
    sc = magnetic_supercell(_square_lattice(), 1, 3)
    assert bands(sc, [[0.1, 0.2]])[0].shape == (3,)
    C = chern_number(sc, mesh=15, n_occ=1)
    assert abs(abs(C) - 1.0) < 1e-12
    sxy = sigma_tensor(sc, [0.0], mu=-1.3, directions=(0, 1), mesh=36,
                       T=10.0, eta=1e-4, spin=1)[0]
    assert abs(sxy.real * np.pi / 2.0 - C) < 1e-4


def test_incompatible_gauge_is_refused_with_a_remedy():
    """Graphene's sublattice offset has fractional a2-coordinate 1/3,
    so p = 1 cannot be represented in this gauge -- refused -- while
    the equivalent flux 3/9 works and p = 0 folds exactly."""
    from hamop import graphene as gr, magnetic_supercell
    g = gr()
    with pytest.raises(ValueError):
        magnetic_supercell(g, 1, 3)
    sc = magnetic_supercell(g, 3, 9)          # same flux, valid gauge
    assert bands(sc, [[0.1, 0.2]])[0].shape == (18,)
    sc0 = magnetic_supercell(g, 0, 2)
    k = np.array([0.21, -0.33])
    recip = 2.0 * np.pi * np.linalg.inv(g.cell).T
    e_sc = np.sort(bands(sc0, [k])[0])
    folded = np.sort(np.concatenate(
        [bands(g, [k + j * recip[0] / 2.0])[0] for j in range(2)]))
    assert np.abs(e_sc - folded).max() < 1e-12
