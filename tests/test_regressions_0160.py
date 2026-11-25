"""Regression tests for the problems fixed in 0.16.0.  Each one fails on
0.15.0 (the old behavior is quoted in the docstrings and in the
CHANGELOG) and passes now.

Most are inputs that gave a silent wrong number: a NaN or an infinite
value let through, or a Chern number returned for bands that touch."""
import numpy as np
import pytest

from hamop import (TightBindingModel, berry_dipole, carrier_count,
                   carrier_density, chern_number, dos, drude_weight,
                   graphene, haldane, kane_mele, landauer_conductance,
                   landauer_current, linear_chain, sigma_optical,
                   thermoelectric)
from hamop.spectrum import _eig_rows


def test_non_finite_blocks_are_refused():
    """0.15.0 accepted a NaN or infinite block: a NaN hopping gave NaN
    bands, and linear_chain(t=inf) an optical conductivity of exactly
    0.0 with no error."""
    m = TightBindingModel([[0.0]], 1, cell=[[1.0]])
    with pytest.raises(ValueError, match="finite"):
        m.add_hop(0, 0, (1,), [[np.nan]])
    with pytest.raises(ValueError, match="finite"):
        m.add_hop(0, 0, (1,), [[-1.0]], [[np.inf]])
    with pytest.raises(ValueError, match="finite"):
        linear_chain(t=np.inf)
    two = TightBindingModel([[0.0, 0.0]], 2)
    with pytest.raises(ValueError, match="finite"):
        two.set_dipole(0, np.full((2, 2, 2), np.nan))
    assert m._hops == [] and two._dipoles == {}


def test_non_finite_mu_and_temperature_are_refused():
    """0.15.0: sigma_optical with mu = NaN returned [0.0], and with a
    NaN photon energy also [0.0]; carrier_count with T = NaN returned
    nan, drude_weight and carrier_density with mu = NaN returned nan;
    dos with eta = inf returned [0.0] and with a NaN energy [nan]."""
    ch = linear_chain()
    with pytest.raises(ValueError, match="mu"):
        sigma_optical(ch, [1.0], np.nan, mesh=10)
    with pytest.raises(ValueError, match="T must be"):
        carrier_count(ch, 0.0, mesh=10, T=np.nan)
    with pytest.raises(ValueError, match="mu"):
        drude_weight(ch, np.nan, mesh=10)
    with pytest.raises(ValueError, match="mu"):
        carrier_density(graphene(), np.nan, 2.0, mesh=12)
    with pytest.raises(ValueError, match="omega"):
        sigma_optical(ch, [np.nan], 0.0, mesh=10)
    with pytest.raises(ValueError, match="eta"):
        dos(ch, [0.0], mesh=10, eta=np.inf)
    with pytest.raises(ValueError, match="energies"):
        dos(ch, [np.nan], mesh=10)
    # finite values work as before
    assert carrier_count(ch, 0.0, mesh=10, T=300.0) == pytest.approx(1.0)


def test_landauer_inputs_are_checked():
    """0.15.0: a NaN chemical potential failed with NumPy's "zero-size
    array to reduction operation maximum", and spin = -2 returned
    G = -2 for a fully open channel.  (An infinite chemical potential
    and a NaN temperature were refused before too, with other
    messages.)"""
    E = np.linspace(-1.0, 1.0, 2001)
    tr = np.ones_like(E)
    with pytest.raises(ValueError, match="finite"):
        landauer_conductance(E, tr, np.nan, 300.0)
    with pytest.raises(ValueError, match="finite"):
        landauer_current(E, tr, 0.0, np.inf, 300.0)
    with pytest.raises(ValueError, match="spin"):
        landauer_conductance(E, tr, 0.0, 300.0, spin=-2)
    with pytest.raises(ValueError, match="T must be"):
        thermoelectric(E, tr, 0.0, np.nan)
    assert landauer_conductance(E, tr, 0.0, 300.0) == pytest.approx(2.0)


def test_berry_dipole_refuses_non_finite_mu():
    """0.15.0 returned D = (nan, nan) for mu = NaN (a NaN kT was
    already refused by method="fermi")."""
    with pytest.raises(ValueError, match="finite"):
        berry_dipole(haldane(), 6, mu=np.nan, kT=0.1)
    with pytest.raises(ValueError, match="finite"):
        berry_dipole(haldane(), 6, mu=0.0, kT=np.nan, method="fermi")


def test_chern_number_refuses_bands_that_touch():
    """The lattice sum gives an integer on any grid.  0.15.0 returned
    0.0 for gapless graphene (its Dirac points lie on a 24 x 24 mesh),
    and for the lowest band of the Kane-Mele model, which is degenerate
    with the second one everywhere, it raised "vanishing link overlap;
    refine the grid", which no grid fixes.  Now the gap is checked at
    every grid point, by both solvers and in both frames."""
    g = graphene(t=-1.0, a=1.0)
    for frame in ("lowdin", "atomic"):
        with pytest.raises(ValueError, match="gap above the 1 occupied"):
            chern_number(g, mesh=24, frame=frame)
    with pytest.raises(ValueError, match="gap above the 1 occupied"):
        chern_number(kane_mele(lam_so=0.06), mesh=12, n_occ=1)
    # the old behavior on request
    assert abs(chern_number(g, mesh=24, gap_min=None)) < 1e-12
    # the sparse solver checks too; a two-site cell is too small for
    # Lanczos with one more state, so a stacked copy is used
    st = TightBindingModel(np.concatenate([g.positions, g.positions]), 1,
                           cell=g.cell)
    for off in (0, 2):
        for i, j, img, Hb, Sb in g._hops:
            st.add_hop(i + off, j + off, img, Hb)
    with pytest.raises(ValueError, match="gap above the 2 occupied"):
        chern_number(st, mesh=12, n_occ=2, solver="sparse")
    # gapped models are unchanged, and all bands need no gap
    h = haldane(t1=-1.0, t2=0.1)
    assert abs(chern_number(h, mesh=18) - 1.0) < 1e-12
    assert abs(chern_number(h, mesh=18, n_occ=2)) < 1e-12
    assert abs(chern_number(kane_mele(lam_so=0.06), n_occ=2)) < 1e-12
    with pytest.raises(ValueError, match="gap_min"):
        chern_number(h, gap_min=-1.0)


def _dense_dos(model, E, mesh, eta, thresh=1e-10):
    """The Gaussian sum of 0.15.0, every energy against every state."""
    k, w = model.monkhorst_pack(mesh)
    rows = _eig_rows(model, k, thresh)
    ev = np.concatenate(rows)
    wv = np.repeat(w, [len(r) for r in rows])
    out = np.zeros(len(E))
    for s in range(0, ev.size, 2000):
        x = (E[:, None] - ev[None, s:s + 2000]) / eta
        out += np.exp(-0.5 * x * x) @ wv[s:s + 2000]
    return out / (eta * np.sqrt(2.0 * np.pi))


def test_windowed_dos_equals_the_full_gaussian_sum():
    """Since 0.16.0 `dos` adds, for each energy, only the states within
    9 Gaussian widths (20 to 28 times faster for graphene with 2001
    energies on a 300 x 300 mesh).  The terms left out are below 2.6e-18 each,
    so the result equals the full sum to rounding: graphene at three
    widths (one wider than the band), unsorted and repeated energies,
    and a model whose overlap drops a state at some k-points (a band
    count that changes with k)."""
    g = graphene()
    for E, mesh, eta in ((np.linspace(-9.0, 9.0, 801), 60, 0.05),
                         (np.linspace(-9.0, 9.0, 101), 30, 20.0),
                         (np.array([8.1, -2.0, 0.3, 0.3, -12.0]), 40, 0.01)):
        new = dos(g, E, mesh=mesh, eta=eta)
        ref = _dense_dos(g, E, mesh, eta)
        # measured 4e-15 relative to the largest value
        assert np.abs(new - ref).max() <= 1e-13 * ref.max()
    # band count that changes with k (the 0.14.0 regression model)
    m = TightBindingModel(positions=[[0.0]], norb=2, cell=[[1.0]])
    m.add_hop(0, 0, (0,), [[0.0, -0.5], [-0.5, 1.0]],
              [[1.0, 0.5], [0.5, 1.0]])
    m.add_hop(0, 0, (1,), [[-0.3, -0.2], [0.0, -0.3]],
              [[0.0, 0.5], [0.0, 0.0]])
    rows = _eig_rows(m, m.monkhorst_pack(24)[0], 1e-10)
    assert len({len(r) for r in rows}) > 1
    E = np.linspace(-3.0, 3.0, 61)
    ref = _dense_dos(m, E, 24, 0.1)
    assert np.abs(dos(m, E, mesh=24, eta=0.1) - ref).max() \
        <= 1e-13 * ref.max()
