"""Regression tests for the bugs fixed in 0.13.0.  Each test fails on
0.12.1 (the failing numbers are quoted) and passes now."""
import numpy as np
import pytest

from hamop import (TightBindingModel, chain_lead_blocks, drude_weight,
                   graphene, kpm_dos, kpm_sigma, principal_layers,
                   transmission, two_site)
from hamop.model import KB


def _dirac_point(g):
    b = 2.0 * np.pi * np.linalg.inv(g.cell).T
    return (2.0 * b[0] + b[1]) / 3.0


def test_drude_weight_at_a_degenerate_k_point_is_basis_independent():
    """At graphene's Dirac point K both bands have the same energy, so
    the intraband sum must run over the degenerate pair, not only over
    the diagonal of whatever eigenvectors the solver returns.

    With mu = 0 and only K on the grid the exact value follows without
    eigenvectors: the degenerate set is the whole space, so the sum is
    Tr[(dH/dk)^2] (-df/de at e = mu, which is 1/(4 kT)).  0.12.1 gave
    3068.84 for x but 0.0 for y; and on a 30 x 30 mesh (through K and
    K') D_xx = 3.5382 and D_yy = 3.2815, 7 % apart, where the hexagonal
    symmetry makes them equal.  Now both directions give the trace form,
    and D_xx = D_yy = 6.8196 on the mesh."""
    g = graphene(t=-2.7, a=2.46)
    K = _dirac_point(g)
    T = 300.0
    for d in (0, 1):
        D = drude_weight(g, 0.0, kpts=[K], weights=[1.0], T=T, direction=d)
        dH, _ = g.bloch_derivative(K, d)
        exact = (2 * 4.0 * np.pi / g.cell_volume
                 * np.real(np.trace(dH @ dH)) / (4.0 * KB * T))
        assert abs(D / exact - 1.0) < 1e-12
    Dx = drude_weight(g, 0.0, mesh=30, T=T, direction=0)
    Dy = drude_weight(g, 0.0, mesh=30, T=T, direction=1)
    assert abs(Dx - Dy) < 1e-12 * Dx


def test_drude_weight_away_from_degeneracies_is_unchanged():
    """On a mesh that misses K (31 x 31) the new sum has no degenerate
    pairs; the result is the plain diagonal sum, computed here
    independently one k-point at a time."""
    g = graphene(t=-2.7, a=2.46)
    kpts, w = g.monkhorst_pack(31)
    T, mu = 300.0, 0.3
    ref = 0.0
    for k, wk in zip(kpts, w):
        H, _ = g.bloch(k)
        dH, _ = g.bloch_derivative(k, 0)
        e, c = np.linalg.eigh(H)
        v = np.real(np.diag(c.conj().T @ dH @ c))
        x = (e - mu) / (KB * T)
        ref += wk * np.sum(v ** 2 / (4 * KB * T * np.cosh(0.5 * x) ** 2))
    ref *= 2 * 4.0 * np.pi / g.cell_volume
    D = drude_weight(g, mu, mesh=31, T=T)
    assert abs(D / ref - 1.0) < 1e-12
    with pytest.raises(ValueError):
        drude_weight(g, mu, mesh=31, T=T, degen_tol=-1.0)


def _spaced_chain(n, a, eps_mid=None):
    m = TightBindingModel(positions=[[i * a] for i in range(n)], norb=1)
    for i in range(n):
        e = eps_mid if (eps_mid is not None and i == n // 2) else 0.0
        m.add_hop(i, i, (0,), [[e]])
    for i in range(n - 1):
        m.add_hop(i, i + 1, (0,), [[-1.0]])
    return m


def test_principal_layers_with_a_non_integer_spacing():
    """A chain of spacing a = 1.42 cut into layers of width a: 0.12.1
    put site 15 in layer 14 because 15 * 1.42 / 1.42 rounds to
    14.999999999999998, and then refused the valid partition ("layers
    14 and 16 couple"); with width 2a the layers came out with 3 and 1
    sites.  Now: 40 one-site layers (20 two-site layers), and the
    impurity chain reproduces its closed-form transmission."""
    a = 1.42
    m = _spaced_chain(40, a)
    P = principal_layers(m, a)
    assert [len(h) for h in P["layers_H"]] == [1] * 40
    P2 = principal_layers(m, 2 * a)
    assert [len(h) for h in P2["layers_H"]] == [2] * 20
    eps = 0.8
    P = principal_layers(_spaced_chain(41, a, eps_mid=eps), a)
    H00, H01 = chain_lead_blocks(t=-1.0)
    E = np.array([-1.2, 0.3, 0.7])
    T = transmission(E, P["layers_H"], P["coup_H"], H00, H01, eta=1e-8)
    v2 = 4.0 - E ** 2
    assert np.abs(T - v2 / (v2 + eps ** 2)).max() < 1e-5
    for bad in (0.0, -1.0, np.inf):
        with pytest.raises(ValueError):
            principal_layers(m, bad)


def test_kpm_refuses_bounds_that_cut_the_spectrum():
    """The 200-site chain has its spectrum in (-2, 2).  With bounds
    (-1.9, 1.9) 0.12.1 returned a DOS of -4.7e7 at E = -1 (and 5e52 with
    (-1, 1)); now the growing moments are detected and refused.
    Correct bounds still work."""
    n = 200
    m = _spaced_chain(n, 1.0)
    E = np.array([-1.0, 0.0, 1.0])
    rho = kpm_dos(m, E, n_moments=100, bounds=(-2.1, 2.1))
    assert np.all(rho > 0)
    for b in ((-1.9, 1.9), (-1.0, 1.0)):
        with pytest.raises(ValueError, match="moments"):
            kpm_dos(m, E, n_moments=100, bounds=b)
        with pytest.raises(ValueError, match="moments"):
            kpm_dos(m, E, n_moments=100, bounds=b, n_random=4)
    with pytest.raises(ValueError, match="moments"):
        kpm_sigma(two_site(t=-1.0), [2.0], mu=0.0, n_moments=64,
                  bounds=(-0.5, 0.5))
