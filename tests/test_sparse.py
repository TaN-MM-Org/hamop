"""Sparse assembly and large-system solvers against exact anchors: the
sparse Bloch matrices equal the dense ones element for element, the
Lanczos band solver reproduces the open chain's closed-form spectrum
2 t cos(pi j / (N+1)) (nonorthogonal variant included), and the KPM
density of states matches the chain's closed form and integrates to
the orbital count."""
import numpy as np
import pytest

from hamop import (TightBindingModel, bloch_sparse, graphene, kpm_dos,
                   linear_chain, lowest_bands)


def _open_chain(n, t=-1.0, s=None):
    m = TightBindingModel([[float(i)] for i in range(n)], 1, None)
    for i in range(n):
        m.add_hop(i, i, (0,), [[0.0]], None if s is None else [[1.0]])
    for i in range(n - 1):
        m.add_hop(i, i + 1, (0,), [[t]], None if s is None else [[s]])
    return m


def test_sparse_assembly_equals_dense_assembly():
    g = graphene()
    k = np.array([0.31, -0.7])
    Hd, _ = g.bloch(k)
    Hs, Ss = bloch_sparse(g, k)
    assert np.abs(Hs.toarray() - Hd).max() == 0.0
    assert Ss is None                        # orthogonal: identity implied
    c = linear_chain(s=0.2)
    k = np.array([0.4])
    Hd, Sd = c.bloch(k)
    Hs, Ss = bloch_sparse(c, k)
    assert np.abs(Hs.toarray() - Hd).max() == 0.0
    assert np.abs(Ss.toarray() - Sd).max() == 0.0


def test_lowest_bands_match_the_open_chain_closed_form():
    """Open-boundary chain: E_j = 2 t cos(pi j / (N+1)) exactly."""
    N = 300
    e = lowest_bands(_open_chain(N), [None], 5)[0]
    j = np.arange(1, N + 1)
    exact = np.sort(2.0 * (-1.0) * np.cos(np.pi * j / (N + 1)))[:5]
    assert np.abs(e - exact).max() < 1e-8


def test_lowest_bands_generalized_nonorthogonal_closed_form():
    """The tridiagonal H and S share eigenvectors, so
    E_j = 2 t cos(theta_j) / (1 + 2 s cos(theta_j)) exactly."""
    N, s = 300, 0.2
    e = lowest_bands(_open_chain(N, s=s), [None], 5)[0]
    th = np.pi * np.arange(1, N + 1) / (N + 1)
    exact = np.sort(2.0 * (-1.0) * np.cos(th) / (1.0 + 2.0 * s
                                                 * np.cos(th)))[:5]
    assert np.abs(e - exact).max() < 1e-8


def test_lowest_bands_periodic_matches_dense():
    """Ten-site periodic supercell of the chain: the sparse low-energy
    window must agree with the dense solver at arbitrary k."""
    from hamop import bands
    n = 10
    m = TightBindingModel([[float(i)] for i in range(n)], 1,
                          cell=[[float(n)]])
    for i in range(n - 1):
        m.add_hop(i, i + 1, (0,), [[-1.0]])
    m.add_hop(n - 1, 0, (1,), [[-1.0]])
    kpts = [np.array([0.07]), np.array([-0.19])]
    e_sparse = lowest_bands(m, kpts, 3)
    e_dense = bands(m, kpts)
    assert np.abs(e_sparse - e_dense[:, :3]).max() < 1e-8


def test_kpm_dos_matches_the_chain_closed_form_and_normalization():
    """rho(0)/N -> 1/(2 pi |t|); the integral equals the orbital count
    (mu_0 is the exact trace of T_0 = 1)."""
    N = 1000
    m = _open_chain(N)
    E = np.linspace(-2.5, 2.5, 1001)
    rho = kpm_dos(m, E, n_moments=300)
    i0 = int(np.argmin(np.abs(E)))
    assert abs(rho[i0] / N - 1.0 / (2.0 * np.pi)) * 2.0 * np.pi < 0.01
    integrate = getattr(np, "trapezoid", getattr(np, "trapz", None))
    assert abs(integrate(rho, E) - N) / N < 0.005


def test_kpm_stochastic_trace_agrees_with_deterministic():
    N = 1000
    m = _open_chain(N)
    E = np.linspace(-2.5, 2.5, 201)
    rho_det = kpm_dos(m, E, n_moments=200)
    rho_sto = kpm_dos(m, E, n_moments=200, n_random=32, seed=1)
    i0 = int(np.argmin(np.abs(E)))
    assert abs(rho_sto[i0] - rho_det[i0]) / rho_det[i0] < 0.25


def test_kpm_refuses_periodic_and_nonorthogonal_models():
    E = np.array([0.0])
    with pytest.raises(ValueError):
        kpm_dos(linear_chain(), E)
    with pytest.raises(ValueError):
        kpm_dos(_open_chain(10, s=0.2), E)


def test_lowest_bands_refuses_full_spectrum_request():
    with pytest.raises(ValueError):
        lowest_bands(_open_chain(10), [None], 10)
