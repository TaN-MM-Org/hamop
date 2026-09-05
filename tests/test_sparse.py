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


def test_kpm_refuses_periodic_models():
    E = np.array([0.0])
    with pytest.raises(ValueError):
        kpm_dos(linear_chain(), E)


def test_lowest_bands_refuses_full_spectrum_request():
    with pytest.raises(ValueError):
        lowest_bands(_open_chain(10), [None], 10)


# ----------------------------------------------------------------------
# nonorthogonal KPM, sparse velocity, KPM optical conductivity

def test_sparse_derivative_equals_dense_derivative():
    from hamop import bloch_derivative_sparse
    g = graphene()
    k = np.array([0.31, -0.7])
    for direction in (0, 1):
        dHd, _ = g.bloch_derivative(k, direction)
        dHs, dSs = bloch_derivative_sparse(g, k, direction)
        assert np.abs(dHs.toarray() - dHd).max() == 0.0
        assert dSs is None
    c = linear_chain(s=0.2)
    dHd, dSd = c.bloch_derivative(np.array([0.4]), 0)
    dHs, dSs = bloch_derivative_sparse(c, np.array([0.4]), 0)
    assert np.abs(dHs.toarray() - dHd).max() == 0.0
    assert np.abs(dSs.toarray() - dSd).max() == 0.0


def test_kpm_dos_nonorthogonal_chain_closed_form():
    """The generalized spectrum E = 2t cos th / (1 + 2s cos th) has
    dE/dth = -2t at the band center, so the per-site DOS there is
    1/(2 pi |t|) -- same as the orthogonal chain; and the band ends at
    2t/(1 +- 2s), outside of which the DOS must vanish."""
    N, s = 1000, 0.2
    m = _open_chain(N, s=s)
    E = np.linspace(-2.0, 4.0, 1201)
    rho = kpm_dos(m, E, n_moments=300)
    i0 = int(np.argmin(np.abs(E)))
    assert abs(rho[i0] / N - 1.0 / (2.0 * np.pi)) * 2.0 * np.pi < 0.01
    integrate = getattr(np, "trapezoid", getattr(np, "trapz", None))
    assert abs(integrate(rho, E) - N) / N < 0.005
    assert rho[E < -1.6].max() == 0.0          # below 2t/(1+2s) = -1.43
    assert rho[E > 3.6].max() == 0.0           # above 2t/(1-2s) = 3.33


def _ssh_chain(n_cells, t1=-1.0, t2=-0.6, a=2.0):
    pos = []
    for i in range(n_cells):
        pos += [[i * a], [i * a + 0.5 * a]]
    m = TightBindingModel(pos, 1, None)
    for i in range(2 * n_cells):
        m.add_hop(i, i, (0,), [[0.0]])
    for i in range(n_cells):
        m.add_hop(2 * i, 2 * i + 1, (0,), [[t1]])
        if i + 1 < n_cells:
            m.add_hop(2 * i + 1, 2 * i + 2, (0,), [[t2]])
    return m


def test_kpm_sigma_molecule_integrated_weight_closed_form():
    """The kernel broadens the line but conserves its weight: the
    integral of Re sigma over the peak is spin 4 pi (a t)^2 / (2|t|),
    kernel-independent."""
    from hamop import kpm_sigma, two_site
    t, a = -0.8, 1.3
    mol = two_site(t=t, a=a)
    om = np.linspace(0.8, 2.4, 401)
    sig = kpm_sigma(mol, om, mu=0.0, n_moments=256, T=10.0)
    integrate = getattr(np, "trapezoid", getattr(np, "trapz", None))
    W = integrate(sig, om)
    expected = 2.0 * 4.0 * np.pi * (a * t) ** 2 / (2.0 * abs(t))
    assert abs(W - expected) / expected < 0.03


def test_kpm_sigma_matches_the_dense_kubo_route():
    """Same finite dimerized chain through two independent code paths
    (double-Chebyshev KPM vs dense eigenpair Kubo) in the smooth part
    of the spectrum."""
    from hamop import kpm_sigma, sigma_optical
    ch = _ssh_chain(120)
    om = np.array([1.9, 2.2, 2.6])
    sK = kpm_sigma(ch, om, mu=0.0, n_moments=192, T=10.0)
    sD = sigma_optical(ch, om, mu=0.0, eta=0.04, T=10.0)
    assert (np.abs(sK - sD) / np.abs(sD)).max() < 0.02


def test_kpm_sigma_stochastic_trace_is_consistent():
    from hamop import kpm_sigma
    ch = _ssh_chain(120)
    om = np.array([2.2])
    sK = kpm_sigma(ch, om, mu=0.0, n_moments=192, T=10.0)
    sS = kpm_sigma(ch, om, mu=0.0, n_moments=192, T=10.0,
                   n_random=64, seed=3)
    assert abs(sS[0] - sK[0]) / abs(sK[0]) < 0.35


def test_kpm_sigma_refusals():
    from hamop import kpm_sigma, linear_chain as lc
    E = np.array([1.0])
    with pytest.raises(ValueError):
        kpm_sigma(lc(), E, mu=0.0)                   # periodic
    with pytest.raises(ValueError):
        kpm_sigma(_open_chain(10, s=0.2), E, mu=0.0)  # overlap


def test_kpm_sigma_includes_the_dipole_velocity():
    """The exact sparse operator i(HX - XH) reproduces the atomic
    dipole line's kernel-independent integrated weight, and the atom
    stays exactly dark without the dipole block."""
    from hamop import kpm_sigma
    D, d = 1.6, 0.7
    m = TightBindingModel([[0.0]], norb=2, cell=None)
    m.add_hop(0, 0, (0,), [[0.0, 0.0], [0.0, D]])
    om = np.linspace(0.8, 2.4, 401)
    assert np.abs(kpm_sigma(m, np.linspace(1.5, 1.7, 11), mu=0.5 * D,
                            n_moments=64)).max() == 0.0
    X = np.zeros((2, 2, 1), dtype=complex)
    X[0, 1, 0] = X[1, 0, 0] = d
    m.set_dipole(0, X)
    sig = kpm_sigma(m, om, mu=0.5 * D, n_moments=256, T=10.0)
    integrate = getattr(np, "trapezoid", getattr(np, "trapz", None))
    W = integrate(sig, om)
    expected = 2.0 * 4.0 * np.pi * (D * d) ** 2 / D
    assert abs(W - expected) / expected < 0.03
