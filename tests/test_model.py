"""Model assembly: Hermiticity, exact dispersions, hand-built lattices."""
import numpy as np
import pytest

from hamop import TightBindingModel, bands, gen_eigh, linear_chain


def test_chain_dispersion_is_exact():
    m = linear_chain(t=-1.0, e0=0.3, a=1.0)
    ks = np.linspace(-np.pi, np.pi, 41)
    e = bands(m, [[k] for k in ks])[:, 0]
    assert np.abs(e - (0.3 - 2.0 * np.cos(ks))).max() < 1e-12


def test_nonorthogonal_chain_dispersion_is_exact():
    """E(k) = (e0 + 2 t cos ka) / (1 + 2 s cos ka), the textbook
    single-band nonorthogonal chain."""
    t, s = -1.0, 0.2
    m = linear_chain(t=t, e0=0.0, a=1.0, s=s)
    ks = np.linspace(-2.0, 2.0, 17)
    e = bands(m, [[k] for k in ks])[:, 0]
    exact = 2.0 * t * np.cos(ks) / (1.0 + 2.0 * s * np.cos(ks))
    assert np.abs(e - exact).max() < 1e-12


def test_bloch_matrices_are_hermitian():
    rng = np.random.default_rng(0)
    cell = np.array([[2.0, 0.1], [-0.3, 1.7]])
    m = TightBindingModel(positions=[[0.0, 0.0], [0.9, 0.6]],
                          norb=[2, 3], cell=cell)
    on0 = rng.normal(size=(2, 2)); on0 = on0 + on0.T
    on1 = rng.normal(size=(3, 3)); on1 = on1 + on1.T
    m.add_hop(0, 0, (0, 0), on0)
    m.add_hop(1, 1, (0, 0), on1)
    m.add_hop(0, 1, (0, 0), rng.normal(size=(2, 3)))
    m.add_hop(0, 1, (-1, 0), rng.normal(size=(2, 3)))
    m.add_hop(0, 0, (1, 0), rng.normal(size=(2, 2)))
    for k in ([0.0, 0.0], [0.3, -1.1], [2.2, 0.7]):
        H, S = m.bloch(k)
        dH, dS = m.bloch_derivative(k, 0)
        assert np.abs(H - H.conj().T).max() < 1e-13
        assert np.abs(dH - dH.conj().T).max() < 1e-13


def test_ssh_chain_gap_closed_form():
    """Dimerized chain: gap = 2 |t1 - t2| at the zone boundary."""
    t1, t2, a = -1.0, -0.6, 2.0
    m = TightBindingModel(positions=[[0.0], [0.7]], norb=1, cell=[[a]])
    m.add_hop(0, 1, (0,), [[t1]])
    m.add_hop(1, 0, (1,), [[t2]])
    e = bands(m, [[np.pi / a]])[0]
    assert np.isclose(e[1] - e[0], 2.0 * abs(abs(t1) - abs(t2)),
                      atol=1e-12)


def test_gen_eigh_matches_scipy_for_well_conditioned_overlap():
    from scipy.linalg import eigh
    rng = np.random.default_rng(1)
    n = 7
    H = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H = H + H.conj().T
    B = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    S = B @ B.conj().T + n * np.eye(n)
    assert np.abs(gen_eigh(H, S) - eigh(H, S, eigvals_only=True)).max() \
        < 1e-10


def test_gen_eigh_drops_null_space_cleanly():
    """A rank-deficient overlap must give finite eigenvalues in the
    kept subspace instead of blowing up."""
    rng = np.random.default_rng(2)
    n = 6
    H = rng.normal(size=(n, n)); H = H + H.T
    U = np.linalg.qr(rng.normal(size=(n, n)))[0]
    s = np.array([2.0, 1.5, 1.0, 0.7, 1e-14, 1e-15])
    S = U @ np.diag(s) @ U.T
    w = gen_eigh(H, S, thresh=1e-6)
    assert len(w) == 4 and np.all(np.isfinite(w)) and np.abs(w).max() < 1e3


def test_input_validation():
    m = TightBindingModel(positions=[[0.0]], norb=1, cell=None)
    with pytest.raises(ValueError):
        m.add_hop(0, 0, (1,), [[1.0]])          # image in finite system
    with pytest.raises(ValueError):
        m.add_hop(0, 0, (0,), [[1.0j]])         # non-Hermitian on-site
    mp = linear_chain()
    with pytest.raises(ValueError):
        mp.add_hop(0, 0, (0,), [[0.0, 0.0]])    # wrong block shape
