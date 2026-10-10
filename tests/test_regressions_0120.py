"""Regression tests for the problems fixed in 0.12.0.  Each test failed
(or the code returned a wrong number without an error) on 0.11.0."""
import os

import numpy as np
import pytest

from hamop import (TightBindingModel, bands, dos, load_wannier90_hr,
                   save_wannier90_hr, sigma_optical, symmetry_fold)


def _chain_with_dipole(x01):
    """One site, two orbitals, real hopping blocks, and an intra-atomic
    dipole block with off-diagonal element x01 (real or complex)."""
    m = TightBindingModel([[0.0]], 2, cell=[[2.0]])
    m.add_hop(0, 0, (0,), np.diag([0.0, 1.5]))
    m.add_hop(0, 0, (1,), [[-1.0, 0.4], [-0.2, 0.3]])
    X = np.zeros((2, 2, 1), dtype=complex)
    X[0, 1, 0] = x01
    X[1, 0, 0] = np.conj(x01)
    m.set_dipole(0, X)
    return m


def _hand_folded_grid(n, a):
    """The k <-> -k fold of the Gamma-centered n-point chain grid, built
    by hand (independently of monkhorst_pack)."""
    ks, ws = [], []
    for i in range(n // 2 + 1):
        ks.append([2.0 * np.pi / a * i / n])
        ws.append((1.0 if (2 * i) % n == 0 else 2.0) / n)
    return np.array(ks), np.array(ws)


def test_time_reversal_fold_refuses_complex_dipole_blocks():
    """monkhorst_pack(time_reversal=True) checked only the H and S
    blocks.  A complex (Hermitian) dipole block keeps the bands even in
    k but not the optical matrix elements, so the folded grid gave a
    wrong sigma_xx for the model below: 1.9489 instead of 2.0963 at
    1 eV (7 % low) and 2.6526 instead of 3.0159 at 2 eV (12 % low).
    The fold is now refused for such a model."""
    w = np.array([1.0, 2.0, 3.0])
    k_tr, w_tr = _hand_folded_grid(40, 2.0)

    m = _chain_with_dipole(0.3j)
    full = sigma_optical(m, w, mu=0.5, mesh=40, eta=0.1)
    folded = sigma_optical(m, w, mu=0.5, kpts=k_tr, weights=w_tr, eta=0.1)
    # the fold really is wrong for this model (what 0.11.0 returned)
    assert np.abs(folded - full).max() > 0.05 * full.max()
    with pytest.raises(ValueError, match="complex intra-atomic"):
        m.monkhorst_pack(40, time_reversal=True)
    # eigenvalue-only quantities can still be folded, with the spectrum
    # verified numerically by symmetry_fold
    k_s, w_s = symmetry_fold(m, 40, [np.eye(1)], time_reversal=True)
    E = np.linspace(-3.0, 4.0, 15)
    assert np.abs(dos(m, E, kpts=k_s, weights=w_s)
                  - dos(m, E, mesh=40)).max() < 1e-12

    # with a real dipole block the fold stays exact and allowed
    m_r = _chain_with_dipole(0.3)
    k, wt = m_r.monkhorst_pack(40, time_reversal=True)
    assert np.allclose(np.sort(k[:, 0]), np.sort(k_tr[:, 0]), atol=1e-14)
    full_r = sigma_optical(m_r, w, mu=0.5, mesh=40, eta=0.1)
    fold_r = sigma_optical(m_r, w, mu=0.5, kpts=k, weights=wt, eta=0.1)
    assert np.abs(fold_r - full_r).max() < 1e-12 * full_r.max()


def test_negative_site_indices_are_refused():
    """add_hop(0, -1, ...) passed the shape check (norb[-1]) but was
    assembled at the orbital offset offsets[-1] = nao, i.e. outside H,
    so the hopping vanished without an error: the two-site molecule
    came out with levels (0, 0) instead of (-1, +1).  set_dipole had
    the same problem.  Negative and out-of-range site indices are now
    refused."""
    m = TightBindingModel([[0.0], [1.0]], 1)
    with pytest.raises(ValueError, match="site index"):
        m.add_hop(0, -1, (0,), [[-1.0]])
    with pytest.raises(ValueError, match="site index"):
        m.add_hop(2, 0, (0,), [[-1.0]])
    with pytest.raises(ValueError, match="site index"):
        m.set_dipole(-1, np.zeros((1, 1, 1)))
    m.add_hop(0, 1, (0,), [[-1.0]])
    assert np.allclose(bands(m, [None])[0], [-1.0, 1.0], atol=1e-15)


def test_hr_dat_round_trip_with_an_empty_or_multiline_comment(tmp_path):
    """save_wannier90_hr wrote the comment verbatim as the header.  An
    empty comment gave a blank first line and a comment with a line
    break gave two header lines; load_wannier90_hr skips blank lines,
    so it could not read either file back ("non-integer degeneracy
    entry" and "lines 2 and 3 must carry num_wann and nrpts")."""
    H_R = {(0,): np.array([[0.1]]), (1,): np.array([[-1.0]]),
           (-1,): np.array([[-1.0]])}
    for comment in ("", "first line\nsecond line", "   "):
        path = os.path.join(tmp_path, "chain_hr.dat")
        save_wannier90_hr(path, H_R, comment=comment)
        back, nw = load_wannier90_hr(path)
        assert nw == 1
        for R, blk in H_R.items():
            assert np.abs(back[R + (0, 0)] - blk).max() < 1e-12
    with open(path) as fh:
        assert fh.readline().strip() == "written by hamop"
    save_wannier90_hr(path, H_R, comment="first line\nsecond line")
    with open(path) as fh:
        assert fh.readline().strip() == "first line second line"


def test_input_checks_added_in_0120():
    """Inputs that used to fail late with an unrelated message, or to
    return a meaningless number, are now refused up front with a
    message that names the problem.  Valid inputs are unaffected."""
    from hamop import (berry_phase, buttiker_transmission,
                       chain_lead_blocks, graphene, k_path,
                       landauer_conductance, linear_chain,
                       multiprobe_transmission, scba_transmission,
                       sigma_tensor)
    g = graphene()
    with pytest.raises(ValueError, match="image"):
        g.add_hop(0, 1, (1,), [[0.1]])          # one index for a 2D cell
    with pytest.raises(ValueError, match="norb >= 1"):
        TightBindingModel([[0.0]], 0)
    with pytest.raises(ValueError, match="Cartesian"):
        TightBindingModel([[0.0, 0.0, 0.0]], 1, cell=np.eye(2))
    with pytest.raises(ValueError, match="mesh"):
        g.monkhorst_pack((4,))
    chain = linear_chain()
    with pytest.raises(ValueError, match="eta"):
        dos(chain, [0.0], mesh=10, eta=-0.05)
    with pytest.raises(ValueError, match="eta"):
        sigma_optical(chain, [1.0], mu=0.0, mesh=10, eta=0.0)
    with pytest.raises(ValueError, match="advanced"):
        sigma_tensor(g, [0.5], mu=0.0, mesh=6, eta=-1e-3)
    # a scalar photon energy now works and equals the array result
    a = sigma_optical(chain, 1.5, mu=0.0, mesh=40, eta=0.1)
    b = sigma_optical(chain, [1.5], mu=0.0, mesh=40, eta=0.1)
    assert a.shape == (1,) and a[0] == b[0]
    with pytest.raises(ValueError, match="two vertices"):
        k_path([[0.0]])
    E = np.linspace(-1.0, 1.0, 2001)
    with pytest.raises(ValueError, match="window"):
        landauer_conductance(E, np.ones_like(E), 0.0, 300.0, window=0.0)
    H00, H01 = chain_lead_blocks()
    dev = ([H00, H00, H00], [H01, H01], H00, H01)
    with pytest.raises(ValueError, match="gamma"):
        buttiker_transmission([0.3], *dev, probe_layer=1, gamma=-0.1)
    with pytest.raises(ValueError, match="gamma"):
        multiprobe_transmission([0.3], *dev, gamma=-0.1)
    with pytest.raises(ValueError, match="W2"):
        scba_transmission([0.3], *dev, W2=-0.01)
    with pytest.raises(ValueError, match="mixing"):
        scba_transmission([0.3], *dev, W2=0.01, mixing=0.0)
    with pytest.raises(ValueError, match="n_occ"):
        berry_phase(chain, [[0.0], [1.0]], n_occ=2)
    with pytest.raises(ValueError, match="1 to 3"):
        save_wannier90_hr(os.devnull, {(0, 0, 0, 0): [[0.0]]})
