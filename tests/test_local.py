"""v0.6 anchors: the local NEGF observables against exact identities.
The retarded plumbing satisfies i(G - G^dag) = G (GamL + GamR +
2 eta) G^dag at ANY eta (an algebraic identity, machine precision);
the closed-device LDOS equals the Lorentzian eigen-sum exactly; and
the bond-current map obeys Kirchhoff at every interior orbital, sums
to the independently computed Caroli transmission through every
inter-layer cut, and injects/drains exactly +T/-T at the contact
layers."""
import numpy as np
import pytest

from hamop.negf import (bond_currents, device_greens, device_ldos,
                        transmission)


def _chain_device(seed=0, nw=3, nlayers=4):
    rng = np.random.default_rng(seed)
    H00 = np.diag(np.full(nw - 1, -1.0), 1) + \
        np.diag(np.full(nw - 1, -1.0), -1)
    H01 = -np.eye(nw)
    layers = [H00 + np.diag(rng.uniform(-0.3, 0.3, nw))
              for _ in range(nlayers)]
    coup = [H01.copy() for _ in range(nlayers - 1)]
    return layers, coup, H00, H01


def test_finite_eta_spectral_identity_is_exact():
    """i(G - G^dag) = G (GamL + GamR + 2 eta) G^dag holds as an
    algebraic identity at ANY eta -- the test that every self-energy
    ended up where it belongs."""
    layers, coup, H00, H01 = _chain_device()
    for eta in (1e-3, 1e-7):
        G, offs, GamL, GamR = device_greens(0.37, layers, coup, H00,
                                            H01, eta=eta)
        lhs = 1j * (G - G.conj().T)
        rhs = G @ (GamL + GamR + 2 * eta * np.eye(offs[-1])) @ \
            G.conj().T
        assert np.abs(lhs - rhs).max() < 1e-12


def test_closed_device_ldos_is_the_exact_lorentzian_sum():
    layers, coup, H00, H01 = _chain_device()
    eta = 0.05
    E = 0.37
    ld = device_ldos([E], layers, coup, H00, H01, eta=eta,
                     attach_leads=False)[0]
    n = sum(len(h) for h in layers)
    Hfull = np.zeros((n, n))
    off = 0
    for i, h in enumerate(layers):
        m = len(h)
        Hfull[off:off + m, off:off + m] = h
        if i < len(layers) - 1:
            Hfull[off:off + m, off + m:off + 2 * m] = coup[i]
            Hfull[off + m:off + 2 * m, off:off + m] = coup[i].T
        off += m
    w, v = np.linalg.eigh(Hfull)
    lor = ((np.abs(v) ** 2) *
           (eta / np.pi / ((E - w) ** 2 + eta ** 2))[None, :]).sum(1)
    assert np.abs(ld - lor).max() < 1e-13


def test_open_ldos_is_positive_and_translation_uniform_when_clean():
    """A clean uniform ribbon between identical leads is translation
    invariant, so the LDOS of every interior layer is identical."""
    nw, nl = 3, 6
    H00 = np.diag(np.full(nw - 1, -1.0), 1) + \
        np.diag(np.full(nw - 1, -1.0), -1)
    H01 = -np.eye(nw)
    layers = [H00.copy() for _ in range(nl)]
    coup = [H01.copy() for _ in range(nl - 1)]
    ld = device_ldos([0.37], layers, coup, H00, H01, eta=1e-8)[0]
    assert np.all(ld > 0.0)
    per_layer = ld.reshape(nl, nw)
    assert np.abs(per_layer - per_layer[0]).max() < 1e-8


def test_bond_currents_kirchhoff_cuts_and_contacts():
    layers, coup, H00, H01 = _chain_device()
    E = 0.37
    eta = 1e-9
    J, offs = bond_currents(E, layers, coup, H00, H01, eta=eta)
    assert np.abs(J + J.T).max() < 1e-14           # antisymmetric
    net = J.sum(axis=1)
    interior = net[offs[1]:offs[len(layers) - 1]]
    assert np.abs(interior).max() < 1e-7           # Kirchhoff (eta leak)
    T = transmission([E], layers, coup, H00, H01, eta=eta)[0]
    for c in range(1, len(layers)):
        cut = J[:offs[c], offs[c]:].sum()
        assert abs(cut - T) < 1e-6
    assert abs(net[:offs[1]].sum() - T) < 1e-6     # left injects +T
    assert abs(net[offs[len(layers) - 1]:].sum() + T) < 1e-6


def test_bond_currents_vanish_in_a_gap():
    """Outside the lead band there are no propagating modes: T = 0 and
    every bond current vanishes with it."""
    layers, coup, H00, H01 = _chain_device()
    E = 25.0                                        # far outside the band
    J, offs = bond_currents(E, layers, coup, H00, H01, eta=1e-9)
    assert np.abs(J).max() < 1e-12
