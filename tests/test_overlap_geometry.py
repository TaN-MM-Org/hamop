"""Overlapping orbitals in the quantum geometry, the Berry curvature
dipole and the bond currents (new in 0.16.0; all three refused overlap
models before), and the Hall conductivity of `sigma_tensor` with
overlap, which is not exactly quantized (a documented limit).

Each check uses something independent of the new code: a closed form
(an overlap S = 1 + r H, which commutes with H and leaves the
orthogonalized states those of H), the plaquette Chern number, the
exact invariance under a shift of the energy zero (H -> H + c S), exact
symmetry zeros, two discretizations that must converge to each other,
and the Caroli transmission of `transmission`."""
import numpy as np
import pytest

from hamop import (TightBindingModel, berry_dipole, chern_number, haldane,
                   quantum_geometric_tensor, quantum_metric, quantum_weight,
                   sigma_tensor)
from hamop.geometry import _qgt_many
from hamop.negf import bond_currents, device_greens, transmission

SQ3 = np.sqrt(3.0)


def _commuting_overlap(model, r):
    """The model with S = 1 + r H: every block gets the overlap r times
    its Hamiltonian block, and each site the on-site overlap 1 + r H."""
    m = TightBindingModel(model.positions, model.norb, cell=model.cell)
    onsite = set()
    for i, j, img, Hb, Sb in model._hops:
        if i == j and not any(img):
            onsite.add(i)
            m.add_hop(i, j, img, Hb, np.eye(Hb.shape[0]) + r * Hb)
        else:
            m.add_hop(i, j, img, Hb, r * Hb)
    for i, n in enumerate(model.norb):
        if i not in onsite:
            m.add_hop(i, i, (0,) * model.cell.shape[0], np.zeros((n, n)),
                      np.eye(n))
    return m


def _shifted(model, c):
    """H -> H + c S (on-site overlap 1 where none was given)."""
    m = TightBindingModel(model.positions, model.norb, cell=model.cell)
    for i, j, img, Hb, Sb in model._hops:
        if Sb is None:
            Sb = np.eye(Hb.shape[0]) if (i == j and not any(img)) \
                else np.zeros_like(Hb)
        m.add_hop(i, j, img, Hb + c * Sb, Sb)
    return m


K30 = np.random.default_rng(1).uniform(-3.0, 3.0, (30, 2))


def test_geometry_with_a_commuting_overlap_equals_the_orthogonal_one():
    """With S = 1 + r H the generalized eigenvectors are those of H
    (rescaled), so the orthogonalized occupied projector is that of the
    orthogonal model and Q is the same at every k, although dS/dk is
    not zero.  Measured 4e-11 (the finite differences)."""
    h = haldane(t1=-1.0, t2=0.1, m_ab=0.2)
    ho = _commuting_overlap(h, 0.08)
    H0, _ = h._bloch_many(K30)
    H1, S1 = ho._bloch_many(K30)
    assert np.abs(H1 - H0).max() < 1e-15
    assert np.abs(S1 - (np.eye(2) + 0.08 * H0)).max() < 1e-15
    Q0 = _qgt_many(h, K30, 1, 1e-5, 1e-8)
    Q1 = _qgt_many(ho, K30, 1, 1e-5, 1e-8)
    assert np.abs(Q1 - Q0).max() < 1e-9 * np.abs(Q0).max()


def test_geometry_with_overlap_is_shift_invariant_and_positive():
    """H -> H + c S moves every energy by c and changes no state
    (measured 9e-11); tr g >= 2 sqrt(det g) >= |Omega| at 25 random
    points (an exact identity of a projector; measured slack -8e-17)."""
    hs = haldane(t1=-1.0, t2=0.1, m_ab=0.2, s=0.15)
    Q = _qgt_many(hs, K30, 1, 1e-5, 1e-8)
    Qc = _qgt_many(_shifted(hs, 3.0), K30, 1, 1e-5, 1e-8)
    assert np.abs(Qc - Q).max() < 1e-9 * np.abs(Q).max()
    for k in np.random.default_rng(2).uniform(-4.0, 4.0, (25, 2)):
        q = quantum_geometric_tensor(hs, k)
        g, om = q.real, 2.0 * q[0, 1].imag
        root = 2.0 * np.sqrt(max(np.linalg.det(g), 0.0))
        assert np.trace(g) >= root - 1e-12
        assert root >= abs(om) - 1e-12
        assert np.allclose(quantum_metric(hs, k), g)


def test_geometry_with_overlap_integrates_to_the_chern_number():
    """The curvature of the overlap geometry integrates to the plaquette
    Chern number (a different algorithm), which is the same in both
    frames.  Mesh 36; measured deviations 1.6e-6, 8e-7 and 6e-10, as
    for the orthogonal model on the same mesh (1.6e-6)."""
    for h, c in ((haldane(s=0.1, m_ab=0.2), 1),
                 (haldane(s=0.15, m_ab=0.9), 0),
                 (haldane(s=0.2, phi=-np.pi / 2), -1)):
        assert chern_number(h, mesh=24) == pytest.approx(c, abs=1e-12)
        assert chern_number(h, mesh=24, frame="atomic") == \
            pytest.approx(c, abs=1e-12)
        q = quantum_weight(h, mesh=36)
        assert abs(q["chern"] - c) < 1e-5
        assert q["tr_K"] >= abs(q["chern"])


def test_hall_conductivity_with_overlap_is_not_exactly_quantized():
    """A documented limit, pinned so that the README number stays true.
    The optics use the velocity dH/dk - (e_n + e_m)/2 dS/dk, whose
    position matrix elements S_ij (tau_i + tau_j)/2 do not commute
    between x and y; the Kubo sigma_xy(0) of the insulating Haldane model
    with overlap s = 0.1 is then 1.00336 e^2/h, not 1, on any fine mesh
    (36 and 72 agree to 2e-6), while without overlap it is 1 to 1e-6.
    The Chern number itself is 1 (above)."""
    vals = {}
    for s in (None, 0.1):
        for N in (36, 72):
            sxy = sigma_tensor(haldane(s=s, m_ab=0.2), [0.0], mu=0.0,
                               directions=(0, 1), mesh=N, T=1.0, eta=1e-5,
                               spin=1)[0].real * np.pi / 2.0
            vals[(s, N)] = sxy
    assert abs(vals[(None, 72)] - 1.0) < 1e-6
    assert abs(vals[(0.1, 36)] - vals[(0.1, 72)]) < 1e-5
    assert abs(vals[(0.1, 72)] - 1.00336) < 1e-5


def _gapped_graphene_overlap(s, c=0.0, mass=0.5, t=-2.7, bond_scale=1.3,
                             a=2.46):
    """Gapped graphene with one strained bond (time reversal, one mirror,
    no inversion) and overlap s on the three bonds; c shifts H -> H + c S."""
    cell = a * np.array([[1.0, 0.0], [0.5, SQ3 / 2.0]])
    pos = np.array([np.zeros(2), (cell[0] + cell[1]) / 3.0])
    m = TightBindingModel(positions=pos, norb=1, cell=cell)
    m.add_hop(0, 0, (0, 0), [[mass + c]])
    m.add_hop(1, 1, (0, 0), [[-mass + c]])
    for img, tb in (((0, 0), t * bond_scale), ((-1, 0), t), ((0, -1), t)):
        m.add_hop(0, 1, img, [[tb + c * s]], [[s]])
    return m


def test_curvature_dipole_with_overlap():
    """The two integral forms (curvature gradient, and curvature times
    Fermi-window derivative times band velocity with its dS term)
    converge to each other: for s = 0.1 they differ by 19 %, 4.5 % and
    2.0 % on meshes 48, 96 and 144.  The single mirror forces D
    perpendicular to it (measured 6e-14 of the other component); a shift
    of the energy zero with mu changes nothing (measured 2e-13); filled
    bands and inversion symmetry give zero."""
    v = np.array([-0.5, SQ3 / 2.0])           # perpendicular to the mirror
    u = np.array([SQ3 / 2.0, 0.5])            # along it
    m = _gapped_graphene_overlap(0.1)
    rel = []
    for N in (48, 144):
        dg = berry_dipole(m, N, 0.9, 0.3, method="grad")["D"]
        df = berry_dipole(m, N, 0.9, 0.3, method="fermi")["D"]
        assert abs(dg @ u) < 1e-8 * abs(dg @ v)
        assert abs(dg @ v) > 1e-3
        rel.append(abs(dg @ v - df @ v) / abs(dg @ v))
    assert rel[1] < 0.03 and rel[1] < 0.2 * rel[0]
    ms = _gapped_graphene_overlap(0.2, c=3.0)
    m2 = _gapped_graphene_overlap(0.2)
    for method in ("grad", "fermi"):
        a = berry_dipole(m2, 24, 0.9, 0.3, method=method)["D"]
        b = berry_dipole(ms, 24, 3.9, 0.3, method=method)["D"]
        assert np.abs(a - b).max() < 1e-10 * np.abs(a).max()
    assert np.abs(berry_dipole(m2, 12, 1e3, 0.0)["D"]).max() < 1e-12
    h = haldane(t1=-1.0, t2=0.15, s=0.1)
    for method in ("grad", "fermi"):
        assert np.abs(berry_dipole(h, 18, 0.0, 0.2,
                                   method=method)["D"]).max() < 1e-12


def _overlap_device(s=0.15, nw=3, nl=4, seed=0):
    rng = np.random.default_rng(seed)
    H00 = np.diag(np.full(nw - 1, -1.0), 1) + np.diag(np.full(nw - 1, -1.0),
                                                       -1)
    H01 = -np.eye(nw)
    S00 = np.eye(nw) + np.diag(np.full(nw - 1, s), 1) \
        + np.diag(np.full(nw - 1, s), -1)
    S01 = s * np.eye(nw)
    layers = [H00 + np.diag(rng.uniform(-0.3, 0.3, nw)) for _ in range(nl)]
    kw = dict(layers_S=[S00] * nl, coup_S=[S01] * (nl - 1), lead_S00=S00,
              lead_S01=S01)
    return layers, [H01.copy()] * (nl - 1), H00, H01, kw


def test_bond_currents_with_overlap():
    """Kirchhoff's law at every interior orbital (measured 1.4e-9, the
    eta leak), every cut and the contacts carry the transmission of
    `transmission` with the same overlap (measured 1e-8), and the -E S
    term matters: with H alone every cut is off by 0.14."""
    layers, coup, H00, H01, kw = _overlap_device()
    E, eta = 0.37, 1e-9
    J, offs = bond_currents(E, layers, coup, H00, H01, eta=eta, **kw)
    T = transmission([E], layers, coup, H00, H01, eta=eta, **kw)[0]
    nl = len(layers)
    assert T > 1.0
    assert np.abs(J + J.T).max() < 1e-14
    net = J.sum(axis=1)
    assert np.abs(net[offs[1]:offs[nl - 1]]).max() < 1e-7
    for c in range(1, nl):
        assert abs(J[:offs[c], offs[c]:].sum() - T) < 1e-6
    assert abs(net[:offs[1]].sum() - T) < 1e-6
    assert abs(net[offs[nl - 1]:].sum() + T) < 1e-6
    # the same Green function with H in place of H - E S
    G, offs2, GamL, _ = device_greens(E, layers, coup, H00, H01,
                                      kw["layers_S"], kw["coup_S"],
                                      kw["lead_S00"], kw["lead_S01"], eta)
    Hf = np.zeros((offs[-1], offs[-1]))
    for i in range(nl):
        Hf[offs[i]:offs[i + 1], offs[i]:offs[i + 1]] = layers[i]
        if i < nl - 1:
            Hf[offs[i]:offs[i + 1], offs[i + 1]:offs[i + 2]] = coup[i]
            Hf[offs[i + 1]:offs[i + 2], offs[i]:offs[i + 1]] = coup[i].T
    J_H = -2.0 * np.imag(Hf * (G @ GamL @ G.conj().T).T)
    for c in range(1, nl):
        assert abs(J_H[:offs[c], offs[c]:].sum() - T) > 0.1


def test_bond_currents_of_the_impurity_chain_with_overlap():
    """A chain with overlap s on its bonds looks, at energy E, like an
    orthogonal chain with hopping t - E s (the coupling is E S - H), so
    one impurity transmits (4 t'^2 - E^2) / ((4 t'^2 - E^2) + eps^2)
    with t' = t - E s, the closed form of the orthogonal chain.  Both
    `transmission` and the current through each bond give it (measured
    6e-9 and 4e-9 at most, the eta = 1e-9 broadening)."""
    t, s, eps = -1.0, 0.2, 0.8
    H00 = np.zeros((1, 1))
    H01 = np.array([[t]])
    S00, S01 = np.eye(1), s * np.eye(1)
    layers, coup = [H00, H00 + eps, H00], [H01, H01]
    ov = dict(layers_S=[S00] * 3, coup_S=[S01] * 2, lead_S00=S00,
              lead_S01=S01)
    for E in (-1.2, 0.3, 1.5):
        tE = t - E * s
        exact = (4 * tE ** 2 - E ** 2) / ((4 * tE ** 2 - E ** 2) + eps ** 2)
        T = transmission([E], layers, coup, H00, H01, eta=1e-9, **ov)[0]
        J, _ = bond_currents(E, layers, coup, H00, H01, eta=1e-9, **ov)
        assert abs(T - exact) < 1e-7
        assert abs(J[0, 1] - exact) < 1e-7 and abs(J[1, 2] - exact) < 1e-7
