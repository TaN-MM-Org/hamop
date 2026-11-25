"""The strong and weak Z2 indices of 3D insulators (new in 0.16.0).

`z2_invariant_3d` takes the Z2 index of six planes of the Brillouin
zone from the Wannier-center flow of `z2_invariant`.  It is checked
against calculations that share nothing with the Wilson loop: the
parity formula of Fu and Kane (Phys. Rev. B 76, 045302 (2007)) at the
eight time-reversal-invariant momenta for a model with inversion
symmetry, combined as in Eqs. (2) and (3) of Fu, Kane and Mele (Phys.
Rev. Lett. 98, 106803 (2007)); and, for weakly coupled layers of a 2D
model, the 2D index of one layer."""
import itertools

import numpy as np
import pytest

from hamop import (PAULI, TightBindingModel, bands, kane_mele,
                   z2_invariant, z2_invariant_3d)
from hamop.berry import _bloch_periodic

S0, SX, SY, SZ = PAULI["0"], PAULI["x"], PAULI["y"], PAULI["z"]
G0 = np.kron(SZ, S0)                         # parity, and the mass term
GI = [np.kron(SX, s) for s in (SX, SY, SZ)]


def _dirac3d(M, t=(1.0, 1.0, 1.0), A=1.0, D=0.0, cell=None):
    """A four-band lattice model on a cubic (or sheared) lattice: two
    orbitals of opposite parity (tau) and spin (s),

        H(k) = [M - sum_i t_i cos k_i] tau_z + A sum_i sin k_i tau_x s_i
               + D tau_x,

    with k_i = k . a_i.  Time reversal (i s_y K) holds for every
    parameter; inversion (P = tau_z) holds for D = 0 and is broken by
    D.  The gap closes where M - sum_i t_i cos k_i = 0 at one of the
    eight time-reversal-invariant momenta (for D = 0)."""
    cell = np.eye(3) if cell is None else cell
    m = TightBindingModel([[0.0, 0.0, 0.0]], 4, cell=cell)
    m.add_hop(0, 0, (0, 0, 0), M * G0 + D * np.kron(SX, S0))
    for i in range(3):
        img = [0, 0, 0]
        img[i] = 1
        m.add_hop(0, 0, tuple(img), -0.5 * t[i] * G0 + A / 2j * GI[i])
    return m


def _parity_indices(model):
    """nu0; (nu1 nu2 nu3) from the parity eigenvalues of the occupied
    Kramers pair at the eight momenta Gamma_n = (n1 b1 + n2 b2 + n3 b3)/2:
    delta_n is the parity of the pair (Fu and Kane), (-1)^nu0 the product
    of all eight and (-1)^nu_i that of the four with n_i = 1 (Fu, Kane
    and Mele, Eqs. 2 and 3)."""
    B = model.reciprocal_vectors
    delta = {}
    for n in itertools.product((0, 1), repeat=3):
        H, _ = _bloch_periodic(model, 0.5 * np.array(n, float) @ B)
        assert np.abs(G0 @ H - H @ G0).max() < 1e-12
        occ = np.linalg.eigh(H)[1][:, :2]
        par = np.linalg.eigvalsh(occ.conj().T @ G0 @ occ)
        assert abs(par[0] - par[1]) < 1e-9        # a pair shares parity
        delta[n] = int(round(par[0]))
    nu0 = 0 if np.prod(list(delta.values())) == 1 else 1
    nu = tuple(0 if np.prod([d for n, d in delta.items() if n[i]]) == 1
               else 1 for i in range(3))
    return nu0, nu


def _min_gap(model, n=12):
    k, _ = model.monkhorst_pack(n)
    e = bands(model, k)
    return float((e[:, 2] - e[:, 1]).min())


CASES = [  # (M, t): one of each kind of phase
    (2.0, (1.0, 1.0, 1.0)),     # strong, 1;(000)
    (0.0, (1.0, 1.0, 1.0)),     # weak,   0;(111)
    (-2.0, (1.0, 1.0, 1.0)),    # strong, 1;(111)
    (4.0, (1.0, 1.0, 1.0)),     # ordinary
    (1.0, (1.0, 1.0, 0.2)),     # weak,   0;(001), layers along a3
    (-0.5, (1.0, 0.3, 1.0)),    # weak,   0;(010)
]


def test_z2_3d_equals_the_parity_indices():
    """All four indices agree with the parity formula in every phase
    of the model: strong with (000) and with (111), weak along one or
    all three directions, and ordinary."""
    seen = set()
    for M, t in CASES:
        m = _dirac3d(M, t)
        assert _min_gap(m) > 0.2
        nu0, nu = _parity_indices(m)
        r = z2_invariant_3d(m, 2)
        assert (r["nu0"], r["nu"]) == (nu0, nu)
        assert r["label"] == f"{nu0};({nu[0]}{nu[1]}{nu[2]})"
        assert len(r["planes"]) == 6
        seen.add(r["label"])
    assert seen == {"1;(000)", "0;(111)", "1;(111)", "0;(000)", "0;(001)",
                    "0;(010)"}


def test_z2_3d_frame_and_cell_do_not_matter():
    """The atomic frame gives the same indices, and so does a sheared
    (non-orthogonal) cell: the indices refer to the reciprocal vectors,
    not to Cartesian axes."""
    sheared = np.array([[1.0, 0.0, 0.0], [0.3, 1.1, 0.0], [0.2, -0.1, 0.9]])
    for M, t in CASES[:2] + CASES[4:5]:
        want = _parity_indices(_dirac3d(M, t))
        r = z2_invariant_3d(_dirac3d(M, t), 2, frame="atomic")
        assert (r["nu0"], r["nu"]) == want
        r = z2_invariant_3d(_dirac3d(M, t, cell=sheared), 2)
        assert (r["nu0"], r["nu"]) == want


def test_z2_3d_without_inversion_while_the_gap_stays_open():
    """D tau_x breaks inversion (so the parity formula no longer
    applies) but keeps time reversal; the indices cannot change while
    the gap stays open."""
    for M, t in CASES[:1] + CASES[4:5]:
        want = _parity_indices(_dirac3d(M, t))
        for D in (0.1, 0.25):
            m = _dirac3d(M, t, D=D)
            assert _min_gap(m) > 0.1
            r = z2_invariant_3d(m, 2)
            assert (r["nu0"], r["nu"]) == want


def _layers(model2d, tz, c=3.0):
    """Copies of a 2D model stacked along z with spacing c and a
    spin-independent hopping tz between neighboring layers."""
    pos = np.hstack([model2d.positions, np.zeros((len(model2d.norb), 1))])
    cell = np.zeros((3, 3))
    cell[:2, :2] = model2d.cell
    cell[2, 2] = c
    m = TightBindingModel(pos, model2d.norb, cell=cell)
    for i, j, img, Hb, Sb in model2d._hops:
        m.add_hop(i, j, tuple(img) + (0,), Hb, Sb)
    for i, n in enumerate(model2d.norb):
        m.add_hop(i, i, (0, 0, 1), tz * np.eye(n))
    return m


def test_stacked_quantum_spin_hall_layers_are_a_weak_insulator():
    """Kane-Mele layers coupled by a hopping that only shifts each plane
    k_z = const in energy: the planes k_z = 0 and 1/2 are copies of one
    layer, so nu_3 is the 2D index of the layer and nu0 = 0; the other
    four planes are crossed by every layer and give 0."""
    for lv, z2 in ((0.1, 1), (0.5, 0)):
        layer = kane_mele(lam_so=0.06, lam_v=lv, lam_r=0.03)
        assert z2_invariant(layer, 2)["z2"] == z2
        r = z2_invariant_3d(_layers(layer, 0.05), 2)
        assert r["label"] == f"0;(00{z2})"
        assert r["planes"][(2, 0.0)]["z2"] == z2


def test_z2_3d_refusals():
    with pytest.raises(ValueError, match="three lattice vectors"):
        z2_invariant_3d(kane_mele(), 2)
    with pytest.raises(ValueError, match="even"):
        z2_invariant_3d(_dirac3d(2.0), 1)
    with pytest.raises(ValueError, match="n_occ"):
        z2_invariant_3d(_dirac3d(2.0), 4)
    with pytest.raises(ValueError, match="frame"):
        z2_invariant_3d(_dirac3d(2.0), 2, frame="x")
    # a Zeeman term breaks time reversal: no Kramers pairs
    m = _dirac3d(2.0)
    m.add_hop(0, 0, (0, 0, 0), 0.3 * np.kron(S0, SZ))
    with pytest.raises(ValueError, match="Kramers|time-reversal"):
        z2_invariant_3d(m, 2)
    # at M = 3 the gap closes at k = 0, which lies on three of the planes
    with pytest.raises(ValueError, match="gap"):
        z2_invariant_3d(_dirac3d(3.0), 2)
    with pytest.raises(ValueError, match="periodic"):
        z2_invariant_3d(TightBindingModel([[0.0, 0.0, 0.0]], 4), 2)
