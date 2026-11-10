"""Hybrid Wannier centers and the Z2 invariant (new in 0.15.0), and the
Rashba and staggered terms of kane_mele.

Every result is checked against something computed another way: the
Z2 index against the spin Chern numbers of the package's plaquette
`chern_number` where S_z is conserved, against the parity formula of
Fu and Kane (Phys. Rev. B 76, 045302 (2007)) for a model with
inversion symmetry, and against its own quantization (it cannot change
while the band gap stays open); the Wannier centers against exact
positions fixed by symmetry, against `berry_phase` and against
`chern_number` (their winding)."""
import numpy as np
import pytest

from hamop import (PAULI, TightBindingModel, bands, berry_phase,
                   chern_number, graphene, haldane, hybrid_wannier_centers,
                   kane_mele, ssh, with_spin, z2_invariant)
from hamop.berry import _bloch_periodic

S0, SX, SY, SZ = PAULI["0"], PAULI["x"], PAULI["y"], PAULI["z"]
LSO = 0.06                       # lambda_so of the Kane-Mele tests, eV
CRIT = 3.0 * np.sqrt(3.0) * LSO  # lambda_v where the gap closes (0.3118)


def _relisted(model):
    """The same crystal with its two lattice vectors in the other order."""
    m = TightBindingModel(positions=model.positions, norb=model.norb,
                          cell=model.cell[::-1])
    for i, j, img, Hb, Sb in model._hops:
        m.add_hop(i, j, img[::-1], Hb, Sb)
    return m


def _stack(models):
    """Decoupled copies of models with the same cell, in one model."""
    pos = np.concatenate([m.positions for m in models])
    norb = np.concatenate([m.norb for m in models])
    st = TightBindingModel(positions=pos, norb=norb, cell=models[0].cell)
    off = 0
    for m in models:
        for i, j, img, Hb, Sb in m._hops:
            st.add_hop(i + off, j + off, img, Hb, Sb)
        off += len(m.norb)
    return st


# ---------------- hybrid Wannier centers ----------------

def test_a_single_orbital_has_its_center_on_the_orbital():
    """One band made of one orbital: in the atomic frame its hybrid
    Wannier function sits exactly on the orbital, at fractional
    coordinates (0.3, 0.6) of this oblique cell, on every line and for
    complex hoppings too; in the periodic (Loewdin) frame every orbital
    counts as sitting at the origin, so the center is 0."""
    cell = np.array([[1.2, 0.1], [0.4, 0.9]])
    r0 = np.array([0.3, 0.6]) @ cell
    m = TightBindingModel([r0, r0 + 0.25 * cell[1]], 1, cell=cell)
    m.add_hop(0, 0, (1, 0), [[-0.7]])
    m.add_hop(0, 0, (0, 1), [[-0.4 + 0.2j]])
    m.add_hop(1, 1, (0, 0), [[3.0]])             # an empty band above
    m.add_hop(1, 1, (0, 1), [[0.2]])
    s = np.linspace(0.0, 1.0, 7)
    x0 = hybrid_wannier_centers(m, 1, s, axis=0, frame="atomic")
    x1 = hybrid_wannier_centers(m, 1, s, axis=1, frame="atomic")
    assert x0.shape == (7, 1)
    assert np.abs(x0 - 0.3).max() < 1e-12
    assert np.abs(x1 - 0.6).max() < 1e-12
    xl = hybrid_wannier_centers(m, 1, s, axis=0)
    assert np.abs(np.mod(xl + 0.5, 1.0) - 0.5).max() < 1e-12


def test_ssh_centers_sit_at_the_middle_of_the_strong_bond():
    """SSH chain, sites at 0 and a/2.  Inversion about the middle of the
    strong bond fixes the center of the filled band there: 1/4 of the
    cell when the bond inside the cell is strong, 3/4 when the bond
    across the cell boundary is (atomic frame).  The periodic frame
    gives the Zak phases 0 and pi as centers 0 and 1/2."""
    x = hybrid_wannier_centers(ssh(t1=-1.0, t2=-0.6), 1, frame="atomic")
    assert x.shape == (1,) and abs(x[0] - 0.25) < 1e-9
    x = hybrid_wannier_centers(ssh(t1=-0.6, t2=-1.0), 1, frame="atomic")
    assert abs(x[0] - 0.75) < 1e-9
    x = hybrid_wannier_centers(ssh(t1=-0.6, t2=-1.0), 1)
    assert abs(x[0] - 0.5) < 1e-9
    x = hybrid_wannier_centers(ssh(t1=-1.0, t2=-0.6), 1)
    assert min(x[0], 1.0 - x[0]) < 1e-9


def test_sum_of_centers_equals_the_berry_phase():
    """The Wilson loop's determinant is the product of the link
    determinants, so -2 pi times the sum of the centers equals
    berry_phase along the same loop, mod 2 pi, for one and for two
    occupied bands and in both frames."""
    cases = [(haldane(m_ab=0.2), 1), (haldane(m_ab=0.2, s=0.1), 1),
             (kane_mele(lam_so=LSO, lam_r=0.03), 2)]
    for model, n in cases:
        B = model.reciprocal_vectors
        for frame in ("lowdin", "atomic"):
            for s in (0.0, 0.13, 0.4):
                x = hybrid_wannier_centers(model, n, [s], n_k=60,
                                           frame=frame)[0]
                loop = [(i / 60) * B[0] + s * B[1] for i in range(60)]
                bp = berry_phase(model, loop, n_occ=n, frame=frame,
                                 closure=B[0])
                d = np.angle(np.exp(1j * (-2 * np.pi * x.sum() - bp)))
                assert abs(d) < 1e-10


def _winding(model, axis, n=101):
    x = hybrid_wannier_centers(model, 1, np.linspace(0, 1, n), axis=axis)
    d = np.mod(np.diff(x[:, 0]) + 0.5, 1.0) - 0.5
    return d.sum()


def test_winding_of_the_centers_is_the_chern_number():
    """As s crosses the zone the center winds C times (C the Chern
    number), counted counterclockwise in Cartesian k-space: with the
    loops along b1 the winding is sign(a1 x a2) C, along b2 its
    negative.  Checked against the plaquette chern_number (a different
    algorithm) for both Haldane phases, both flux directions and both
    orders of the lattice vectors."""
    for h in (haldane(), haldane(phi=-np.pi / 2), haldane(m_ab=0.9),
              _relisted(haldane())):
        C = chern_number(h, mesh=24)
        hand = np.sign(np.linalg.det(h.cell))
        assert abs(hand * _winding(h, 0) - C) < 1e-9
        assert abs(-hand * _winding(h, 1) - C) < 1e-9
    assert abs(abs(chern_number(haldane(), mesh=24)) - 1.0) < 1e-12


def test_hybrid_wannier_centers_refusals():
    h = haldane()
    with pytest.raises(ValueError, match="give s"):
        hybrid_wannier_centers(h, 1)
    with pytest.raises(ValueError, match="leave s"):
        hybrid_wannier_centers(ssh(), 1, s=[0.0])
    with pytest.raises(ValueError, match="n_occ"):
        hybrid_wannier_centers(h, 2, [0.0])          # no empty band
    with pytest.raises(ValueError, match="periodic"):
        hybrid_wannier_centers(TightBindingModel([[0.0]], 1), 1)
    with pytest.raises(ValueError, match="frame"):
        hybrid_wannier_centers(h, 1, [0.0], frame="other")
    with pytest.raises(ValueError, match="axis"):
        hybrid_wannier_centers(h, 1, [0.0], axis=2)
    with pytest.raises(ValueError, match="gap"):     # graphene: K on a line
        hybrid_wannier_centers(graphene(t=-1.0, a=1.0), 1, [1.0 / 3.0])


def test_fewer_lattice_vectors_than_dimensions():
    """An SSH chain lying at 30 degrees in a plane has the centers of
    the straight chain (1/4 and 3/4) to 1e-12, and the Kane-Mele sheet
    placed in 3D space gives the same Z2 index and the same centers as
    the 2D model to 1e-12."""
    c, sn = np.cos(np.pi / 6), np.sin(np.pi / 6)
    rot = np.array([[c, -sn], [sn, c]])
    for t1, t2, want in ((-1.0, -0.6, 0.25), (-0.6, -1.0, 0.75)):
        chain = ssh(t1, t2)
        m = TightBindingModel(
            positions=np.hstack([chain.positions, np.zeros((2, 1))]) @ rot.T,
            norb=1, cell=np.array([[2.0, 0.0]]) @ rot.T)
        for i, j, img, Hb, Sb in chain._hops:
            m.add_hop(i, j, img, Hb, Sb)
        x = hybrid_wannier_centers(m, 1, frame="atomic")
        assert abs(x[0] - want) < 1e-12
    km = kane_mele(lam_so=LSO, lam_r=0.03, lam_v=0.1)
    m3 = TightBindingModel(positions=np.hstack([km.positions,
                                                np.zeros((2, 1))]),
                           norb=km.norb,
                           cell=np.hstack([km.cell, np.zeros((2, 1))]))
    for i, j, img, Hb, Sb in km._hops:
        m3.add_hop(i, j, img, Hb, Sb)
    r3, r2 = z2_invariant(m3, 2), z2_invariant(km, 2)
    assert r3["z2"] == r2["z2"] == 1
    assert r3["centers"].shape == r2["centers"].shape
    assert np.abs(r3["centers"] - r2["centers"]).max() < 1e-12


# ---------------- kane_mele: staggered and Rashba terms ----------------

def test_kane_mele_default_is_unchanged():
    """lam_r = lam_v = 0 builds exactly the blocks of 0.14.0 (copied
    here), so every earlier result is unchanged."""
    old = TightBindingModel(positions=kane_mele().positions, norb=2,
                            cell=kane_mele().cell)
    for img in [(0, 0), (-1, 0), (0, -1)]:
        old.add_hop(0, 1, img, -1.0 * S0)
    tc = 1.0j * 0.05
    for img in [(1, 0), (-1, 1), (0, -1)]:
        old.add_hop(0, 0, img, tc * SZ)
    for img in [(1, 0), (-1, 1), (0, -1)]:
        old.add_hop(1, 1, img, np.conj(tc) * SZ)
    new = kane_mele(lam_r=0.0, lam_v=0.0)
    assert len(new._hops) == len(old._hops)
    for (a, b) in zip(new._hops, old._hops):
        assert a[:3] == b[:3] and np.array_equal(a[3], b[3])


def test_kane_mele_staggered_term_is_two_haldane_copies():
    """With lam_v (and no Rashba) the model is haldane(t2=lam_so,
    phi=+pi/2, m_ab=lam_v) for spin up and the phi = -pi/2 copy for spin
    down, and the smaller of the gaps at K and K' is
    2 |3 sqrt(3) lam_so - |lam_v||."""
    rng = np.random.default_rng(3)
    k = rng.uniform(-4, 4, (12, 2))
    for lv in (0.1, 0.25, 0.4):
        km = kane_mele(lam_so=LSO, lam_v=lv)
        up = haldane(t1=-1.0, t2=LSO, phi=np.pi / 2, m_ab=lv)
        dn = haldane(t1=-1.0, t2=LSO, phi=-np.pi / 2, m_ab=lv)
        ref = np.sort(np.concatenate([bands(up, k), bands(dn, k)], axis=1),
                      axis=1)
        assert np.abs(bands(km, k) - ref).max() < 1e-12
        B = km.reciprocal_vectors
        gaps = [np.diff(bands(km, [K])[0])[1]
                for K in ((2 * B[0] + B[1]) / 3, (B[0] + 2 * B[1]) / 3)]
        assert abs(min(gaps) - 2 * abs(CRIT - lv)) < 1e-12


def test_kane_mele_rashba_term():
    """The Rashba block of each bond is i lam_r (s x d)_z with d the unit
    vector from site A to site B along that bond (read off the model's
    own positions); it keeps time reversal, H(-k) = Theta H(k) Theta^-1
    with Theta = (1 x i s_y) K, so the bands are Kramers degenerate at
    the time-reversal-invariant momenta; and it mixes the spins."""
    lr = 0.05
    km = kane_mele(lam_so=LSO, lam_r=lr, lam_v=0.1)
    base = kane_mele(lam_so=LSO, lam_v=0.1)
    rashba = [(h[2], h[3] - g[3]) for h, g in zip(km._hops, base._hops)
              if not np.array_equal(h[3], g[3])]
    assert len(rashba) == 3
    for img, blk in rashba:
        d = km.positions[1] + np.asarray(img, float) @ km.cell \
            - km.positions[0]
        d = d / np.linalg.norm(d)
        assert np.abs(blk - 1j * lr * (SX * d[1] - SY * d[0])).max() < 1e-15
    Th = np.kron(np.eye(2), 1j * SY)
    rng = np.random.default_rng(5)
    for k in rng.uniform(-4, 4, (5, 2)):
        H1, _ = km.bloch(k)
        H2, _ = km.bloch(-k)
        assert np.abs(Th @ H1.conj() @ Th.conj().T - H2).max() < 1e-12
    B = km.reciprocal_vectors
    for G in (np.zeros(2), B[0] / 2, B[1] / 2, (B[0] + B[1]) / 2):
        e = bands(km, [G])[0]
        assert np.abs(e[0::2] - e[1::2]).max() < 1e-12
    H, _ = km.bloch(np.array([0.3, 0.2]))
    Sz = np.kron(np.eye(2), SZ)
    assert np.abs(H @ Sz - Sz @ H).max() > 0.01
    # the opposite bond direction (lam_r -> -lam_r) is the same model up
    # to a rotation of the spin by pi about z: identical bands
    k = rng.uniform(-4, 4, (8, 2))
    flip = kane_mele(lam_so=LSO, lam_r=-lr, lam_v=0.1)
    assert np.abs(bands(km, k) - bands(flip, k)).max() < 1e-12


# ---------------- the Z2 invariant ----------------

def test_z2_of_kane_mele_equals_its_spin_chern_number():
    """Without Rashba, S_z is conserved and the Z2 index is half the
    difference of the two spin Chern numbers, mod 2, each computed by
    the plaquette chern_number of its Haldane copy: 1 below
    lam_v = 3 sqrt(3) lam_so = 0.3118, 0 above.  The same for both
    frames and both axes."""
    for lv in (0.0, 0.1, 0.25, 0.35, 0.5):
        up = chern_number(haldane(t2=LSO, phi=np.pi / 2, m_ab=lv), mesh=24)
        dn = chern_number(haldane(t2=LSO, phi=-np.pi / 2, m_ab=lv), mesh=24)
        want = int(round(abs(up - dn) / 2)) % 2
        assert want == (1 if lv < CRIT else 0)
        km = kane_mele(lam_so=LSO, lam_v=lv)
        for frame in ("lowdin", "atomic"):
            for axis in (0, 1):
                r = z2_invariant(km, 2, axis=axis, frame=frame)
                assert r["z2"] == want


def _four_band(M, D=0.0, A=1.0, B=1.0):
    """Two orbitals (even and odd under inversion) and two spins on the
    square lattice, a lattice model of the kind introduced by Bernevig,
    Hughes and Zhang (Science 314, 1757 (2006)): spin up is
    A sin kx tau_x + A sin ky tau_y + [M - 2B(2 - cos kx - cos ky)] tau_z,
    spin down its time-reversed partner.  D sin kx tau_x s_x mixes the
    spins and keeps both time reversal and inversion (P = tau_z)."""
    m = TightBindingModel(positions=[[0.0, 0.0]], norb=4, cell=np.eye(2))
    m.add_hop(0, 0, (0, 0), (M - 4 * B) * np.kron(SZ, S0))
    m.add_hop(0, 0, (1, 0), A / 2j * np.kron(SX, SZ) + B * np.kron(SZ, S0)
              + D / 2j * np.kron(SX, SX))
    m.add_hop(0, 0, (0, 1), A / 2j * np.kron(SY, S0) + B * np.kron(SZ, S0))
    return m


def _fu_kane(model, P):
    """(-1)^Z2 = product over the four time-reversal-invariant momenta
    of the parity eigenvalues of the occupied Kramers pairs (Fu and
    Kane), from the numerical eigenvectors."""
    sign = 1
    for kt in ((0, 0), (np.pi, 0), (0, np.pi), (np.pi, np.pi)):
        H, _ = _bloch_periodic(model, np.array(kt, dtype=float))
        assert np.abs(P @ H - H @ P).max() < 1e-12
        occ = np.linalg.eigh(H)[1][:, :2]
        parity = np.linalg.eigvalsh(occ.conj().T @ P @ occ)
        n_odd = int(np.sum(parity < 0))
        assert n_odd % 2 == 0                  # pairs share the parity
        sign *= (-1) ** (n_odd // 2)
    return 0 if sign == 1 else 1


def test_z2_equals_the_fu_kane_parity_index():
    """With inversion symmetry the Z2 index is fixed by parities at four
    momenta, a calculation that shares nothing with the Wilson loop.
    Both agree on both sides of every gap closing (M = 0, 4, 8), with
    and without the spin-mixing term; without it the spin-up Chern
    number (plaquette method) gives the same parity."""
    P = np.kron(SZ, S0)
    for D in (0.0, 0.3):
        for M in (-1.0, 1.0, 3.0, 5.0, 7.0, 9.0):
            m = _four_band(M, D)
            fk = _fu_kane(m, P)
            assert fk == (1 if 0.0 < M < 8.0 else 0)
            assert z2_invariant(m, 2)["z2"] == fk
            assert z2_invariant(m, 2, axis=1, frame="atomic")["z2"] == fk
            if D == 0.0:
                up = TightBindingModel([[0.0, 0.0]], 2, cell=np.eye(2))
                up.add_hop(0, 0, (0, 0), (M - 4) * SZ)
                up.add_hop(0, 0, (1, 0), SX / 2j + SZ)
                up.add_hop(0, 0, (0, 1), SY / 2j + SZ)
                assert int(round(abs(chern_number(up, mesh=24)))) % 2 == fk


def _min_gap(model, n=60):
    """Smallest direct gap between bands 2 and 3 on an n x n mesh and at
    K and K'."""
    k, _ = model.monkhorst_pack(n)
    B = model.reciprocal_vectors
    k = np.concatenate([k, [(2 * B[0] + B[1]) / 3, (B[0] + 2 * B[1]) / 3]])
    e = bands(model, k)
    return float((e[:, 2] - e[:, 1]).min())


def test_z2_does_not_change_while_the_gap_stays_open():
    """Rashba coupling breaks S_z, so no spin Chern number exists, but
    the Z2 index is quantized: along a path in lam_r on which the bulk
    gap stays open it cannot change.  Topological at lam_v = 0.1 and
    ordinary at lam_v = 0.5 (from the S_z-conserving start, checked
    above), for every lam_r up to 0.06 eV."""
    for lv, want in ((0.1, 1), (0.5, 0)):
        for lr in (0.0, 0.02, 0.04, 0.06):
            km = kane_mele(lam_so=LSO, lam_v=lv, lam_r=lr)
            assert _min_gap(km) > 0.05
            assert z2_invariant(km, 2)["z2"] == want
            assert z2_invariant(km, 2, axis=1, frame="atomic")["z2"] == want


def test_z2_adds_modulo_two_and_does_not_depend_on_the_settings():
    """Two decoupled copies of a quantum spin Hall insulator are
    ordinary (1 + 1 = 0 mod 2; the centers cross twice), a copy next
    to an ordinary insulator is not; and the result does not depend
    on the number of starting lines (the coarse starts are refined) or
    of k-points per loop."""
    qsh = kane_mele(lam_so=LSO, lam_r=0.03)
    triv = kane_mele(lam_so=LSO, lam_v=0.5)
    for model, want, cross in ((_stack([qsh, qsh]), 0, 2),
                               (_stack([qsh, triv]), 1, 1)):
        for n_lines, n_k in ((11, 48), (2, 48), (3, 24)):
            r = z2_invariant(model, 4, n_lines=n_lines, n_k=n_k)
            assert (r["z2"], r["crossings"]) == (want, cross)
            assert r["centers"].shape == (len(r["s"]), 4)
            assert r["s"][0] == 0.0 and r["s"][-1] == 0.5
    r = z2_invariant(_relisted(qsh), 2)
    assert r["z2"] == 1


def _with_overlap(model, s):
    """The model with a spin-independent overlap s on every bond between
    different sites (nearest neighbors in kane_mele)."""
    m = TightBindingModel(positions=model.positions, norb=model.norb,
                          cell=model.cell)
    for i, j, img, Hb, Sb in model._hops:
        m.add_hop(i, j, img, Hb, s * np.eye(Hb.shape[0]) if i != j else None)
    return m


def test_z2_survives_a_nonorthogonal_basis():
    """A nearest-neighbor overlap deforms the bands without closing the
    gap, so the index must not change; both frames handle the overlap
    (the Loewdin map in the periodic frame, the bond-midpoint overlap in
    the atomic one)."""
    for lv, lr, want in ((0.1, 0.0, 1), (0.1, 0.04, 1), (0.5, 0.04, 0)):
        km = kane_mele(lam_so=LSO, lam_v=lv, lam_r=lr)
        for s in (0.1, 0.2):
            m = _with_overlap(km, s)
            assert m.has_overlap() and _min_gap(m) > 0.3
            for frame in ("lowdin", "atomic"):
                assert z2_invariant(m, 2, frame=frame)["z2"] == want


def test_z2_refusals():
    """Models without a Z2 index are refused, not answered."""
    # two spin copies of a Chern insulator: no time reversal (the centers
    # at s and 1 - s differ), even though the bands are spin degenerate
    with pytest.raises(ValueError, match="not time-reversal symmetric"):
        z2_invariant(with_spin(haldane()), 2)
    # spinless orbitals with real hoppings: time reversal squares to +1,
    # no Kramers pairs
    g = haldane(t2=0.0, m_ab=0.3)
    m = TightBindingModel(positions=g.positions, norb=2, cell=g.cell)
    for i, j, img, Hb, Sb in g._hops:
        if i == j and not any(img):
            m.add_hop(i, j, img, np.diag([Hb[0, 0], 2 * Hb[0, 0] + 0.4]))
        else:
            m.add_hop(i, j, img, np.kron(Hb, np.eye(2)))
    with pytest.raises(ValueError, match="Kramers"):
        z2_invariant(m, 2)
    with pytest.raises(ValueError, match="even"):
        z2_invariant(kane_mele(), 1)
    with pytest.raises(ValueError, match="gap"):      # gapless graphene
        z2_invariant(with_spin(graphene(t=-1.0, a=1.0)), 2)
    with pytest.raises(ValueError, match="two lattice vectors"):
        z2_invariant(with_spin(ssh()), 2)
    with pytest.raises(ValueError, match="n_occ"):
        z2_invariant(kane_mele(), 4)
    with pytest.raises(ValueError, match="max_lines"):
        z2_invariant(kane_mele(), 2, n_lines=11, max_lines=5)
    # two starting lines need refining (to 6 for this model): a limit of
    # 3 lines is refused instead of counting on an unresolved flow
    assert len(z2_invariant(kane_mele(lam_so=LSO), 2, n_lines=2)["s"]) > 3
    with pytest.raises(ValueError, match="could not be followed"):
        z2_invariant(kane_mele(lam_so=LSO), 2, n_lines=2, max_lines=3)
    with pytest.raises(ValueError, match="periodic"):
        z2_invariant(TightBindingModel([[0.0, 0.0]], 2), 2)
