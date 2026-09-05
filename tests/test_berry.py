"""Topology against its exact anchors: integer Chern quantization on
the lattice, the known Haldane phase diagram, zero total Chern number,
the quantized Zak phase of the SSH chain (whose convention-free
statement is the pi difference between the two dimerizations), and the
same statements in a nonorthogonal basis through the Loewdin frame,
where the Chern number is invariant because the frame map is a smooth
bundle isomorphism."""
import numpy as np

from hamop import (berry_phase, chern_number, haldane, linear_chain, ssh)


def test_haldane_chern_number_is_exactly_one_in_the_topological_phase():
    """|m| < 3 sqrt(3) |t2 sin phi| (= 0.52 here): Chern = +/-1, and
    the lattice formula returns the integer to 1e-12."""
    C = chern_number(haldane(t1=-1.0, t2=0.1, phi=np.pi / 2, m_ab=0.0),
                     mesh=18)
    assert abs(abs(C) - 1.0) < 1e-12


def test_haldane_chern_number_vanishes_in_the_trivial_phase():
    """m = 0.9 > 3 sqrt(3) * 0.1: the mass gap wins and C = 0."""
    C = chern_number(haldane(t1=-1.0, t2=0.1, phi=np.pi / 2, m_ab=0.9),
                     mesh=18)
    assert abs(C) < 1e-12


def test_haldane_chern_number_flips_with_flux_direction():
    Cp = chern_number(haldane(phi=+np.pi / 2), mesh=18)
    Cm = chern_number(haldane(phi=-np.pi / 2), mesh=18)
    assert abs(Cp + Cm) < 1e-12 and abs(abs(Cp) - 1.0) < 1e-12


def test_total_chern_number_of_all_bands_is_zero():
    C = chern_number(haldane(), mesh=18, n_occ=2)
    assert abs(C) < 1e-12


def _zak(model, N=60, a=2.0):
    b = 2.0 * np.pi / a
    return berry_phase(model, [[i * b / N] for i in range(N)], n_occ=1)


def test_ssh_zak_phase_is_quantized_and_dimerizations_differ_by_pi():
    """Inversion symmetry quantizes the Zak phase to 0 or pi; which one
    is convention-dependent, but the two dimerizations always differ by
    exactly pi (mod 2 pi)."""
    z1 = _zak(ssh(t1=-1.0, t2=-0.6))
    z2 = _zak(ssh(t1=-0.6, t2=-1.0))
    for z in (z1, z2):
        q = min(abs(z) % np.pi, np.pi - abs(z) % np.pi)
        assert q < 1e-9                       # quantized to 0 or pi
    d = abs(z1 - z2) % (2.0 * np.pi)
    assert abs(d - np.pi) < 1e-9


def test_chern_number_survives_a_nonorthogonal_basis():
    """A small real overlap deforms the Haldane bands smoothly without
    closing the gap, so the Chern number computed in the Loewdin frame
    must be the same exact integer as at s = 0 -- in both phases."""
    for s in (0.05, 0.1):
        C = chern_number(haldane(t1=-1.0, t2=0.1, phi=np.pi / 2, s=s),
                         mesh=18)
        assert abs(C - 1.0) < 1e-12
    C = chern_number(haldane(t1=-1.0, t2=0.1, phi=np.pi / 2, m_ab=0.9,
                             s=0.1), mesh=18)
    assert abs(C) < 1e-12


def test_ssh_zak_quantization_survives_a_nonorthogonal_basis():
    """The bond overlap preserves inversion symmetry, so the Zak phase
    stays quantized and the two dimerizations still differ by pi."""
    z1 = _zak(ssh(t1=-1.0, t2=-0.6, s=0.1))
    z2 = _zak(ssh(t1=-0.6, t2=-1.0, s=0.1))
    for z in (z1, z2):
        q = min(abs(z) % np.pi, np.pi - abs(z) % np.pi)
        assert q < 1e-9
    d = abs(z1 - z2) % (2.0 * np.pi)
    assert abs(d - np.pi) < 1e-9


def test_zak_phase_of_the_nonorthogonal_chain_is_quantized():
    z = _zak(linear_chain(t=-1.0, s=0.2), a=1.0)
    q = min(abs(z) % np.pi, np.pi - abs(z) % np.pi)
    assert q < 1e-9


# ----------------------------------------------------------------------
# the atomic frame (beyond the Loewdin convention) and the sparse solver

def test_chern_number_is_frame_independent():
    """Two genuinely different link conventions -- periodic Loewdin
    frame vs atomic-gauge links with the midpoint overlap metric --
    must return the same integer, orthogonal and overlap cases alike."""
    for s in (None, 0.1):
        h = haldane(t1=-1.0, t2=0.1, phi=np.pi / 2, s=s)
        Cl = chern_number(h, mesh=18, frame="lowdin")
        Ca = chern_number(h, mesh=18, frame="atomic")
        assert abs(Cl - 1.0) < 1e-12
        assert abs(Ca - 1.0) < 1e-12
    Ca = chern_number(haldane(m_ab=0.9, s=0.1), mesh=18, frame="atomic")
    assert abs(Ca) < 1e-12


def _zak_atomic(model, N=60, a=2.0):
    b = 2.0 * np.pi / a
    return berry_phase(model, [[i * b / N] for i in range(N)], n_occ=1,
                       frame="atomic", closure=[b])


def test_atomic_frame_zak_carries_the_intracell_position():
    """Orthogonal SSH with sites at 0 and a/2: the atomic frame adds
    the intracell-position contribution (the occupied band sits with
    equal weight on both sites, mean position a/4 per site pair), so
    the two dimerizations give -/+ pi/2 -- still differing by exactly
    pi, with inversion mapping one onto minus the other."""
    z1 = _zak_atomic(ssh(t1=-1.0, t2=-0.6))
    z2 = _zak_atomic(ssh(t1=-0.6, t2=-1.0))
    assert abs(abs(z1) - np.pi / 2) < 1e-9
    assert abs(z1 + z2) < 1e-9                    # inversion antisymmetry
    assert abs(abs(z1 - z2) % (2 * np.pi) - np.pi) < 1e-9


def test_atomic_frame_zak_inversion_antisymmetry_with_overlap():
    """With overlap the atomic-frame Zak values are no longer
    quantized (the frame convention differs), but inversion still maps
    the two dimerizations onto opposite phases -- the frame-robust
    statement, asserted here."""
    z1 = _zak_atomic(ssh(t1=-1.0, t2=-0.6, s=0.1))
    z2 = _zak_atomic(ssh(t1=-0.6, t2=-1.0, s=0.1))
    assert abs(z1 + z2) < 1e-9


def test_sparse_solver_returns_the_same_integers():
    from hamop import kane_mele, with_spin
    hh = with_spin(haldane(t1=-1.0, t2=0.1, phi=np.pi / 2))
    Cd = chern_number(hh, mesh=18, n_occ=2, solver="dense")
    Cs = chern_number(hh, mesh=18, n_occ=2, solver="sparse")
    assert abs(Cd - 2.0) < 1e-12 and abs(Cs - 2.0) < 1e-12
    km = kane_mele(lam_so=0.06)
    assert abs(chern_number(km, mesh=18, n_occ=2, solver="sparse")) < 1e-12


def test_sparse_solver_refusals():
    import pytest
    with pytest.raises(ValueError):
        chern_number(haldane(), mesh=6, solver="sparse")     # nao too small
    with pytest.raises(ValueError):
        chern_number(haldane(s=0.1), mesh=6, solver="sparse")  # overlap


# ----------------------------------------------------------------------
# real-space topology of finite systems

def _haldane_flake(nx, ny, m_ab=0.0):
    from hamop import TightBindingModel
    h = haldane(t1=-1.0, t2=0.1, phi=np.pi / 2, m_ab=m_ab)
    pos, idx = [], {}
    for i in range(nx):
        for j in range(ny):
            for s in range(2):
                idx[(i, j, s)] = len(pos)
                pos.append(h.positions[s] + i * h.cell[0] + j * h.cell[1])
    m = TightBindingModel(np.array(pos), 1, None)
    for i in range(nx):
        for j in range(ny):
            for (si, sj, img, Hb, Sb) in h._hops:
                ti, tj = i + img[0], j + img[1]
                if 0 <= ti < nx and 0 <= tj < ny:
                    m.add_hop(idx[(i, j, si)], idx[(ti, tj, sj)],
                              (0, 0), Hb)
    return m, h, idx


def test_finite_dc_hall_vanishes_identically():
    """Im Tr[P x Q y] = 0 for any Hermitian projector and real diagonal
    x, y -- the reason a bounded system's DC Hall response is zero in
    the site-diagonal position formulation, and the reason the package
    offers the Chern marker instead of a finite-system KPM sigma_xy."""
    rng = np.random.default_rng(1)
    n = 10
    H = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H = H + H.conj().T
    e, c = np.linalg.eigh(H)
    P = c[:, :4] @ c[:, :4].conj().T
    Q = np.eye(n) - P
    x = np.diag(rng.uniform(0, 3, n))
    y = np.diag(rng.uniform(0, 3, n))
    assert abs(np.imag(np.trace(P @ x @ Q @ y))) < 1e-12


def test_chern_marker_bulk_value_and_exact_total():
    """Bianco-Resta marker (PRB 84, 241106(R) (2011)): bulk average
    equals the periodic Chern number to a few percent on a 10 x 10
    flake, and the total over the finite system vanishes exactly."""
    from hamop import chern_marker
    m, h, idx = _haldane_flake(10, 10)
    mk = chern_marker(m, 0.0)
    assert abs(mk.sum()) < 1e-8
    Ac = abs(np.linalg.det(h.cell))
    bulk = sum(mk[idx[(i, j, s)]] for i in range(3, 7)
               for j in range(3, 7) for s in range(2)) / (16.0 * Ac)
    C = chern_number(haldane(t1=-1.0, t2=0.1, phi=np.pi / 2), mesh=18)
    assert abs(bulk - C) < 0.05                # sign included
    m_t, h_t, idx_t = _haldane_flake(10, 10, m_ab=0.9)
    mk_t = chern_marker(m_t, 0.0)
    bulk_t = sum(mk_t[idx_t[(i, j, s)]] for i in range(3, 7)
                 for j in range(3, 7) for s in range(2)) / (16.0 * Ac)
    assert abs(bulk_t) < 0.05


def test_chern_marker_refusals():
    import pytest
    from hamop import chern_marker
    with pytest.raises(ValueError):
        chern_marker(haldane(), 0.0)           # periodic
