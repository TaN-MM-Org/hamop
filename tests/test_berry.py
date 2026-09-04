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
