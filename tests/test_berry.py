"""Topology against its exact anchors: integer Chern quantization on
the lattice, the known Haldane phase diagram, zero total Chern number,
and the quantized Zak phase of the SSH chain (whose convention-free
statement is the pi difference between the two dimerizations)."""
import numpy as np
import pytest

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


def test_berry_refuses_nonorthogonal_models():
    with pytest.raises(ValueError):
        _zak(linear_chain(t=-1.0, s=0.2), a=1.0)
