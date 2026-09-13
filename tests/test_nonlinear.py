"""v0.8 Berry-curvature-dipole anchors (Sodemann and Fu, PRL 115,
216806 (2015)): exact zeros from inversion and from filled bands,
near-exact zero from time reversal + C3 (unstrained gapped graphene),
the single-mirror constraint D perpendicular to the mirror line, and
the two independent integral forms agreeing where the answer is
nonzero -- symmetry doing the verifying, never a stored number."""
import numpy as np
import pytest

from hamop import (TightBindingModel, band_curvatures, berry_dipole,
                   haldane)

A = 2.46
SQ3 = np.sqrt(3.0)


def _gapped_graphene(mass=0.5, t=-2.7, bond_scale=1.0):
    """Honeycomb with sublattice mass (breaks inversion, keeps time
    reversal). bond_scale != 1 strengthens the (0, 0) bond only,
    breaking C3 down to the single mirror along that bond."""
    cell = A * np.array([[1.0, 0.0], [0.5, SQ3 / 2.0]])
    pos = np.array([np.zeros(2), (cell[0] + cell[1]) / 3.0])
    m = TightBindingModel(positions=pos, norb=1, cell=cell)
    m.add_hop(0, 0, (0, 0), [[+mass]])
    m.add_hop(1, 1, (0, 0), [[-mass]])
    m.add_hop(0, 1, (0, 0), [[t * bond_scale]])
    m.add_hop(0, 1, (-1, 0), [[t]])
    m.add_hop(0, 1, (0, -1), [[t]])
    return m


def test_band_curvatures_complete_set_sums_to_zero():
    model = _gapped_graphene()
    _, omega = band_curvatures(model, 12)
    assert np.max(np.abs(omega.sum(axis=2))) < 1e-10   # exact identity


def test_filled_bands_carry_no_dipole():
    """A gradient averaged over the torus is exactly zero: with every
    band filled the dipole vanishes to machine precision, however low
    the symmetry."""
    model = _gapped_graphene(bond_scale=1.3)           # no symmetry help
    out = berry_dipole(model, 12, mu=1e3, kT=0.0, method="grad")
    assert np.max(np.abs(out["D"])) < 1e-12
    assert np.max(np.abs(out["per_band"])) < 1e-12     # per band, even


def test_inversion_symmetry_kills_dipole_exactly():
    """Haldane at zero mass: gapped, Chern bands, broken time
    reversal -- but inversion makes Omega even in k, so the dipole
    cancels EXACTLY on the inversion-symmetric mesh."""
    model = haldane(t1=-1.0, t2=0.15, m_ab=0.0)
    out = berry_dipole(model, 18, mu=0.0, kT=0.2, method="grad")
    assert np.max(np.abs(out["D"])) < 1e-10


def test_c3_kills_dipole_and_strain_restores_it():
    """Unstrained gapped graphene has time reversal + three mirrors
    (C3v): D = 0 -- which is exactly why the nonlinear Hall effect is
    a STRAIN probe. The C3 map does not tile the half-shifted
    plaquette-center mesh, so the discrete residual is not machine
    zero, but it must CONVERGE to zero with the mesh while the
    strain-broken dipole converges to a nonzero value well above it.
    (This anchor is what exposed that a local curvature field needs
    the atomic frame; see the module docstring.)"""
    mu, kT = 0.9, 0.3
    d0_24 = berry_dipole(_gapped_graphene(), 24, mu, kT, method="grad")["D"]
    d0_48 = berry_dipole(_gapped_graphene(), 48, mu, kT, method="grad")["D"]
    d1_48 = berry_dipole(_gapped_graphene(bond_scale=1.3), 48, mu, kT,
                         method="grad")["D"]
    assert np.linalg.norm(d0_48) < 0.5 * np.linalg.norm(d0_24)
    assert np.linalg.norm(d1_48) > 10.0 * np.linalg.norm(d0_48)


def test_mirror_constrains_dipole_direction():
    """One bond scaled leaves one mirror line, along that bond
    (30 degrees). Time reversal + a single mirror force D onto the
    axis PERPENDICULAR to the mirror line: the component along the
    bond cancels by symmetry of the mesh, the perpendicular one
    survives."""
    d = berry_dipole(_gapped_graphene(bond_scale=1.3), 24, mu=0.9,
                     kT=0.3, method="grad")["D"]
    u = np.array([SQ3 / 2.0, 0.5])              # along the (0,0) bond
    v = np.array([-0.5, SQ3 / 2.0])             # perpendicular
    assert abs(d @ u) < 1e-8 * abs(d @ v)
    assert abs(d @ v) > 1e-3


def test_two_integral_forms_agree_where_nonzero():
    """grad path: central differences of the plaquette-flux field.
    fermi path: analytic Fermi-window derivative with Hellmann-
    Feynman velocities. Independent discretizations of the two sides
    of the integration by parts; they must meet in the middle."""
    model = _gapped_graphene(bond_scale=1.3)
    mu, kT, N = 0.9, 0.3, 36
    dg = berry_dipole(model, N, mu, kT, method="grad")["D"]
    df = berry_dipole(model, N, mu, kT, method="fermi")["D"]
    v = np.array([-0.5, SQ3 / 2.0])
    assert abs(dg @ v) > 1e-3                    # genuinely nonzero
    assert abs(dg @ v - df @ v) < 0.1 * abs(dg @ v)


def test_refusals():
    model = _gapped_graphene()
    with pytest.raises(ValueError):
        berry_dipole(model, 6, mu=0.8, kT=0.0, method="fermi")  # kT = 0
    with pytest.raises(ValueError):
        berry_dipole(model, 6, mu=0.8, kT=0.1, method="nope")
    with pytest.raises(ValueError):
        berry_dipole(model, 6, mu=0.8, kT=-0.1, method="grad")
    from hamop import linear_chain
    with pytest.raises(ValueError):
        band_curvatures(linear_chain(), 6)       # not 2D
    ov = TightBindingModel(
        positions=np.array([np.zeros(2), (A / 3.0) * np.array([1.5, SQ3 / 2])]),
        norb=1, cell=A * np.array([[1.0, 0.0], [0.5, SQ3 / 2.0]]))
    ov.add_hop(0, 1, (0, 0), [[-1.0]], [[0.1]])
    with pytest.raises(ValueError):
        band_curvatures(ov, 6)                   # overlap refused
