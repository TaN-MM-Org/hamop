"""Spin as a first-class convention: spin doubling and spin-orbit terms.

A spinless model becomes spinful by tensoring every block with the
2 x 2 identity; spin-dependent terms (Zeeman fields, intrinsic
spin-orbit coupling) are then added as ordinary hopping blocks built
from the Pauli matrices exported here.  The convention, stated once:
spin is the *fastest* (innermost) orbital index, i.e. a block acting on
(orbital o, spin s) is ``np.kron(orbital_block, spin_block)``, and
orbital 2m is (m, up), orbital 2m+1 is (m, down).

With spin explicit, the degeneracy factor in the observables must not
be applied twice: pass ``spin=1`` to :func:`hamop.sigma_optical`,
:func:`hamop.drude_weight` and friends for a model built here.

The canonical anchor is the Kane-Mele model (Kane and Mele, Phys. Rev.
Lett. 95, 226801 (2005)) with S_z conserved (no Rashba term): it is
exactly two Haldane copies with opposite chirality, its spin-orbit gap
at the K point is exactly 6 sqrt(3) lambda_so, its total Chern number
vanishes, and its two spin sectors carry Chern numbers +/-1 -- all of
which the test suite asserts against the already-validated Haldane
machinery.  Since 0.15.0 ``kane_mele`` also takes the Rashba and
staggered terms of Kane and Mele, Phys. Rev. Lett. 95, 146802 (2005);
with Rashba coupling S_z is no longer conserved, and the model is
classified by :func:`hamop.z2_invariant` instead of spin Chern numbers.
"""
from __future__ import annotations

import numpy as np

from .model import TightBindingModel

__all__ = ["PAULI", "with_spin", "kane_mele"]

PAULI = {
    "0": np.eye(2, dtype=complex),
    "x": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


def with_spin(model: TightBindingModel) -> TightBindingModel:
    """Spinful copy of a spinless model: every H and S block is tensored
    with the 2 x 2 identity (spin innermost), so each band becomes an
    exact doublet.  Intra-atomic dipole blocks (``set_dipole``) are
    carried over the same way, since the position operator does not
    act on spin.  Spin-dependent terms are then added to the returned
    model with ordinary ``add_hop`` calls on 2 norb x 2 norb blocks,
    e.g. ``np.kron(block, PAULI["z"])``."""
    m = TightBindingModel(positions=model.positions.copy(),
                          norb=2 * model.norb,
                          cell=None if model.cell is None
                          else model.cell.copy())
    for i, j, image, Hb, Sb in model._hops:
        m.add_hop(i, j, image, np.kron(Hb, PAULI["0"]),
                  None if Sb is None else np.kron(Sb, PAULI["0"]))
    for i, X in model._dipoles.items():
        m.set_dipole(i, np.stack([np.kron(X[:, :, a], PAULI["0"])
                                  for a in range(X.shape[2])], axis=-1))
    return m


def kane_mele(t1: float = -1.0, lam_so: float = 0.05,
              a: float = 1.0, lam_r: float = 0.0,
              lam_v: float = 0.0) -> TightBindingModel:
    """Kane-Mele model (Kane and Mele, PRL 95, 226801 (2005); the
    Rashba and staggered terms as in Eq. (1) of Kane and Mele, PRL 95,
    146802 (2005)): nearest-neighbor hop t1 (spin diagonal) plus the
    intrinsic spin-orbit term i lambda_so nu_ij s_z on second
    neighbors.

    With lam_r = lam_v = 0 (the default) S_z is conserved and the model
    decouples exactly into two Haldane copies -- spin up =
    haldane(t1, t2=lam_so, phi=+pi/2), spin down the conjugate copy --
    so the spin-orbit gap at K is 6 sqrt(3) lambda_so and the two spin
    sectors carry opposite unit Chern numbers while the total Chern
    number is zero.

    lam_v (new in 0.15.0): a staggered on-site energy, +lam_v on
    sublattice A (site 0) and -lam_v on B, the m_ab of
    :func:`hamop.haldane`.  It keeps S_z conserved; the two copies are
    then haldane(..., m_ab=lam_v), so the gap closes at
    |lam_v| = 3 sqrt(3) |lam_so| and, with lam_r = 0, the model is a
    quantum spin Hall insulator (Z2 = 1) below that value and an
    ordinary insulator (Z2 = 0) above it.

    lam_r (new in 0.15.0): the nearest-neighbor Rashba term
    i lam_r (s x d_ij)_z = i lam_r (s_x d_y - s_y d_x) on the bond from
    site i to site j, with d_ij the unit vector along that bond.  It
    mixes the two spins (S_z is no longer conserved) but keeps time
    reversal, so the Z2 invariant (:func:`hamop.z2_invariant`) still
    classifies the model.  Taking d_ij the other way round is the same
    model with lam_r -> -lam_r, which a rotation of the spin by pi
    about z maps onto this one.  Both new terms are added only when
    they are nonzero, so the default model is unchanged.

    Spin degeneracy is explicit here: use spin=1 in the observables.
    """
    cell = a * np.array([[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
    pos = np.array([np.zeros(2), (cell[0] + cell[1]) / 3.0])
    mdl = TightBindingModel(positions=pos, norb=2, cell=cell)
    s0, sz = PAULI["0"], PAULI["z"]
    for img in [(0, 0), (-1, 0), (0, -1)]:
        blk = t1 * s0
        if lam_r != 0.0:
            d = pos[1] + np.asarray(img, dtype=float) @ cell - pos[0]
            d = d / np.linalg.norm(d)
            blk = blk + 1.0j * lam_r * (PAULI["x"] * d[1] - PAULI["y"] * d[0])
        mdl.add_hop(0, 1, img, blk)
    tc = 1.0j * lam_so
    for img in [(1, 0), (-1, 1), (0, -1)]:      # chirality on sublattice A
        mdl.add_hop(0, 0, img, tc * sz)
    for img in [(1, 0), (-1, 1), (0, -1)]:      # opposite on sublattice B
        mdl.add_hop(1, 1, img, np.conj(tc) * sz)
    if lam_v != 0.0:
        mdl.add_hop(0, 0, (0, 0), lam_v * s0)
        mdl.add_hop(1, 1, (0, 0), -lam_v * s0)
    return mdl
