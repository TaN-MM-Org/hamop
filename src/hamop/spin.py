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
machinery.
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


def with_spin(model):
    """Spinful copy of a spinless model: every H and S block is tensored
    with the 2 x 2 identity (spin innermost), so each band becomes an
    exact doublet.  Spin-dependent terms are then added to the returned
    model with ordinary ``add_hop`` calls on 2 norb x 2 norb blocks,
    e.g. ``np.kron(block, PAULI["z"])``."""
    m = TightBindingModel(positions=model.positions.copy(),
                          norb=2 * model.norb,
                          cell=None if model.cell is None
                          else model.cell.copy())
    for i, j, image, Hb, Sb in model._hops:
        m.add_hop(i, j, image, np.kron(Hb, PAULI["0"]),
                  None if Sb is None else np.kron(Sb, PAULI["0"]))
    return m


def kane_mele(t1=-1.0, lam_so=0.05, a=1.0):
    """Kane-Mele model with S_z conserved (Kane and Mele, PRL 95,
    226801 (2005)): nearest-neighbour hop t1 (spin diagonal) plus the
    intrinsic spin-orbit term i lambda_so nu_ij s_z on second
    neighbours.  Decouples exactly into two Haldane copies --
    spin up = haldane(t1, t2=lam_so, phi=+pi/2), spin down the
    conjugate copy -- so the spin-orbit gap at K is 6 sqrt(3) lambda_so
    and the two spin sectors carry opposite unit Chern numbers while
    the total Chern number is zero.

    Spin degeneracy is explicit here: use spin=1 in the observables.
    """
    cell = a * np.array([[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
    pos = np.array([np.zeros(2), (cell[0] + cell[1]) / 3.0])
    mdl = TightBindingModel(positions=pos, norb=2, cell=cell)
    s0, sz = PAULI["0"], PAULI["z"]
    for img in [(0, 0), (-1, 0), (0, -1)]:
        mdl.add_hop(0, 1, img, t1 * s0)
    tc = 1.0j * lam_so
    for img in [(1, 0), (-1, 1), (0, -1)]:      # chirality on sublattice A
        mdl.add_hop(0, 0, img, tc * sz)
    for img in [(1, 0), (-1, 1), (0, -1)]:      # opposite on sublattice B
        mdl.add_hop(1, 1, img, np.conj(tc) * sz)
    return mdl
