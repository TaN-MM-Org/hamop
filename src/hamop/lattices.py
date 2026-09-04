"""Reference lattices with closed-form physics, for tests and examples.

These builders exist because every one of them has textbook exact
results the package is validated against: the chain's dispersion, its
density of states and its unit transmission; the two-site molecule's
single absorption line; graphene's Dirac cones and universal optical
sheet conductivity.  They double as templates for building your own
models.
"""
from __future__ import annotations

import numpy as np

from .model import TightBindingModel

__all__ = ["linear_chain", "two_site", "graphene", "ssh", "haldane",
           "chain_lead_blocks"]


def linear_chain(t=-1.0, e0=0.0, a=1.0, s=None):
    """Infinite single-orbital chain: E(k) = e0 + 2 t cos(k a).

    s: optional nearest-neighbour overlap (nonorthogonal chain).
    """
    m = TightBindingModel(positions=[[0.0]], norb=1, cell=[[a]])
    m.add_hop(0, 0, (0,), [[e0]],
              None if s is None else [[1.0]])
    m.add_hop(0, 0, (1,), [[t]],
              None if s is None else [[s]])
    return m


def two_site(t=-1.0, e0=0.0, a=1.0, s=None):
    """Finite two-site molecule: levels e0 -/+ |t| (orthogonal case),
    one optical transition at 2|t|."""
    m = TightBindingModel(positions=[[0.0], [a]], norb=1, cell=None)
    m.add_hop(0, 0, (0,), [[e0]], None if s is None else [[1.0]])
    m.add_hop(1, 1, (0,), [[e0]], None if s is None else [[1.0]])
    m.add_hop(0, 1, (0,), [[t]], None if s is None else [[s]])
    return m


def graphene(t=-2.7, a=2.46):
    """Nearest-neighbour graphene: Dirac cones at K, bandwidth 6|t|,
    and the universal optical sheet conductivity e^2/(4 hbar) on the
    interband plateau."""
    cell = a * np.array([[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
    pos = np.array([np.zeros(2), (cell[0] + cell[1]) / 3.0])
    m = TightBindingModel(positions=pos, norb=1, cell=cell)
    m.add_hop(0, 1, (0, 0), [[t]])
    m.add_hop(0, 1, (-1, 0), [[t]])
    m.add_hop(0, 1, (0, -1), [[t]])
    return m


def chain_lead_blocks(t=-1.0, e0=0.0, per_layer=1):
    """Principal-layer blocks of the single-orbital chain lead, with
    ``per_layer`` sites per layer, for the NEGF module."""
    n = per_layer
    H00 = np.zeros((n, n), dtype=complex)
    for i in range(n):
        H00[i, i] = e0
        if i + 1 < n:
            H00[i, i + 1] = t
            H00[i + 1, i] = np.conj(t)
    H01 = np.zeros((n, n), dtype=complex)
    H01[n - 1, 0] = t
    return H00, H01


def ssh(t1=-1.0, t2=-0.6, a=2.0):
    """Su-Schrieffer-Heeger dimerized chain: intra-cell hop t1, inter-cell
    hop t2, gap 2 | |t1| - |t2| | at the zone boundary, and a Zak phase
    that differs by pi between the two dimerizations."""
    m = TightBindingModel(positions=[[0.0], [0.5 * a]], norb=1, cell=[[a]])
    m.add_hop(0, 1, (0,), [[t1]])
    m.add_hop(1, 0, (1,), [[t2]])
    return m


def haldane(t1=-1.0, t2=0.1, phi=0.5 * np.pi, m_ab=0.0, a=1.0):
    """Haldane honeycomb model (Haldane, PRL 61, 2015 (1988)): real
    nearest-neighbour hop t1, complex second-neighbour hop t2 e^{i phi}
    with opposite chirality on the two sublattices, and a sublattice
    mass +/- m_ab.  The lower band carries Chern number +/-1 when
    |m_ab| < 3 sqrt(3) |t2 sin phi| and 0 outside -- the anchor the
    topology tests are pinned to."""
    cell = a * np.array([[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
    pos = np.array([np.zeros(2), (cell[0] + cell[1]) / 3.0])
    mdl = TightBindingModel(positions=pos, norb=1, cell=cell)
    mdl.add_hop(0, 0, (0, 0), [[+m_ab]])
    mdl.add_hop(1, 1, (0, 0), [[-m_ab]])
    mdl.add_hop(0, 1, (0, 0), [[t1]])
    mdl.add_hop(0, 1, (-1, 0), [[t1]])
    mdl.add_hop(0, 1, (0, -1), [[t1]])
    tc = t2 * np.exp(1j * phi)
    for img in [(1, 0), (-1, 1), (0, -1)]:      # chirality on sublattice A
        mdl.add_hop(0, 0, img, [[tc]])
    for img in [(1, 0), (-1, 1), (0, -1)]:      # opposite on sublattice B
        mdl.add_hop(1, 1, img, [[np.conj(tc)]])
    return mdl
