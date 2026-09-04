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

__all__ = ["linear_chain", "two_site", "graphene", "chain_lead_blocks"]


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
