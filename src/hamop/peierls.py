"""Uniform out-of-plane magnetic fields by Peierls substitution.

Each hopping block acquires the phase exp(i (2 pi / Phi_0) integral of
A along the straight bond) (R. Peierls, Z. Phys. 80, 763 (1933)).  The
line integral is evaluated by the midpoint rule, which is *exact* for
any linear gauge field -- in particular for the Landau and symmetric
gauges of a uniform field -- so gauge equivalence holds to machine
precision, not just approximately: the test suite asserts identical
spectra in the two gauges, the exact closed-form spectrum of a
flux-threaded ring, exact plaquette flux, and exact periodicity in the
flux quantum.

The field strength is given as ``phi``: flux per unit area in units of
the flux quantum per Angstrom^2 (dimensionless), i.e. the phase around
a region of area F (Angstrom^2) is exactly 2 pi phi F.

Finite models only.  A uniform field breaks lattice periodicity, and
magnetic unit cells (Hofstadter physics at rational flux) are not
implemented -- a periodic model is refused with an explicit error
rather than silently mistreated.
"""
from __future__ import annotations

import numpy as np

from .model import TightBindingModel

__all__ = ["with_peierls"]


def _line_integral(phi, r_i, r_j, gauge):
    """(2 pi / Phi_0) * integral of A along the straight bond, midpoint
    rule (exact for linear A).  Uses the first two Cartesian
    coordinates; the field points out of that plane."""
    mid = 0.5 * (r_i + r_j)
    d = r_j - r_i
    if gauge == "landau":            # A = phi_0-scaled (0, B x)
        val = mid[0] * d[1]
    elif gauge == "symmetric":       # A = (B/2)(-y, x)
        val = 0.5 * (mid[0] * d[1] - mid[1] * d[0])
    else:
        raise ValueError("gauge must be 'landau' or 'symmetric'")
    return 2.0 * np.pi * phi * val


def with_peierls(model, phi, gauge="landau"):
    """Copy of a *finite* model with a uniform out-of-plane magnetic
    field applied by Peierls substitution.

    phi: flux per Angstrom^2 in units of the flux quantum
    (dimensionless); the accumulated phase around any closed loop of
    bonds equals exactly 2 pi phi times the enclosed area.
    gauge: 'landau' (A proportional to (0, x)) or 'symmetric'
    (A proportional to (-y, x)/2); physical results are gauge
    independent to machine precision (asserted in the tests).

    Overlap blocks acquire the same phase (the substitution acts on the
    basis functions).  Dipole blocks are on-site and unchanged.
    Periodic models are refused: a uniform field breaks lattice
    periodicity, and magnetic unit cells are not implemented.
    """
    if model.cell is not None:
        raise ValueError(
            "with_peierls works on finite models only; a uniform field "
            "breaks lattice periodicity and magnetic unit cells "
            "(Hofstadter) are not implemented")
    if model.positions.shape[1] < 2:
        raise ValueError("need at least two Cartesian coordinates for "
                         "an out-of-plane field")
    m = TightBindingModel(positions=model.positions.copy(),
                          norb=model.norb.copy(), cell=None)
    for i, j, image, Hb, Sb in model._hops:
        theta = _line_integral(phi, model.positions[i],
                               model.positions[j], gauge)
        ph = np.exp(1j * theta)
        onsite = (i == j) and not any(image)
        if onsite:
            m.add_hop(i, j, image, Hb, Sb)     # zero bond length: phase 1
        else:
            m.add_hop(i, j, image, ph * Hb,
                      None if Sb is None else ph * Sb)
    for i, X in model._dipoles.items():
        m.set_dipole(i, X)
    return m
