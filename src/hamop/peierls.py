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


def magnetic_supercell(model, p, q):
    """Magnetic supercell of a *periodic 2D* model at rational flux
    p/q per unit cell (in units of the flux quantum): q cells along the
    first lattice vector, Peierls phases in the oblique-coordinate
    Landau gauge theta = 2 pi (p/q) u_mid dv, where (u, v) are the
    fractional coordinates along (a1, a2).  Around any closed loop the
    accumulated phase is exactly 2 pi p/q times the enclosed area in
    unit cells -- the Hofstadter construction.

    The construction is self-validating: for the q-cell supercell to
    represent the infinite field-threaded lattice, every hopping's
    phase must be invariant under translation by q a1, which requires
    p * frac(dv) to be an integer for every hop.  A model where that
    fails (e.g. sublattice offsets with fractional v, like graphene at
    p not divisible by 3) is refused with the exact remedy: scale to an
    equivalent flux (n p)/(n q) that clears the fractions -- same
    physics, compatible gauge.

    Returns a TightBindingModel with q times the sites and cell
    [q a1, a2].  Anchors in the tests: p = 0 reproduces exact band
    folding of the original model; the pi-flux square lattice
    reproduces its closed form E = +-2|t| sqrt(cos^2 kx + cos^2 ky) to
    machine precision; there are exactly q magnetic bands per orbital;
    and the lowest Hofstadter band's Chern number is consistent with
    sigma_xy(0) on the magnetic cell through the package's own TKNN
    anchor.
    """
    if model.cell is None or model.cell.shape != (2, 2):
        raise ValueError("magnetic_supercell needs a periodic 2D model")
    p, q = int(p), int(q)
    if q < 1:
        raise ValueError("q must be a positive integer")
    inv_cell = np.linalg.inv(model.cell)
    frac_pos = model.positions @ inv_cell           # (u, v) per site
    # translation-consistency check: p * frac(dv) integer for all hops
    for i, j, image, Hb, Sb in model._hops:
        dv = (frac_pos[j][1] + image[1]) - frac_pos[i][1]
        mismatch = p * dv
        if abs(mismatch - round(mismatch)) > 1e-9:
            raise ValueError(
                "gauge inconsistency: hop with fractional a2-displacement "
                f"dv = {dv:.6f} gives non-integer p*dv = {mismatch:.6f}; "
                f"use the equivalent flux ({p}*n)/({q}*n) with n clearing "
                "the fraction (same physics, compatible gauge)")
    nsite = len(model.norb)
    pos_sc, norb_sc = [], []
    for n in range(q):
        for s in range(nsite):
            pos_sc.append(model.positions[s] + n * model.cell[0])
            norb_sc.append(model.norb[s])
    cell_sc = np.array([q * model.cell[0], model.cell[1]])
    sc = TightBindingModel(positions=np.array(pos_sc), norb=norb_sc,
                           cell=cell_sc)
    for n in range(q):
        for i, j, image, Hb, Sb in model._hops:
            ui = frac_pos[i][0] + n
            uj = frac_pos[j][0] + n + image[0]
            dv = (frac_pos[j][1] + image[1]) - frac_pos[i][1]
            theta = 2.0 * np.pi * (p / q) * 0.5 * (ui + uj) * dv
            ph = np.exp(1j * theta)
            n_tgt = n + image[0]
            img_sc = (n_tgt // q, image[1])
            onsite = (i == j) and not any(image)
            if onsite:
                sc.add_hop(n * nsite + i, n * nsite + j, (0, 0), Hb, Sb)
            else:
                sc.add_hop(n * nsite + i, (n_tgt % q) * nsite + j,
                           img_sc, ph * Hb,
                           None if Sb is None else ph * Sb)
    for i, X in model._dipoles.items():
        for n in range(q):
            sc.set_dipole(n * nsite + i, X)
    return sc
