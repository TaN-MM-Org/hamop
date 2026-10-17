"""Carrier density, chemical potential and quantum capacitance of a
gated or doped periodic model (new in 0.13.0).

A transport lab rarely knows the chemical potential of its sample.  It
knows the carrier density, from the Hall effect or from the gate
voltage and the gate capacitance.  The optics and transport functions
of this package, however, take ``mu``.  This module connects the two:

- `carrier_density` gives the net density of added electrons (negative
  for holes) at a chemical potential, per cm^dim (per cm^2 for a 2D
  sheet), measured from a neutral filling you state;
- `chemical_potential_from_density` inverts it: the chemical potential
  that holds a measured density, with the interval that the density's
  error bar maps to, and the thermodynamic density of states dn/dmu,
  which gives the quantum capacitance C_Q = e^2 dn/dmu.

Both use the same eigenvalues on the same k-mesh as `carrier_count`
and `fermi_level`:

    n(mu) = [spin * sum_k w_k sum_j f(e_j(k) - mu) - n_neutral] / A,
    dn/dmu = spin * sum_k w_k sum_j (-df/de)(e_j(k)) / A,

with A the cell area (length or volume in 1D or 3D).  For graphene's
Dirac cones this gives n = mu^2 / (pi (hbar v_F)^2) and
C_Q = 2 e^2 |mu| / (pi (hbar v_F)^2) at low temperature (the standard
Dirac-cone result; the quantum capacitance of graphene was measured by
J. Xia, F. Chen, J. Li and N. Tao, Nature Nanotechnology 4, 505
(2009)); the tests check the lattice model against these closed forms, and the
chain against the exact integral of its density of states.

Units: densities per cm^dim (1 Angstrom = 1e-8 cm), dn/dmu per eV per
cm^dim, C_Q in F/cm^dim (C_Q = e * dn/dmu with e = 1.602176634e-19 C,
exact in the SI, because one eV is e joule).  The neutral filling
``n_neutral`` is in electrons per cell with spin included, the unit of
`carrier_count` (2.0 for graphene's two pi orbitals per cell).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq

from .model import E_CHARGE, KB, check_kgrid, fermi_dirac
from .spectrum import _eigs

if TYPE_CHECKING:
    from .model import TightBindingModel

__all__ = ["carrier_density", "chemical_potential_from_density"]

_CM = 1e8          # Angstrom per cm


def _setup(model, n_neutral, mesh, kpts, weights, spin, thresh):
    if model.cell is None:
        raise ValueError("a carrier density needs a periodic model (per "
                         "unit area of the cell); for a finite model use "
                         "carrier_count")
    if not (np.isfinite(spin) and spin > 0):
        raise ValueError("spin (the degeneracy factor) must be > 0")
    if mesh is not None:
        kpts, weights = model.monkhorst_pack(mesh)
    elif kpts is not None:
        kpts, weights = check_kgrid(kpts, weights)
    else:
        raise ValueError("periodic model: give mesh or kpts")
    eigs = _eigs(model, kpts, thresh)
    if eigs.ndim != 2:
        raise ValueError("the overlap matrix drops states at some k-points "
                         "(canonical orthogonalization); a density needs "
                         "the same number of bands everywhere")
    n_full = spin * eigs.shape[1]
    n_neutral = float(n_neutral)
    if not 0.0 <= n_neutral <= n_full:
        raise ValueError(f"n_neutral must lie in [0, {n_full:g}] electrons "
                         "per cell (spin included)")
    dim = model.cell.shape[0]
    scale = _CM ** dim / model.cell_volume      # per cell -> per cm^dim
    return eigs, np.asarray(weights, dtype=float), n_neutral, n_full, \
        scale, dim


def _density(eigs, w, mu, T, spin, n_neutral, scale):
    return (spin * float(w @ fermi_dirac(eigs, mu, T).sum(axis=1))
            - n_neutral) * scale


def _mdfde(eigs, mu, T):
    x = np.clip((eigs - mu) / (KB * T), -60.0, 60.0)
    return 1.0 / (4.0 * KB * T * np.cosh(0.5 * x) ** 2)


def carrier_density(model: TightBindingModel, mu: float, n_neutral: float,
                    mesh: Optional[Union[int, Sequence[int]]] = None,
                    kpts: Optional[ArrayLike] = None,
                    weights: Optional[ArrayLike] = None, T: float = 300.0,
                    spin: float = 2, thresh: float = 1e-10) -> float:
    """Net carrier density at chemical potential ``mu`` (eV), per cm^dim.

    n = (carrier_count(mu) - n_neutral) / (cell area), positive when
    electrons are added (n-type), negative for holes.  ``n_neutral`` is
    the electron count per cell (spin included) of the undoped sample,
    for example 2.0 for graphene.  T in kelvin (T = 0 is the exact step).
    A 2D model gives cm^-2, a 1D model cm^-1, a 3D model cm^-3.
    """
    if not T >= 0:
        raise ValueError("T must be >= 0 K")
    eigs, w, n0, _, scale, _ = _setup(model, n_neutral, mesh, kpts,
                                      weights, spin, thresh)
    return _density(eigs, w, float(mu), T, spin, n0, scale)


def chemical_potential_from_density(
        model: TightBindingModel, density: float, n_neutral: float,
        density_err: Optional[float] = None,
        mesh: Optional[Union[int, Sequence[int]]] = None,
        kpts: Optional[ArrayLike] = None,
        weights: Optional[ArrayLike] = None, T: float = 300.0,
        spin: float = 2, thresh: float = 1e-10, tol: float = 1e-10,
        max_share: float = 0.05) -> Dict[str, Any]:
    """Chemical potential that holds a measured carrier density.

    density : net density in cm^-dim (cm^-2 for a 2D sheet), positive
        for electrons, as from a Hall measurement or C_gate (V_g - V_0)
        / e.  n_neutral : as in :func:`carrier_density`.
    density_err : optional 1-sigma error of the density (same unit).
        The interval is found by solving for density -/+ density_err
        exactly (no linearization), so it is right even where n(mu)
        curves, for example near a band edge or a Dirac point.
    T : temperature in kelvin, > 0 (at T = 0 the density on a finite
        k-mesh is a staircase and does not fix mu).
    tol : accuracy of mu in eV.

    Returns dict(mu, interval (lo, hi) or None, mu_err (half-width of
    the interval) or None, dn_dmu (per eV per cm^dim), quantum_capacitance
    (e^2 dn/dmu in F/cm^dim), density_range (the lowest and highest
    density the bands can hold), unit (e.g. "cm^-2")).

    Refused, with an explanation: a density outside the range the bands
    can hold; T <= 0; and a k-mesh too coarse for the temperature, which
    is detected when one (k-point, band) state carries more than
    ``max_share`` (5 %) of dn/dmu at the solution, so that mu would jump
    from one mesh state to the next instead of following the density.
    """
    if not T > 0:
        raise ValueError("T must be > 0 K: at T = 0 the density on a "
                         "k-mesh is a staircase and does not determine mu")
    if not (0.0 < max_share <= 1.0):
        raise ValueError("max_share must lie in (0, 1]")
    eigs, w, n0, n_full, scale, dim = _setup(model, n_neutral, mesh, kpts,
                                             weights, spin, thresh)
    density = float(density)
    if not np.isfinite(density):
        raise ValueError("density must be finite")
    err = None
    if density_err is not None:
        err = float(density_err)
        if not (np.isfinite(err) and err >= 0.0):
            raise ValueError("density_err must be finite and >= 0")
    n_lo, n_hi = -n0 * scale, (n_full - n0) * scale
    targets = [density] if err is None else [density - err, density,
                                             density + err]
    for x in targets:
        if not n_lo < x < n_hi:
            raise ValueError(
                f"density {x:.6g} cm^-{dim} is outside the range "
                f"({n_lo:.6g}, {n_hi:.6g}) cm^-{dim} that these bands can "
                "hold for this n_neutral")
    kT = KB * T
    lo = float(eigs.min()) - 60.0 * kT - 1.0
    hi = float(eigs.max()) + 60.0 * kT + 1.0

    def solve(target):
        def g(mu):
            return _density(eigs, w, mu, T, spin, n0, scale) - target
        return float(brentq(g, lo, hi, xtol=tol, rtol=4 * np.finfo(float).eps,
                            maxiter=500))

    mu = solve(density)
    terms = w[:, None] * _mdfde(eigs, mu, T)
    total = float(terms.sum())
    if not total > 0 or float(terms.max()) > max_share * total:
        raise ValueError(
            "the thermal window around mu holds too few k-point states on "
            f"this mesh (one state carries more than {max_share:.0%} of "
            "dn/dmu), so mu would follow the mesh, not the density; "
            "refine the mesh or raise T")
    dn_dmu = spin * total * scale
    interval = mu_err = None
    if err is not None:
        interval = (solve(density - err), solve(density + err))
        mu_err = 0.5 * (interval[1] - interval[0])
    return {"mu": mu, "interval": interval, "mu_err": mu_err,
            "dn_dmu": dn_dmu, "quantum_capacitance": E_CHARGE * dn_dmu,
            "density_range": (n_lo, n_hi), "unit": f"cm^-{dim}"}
