"""From T(E) to what a transport measurement reports (new in v0.11).

The NEGF functions return the transmission T(E) at the energies you
ask for.  A measurement, however, is made at a finite temperature and
often at a finite bias, and it reports a conductance, a current or a
thermovoltage.  This module turns a sampled T(E) into those numbers
with the Landauer formulas (e.g. S. Datta, *Electronic Transport in
Mesoscopic Systems*, Cambridge University Press (1995), ch. 2; for the
thermoelectric moments U. Sivan and Y. Imry, Phys. Rev. B 33, 551
(1986)):

    L_n = integral dE  T(E) (E - mu)^n (-df/dE),     n = 0, 1, 2,

    G      = spin (e^2/h) L_0                           (conductance)
    S      = -L_1 / (e T L_0)                           (Seebeck)
    kappa  = spin (1/h) (L_2 - L_1^2 / L_0) / T         (electronic
                                                         thermal conductance,
                                                         zero current)
    I      = spin (e/h) integral dE T(E) [f_L(E) - f_R(E)]   (current)

with f the Fermi function at temperature T (kelvin).  Units, chosen so
that no constant beyond the Boltzmann constant enters the code:

- G is returned in units of e^2/h (multiply by e^2/h = 3.874...e-5 S
  for siemens; with ``spin=2``, G = 2 means one fully open channel);
- S in V/K (because L_1 is in eV, L_1 / e is in volts);
- kappa in units of (e^2/h) V^2/K (multiply by e^2/h in siemens for
  W/K);
- the Lorenz ratio kappa / (G T) in V^2/K^2;
- I in units of (e^2/h) V (multiply by e^2/h in siemens for amperes).

The integrals are trapezoid sums over the energy grid you sampled
T(E) on.  For a T(E) that is smooth on the scale kT the trapezoid rule
is extremely accurate here (the integrand decays exponentially and the
sum converges faster than any power of the step); a step in T(E), such
as a band edge inside the Fermi window, converges only linearly in the
step.  Two things are checked, not assumed: the grid must reach
``window`` kT beyond every chemical potential on both sides (the
neglected tail is then below ~1e-10 of the result for window = 30),
and its largest step inside that window must not exceed kT, so the
Fermi window is actually resolved.

Anchors asserted in the tests: for an energy-independent T(E) = tau,
G = spin tau, S = 0, and the Lorenz ratio equals the Sommerfeld value
(pi^2/3)(k_B/e)^2 -- exact at every temperature for constant tau,
because integral x^2 (-df/dx) dx = pi^2/3; for a linear T(E) = a + b E
the Seebeck coefficient equals the Mott form
-(pi^2/3)(k_B^2 T/e) b / (a + b mu), again exact; the current through a
constant tau equals spin tau (mu_L - mu_R) at any temperature; and at
low temperature the conductance of the single-impurity chain, computed
from `transmission`, approaches spin times its closed-form T(mu).
"""
from __future__ import annotations

import numpy as np

from .model import KB, fermi_dirac

__all__ = ["landauer_conductance", "landauer_current",
           "thermoelectric"]


def _trapz(y, x):
    """Plain trapezoid sum (independent of the NumPy version)."""
    return float(0.5 * np.sum((y[1:] + y[:-1]) * np.diff(x)))


def _minus_dfde(E, mu, T):
    """-df/dE = e^-|x| / (kT (1 + e^-|x|)^2), x = (E - mu)/kT, in 1/eV;
    written with e^-|x| so it never overflows far from mu."""
    kT = KB * T
    q = np.exp(-np.abs((E - mu) / kT))
    return q / (kT * (1.0 + q) ** 2)


def _prepare(E, trans, mus, T, window):
    E = np.asarray(E, dtype=float).ravel()
    tr = np.asarray(trans, dtype=float).ravel()
    if E.shape != tr.shape:
        raise ValueError(f"{E.size} energies but {tr.size} "
                         "transmission values")
    if E.size < 2:
        raise ValueError("need T(E) on at least two energies")
    if not (np.all(np.isfinite(E)) and np.all(np.isfinite(tr))):
        raise ValueError("energies and transmission must be finite")
    if np.any(np.diff(E) <= 0.0):
        raise ValueError("energies must be strictly increasing")
    if not T > 0:
        raise ValueError("T must be > 0 K: the Fermi window is resolved "
                         "on the energy grid (at T = 0 the conductance "
                         "is simply spin * T(mu))")
    kT = KB * T
    lo = min(mus) - window * kT
    hi = max(mus) + window * kT
    if E[0] > lo or E[-1] < hi:
        raise ValueError(
            f"the energy grid [{E[0]:.6g}, {E[-1]:.6g}] eV does not cover "
            f"the Fermi window [{lo:.6g}, {hi:.6g}] eV ({window:g} kT "
            "beyond the chemical potentials); compute T(E) further out")
    inside = (E[1:] >= lo) & (E[:-1] <= hi)
    step = float(np.diff(E)[inside].max())
    if step > kT:
        raise ValueError(
            f"energy step {step:.3g} eV inside the Fermi window exceeds "
            f"kT = {kT:.3g} eV; refine the grid so the window is "
            "resolved")
    return E, tr


def _moments(E, tr, mu, T):
    w = tr * _minus_dfde(E, mu, T)
    x = E - mu
    return (_trapz(w, E), _trapz(w * x, E), _trapz(w * x * x, E))


def landauer_conductance(E, trans, mu, T, spin=2, window=30.0):
    """Linear-response conductance at temperature T, in units of e^2/h.

    E : increasing energies (eV) at which T(E) was computed.
    trans : the transmission at those energies (from `transmission`,
        `transmission_direct`, `transmission_sparse`, the dephasing
        functions, or your own code).
    mu : chemical potential (eV).  T : temperature (K, > 0).
    spin : degeneracy factor (2 for a spinless model; 1 when spin is
        explicit in the model, as after `with_spin`).
    window : how many kT the grid must reach beyond mu on each side.

    G = spin * integral T(E) (-df/dE) dE.
    """
    E, tr = _prepare(E, trans, [mu], T, window)
    return spin * _moments(E, tr, mu, T)[0]


def thermoelectric(E, trans, mu, T, spin=2, window=30.0):
    """Linear-response thermoelectric coefficients from T(E).

    Returns a dict with
    ``G`` (e^2/h), ``S`` (Seebeck coefficient, V/K; negative when T(E)
    rises with energy, i.e. for electron-like transport),
    ``kappa`` (electronic thermal conductance at zero current, in
    (e^2/h) V^2/K), ``lorenz`` (kappa / (G T), V^2/K^2; the Sommerfeld
    value is (pi^2/3)(k_B/e)^2 = 2.443e-8 V^2/K^2) and the moments
    ``L0`` (dimensionless), ``L1`` (eV), ``L2`` (eV^2).  Arguments as
    in :func:`landauer_conductance`.  Refuses a grid where L0 = 0
    (nothing transmits in the Fermi window, so S is undefined).
    """
    E, tr = _prepare(E, trans, [mu], T, window)
    L0, L1, L2 = _moments(E, tr, mu, T)
    if not L0 > 0:
        raise ValueError("no transmission inside the Fermi window "
                         "(L0 = 0); S and the Lorenz ratio are undefined")
    heat = L2 - L1 * L1 / L0
    return {"G": spin * L0, "S": -L1 / (T * L0),
            "kappa": spin * heat / T, "lorenz": heat / (L0 * T * T),
            "L0": L0, "L1": L1, "L2": L2}


def landauer_current(E, trans, mu_L, mu_R, T, spin=2, window=30.0):
    """Current from left to right at bias mu_L - mu_R, in units of
    (e^2/h) V.

    I = spin * integral T(E) [f(E; mu_L, T) - f(E; mu_R, T)] dE, with
    T(E) as given: any change of T(E) with the bias itself (a
    voltage drop across the device) must already be in ``trans``.
    Positive when mu_L > mu_R and T(E) > 0.  The grid must cover
    ``window`` kT beyond both chemical potentials.
    """
    E, tr = _prepare(E, trans, [mu_L, mu_R], T, window)
    df = fermi_dirac(E, mu_L, T) - fermi_dirac(E, mu_R, T)
    return spin * _trapz(tr * df, E)
