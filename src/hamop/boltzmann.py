"""Bulk transport coefficients in the Boltzmann relaxation-time
approximation (new in 0.14.0).

The Landauer functions describe a small device between two leads.  A
lab that measures a bulk crystal or a gated 2D sheet with four probes
and a temperature gradient reports instead the conductivity, the
Seebeck coefficient (thermopower) and the electronic thermal
conductivity of the material itself.  In the semiclassical (Boltzmann)
picture with a relaxation time tau these follow from the band
velocities alone, as in the BoltzTraP code (G. K. H. Madsen and D. J.
Singh, Comput. Phys. Commun. 175, 67 (2006), Eqs. 9 to 18):

    L_j^ab = spin / A * sum_k w_k sum_n tau(e_n) v_a v_b (e_n - mu)^j
             (-df/de)(e_n),                         j = 0, 1, 2,

    sigma   = e^2 L_0,
    S       = -(1 / T) L_0^-1 L_1           (charge -e; S < 0 for
                                             electron-like transport),
    kappa_e = (L_2 - L_1 L_0^-1 L_1) / T    (zero electric current),

with v = (1/hbar) dE/dk the band velocity and A the cell length, area
or volume.  Madsen and Singh note that with a direction-independent
tau the Seebeck (and Hall) coefficients do not depend on tau; here tau
may depend on the energy of the state (a callable), which changes S,
and a constant tau drops out of S and of the Lorenz ratio
kappa_e / (sigma T).

The band velocities are the diagonal velocity matrix elements of the
Kubo functions, <n| dH/dk - e_n dS/dk |n> / hbar, so overlap models
are handled.  Where bands are degenerate at a k-point the diagonal is
not defined by the eigenvalues alone (it depends on which eigenvectors
the solver returns inside the degenerate set); as in `drude_weight`,
the sum then runs over all pairs n, m of the degenerate set,
Re(v^a_nm v^b_mn), which is basis independent.  With a constant tau the
conductivity equals the Drude weight times tau / (pi hbar), and the
tests check that identity.

Units: SI.  For p lattice vectors the conductivity is in S m^(2-p)
(S/m in 3D, S per square for a 2D sheet, S m for a 1D chain or
ribbon), kappa_e in W m^(2-p) / K, S in V/K and the Lorenz ratio in
V^2/K^2.  tau is in seconds.  Only the constants e and h enter, both
exact in the SI.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence, \
    Union

import numpy as np
from numpy.typing import ArrayLike

from .eigsolve import _eig_chunks
from .kubo import _velocity
from .model import E_CHARGE, HBAR_EVS, KB, check_kgrid

if TYPE_CHECKING:
    from .model import TightBindingModel

__all__ = ["boltzmann_transport"]


def _states(model, kpts, weights, thresh, degen_tol):
    """Every kept state's energy (n_states,) and its weighted velocity
    products w_k * sum_{m degenerate with n} Re(M^a_nm M^b_mn), shape
    (n_states, dim, dim), in eV^2 Angstrom^2 (M = hbar v)."""
    dim = model.positions.shape[1]
    dirs = tuple(range(dim))
    energies, products = [], []
    for w, e, c, der in _eig_chunks(model, kpts, weights, thresh=thresh,
                                    vectors=True, directions=dirs):
        M = [_velocity(model, e, c, der[a], a) for a in dirs]
        pair = np.abs(e[:, :, None] - e[:, None, :]) <= degen_tol
        X = np.empty(e.shape + (dim, dim))
        for a in dirs:
            for b in dirs[a:]:
                xab = np.real(M[a] * np.swapaxes(M[b], 1, 2))
                xab = (xab * pair).sum(axis=2)          # (m, nb)
                X[:, :, a, b] = xab
                X[:, :, b, a] = xab
        energies.append(e.ravel())
        products.append((w[:, None, None, None] * X).reshape(-1, dim, dim))
    return np.concatenate(energies), np.concatenate(products)


def boltzmann_transport(model: TightBindingModel, mu: ArrayLike, T: float,
                        mesh: Optional[Union[int, Sequence[int]]] = None,
                        kpts: Optional[ArrayLike] = None,
                        weights: Optional[ArrayLike] = None,
                        tau: Union[float, Callable[[np.ndarray],
                                                   ArrayLike]] = 1e-14,
                        spin: float = 2, thresh: float = 1e-10,
                        degen_tol: float = 1e-9,
                        max_share: float = 0.05) -> Dict[str, Any]:
    """Conductivity, Seebeck coefficient and electronic thermal
    conductivity of a periodic model in the relaxation-time
    approximation of the Boltzmann equation.

    mu : chemical potential (eV), a number or a 1D array; the bands are
        diagonalized once for all of them.
    T : temperature in kelvin, > 0.
    mesh, kpts, weights : the k-grid, as in :func:`hamop.dos`.  The
        grid must resolve the thermal window: a mesh on which one
        (k-point, band) state carries more than ``max_share`` (5 %) of
        the conductivity is refused, because the result would then
        follow the mesh, not the bands.
    tau : relaxation time in seconds, a positive number or a callable
        tau(E) of the state energy E (eV, an array) returning seconds,
        for example ``lambda E: 1e-14 * np.abs(E) / 0.1`` for a tau
        that grows linearly with the distance from E = 0.  A constant
        tau scales sigma and kappa_e and drops out of S and the Lorenz
        ratio.
    spin : degeneracy factor (2; 1 for a model with explicit spin).
    degen_tol : energy window (eV) within which two states at one
        k-point count as degenerate (see the module docstring).

    Returns dict with, for each mu (a leading axis when mu is an array):
    ``sigma`` (dim, dim) conductivity tensor in S m^(2-p) for p lattice
    vectors (S per square for a 2D model), ``seebeck`` (dim, dim) in V/K,
    ``kappa`` (dim, dim) electronic thermal conductivity at zero current
    in W m^(2-p)/K, ``lorenz`` (dim,) kappa_aa / (sigma_aa T) in V^2/K^2
    (NaN along a direction that does not conduct), and ``units``.
    The tensors use the Cartesian axes of the model; along an axis in
    which the model is not periodic (across a ribbon) nothing conducts,
    and S is taken in the conducting subspace (a pseudo-inverse of L_0).

    Refused: a finite model, T <= 0, a non-positive or non-finite tau,
    a mu with no state within 60 kT (far outside the bands, or deep in
    a wide gap) or at which nothing conducts (L_0 = 0, a flat band), and
    a mesh too coarse for T (see above).
    """
    if model.cell is None:
        raise ValueError("Boltzmann transport needs a periodic model; for a "
                         "finite device use transmission and the Landauer "
                         "functions")
    if not (np.isfinite(T) and T > 0):
        raise ValueError("T must be > 0 K: the coefficients sample -df/de "
                         "on the k-mesh and need thermal smearing")
    if not (np.isfinite(spin) and spin > 0):
        raise ValueError("spin (the degeneracy factor) must be > 0")
    if not degen_tol >= 0:
        raise ValueError("degen_tol (eV) must be >= 0")
    if not 0.0 < max_share <= 1.0:
        raise ValueError("max_share must lie in (0, 1]")
    mus = np.atleast_1d(np.asarray(mu, dtype=float))
    if mus.ndim != 1 or not np.all(np.isfinite(mus)):
        raise ValueError("mu must be a finite number or a 1D array")
    if mesh is not None:
        kpts, weights = model.monkhorst_pack(mesh)
    elif kpts is not None:
        kpts, weights = check_kgrid(kpts, weights)
    else:
        raise ValueError("periodic model: give mesh or kpts")

    e, X = _states(model, kpts, weights, thresh, degen_tol)
    if callable(tau):
        tau_n = np.asarray(tau(e.copy()), dtype=float)
        if tau_n.shape != e.shape:
            tau_n = np.broadcast_to(tau_n, e.shape).astype(float)
    else:
        tau_n = np.full(e.shape, float(tau))
    if not (np.all(np.isfinite(tau_n)) and np.all(tau_n > 0.0)):
        raise ValueError("tau must be a positive, finite relaxation time "
                         "(seconds) for every state")

    p = model.cell.shape[0]
    vol = model.cell_volume
    kT = KB * T
    # sigma = spin e L0 1e-10^(2-p) / hbar^2, with L0 in eV Angstrom^(2-p) s
    to_si = spin * E_CHARGE * 1e-10 ** (2 - p) / (vol * HBAR_EVS ** 2)
    dim = X.shape[1]
    Xf = X.reshape(X.shape[0], dim * dim)
    trX = np.trace(X, axis1=1, axis2=2)
    out = {"sigma": [], "seebeck": [], "kappa": [], "lorenz": []}
    for m in mus:
        x = (e - m) / kT
        if not np.any(np.abs(x) < 60.0):
            raise ValueError(
                f"no state lies within 60 kT of mu = {m:.6g} eV at T = "
                f"{T:g} K, so nothing conducts and S and the Lorenz ratio "
                "are undefined (the thermal tails beyond 60 kT are not "
                "resolved)")
        x = np.clip(x, -60.0, 60.0)
        g = tau_n / (4.0 * kT * np.cosh(0.5 * x) ** 2)     # tau (-df/de)
        d = e - m
        gd = g * d
        L0, L1, L2 = (np.stack([g, gd, gd * d]) @ Xf).reshape(3, dim, dim)
        share = g * trX
        total = float(np.trace(L0))
        if not total > 0:
            raise ValueError(
                f"nothing conducts at mu = {m:.6g} eV and T = {T:g} K "
                "(L_0 = 0: the states in the thermal window do not move), "
                "so S and the Lorenz ratio are undefined")
        if float(share.max()) > max_share * total:
            raise ValueError(
                "the thermal window around mu holds too few k-point states "
                f"on this mesh (one state carries more than {max_share:.0%}"
                " of the conductivity at mu = "
                f"{m:.6g} eV); refine the mesh or raise T")
        inv0 = np.linalg.pinv(L0, rcond=1e-12)
        heat = L2 - L1 @ inv0 @ L1
        sig = to_si * L0
        kap = to_si * heat / T
        diag = np.diag(L0)
        with np.errstate(invalid="ignore", divide="ignore"):
            lor = np.where(diag > 1e-12 * total,
                           np.diag(heat) / (diag * T * T), np.nan)
        out["sigma"].append(sig)
        out["seebeck"].append(-(inv0 @ L1) / T)
        out["kappa"].append(kap)
        out["lorenz"].append(lor)
    res = {key: (np.array(val) if np.ndim(mu) else val[0])
           for key, val in out.items()}
    unit = {1: "S m", 2: "S", 3: "S/m"}[p]
    kunit = {1: "W m/K", 2: "W/K", 3: "W/(m K)"}[p]
    res["units"] = {"sigma": unit, "seebeck": "V/K", "kappa": kunit,
                    "lorenz": "V^2/K^2"}
    return res
