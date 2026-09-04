"""Kubo-Greenwood optical conductivity from the same Bloch matrices.

The velocity operator uses the standard atomistic (Peierls-like)
position gauge: the position operator is taken diagonal at the sites,
so dH/dk carries a factor of the inter-site displacement and the
intra-atomic dipole contribution is neglected -- the common
approximation in tight-binding optics.  In a nonorthogonal basis the
interband matrix element at k is

    M_nm = <n| dH/dk - (e_n + e_m)/2  dS/dk |m>,

which is what makes the result invariant under a rigid shift
H -> H + c S of the energy zero; the test suite asserts that
invariance to near machine precision, along with the textbook anchors
described in the README.

Convention: the real part of the sheet conductivity is returned in
units of e^2 / (4 hbar) -- for reference, that unit is exactly the
universal optical sheet conductivity of graphene (Kuzmenko et al.,
Phys. Rev. Lett. 100, 117401 (2008)), and the test suite reproduces
sigma = 1 on the graphene plateau from the nearest-neighbour model.
Spin degeneracy enters as the explicit factor ``spin`` (default 2).
"""
from __future__ import annotations

import numpy as np

from .eigsolve import gen_eigh

__all__ = ["sigma_optical", "carrier_count"]

KB = 8.617333262e-5  # eV / K


def sigma_optical(model, omega, mu, mesh=None, kpts=None, weights=None,
                  T=300.0, eta=0.05, direction=0, spin=2, thresh=1e-10):
    """Real part of the optical sheet conductivity, in units of e^2/(4 hbar).

    omega: photon energies (eV, > 0).  mu: chemical potential (eV).
    eta: Gaussian broadening of the energy-conservation delta (eV).
    direction: Cartesian polarization axis.  For a finite system
    (cell=None) the "cell volume" is absent and the result is the
    conductivity times the system area; divide by your geometric area.
    """
    omega = np.asarray(omega, dtype=float)
    if np.any(omega <= 0):
        raise ValueError("omega must be positive photon energies")
    if mesh is not None or kpts is not None:
        if mesh is not None:
            kpts, weights = model.monkhorst_pack(mesh)
        elif weights is None:
            weights = np.full(len(kpts), 1.0 / len(kpts))
        area = model.cell_volume
    else:
        if model.cell is not None:
            raise ValueError("periodic model: give mesh or kpts")
        kpts, weights, area = [None], [1.0], 1.0

    sig = np.zeros_like(omega)
    for k, w in zip(kpts, weights):
        H, S = model.bloch(k)
        dH, dS = model.bloch_derivative(k, direction)
        e, c = gen_eigh(H, S, thresh=thresh, eigvals_only=False)
        x = np.clip((e - mu) / (KB * T), -60.0, 60.0)
        f = 1.0 / (1.0 + np.exp(x))
        M = c.conj().T @ dH @ c
        Sd = c.conj().T @ dS @ c
        M = M - 0.5 * (e[:, None] + e[None, :]) * Sd
        dE = e[None, :] - e[:, None]          # E_m - E_n
        df = f[:, None] - f[None, :]          # f_n - f_m
        A2 = np.abs(M) ** 2
        mask = dE > 1e-3
        for iw, hw in enumerate(omega):
            g = np.exp(-0.5 * ((dE - hw) / eta) ** 2) \
                / (eta * np.sqrt(2.0 * np.pi))
            sig[iw] += w * (df * A2 * g / np.where(mask, dE, 1.0))[mask].sum()
    # sigma / (e^2 / 4 hbar) = spin * 4 pi / area * sum, M in eV*A, dE in eV
    return sig * spin * 4.0 * np.pi / area


def carrier_count(model, mu, mesh=None, kpts=None, weights=None, T=300.0,
                  spin=2, thresh=1e-10):
    """Mean number of occupied states per unit cell (spin included)."""
    if mesh is not None:
        kpts, weights = model.monkhorst_pack(mesh)
    elif kpts is None:
        kpts, weights = [None], [1.0]
    elif weights is None:
        weights = np.full(len(kpts), 1.0 / len(kpts))
    n = 0.0
    for k, w in zip(kpts, weights):
        H, S = model.bloch(k)
        e = gen_eigh(H, S, thresh=thresh)
        x = np.clip((e - mu) / (KB * T), -60.0, 60.0)
        n += w * spin * float((1.0 / (1.0 + np.exp(x))).sum())
    return n
