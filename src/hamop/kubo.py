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

__all__ = ["sigma_optical", "sigma_tensor", "carrier_count",
           "drude_weight"]

KB = 8.617333262e-5  # eV / K


def sigma_optical(model, omega, mu, mesh=None, kpts=None, weights=None,
                  T=300.0, eta=0.05, direction=0, spin=2, thresh=1e-10,
                  lineshape="gaussian"):
    """Real part of the optical sheet conductivity, in units of e^2/(4 hbar).

    omega: photon energies (eV, > 0).  mu: chemical potential (eV).
    eta: broadening of the energy-conservation delta (eV);
    lineshape: "gaussian" (default) or "lorentzian".
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
            if lineshape == "gaussian":
                g = np.exp(-0.5 * ((dE - hw) / eta) ** 2) \
                    / (eta * np.sqrt(2.0 * np.pi))
            elif lineshape == "lorentzian":
                g = (eta / np.pi) / ((dE - hw) ** 2 + eta ** 2)
            else:
                raise ValueError("lineshape must be gaussian or lorentzian")
            sig[iw] += w * (df * A2 * g / np.where(mask, dE, 1.0))[mask].sum()
    # sigma / (e^2 / 4 hbar) = spin * 4 pi / area * sum, M in eV*A, dE in eV
    return sig * spin * 4.0 * np.pi / area


def sigma_tensor(model, omega, mu, directions=(0, 1), mesh=None, kpts=None,
                 weights=None, T=300.0, eta=1e-3, spin=2, thresh=1e-10):
    """Complex interband conductivity tensor component sigma_ab(omega),
    in units of e^2 / (4 hbar).

    directions: the pair (a, b) of Cartesian axes; (0, 1) is sigma_xy,
    the finite-frequency Hall conductivity, and (0, 0) recovers the
    longitudinal component (its real part agrees with
    :func:`sigma_optical` with the Lorentzian lineshape up to the
    antiresonant terms sigma_optical drops -- asserted in the tests).

    The Kubo formula summed over interband pairs only,

        sigma_ab = -(4 i spin / A) sum_k w_k sum_{n != m}
                   (f_n - f_m) M^a_nm M^b_mn
                   / [dE_mn (dE_mn - omega - i eta)],

    with the same nonorthogonal velocity M as sigma_optical.  omega = 0
    is allowed (eta regularizes).  The intraband (Drude) part is *not*
    included; combine with :func:`drude_weight` for metals.

    Anchor (TKNN; Thouless, Kohmoto, Nightingale and den Nijs, Phys.
    Rev. Lett. 49, 405 (1982)): for a Chern insulator at T -> 0 with mu
    in the gap, sigma_xy(0) = C e^2/h = (2 C / pi) in these units, with
    C exactly the package's :func:`hamop.chern_number` of the occupied
    bands -- the test suite asserts that internal consistency on the
    Haldane model, sign included.
    """
    a, b = directions
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
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
    sig = np.zeros(len(omega), dtype=complex)
    for k, w in zip(kpts, weights):
        H, S = model.bloch(k)
        dHa, dSa = model.bloch_derivative(k, a)
        e, c = gen_eigh(H, S, thresh=thresh, eigvals_only=False)
        Ma = c.conj().T @ dHa @ c \
            - 0.5 * (e[:, None] + e[None, :]) * (c.conj().T @ dSa @ c)
        if b == a:
            Mb = Ma
        else:
            dHb, dSb = model.bloch_derivative(k, b)
            Mb = c.conj().T @ dHb @ c \
                - 0.5 * (e[:, None] + e[None, :]) * (c.conj().T @ dSb @ c)
        x = np.clip((e - mu) / (KB * T), -60.0, 60.0)
        f = 1.0 / (1.0 + np.exp(x))
        dE = e[None, :] - e[:, None]          # E_m - E_n
        df = f[:, None] - f[None, :]          # f_n - f_m
        num = df * Ma * Mb.T                  # (f_n - f_m) M^a_nm M^b_mn
        mask = np.abs(dE) > 1e-9              # strictly interband
        for iw, hw in enumerate(omega):
            den = dE * (dE - hw - 1j * eta)
            sig[iw] += w * (num[mask] / den[mask]).sum()
    return sig * (-4j) * spin / area


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


def drude_weight(model, mu, mesh=None, kpts=None, weights=None, T=300.0,
                 direction=0, spin=2, thresh=1e-10):
    """Intraband (Drude) weight, in units of e^2/(4 hbar) times eV.

    Defined so that the intraband part of the conductivity, in the same
    units as :func:`sigma_optical`, is sigma_intra(hw) = D delta(hw):

        D = spin * (4 pi / A) * sum_k w_k sum_n |M_nn|^2 (-df/de)|_{e_n},

    with the same nonorthogonal velocity M_nn = <n| dH/dk - e_n dS/dk |n>
    as the interband formula, so D shares its exact invariance under a
    shift of the energy zero.  D vanishes for a filled or empty band and
    is pinned in the tests to the closed form of the half-filled chain,
    D = 8 spin |t| a  (from (4 pi / a) (a / 2 pi) * integral dk
    v(k)^2 delta(E(k)) with v = -2 t a sin ka).
    """
    if mesh is not None:
        kpts, weights = model.monkhorst_pack(mesh)
        area = model.cell_volume
    elif kpts is not None:
        if weights is None:
            weights = np.full(len(kpts), 1.0 / len(kpts))
        area = model.cell_volume
    else:
        if model.cell is not None:
            raise ValueError("periodic model: give mesh or kpts")
        kpts, weights, area = [None], [1.0], 1.0
    D = 0.0
    for k, w in zip(kpts, weights):
        H, S = model.bloch(k)
        dH, dS = model.bloch_derivative(k, direction)
        e, c = gen_eigh(H, S, thresh=thresh, eigvals_only=False)
        x = np.clip((e - mu) / (KB * T), -60.0, 60.0)
        mdfde = 1.0 / (4.0 * KB * T * np.cosh(0.5 * x) ** 2)
        v_nn = np.real(np.einsum("in,ij,jn->n", c.conj(), dH, c)
                       - e * np.einsum("in,ij,jn->n", c.conj(), dS, c))
        D += w * float((v_nn ** 2 * mdfde).sum())
    return D * spin * 4.0 * np.pi / area
