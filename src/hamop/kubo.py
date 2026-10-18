"""Kubo-Greenwood optical conductivity from the same Bloch matrices.

The velocity operator uses the standard atomistic (Peierls-like)
position gauge: the position operator is taken diagonal at the sites,
so dH/dk carries a factor of the inter-site displacement -- the common
approximation in tight-binding optics.  The intra-atomic dipole
contribution is left out unless the model carries on-site position
blocks (``TightBindingModel.set_dipole``), which then add
i (E_n - E_m) <n|X|m> to the matrix element.  In a nonorthogonal basis the
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

from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from .eigsolve import _eig_chunks
from .model import KB, check_kgrid, fermi_dirac

if TYPE_CHECKING:
    from .model import TightBindingModel

__all__ = ["sigma_optical", "sigma_tensor", "carrier_count",
           "drude_weight"]


def _check_T(T, strict=False):
    """Refuse temperatures the Fermi factors cannot use: T < 0 always,
    and T = 0 where -df/de must be sampled on a k-mesh (strict)."""
    if strict and not T > 0:
        raise ValueError("T must be > 0 K here: the Fermi-surface "
                         "factor -df/de is sampled on the k-mesh and "
                         "needs thermal smearing")
    if T < 0:
        raise ValueError("T must be >= 0 K")


def _dipole_term(model, c, e, direction):
    """Intra-atomic dipole contribution to the velocity matrix element,
    i (E_n - E_m) <n|X_a|m>, from v = (i/hbar)[H, r] with the on-site
    position blocks the model carries.  None when the model has no
    dipole blocks.

    Nonorthogonal bases are supported: <n|[H, x]|m> = (E_n - E_m)
    <n|x|m> is an operator identity on the (generalized) eigenstates,
    the on-site dipole blocks are basis matrix elements <alpha|x|beta>,
    and the eigenvectors ``c`` are S-normalized -- so the same
    expression applies verbatim, and it inherits the exact invariance
    under H -> H + c S with mu -> mu + c (eigenvector-, X- and
    energy-difference-preserving), which the tests assert."""
    if not model.has_dipoles():
        return None
    ch = np.conj(np.swapaxes(c, -1, -2))
    Xn = ch @ model.dipole_matrix(direction) @ c
    return 1j * (e[..., :, None] - e[..., None, :]) * Xn


def _velocity(model, e, c, deriv, direction):
    """Interband velocity matrix elements M_nm for a block of k-points
    (shapes: e (m, nb), c (m, nao, nb), deriv = (dH, dS) at those
    k-points): <n| dH/dk - (e_n + e_m)/2 dS/dk |m> plus the intra-atomic
    dipole term.  The overlap term is skipped when dS is None (an
    orthogonal model, where dS/dk = 0)."""
    dH, dS = deriv
    ch = np.conj(np.swapaxes(c, 1, 2))
    M = ch @ dH @ c
    if dS is not None:
        M = M - 0.5 * (e[:, :, None] + e[:, None, :]) * (ch @ dS @ c)
    Xt = _dipole_term(model, c, e, direction)
    if Xt is not None:
        M = M + Xt
    return M


def sigma_optical(model: TightBindingModel, omega: ArrayLike, mu: float,
                  mesh: Optional[Union[int, Sequence[int]]] = None,
                  kpts: Optional[ArrayLike] = None,
                  weights: Optional[ArrayLike] = None, T: float = 300.0,
                  eta: float = 0.05, direction: int = 0,
                  spin: float = 2, thresh: float = 1e-10,
                  lineshape: str = "gaussian") -> np.ndarray:
    """Real part of the optical sheet conductivity, in units of e^2/(4 hbar).

    omega: photon energies (eV, > 0).  mu: chemical potential (eV).
    eta: broadening of the energy-conservation delta (eV);
    lineshape: "gaussian" (default) or "lorentzian".
    direction: Cartesian polarization axis.  For a finite system
    (cell=None) the "cell volume" is absent and the result is the
    conductivity times the system area; divide by your geometric area.
    For a model with fewer lattice vectors than dimensions (a ribbon,
    since 0.14.0) the sum is divided by the length of the cell instead
    of an area (unit e^2/(4 hbar) times Angstrom), as for a 1D chain.
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    if np.any(omega <= 0):
        raise ValueError("omega must be positive photon energies")
    if not eta > 0:
        raise ValueError("eta (the line width, eV) must be > 0")
    if lineshape not in ("gaussian", "lorentzian"):
        raise ValueError("lineshape must be gaussian or lorentzian")
    _check_T(T)
    if mesh is not None or kpts is not None:
        if mesh is not None:
            kpts, weights = model.monkhorst_pack(mesh)
        else:
            kpts, weights = check_kgrid(kpts, weights)
        area = model.cell_volume
    else:
        if model.cell is not None:
            raise ValueError("periodic model: give mesh or kpts")
        kpts, weights, area = [None], [1.0], 1.0

    sig = np.zeros_like(omega)
    for w, e, c, der in _eig_chunks(model, kpts, weights, thresh=thresh,
                                    vectors=True, directions=(direction,)):
        M = _velocity(model, e, c, der[direction], direction)
        f = fermi_dirac(e, mu, T)
        dE = e[:, None, :] - e[:, :, None]          # E_m - E_n
        df = f[:, :, None] - f[:, None, :]          # f_n - f_m
        mask = dE > 1e-3
        dEm = dE[mask]
        W = (np.broadcast_to(w[:, None, None], dE.shape)[mask]
             * df[mask] * np.abs(M[mask]) ** 2 / dEm)
        for iw, hw in enumerate(omega):
            if lineshape == "gaussian":
                g = np.exp(-0.5 * ((dEm - hw) / eta) ** 2) \
                    / (eta * np.sqrt(2.0 * np.pi))
            else:
                g = (eta / np.pi) / ((dEm - hw) ** 2 + eta ** 2)
            sig[iw] += float(W @ g)
    # sigma / (e^2 / 4 hbar) = spin * 4 pi / area * sum, M in eV*A, dE in eV
    return sig * spin * 4.0 * np.pi / area


def sigma_tensor(model: TightBindingModel, omega: ArrayLike, mu: float,
                 directions: Tuple[int, int] = (0, 1),
                 mesh: Optional[Union[int, Sequence[int]]] = None,
                 kpts: Optional[ArrayLike] = None,
                 weights: Optional[ArrayLike] = None, T: float = 300.0,
                 eta: float = 1e-3, spin: float = 2,
                 thresh: float = 1e-10) -> np.ndarray:
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

    For a != b, do not pass a grid folded by time reversal
    (``monkhorst_pack(..., time_reversal=True)``): the fold keeps one of
    each pair k, -k, whose Hall integrands are complex conjugates of
    each other, so the imaginary parts that cancel on the full grid are
    doubled instead.  Gapped graphene (``haldane(t2=0.0, m_ab=0.3)``,
    mesh 30) gives sigma_xy(0) = -0.912 on the folded grid and 0 on the
    full one.  The diagonal components (a == b) fold exactly.

    Anchor (TKNN; Thouless, Kohmoto, Nightingale and den Nijs, Phys.
    Rev. Lett. 49, 405 (1982)): for a Chern insulator at T -> 0 with mu
    in the gap, sigma_xy(0) = C e^2/h = (2 C / pi) in these units, with
    C exactly the package's :func:`hamop.chern_number` of the occupied
    bands -- the test suite asserts that internal consistency on the
    Haldane model, sign included.
    """
    a, b = directions
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    if not eta >= 0:
        raise ValueError("eta must be >= 0 (a negative eta gives the "
                         "advanced, not the retarded, response)")
    _check_T(T)
    if mesh is not None or kpts is not None:
        if mesh is not None:
            kpts, weights = model.monkhorst_pack(mesh)
        else:
            kpts, weights = check_kgrid(kpts, weights)
        area = model.cell_volume
    else:
        if model.cell is not None:
            raise ValueError("periodic model: give mesh or kpts")
        kpts, weights, area = [None], [1.0], 1.0
    sig = np.zeros(len(omega), dtype=complex)
    dirs = (a,) if b == a else (a, b)
    for w, e, c, der in _eig_chunks(model, kpts, weights, thresh=thresh,
                                    vectors=True, directions=dirs):
        Ma = _velocity(model, e, c, der[a], a)
        Mb = Ma if b == a else _velocity(model, e, c, der[b], b)
        f = fermi_dirac(e, mu, T)
        dE = e[:, None, :] - e[:, :, None]          # E_m - E_n
        df = f[:, :, None] - f[:, None, :]          # f_n - f_m
        num = (w[:, None, None] * df) * Ma * np.swapaxes(Mb, 1, 2)
        mask = np.abs(dE) > 1e-9                    # strictly interband
        num, dEm = num[mask], dE[mask]
        for iw, hw in enumerate(omega):
            sig[iw] += (num / (dEm * (dEm - hw - 1j * eta))).sum()
    return sig * (-4j) * spin / area


def carrier_count(model: TightBindingModel, mu: float,
                  mesh: Optional[Union[int, Sequence[int]]] = None,
                  kpts: Optional[ArrayLike] = None,
                  weights: Optional[ArrayLike] = None, T: float = 300.0,
                  spin: float = 2, thresh: float = 1e-10) -> float:
    """Mean number of occupied states per unit cell (spin included).

    A periodic model needs ``mesh`` or ``kpts``; a finite model
    (cell=None) is evaluated at its single k = 0."""
    _check_T(T)
    if mesh is not None:
        kpts, weights = model.monkhorst_pack(mesh)
    elif kpts is None:
        if model.cell is not None:
            raise ValueError("periodic model: give mesh or kpts")
        kpts, weights = [None], [1.0]
    else:
        kpts, weights = check_kgrid(kpts, weights)
    n = 0.0
    for w, e, _, _ in _eig_chunks(model, kpts, weights, thresh=thresh):
        n += spin * float(w @ fermi_dirac(e, mu, T).sum(axis=1))
    return n


def drude_weight(model: TightBindingModel, mu: float,
                 mesh: Optional[Union[int, Sequence[int]]] = None,
                 kpts: Optional[ArrayLike] = None,
                 weights: Optional[ArrayLike] = None, T: float = 300.0,
                 direction: int = 0, spin: float = 2,
                 thresh: float = 1e-10, degen_tol: float = 1e-9) -> float:
    """Intraband (Drude) weight, in units of e^2/(4 hbar) times eV.

    Defined so that the intraband part of the conductivity, in the same
    units as :func:`sigma_optical`, is sigma_intra(hw) = D delta(hw):

        D = spin * (4 pi / A) * sum_k w_k sum_{n, m: e_n = e_m}
            |M_nm|^2 (-df/de)|_{e_n},

    with the same nonorthogonal velocity M_nm = <n| dH/dk - (e_n + e_m)/2
    dS/dk |m> as the interband formula, so D shares its exact
    invariance under a shift of the energy zero.  The sum runs over the
    pairs of states with equal energy (|e_n - e_m| <= ``degen_tol``,
    eV): the diagonal n = m and, where bands are degenerate at a
    k-point, the pairs inside the degenerate set (the zero-frequency
    limit (f_n - f_m)/(e_m - e_n) -> -df/de of the Kubo formula).  Away
    from degeneracies only n = m contributes.  D vanishes for a filled
    or empty band and is pinned in the tests to the closed form of the
    half-filled chain, D = 8 spin |t| a  (from (4 pi / a) (a / 2 pi) *
    integral dk v(k)^2 delta(E(k)) with v = -2 t a sin ka).

    Before 0.13.0 only n = m was summed.  At a k-point where two bands
    are degenerate (a Dirac point on the mesh, for example) that
    depended on which eigenvectors the solver returned inside the
    degenerate set, and graphene at mu = 0 on a mesh through K came out
    anisotropic (D_xx and D_yy 7 % apart, where the hexagonal symmetry
    makes them equal).

    T must be > 0: the factor -df/de is sampled on the k-points, so it
    needs thermal smearing (T = 0 is refused).
    """
    _check_T(T, strict=True)
    if not degen_tol >= 0:
        raise ValueError("degen_tol (eV) must be >= 0")
    if mesh is not None:
        kpts, weights = model.monkhorst_pack(mesh)
        area = model.cell_volume
    elif kpts is not None:
        kpts, weights = check_kgrid(kpts, weights)
        area = model.cell_volume
    else:
        if model.cell is not None:
            raise ValueError("periodic model: give mesh or kpts")
        kpts, weights, area = [None], [1.0], 1.0
    D = 0.0
    for w, e, c, der in _eig_chunks(model, kpts, weights, thresh=thresh,
                                    vectors=True, directions=(direction,)):
        V = _velocity(model, e, c, der[direction], direction)
        x = np.clip((e - mu) / (KB * T), -60.0, 60.0)
        mdfde = 1.0 / (4.0 * KB * T * np.cosh(0.5 * x) ** 2)
        v_nn = np.real(np.diagonal(V, axis1=1, axis2=2))
        D += float(w @ (v_nn ** 2 * mdfde).sum(axis=1))
        nb = e.shape[1]
        pair = np.abs(e[:, :, None] - e[:, None, :]) <= degen_tol
        pair &= ~np.eye(nb, dtype=bool)[None]
        if np.any(pair):                  # degenerate sets at some k
            extra = (np.abs(V) ** 2 * mdfde[:, :, None]) * pair
            D += float(w @ extra.sum(axis=(1, 2)))
    return D * spin * 4.0 * np.pi / area
