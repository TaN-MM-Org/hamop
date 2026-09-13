"""Berry curvature dipole: the band-geometric generator of the
nonlinear Hall effect (new in v0.8).

A time-reversal-symmetric crystal has zero anomalous Hall response,
but its SECOND-order Hall response survives, controlled not by the
integrated Berry curvature (which vanishes) but by its first moment
over the occupied states -- the Berry curvature dipole of Sodemann
and Fu, Phys. Rev. Lett. 115, 216806 (2015):

    D_alpha = integral d^2k/(2 pi)^2  sum_n f_n(k) d Omega_n / d k_alpha
            = -integral d^2k/(2 pi)^2 sum_n Omega_n(k) f'(eps_n) v_n,alpha

(equal by integration by parts on the Brillouin torus; in 2D, D has
dimension of length and Omega is the scalar curvature). Symmetry does
real work here, and the tests assert it rather than state it: with
inversion symmetry D vanishes identically (Omega even, its gradient
odd); with time-reversal AND three-fold rotation it vanishes too, so
trigonal crystals like unstrained gapped graphene give exactly zero
and STRAIN is what switches the effect on; a single surviving mirror
line forces D perpendicular to that line; and a completely filled
band contributes exactly nothing (the BZ average of a gradient).

Both integral forms are implemented as genuinely independent paths --
one differentiates the lattice (plaquette-flux) curvature field, the
other uses analytic Fermi-window derivatives with Hellmann-Feynman
velocities -- and the tests require them to agree on a low-symmetry
model where the answer is nonzero.

Conventions: the model's cell is in Angstrom, so Omega is in A^2 and
D in A. Only the intrinsic band-geometry object is computed; turning
D into a measured nonlinear Hall voltage multiplies in a scattering
time and device geometry this package has no business guessing.
"""
from __future__ import annotations

import numpy as np

from .berry import berry_curvature

__all__ = ["band_curvatures", "berry_dipole", "fermi_occupation"]


def fermi_occupation(e, mu, kT):
    """Fermi-Dirac occupation; kT = 0 gives the exact step."""
    e = np.asarray(e, dtype=float)
    if kT < 0:
        raise ValueError("kT must be >= 0")
    if kT == 0:
        return (e < mu).astype(float) + 0.5 * (e == mu)
    x = (e - mu) / kT
    return 0.5 * (1.0 - np.tanh(0.5 * x))       # stable at |x| >> 1


def _dfermi_deps(e, mu, kT):
    """df/d eps, analytic: -1/(4 kT) sech^2((e-mu)/2kT)."""
    x = (np.asarray(e, dtype=float) - mu) / (2.0 * kT)
    return -1.0 / (4.0 * kT * np.cosh(x) ** 2)


def band_curvatures(model, mesh, solver="dense"):
    """Band-resolved lattice Berry curvature on an N x N mesh.

    Returns (energies, omega): energies (N, N, nb) from the Bloch
    Hamiltonian at the plaquette CENTERS -- the same points the
    plaquette fluxes naturally live on, so the discrete symmetry
    cancellations below are exact -- and omega (N, N, nb),
    the curvature DENSITY of each band in A^2 (plaquette flux divided
    by the plaquette area). Band n's flux is obtained as the exact
    abelian difference of the tested multi-band fluxes
    F(n_occ = n+1) - F(n_occ = n), and the top band comes for free:
    the total flux of a complete band set is identically zero (the
    loop product of full-rank links is det of a unitary loop = 1),
    which is also asserted in the tests. Orthogonal-basis models
    only, refused otherwise (the velocity form below would need the
    generalized Hellmann-Feynman term this refuses to half-implement).

    The atomic frame is used deliberately, not by preference: the
    Loewdin frame's periodic identification u(k+G) = u(k) is a gauge
    choice that leaves TOTALS (Chern numbers) exact but redistributes
    flux locally across the zone boundary, and a dipole is a first
    moment -- the C3 symmetry anchor in the tests is precisely what
    caught this. Local curvature fields must come from the frame that
    evaluates the wrapped boundary points explicitly.
    """
    if model.cell is None or model.cell.shape != (2, 2):
        raise ValueError("band_curvatures needs a 2D periodic model")
    if model.has_overlap():
        raise ValueError("band-resolved curvature dipole is implemented "
                         "for orthogonal bases only; refused for overlap "
                         "models rather than dropping the dS terms")
    N = int(mesh)
    recip = 2.0 * np.pi * np.linalg.inv(model.cell).T
    b1, b2 = recip[0], recip[1]
    a_bz = abs(b1[0] * b2[1] - b1[1] * b2[0])
    a_plaq = a_bz / N ** 2
    nb = model.nao
    energies = np.empty((N, N, nb))
    for i in range(N):
        for j in range(N):
            k = ((i + 0.5) / N) * b1 + ((j + 0.5) / N) * b2
            H, _ = model.bloch(k)
            energies[i, j] = np.linalg.eigvalsh(H)
    F = np.zeros((N, N, nb))
    below = np.zeros((N, N))
    for n in range(nb - 1):
        Fn = berry_curvature(model, N, n_occ=n + 1, frame="atomic",
                             solver=solver)
        F[:, :, n] = Fn - below
        below = Fn
    F[:, :, nb - 1] = -below                    # complete set: zero total
    return energies, F / a_plaq


def berry_dipole(model, mesh, mu, kT, method="grad", solver="dense"):
    """Berry curvature dipole D = (D_x, D_y) in Angstrom.

    mesh : N for the N x N Brillouin-zone mesh (symmetry cancellations
    are exact only on meshes the point group maps to itself, which a
    Gamma-centered N x N mesh in reduced coordinates always is).
    mu, kT : chemical potential and temperature (energy units of the
    model).
    method : "grad" sums f * dOmega/dk with central differences of the
    plaquette-flux field (works at kT = 0); "fermi" sums
    -Omega * f'(eps) * v with the analytic Fermi-window derivative and
    Hellmann-Feynman velocities v = <u|dH/dk|u> (needs kT > 0). The
    two are independent discretizations of the two sides of the
    integration by parts, and the tests require their agreement.

    Returns dict(D, per_band) with per_band an (nb, 2) array of the
    band contributions.
    """
    N = int(mesh)
    energies, omega = band_curvatures(model, N, solver=solver)
    nb = energies.shape[2]
    recip = 2.0 * np.pi * np.linalg.inv(model.cell).T
    b1, b2 = recip[0], recip[1]
    a_cell = abs(np.linalg.det(model.cell))
    # d/dk_alpha = sum_m (B^-1)_{alpha m} d/ds_m  with k = B^T s
    Binv = np.linalg.inv(np.stack([b1, b2]))
    per_band = np.zeros((nb, 2))
    if method == "grad":
        f = fermi_occupation(energies, mu, kT)
        for n in range(nb):
            dO_ds1 = (np.roll(omega[:, :, n], -1, axis=0)
                      - np.roll(omega[:, :, n], 1, axis=0)) * (N / 2.0)
            dO_ds2 = (np.roll(omega[:, :, n], -1, axis=1)
                      - np.roll(omega[:, :, n], 1, axis=1)) * (N / 2.0)
            for a in range(2):
                dO_dka = Binv[a, 0] * dO_ds1 + Binv[a, 1] * dO_ds2
                per_band[n, a] = (f[:, :, n] * dO_dka).sum() / (N ** 2 * a_cell)
    elif method == "fermi":
        if not kT > 0:
            raise ValueError('method="fermi" differentiates the Fermi '
                             "function and needs kT > 0")
        dfde = _dfermi_deps(energies, mu, kT)
        vel = np.empty((N, N, nb, 2))
        for i in range(N):
            for j in range(N):
                k = ((i + 0.5) / N) * b1 + ((j + 0.5) / N) * b2
                H, _ = model.bloch(k)
                _, U = np.linalg.eigh(H)
                for a in range(2):
                    dH, _ = model.bloch_derivative(k, direction=a)
                    vel[i, j, :, a] = np.real(
                        np.einsum("in,ij,jn->n", U.conj(), dH, U))
        for n in range(nb):
            for a in range(2):
                per_band[n, a] = -(omega[:, :, n] * dfde[:, :, n]
                                   * vel[:, :, n, a]).sum() / (N ** 2 * a_cell)
    else:
        raise ValueError('method must be "grad" or "fermi"')
    return dict(D=per_band.sum(axis=0), per_band=per_band)
