"""Landauer transmission by nonequilibrium Green functions.

Two-probe geometry: a device of N principal layers between two
semi-infinite periodic leads, with only nearest-layer coupling (choose
the principal layer at least as wide as the interaction range).  The
surface Green function of each lead is computed by the Sancho-Rubio
decimation (M. P. Lopez Sancho, J. M. Lopez Sancho and J. Rubio,
J. Phys. F 15, 851 (1985)); the device is traversed by the standard
recursive Green function sweep, and the transmission is the Caroli
trace  T = Tr[ Gamma_R G Gamma_L G^dagger ].

Everything takes explicit layer blocks, so any Hamiltonian source --
built by hand, assembled from a TightBindingModel supercell, or
imported from an LCAO code -- can be pushed through the same
transmission function.  Nonorthogonal bases are supported throughout
(energy-dependent coupling z S - H).

The test suite checks the analytic single-band chain: unit transmission
across the band and zero outside, the closed-form surface Green
function, the closed-form single-impurity transmission, and exact
agreement between the recursive sweep and a direct inversion of the
full device Green function.
"""
from __future__ import annotations

import numpy as np

__all__ = ["sancho_rubio", "transmission", "transmission_direct"]


def sancho_rubio(E, H00, H01, S00=None, S01=None, eta=1e-6, maxiter=400,
                 tol=1e-12):
    """Retarded surface Green function of a semi-infinite periodic lead.

    H00: principal-layer block; H01: coupling from one layer to the
    next deeper layer.  S blocks default to identity / zero
    (orthogonal basis).
    """
    n = len(H00)
    S00 = np.eye(n, dtype=complex) if S00 is None else S00
    S01 = np.zeros_like(H01) if S01 is None else S01
    z = E + 1j * eta
    a = z * S01 - H01
    b = a.conj().T
    es = e = z * S00 - H00
    I = np.eye(n, dtype=complex)
    for _ in range(maxiter):
        g = np.linalg.solve(e, I)
        ab = a @ g @ b
        ba = b @ g @ a
        es = es - ab
        e = e - ab - ba
        a = a @ g @ a
        b = b @ g @ b
        if np.abs(a).max() + np.abs(b).max() < tol:
            break
    return np.linalg.solve(es, I)


def _lead_sigmas(E, lead_H00, lead_H01, lead_S00, lead_S01, eta):
    """Self-energies and broadenings of the left and right leads."""
    z = E + 1j * eta
    gL = sancho_rubio(E, lead_H00, lead_H01.conj().T, lead_S00,
                      None if lead_S01 is None else lead_S01.conj().T, eta)
    gR = sancho_rubio(E, lead_H00, lead_H01, lead_S00, lead_S01, eta)
    S01 = np.zeros_like(lead_H01) if lead_S01 is None else lead_S01
    tau = z * S01 - lead_H01
    sigL = tau.conj().T @ gL @ tau
    sigR = tau @ gR @ tau.conj().T
    gamL = 1j * (sigL - sigL.conj().T)
    gamR = 1j * (sigR - sigR.conj().T)
    return sigL, sigR, gamL, gamR


def transmission(E_list, layers_H, coup_H, lead_H00, lead_H01,
                 layers_S=None, coup_S=None, lead_S00=None, lead_S01=None,
                 eta=1e-6):
    """T(E) by the recursive Green function sweep.

    layers_H[i]: on-layer Hamiltonian of device layer i.
    coup_H[i]: coupling from layer i to layer i+1 (N-1 blocks).
    lead_H00 / lead_H01: principal layer of the identical left and right
    leads.  The outermost device layers must couple to the leads through
    lead_H01, i.e. they must be lead-like at their outer edge.
    """
    N = len(layers_H)
    layers_S = [None] * N if layers_S is None else layers_S
    coup_S = [None] * (N - 1) if coup_S is None else coup_S
    T = np.zeros(len(E_list))
    for iE, E in enumerate(E_list):
        z = E + 1j * eta
        sigL, sigR, gamL, gamR = _lead_sigmas(
            E, lead_H00, lead_H01, lead_S00, lead_S01, eta)
        Gs = []
        g_prev = None
        for i in range(N):
            Si = layers_S[i]
            h_eff = (z * (np.eye(len(layers_H[i])) if Si is None else Si)
                     - layers_H[i])
            if i == 0:
                h_eff = h_eff - sigL
            if i == N - 1:
                h_eff = h_eff - sigR
            if i == 0:
                g_prev = np.linalg.inv(h_eff)
            else:
                Sc = coup_S[i - 1]
                tau = (z * (np.zeros_like(coup_H[i - 1]) if Sc is None
                            else Sc) - coup_H[i - 1])
                g_prev = np.linalg.inv(h_eff - tau.conj().T @ g_prev @ tau)
            Gs.append(g_prev)
        prod = Gs[-1]
        for i in range(N - 2, -1, -1):
            Sc = coup_S[i]
            tau = (z * (np.zeros_like(coup_H[i]) if Sc is None else Sc)
                   - coup_H[i])
            prod = prod @ tau.conj().T @ Gs[i]
        G1N = prod          # G_{N,1}: right edge <- left edge
        T[iE] = float(np.real(np.trace(
            gamR @ G1N @ gamL @ G1N.conj().T)))
    return T


def transmission_direct(E_list, layers_H, coup_H, lead_H00, lead_H01,
                        layers_S=None, coup_S=None, lead_S00=None,
                        lead_S01=None, eta=1e-6):
    """T(E) by direct inversion of the full device Green function.

    Numerically exact reference for :func:`transmission` on small
    devices; the recursive sweep must agree with this to machine
    precision, and the test suite asserts that it does.
    """
    N = len(layers_H)
    layers_S = [None] * N if layers_S is None else layers_S
    coup_S = [None] * (N - 1) if coup_S is None else coup_S
    sizes = [len(h) for h in layers_H]
    offs = np.concatenate([[0], np.cumsum(sizes)])
    ntot = offs[-1]
    T = np.zeros(len(E_list))
    for iE, E in enumerate(E_list):
        z = E + 1j * eta
        sigL, sigR, gamL, gamR = _lead_sigmas(
            E, lead_H00, lead_H01, lead_S00, lead_S01, eta)
        A = np.zeros((ntot, ntot), dtype=complex)
        for i in range(N):
            Si = layers_S[i]
            blk = (z * (np.eye(sizes[i]) if Si is None else Si)
                   - layers_H[i])
            A[offs[i]:offs[i + 1], offs[i]:offs[i + 1]] = blk
            if i < N - 1:
                Sc = coup_S[i]
                tau = (z * (np.zeros_like(coup_H[i]) if Sc is None else Sc)
                       - coup_H[i])
                A[offs[i]:offs[i + 1], offs[i + 1]:offs[i + 2]] = tau
                A[offs[i + 1]:offs[i + 2], offs[i]:offs[i + 1]] = \
                    tau.conj().T
        A[offs[0]:offs[1], offs[0]:offs[1]] -= sigL
        A[offs[N - 1]:offs[N], offs[N - 1]:offs[N]] -= sigR
        G = np.linalg.inv(A)
        G1N = G[offs[N - 1]:offs[N], offs[0]:offs[1]]
        T[iE] = float(np.real(np.trace(
            gamR @ G1N @ gamL @ G1N.conj().T)))
    return T
