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

__all__ = ["sancho_rubio", "transmission", "transmission_direct",
           "buttiker_transmission"]


def _sigma_at(sigma_int, i, E, n):
    """Retarded interaction self-energy of layer i at energy E, or None.

    sigma_int maps layer indices to either a constant (n, n) array or a
    callable E -> (n, n) array (e.g. a self-energy computed by an
    external many-body treatment)."""
    if not sigma_int or i not in sigma_int:
        return None
    s = sigma_int[i]
    s = s(E) if callable(s) else s
    s = np.asarray(s, dtype=complex)
    if s.shape != (n, n):
        raise ValueError(
            f"sigma_int[{i}] has shape {s.shape}, layer needs ({n}, {n})")
    return s


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
                 eta=1e-6, sigma_int=None):
    """T(E) by the recursive Green function sweep.

    layers_H[i]: on-layer Hamiltonian of device layer i.
    coup_H[i]: coupling from layer i to layer i+1 (N-1 blocks).
    lead_H00 / lead_H01: principal layer of the identical left and right
    leads.  The outermost device layers must couple to the leads through
    lead_H01, i.e. they must be lead-like at their outer edge.

    sigma_int: optional dict mapping a layer index to a retarded
    interaction self-energy on that layer -- a constant (n, n) array or
    a callable E -> (n, n) array, supplied by whatever many-body
    treatment produced it.  It is added to the layer verbatim
    (H_i -> H_i + Sigma_i(E)); no self-consistency is performed here,
    and the coherent Caroli trace is what is returned -- with a
    non-Hermitian Sigma it is the coherent part of the current only.
    For phenomenological dephasing with current conservation use
    :func:`buttiker_transmission`.
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
            sg = _sigma_at(sigma_int, i, E, len(layers_H[i]))
            if sg is not None:
                h_eff = h_eff - sg
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
                        lead_S01=None, eta=1e-6, sigma_int=None):
    """T(E) by direct inversion of the full device Green function.

    Numerically exact reference for :func:`transmission` on small
    devices; the recursive sweep must agree with this to machine
    precision, and the test suite asserts that it does.  ``sigma_int``
    as in :func:`transmission`.
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
            sg = _sigma_at(sigma_int, i, E, sizes[i])
            if sg is not None:
                blk = blk - sg
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


def buttiker_transmission(E_list, layers_H, coup_H, lead_H00, lead_H01,
                          probe_layer, gamma, layers_S=None, coup_S=None,
                          lead_S00=None, lead_S01=None, eta=1e-6,
                          return_parts=False):
    """Two-terminal transmission with one current-conserving dephasing
    probe (Buttiker, Phys. Rev. B 33, 3020 (1986)).

    A fictitious voltage probe with broadening ``gamma`` (self-energy
    -i gamma / 2 on every orbital of ``probe_layer``) absorbs and
    reinjects carriers; its chemical potential floats so that it draws
    no net current, which in linear response gives the closed
    composition

        T_eff = T_LR + T_Lp T_pR / (T_Lp + T_pR).

    The three transmissions are Caroli traces of the same full Green
    function, computed by dense inversion.  gamma = 0 recovers the
    coherent T_LR exactly.  Returns T_eff, or (with
    ``return_parts=True``) a dict with T_eff, T_LR, T_Lp, T_pR.
    """
    N = len(layers_H)
    layers_S = [None] * N if layers_S is None else layers_S
    coup_S = [None] * (N - 1) if coup_S is None else coup_S
    p = int(probe_layer)
    if not 0 <= p < N:
        raise ValueError("probe_layer outside the device")
    sizes = [len(h) for h in layers_H]
    offs = np.concatenate([[0], np.cumsum(sizes)])
    ntot = offs[-1]
    out = {key: np.zeros(len(E_list))
           for key in ("T_eff", "T_LR", "T_Lp", "T_pR")}
    for iE, E in enumerate(E_list):
        z = E + 1j * eta
        sigL, sigR, gamL, gamR = _lead_sigmas(
            E, lead_H00, lead_H01, lead_S00, lead_S01, eta)
        A = np.zeros((ntot, ntot), dtype=complex)
        for i in range(N):
            Si = layers_S[i]
            A[offs[i]:offs[i + 1], offs[i]:offs[i + 1]] = \
                z * (np.eye(sizes[i]) if Si is None else Si) - layers_H[i]
            if i < N - 1:
                Sc = coup_S[i]
                tau = (z * (np.zeros_like(coup_H[i]) if Sc is None else Sc)
                       - coup_H[i])
                A[offs[i]:offs[i + 1], offs[i + 1]:offs[i + 2]] = tau
                A[offs[i + 1]:offs[i + 2], offs[i]:offs[i + 1]] = \
                    tau.conj().T
        A[offs[0]:offs[1], offs[0]:offs[1]] -= sigL
        A[offs[N - 1]:offs[N], offs[N - 1]:offs[N]] -= sigR
        A[offs[p]:offs[p + 1], offs[p]:offs[p + 1]] += \
            0.5j * gamma * np.eye(sizes[p])
        G = np.linalg.inv(A)
        gamP = gamma * np.eye(sizes[p])
        G_RL = G[offs[N - 1]:offs[N], offs[0]:offs[1]]
        G_pL = G[offs[p]:offs[p + 1], offs[0]:offs[1]]
        G_Rp = G[offs[N - 1]:offs[N], offs[p]:offs[p + 1]]
        T_LR = float(np.real(np.trace(
            gamR @ G_RL @ gamL @ G_RL.conj().T)))
        T_Lp = float(np.real(np.trace(
            gamP @ G_pL @ gamL @ G_pL.conj().T)))
        T_pR = float(np.real(np.trace(
            gamR @ G_Rp @ gamP @ G_Rp.conj().T)))
        denom = T_Lp + T_pR
        T_eff = T_LR + (T_Lp * T_pR / denom if denom > 1e-300 else 0.0)
        out["T_LR"][iE], out["T_Lp"][iE], out["T_pR"][iE] = T_LR, T_Lp, T_pR
        out["T_eff"][iE] = T_eff
    return out if return_parts else out["T_eff"]
