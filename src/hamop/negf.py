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
           "buttiker_transmission", "scba_transmission",
           "transmission_sparse", "multiprobe_transmission"]


def _back(z, Hc, Sc):
    """The reverse coupling block of z S - H: z S_c^dag - H_c^dag.

    It is NOT the conjugate transpose of z S_c - H_c, because z is
    complex (z = E + i eta); the two differ by 2 i eta S_c^dag, which
    matters whenever the overlap couples the two blocks."""
    Hc = np.asarray(Hc)
    back = -Hc.conj().T.astype(complex)
    if Sc is not None:
        back = back + z * np.asarray(Sc).conj().T
    return back


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
                 tol=1e-12, check_tol=1e-10):
    """Retarded surface Green function of a semi-infinite periodic lead.

    H00: principal-layer block; H01: coupling from one layer to the
    next deeper layer.  S blocks default to identity / zero
    (orthogonal basis).

    The result is checked, not trusted (since 0.11.0).  The surface
    Green function g must satisfy its own Dyson equation

        g = (A - a g b)^-1,   A = z S00 - H00,  a = z S01 - H01,
                              b = z S01^dag - H01^dag,

    and, being retarded, i (g - g^dag) must have no negative
    eigenvalues.  The decimation (Lopez Sancho et al.) is tried first.
    It can fail in two ways: at eta = 0 inside a band it never
    converges, and when E coincides with an eigenvalue of H00 (e.g.
    E = 0 for a chain with on-site energy 0) its first steps divide by
    a number of size eta, so with a small eta it loses up to
    log10(1/eta) digits and can reach a wrong, self-inconsistent fixed
    point (for the chain at E = 0, eta = 1e-8 it gives -6.7e7 i
    instead of -i).  If the decimation fails the Dyson check
    (relative residual above ``check_tol``) and eta > 0, the surface
    Green function is computed a second, independent way from the
    lead's decaying modes: the solutions psi_{n+1} = lambda psi_n of
    b + A lambda + a lambda^2 = 0 with |lambda| < 1, giving
    g = (A + a F)^-1 with F = U diag(lambda) U^-1 (the mode-matching
    form of D. H. Lee and J. D. Joannopoulos, Phys. Rev. B 23, 4988 and
    4997 (1981)).
    That result must pass the same checks.  If neither route passes,
    RuntimeError is raised instead of returning a wrong number; this
    always happens at eta = 0 inside a band.
    """
    n = len(H00)
    S00 = np.eye(n, dtype=complex) if S00 is None else np.asarray(S00)
    S01 = np.zeros_like(H01) if S01 is None else np.asarray(S01)
    z = E + 1j * eta
    A = z * S00 - H00
    a0 = z * S01 - H01
    b0 = _back(z, H01, S01)
    I = np.eye(n, dtype=complex)

    def dyson_residual(g):
        R = (A - a0 @ g @ b0) @ g - I
        return float(np.abs(R).max())

    def retarded(g):
        spec = 1j * (g - g.conj().T)
        w = np.linalg.eigvalsh(0.5 * (spec + spec.conj().T))
        return bool(w.min() >= -check_tol * max(1.0, float(np.abs(g).max())))

    # route 1: decimation
    a, b = a0, b0
    es = e = A
    converged = False
    with np.errstate(all="ignore"):
        for _ in range(int(maxiter)):
            g = np.linalg.solve(e, I)
            ab = a @ g @ b
            ba = b @ g @ a
            es = es - ab
            e = e - ab - ba
            a = a @ g @ a
            b = b @ g @ b
            if np.abs(a).max() + np.abs(b).max() < tol:
                converged = True
                break
        g_dec = np.linalg.solve(es, I) if converged else None
    res_dec = np.inf
    if converged and np.all(np.isfinite(g_dec)):
        res_dec = dyson_residual(g_dec)
        if res_dec <= check_tol and retarded(g_dec):
            return g_dec
    # route 2: decaying modes (needs eta > 0 to tell retarded from advanced)
    res_mod = np.inf
    if eta > 0:
        g_mod = _surface_from_modes(A, a0, b0)
        if g_mod is not None:
            res_mod = dyson_residual(g_mod)
            if res_mod <= check_tol and retarded(g_mod):
                return g_mod
    why = ("the decimation did not converge" if not converged else
           f"the decimation's Dyson residual is {res_dec:.2e}")
    if eta > 0:
        why += f"; the mode route's residual is {res_mod:.2e}"
    raise RuntimeError(
        f"lead surface Green function failed at E = {E}, eta = {eta} "
        f"({why}; check_tol = {check_tol:g}).  Inside a lead band the "
        "calculation needs eta > 0; a wrong surface Green function is "
        "never returned")


def _surface_from_modes(A, a, b):
    """Surface Green function (A + a F)^-1 from the n decaying modes of
    b + A lam + a lam^2 = 0 (generalized companion eigenproblem, so a
    singular coupling a gives infinite eigenvalues that are discarded
    and a singular b gives lam = 0 modes that are kept).  None when the
    modes do not split cleanly into n with |lam| < 1 and n with
    |lam| > 1, or when their eigenvectors are singular."""
    from scipy.linalg import eig
    n = A.shape[0]
    I = np.eye(n, dtype=complex)
    Z = np.zeros((n, n), dtype=complex)
    M1 = np.block([[Z, I], [-b, -A]])
    M2 = np.block([[I, Z], [Z, a]])
    with np.errstate(all="ignore"):
        lam, V = eig(M1, M2)
    mag = np.where(np.isfinite(lam), np.abs(lam), np.inf)
    order = np.argsort(mag)
    if not (mag[order[n - 1]] < 1.0 < mag[order[n]]):
        return None
    idx = order[:n]
    U = V[:n, idx]
    if np.linalg.cond(U) > 1e12:
        return None
    F = (U * lam[idx]) @ np.linalg.inv(U)
    try:
        return np.linalg.solve(A + a @ F, I)
    except np.linalg.LinAlgError:
        return None


def _lead_sigmas(E, lead_H00, lead_H01, lead_S00, lead_S01, eta):
    """Self-energies and broadenings of the left and right leads."""
    z = E + 1j * eta
    gL = sancho_rubio(E, lead_H00, lead_H01.conj().T, lead_S00,
                      None if lead_S01 is None else lead_S01.conj().T, eta)
    gR = sancho_rubio(E, lead_H00, lead_H01, lead_S00, lead_S01, eta)
    S01 = np.zeros_like(lead_H01) if lead_S01 is None else lead_S01
    tau = z * S01 - lead_H01
    tau_b = _back(z, lead_H01, lead_S01)
    sigL = tau_b @ gL @ tau
    sigR = tau @ gR @ tau_b
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
                g_prev = np.linalg.inv(
                    h_eff - _back(z, coup_H[i - 1], Sc) @ g_prev @ tau)
            Gs.append(g_prev)
        prod = Gs[-1]
        for i in range(N - 2, -1, -1):
            Sc = coup_S[i]
            tau = (z * (np.zeros_like(coup_H[i]) if Sc is None else Sc)
                   - coup_H[i])
            prod = prod @ _back(z, coup_H[i], Sc) @ Gs[i]
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
                    _back(z, coup_H[i], Sc)
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
                    _back(z, coup_H[i], Sc)
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


def _assemble_device(E, layers_H, coup_H, layers_S, coup_S, eta,
                     sigma_int=None):
    """Full device matrix A = z S - H - Sigma_int(E) (leads not yet
    attached) plus the block offsets."""
    N = len(layers_H)
    sizes = [len(h) for h in layers_H]
    offs = np.concatenate([[0], np.cumsum(sizes)])
    z = E + 1j * eta
    A = np.zeros((offs[-1], offs[-1]), dtype=complex)
    for i in range(N):
        Si = layers_S[i]
        blk = z * (np.eye(sizes[i]) if Si is None else Si) - layers_H[i]
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
                _back(z, coup_H[i], Sc)
    return A, offs


def scba_transmission(E_list, layers_H, coup_H, lead_H00, lead_H01, W2,
                      layers_S=None, coup_S=None, lead_S00=None,
                      lead_S01=None, eta=1e-6, mixing=0.5, tol=1e-10,
                      maxiter=1000, return_info=False):
    """Transmission through a device with an elastic self-consistent
    Born (SCBA) self-energy for uncorrelated on-site disorder.

    Convention: on-site disorder of variance W2 (eV^2) per orbital,
    uncorrelated between orbitals, gives the retarded self-energy

        Sigma_i(E) = W2_i * diag( G_ii(E) )      (diagonal per orbital),

    iterated to self-consistency with the full device Green function
    (leads attached) by damped fixed-point iteration.  W2: scalar, or
    one value per layer.  This is the standard first-order
    self-consistent Born treatment of disorder averaging; the returned
    T is the Caroli trace of the *averaged* Green function -- vertex
    corrections to the conductance are not included, stated plainly.
    Only elastic (energy-diagonal) self-energies are treated;
    inelastic (Keldysh) electron-phonon SCBA is not implemented.

    Anchors asserted in the tests: W2 = 0 reproduces the coherent
    transmission exactly; the converged Sigma satisfies its own
    equation to below ``tol``; Im Sigma <= 0 (retarded causality); and
    the central-layer Sigma of a long uniform chain reproduces the
    *bulk* scalar SCBA equation Sigma = W2 * g_loc(E - Sigma), solved
    independently from the closed-form chain Green function.

    Returns T(E), or (T, info) with info["sigma"] the converged
    per-layer self-energies, info["niter"], info["residual"].
    """
    N = len(layers_H)
    layers_S = [None] * N if layers_S is None else layers_S
    coup_S = [None] * (N - 1) if coup_S is None else coup_S
    sizes = [len(h) for h in layers_H]
    W2 = np.broadcast_to(np.asarray(W2, dtype=float), (N,))
    T = np.zeros(len(E_list))
    info = {"sigma": [], "niter": [], "residual": []}
    for iE, E in enumerate(E_list):
        sigL, sigR, gamL, gamR = _lead_sigmas(
            E, lead_H00, lead_H01, lead_S00, lead_S01, eta)
        A0, offs = _assemble_device(E, layers_H, coup_H, layers_S,
                                    coup_S, eta)
        A0[offs[0]:offs[1], offs[0]:offs[1]] -= sigL
        A0[offs[N - 1]:offs[N], offs[N - 1]:offs[N]] -= sigR
        sig = [np.zeros(sizes[i], dtype=complex) for i in range(N)]
        res = np.inf
        for it in range(maxiter):
            A = A0.copy()
            for i in range(N):
                idx = np.arange(offs[i], offs[i + 1])
                A[idx, idx] -= sig[i]
            G = np.linalg.inv(A)
            res = 0.0
            new = []
            for i in range(N):
                gd = np.diag(G[offs[i]:offs[i + 1], offs[i]:offs[i + 1]])
                target = W2[i] * gd
                res = max(res, float(np.abs(target - sig[i]).max()))
                new.append((1.0 - mixing) * sig[i] + mixing * target)
            sig = new
            if res < tol:
                break
        else:
            raise RuntimeError(
                f"SCBA did not converge at E = {E} "
                f"(residual {res:.2e} after {maxiter} iterations); "
                "try smaller mixing or larger eta")
        # transmission of the averaged Green function
        A = A0.copy()
        for i in range(N):
            idx = np.arange(offs[i], offs[i + 1])
            A[idx, idx] -= sig[i]
        G = np.linalg.inv(A)
        G1N = G[offs[N - 1]:offs[N], offs[0]:offs[1]]
        T[iE] = float(np.real(np.trace(
            gamR @ G1N @ gamL @ G1N.conj().T)))
        info["sigma"].append(sig)
        info["niter"].append(it + 1)
        info["residual"].append(res)
    return (T, info) if return_info else T


def transmission_sparse(E_list, layers_H, coup_H, lead_H00, lead_H01,
                        layers_S=None, coup_S=None, lead_S00=None,
                        lead_S01=None, eta=1e-6, sigma_int=None):
    """T(E) with the full device matrix stored sparse and factorized by
    sparse LU -- the large-device counterpart of
    :func:`transmission_direct`, solving only for the left-edge block
    of the Green function.  Must agree with the dense routines to
    machine precision (asserted in the tests), including overlap and
    ``sigma_int``.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import splu
    N = len(layers_H)
    layers_S = [None] * N if layers_S is None else layers_S
    coup_S = [None] * (N - 1) if coup_S is None else coup_S
    T = np.zeros(len(E_list))
    for iE, E in enumerate(E_list):
        sigL, sigR, gamL, gamR = _lead_sigmas(
            E, lead_H00, lead_H01, lead_S00, lead_S01, eta)
        A, offs = _assemble_device(E, layers_H, coup_H, layers_S,
                                   coup_S, eta, sigma_int)
        A[offs[0]:offs[1], offs[0]:offs[1]] -= sigL
        A[offs[N - 1]:offs[N], offs[N - 1]:offs[N]] -= sigR
        lu = splu(csr_matrix(A).tocsc())
        rhs = np.zeros((offs[-1], offs[1] - offs[0]), dtype=complex)
        rhs[offs[0]:offs[1]] = np.eye(offs[1] - offs[0])
        X = lu.solve(rhs)
        G1N = X[offs[N - 1]:offs[N], :]      # G_{N,1}
        T[iE] = float(np.real(np.trace(
            gamR @ G1N @ gamL @ G1N.conj().T)))
    return T


def multiprobe_transmission(E_list, layers_H, coup_H, lead_H00, lead_H01,
                            gamma, probe_layers=None, layers_S=None,
                            coup_S=None, lead_S00=None, lead_S01=None,
                            eta=1e-6, return_parts=False):
    """Two-terminal conductance with current-conserving dephasing
    probes on many layers -- the D'Amato-Pastawski model (J. L. D'Amato
    and H. M. Pastawski, Phys. Rev. B 41, 7411 (1990)).

    Each probe layer carries the self-energy -i gamma / 2 per orbital;
    all pairwise Caroli transmissions between {left lead, right lead,
    probes} are computed from one dense Green function, and the probe
    chemical potentials are solved from exact linear-response current
    conservation (zero net current into every probe).  Returns the
    effective transmission T_eff = I_L / (V_L - V_R).

    probe_layers: iterable of layer indices (default: every layer).
    Anchors in the tests: gamma = 0 recovers the coherent result; a
    single probe equals :func:`buttiker_transmission` to machine
    precision; the total current is conserved to machine precision;
    and uniform dephasing produces the Ohmic (linear-in-length)
    resistance of the model's namesake paper.
    """
    N = len(layers_H)
    layers_S = [None] * N if layers_S is None else layers_S
    coup_S = [None] * (N - 1) if coup_S is None else coup_S
    probes = list(range(N)) if probe_layers is None else \
        sorted(int(x) for x in probe_layers)
    if any(not 0 <= x < N for x in probes):
        raise ValueError("probe layer outside the device")
    sizes = [len(h) for h in layers_H]
    T_eff = np.zeros(len(E_list))
    parts = []
    for iE, E in enumerate(E_list):
        sigL, sigR, gamL, gamR = _lead_sigmas(
            E, lead_H00, lead_H01, lead_S00, lead_S01, eta)
        A, offs = _assemble_device(E, layers_H, coup_H, layers_S,
                                   coup_S, eta)
        A[offs[0]:offs[1], offs[0]:offs[1]] -= sigL
        A[offs[N - 1]:offs[N], offs[N - 1]:offs[N]] -= sigR
        for pL in probes:
            idx = np.arange(offs[pL], offs[pL + 1])
            A[idx, idx] += 0.5j * gamma
        G = np.linalg.inv(A)
        # terminals: 0 = L, 1 = R, 2.. = probes
        gams = [gamL, gamR] + [gamma * np.eye(sizes[pL]) for pL in probes]
        blocks = [(offs[0], offs[1]), (offs[N - 1], offs[N])] + \
            [(offs[pL], offs[pL + 1]) for pL in probes]
        nt = len(gams)
        T = np.zeros((nt, nt))
        for a in range(nt):
            ra = slice(*blocks[a])
            for b in range(nt):
                if a == b:
                    continue
                rb = slice(*blocks[b])
                Gab = G[ra, rb]
                T[a, b] = float(np.real(np.trace(
                    gams[a] @ Gab @ gams[b] @ Gab.conj().T)))
        # linear response: I_a = sum_b T_ab (V_a - V_b); V_L = 1, V_R = 0,
        # probes float with I_p = 0
        np_probe = nt - 2
        if np_probe == 0 or gamma == 0.0:
            T_eff[iE] = T[0, 1]
            parts.append({"T": T, "V": np.array([1.0, 0.0])})
            continue
        M = np.zeros((np_probe, np_probe))
        rhs = np.zeros(np_probe)
        for ip in range(np_probe):
            a = ip + 2
            M[ip, ip] = T[a].sum()
            for jp in range(np_probe):
                if jp != ip:
                    M[ip, jp] = -T[a, jp + 2]
            rhs[ip] = T[a, 0] * 1.0 + T[a, 1] * 0.0
        Vp = np.linalg.solve(M, rhs)
        V = np.concatenate([[1.0, 0.0], Vp])
        I_L = float(sum(T[0, b] * (V[0] - V[b]) for b in range(nt)))
        T_eff[iE] = I_L
        parts.append({"T": T, "V": V})
    return (T_eff, parts) if return_parts else T_eff


# ---------------- local spectroscopy and current imaging ----------------

def device_greens(E, layers_H, coup_H, lead_H00, lead_H01, layers_S=None,
                  coup_S=None, lead_S00=None, lead_S01=None, eta=1e-6,
                  sigma_int=None, attach_leads=True):
    """Full retarded device Green function with (optionally) the lead
    self-energies attached, plus the embedded broadening matrices.

    Returns (G, offs, GamL, GamR) with G the (n, n) retarded Green
    function over all device orbitals, offs the layer offsets, and
    GamL/GamR the lead broadenings embedded at full device size (zero
    when attach_leads=False). The plumbing is held to an exact
    identity in the tests rather than trusted:
    i (G - G^dag) = G (GamL + GamR + 2 eta S) G^dag at ANY eta.
    """
    N = len(layers_H)
    layers_S = [None] * N if layers_S is None else layers_S
    coup_S = [None] * (N - 1) if coup_S is None else coup_S
    A, offs = _assemble_device(E, layers_H, coup_H, layers_S, coup_S,
                               eta, sigma_int)
    n = offs[-1]
    GamL = np.zeros((n, n), dtype=complex)
    GamR = np.zeros((n, n), dtype=complex)
    if attach_leads:
        sigL, sigR, gamL, gamR = _lead_sigmas(
            E, lead_H00, lead_H01, lead_S00, lead_S01, eta)
        nl = offs[1]
        nr = n - offs[N - 1]
        A[:nl, :nl] -= sigL
        A[offs[N - 1]:, offs[N - 1]:] -= sigR
        GamL[:nl, :nl] = gamL
        GamR[offs[N - 1]:, offs[N - 1]:] = gamR
    G = np.linalg.inv(A)
    return G, offs, GamL, GamR


def device_ldos(E_list, layers_H, coup_H, lead_H00, lead_H01,
                layers_S=None, coup_S=None, lead_S00=None, lead_S01=None,
                eta=1e-6, sigma_int=None, attach_leads=True):
    """Orbital-resolved local density of states of the device,
    LDOS_i(E) = -Im (G S)_ii / pi (S = identity for orthogonal bases;
    Mulliken convention otherwise). Shape (len(E_list), n_orbitals).

    Anchors in the tests: with the leads detached the LDOS equals the
    exact Lorentzian eigen-sum of the isolated device at the same eta
    to machine precision, and with leads attached the underlying
    spectral function satisfies the exact finite-eta identity of
    `device_greens`.
    """
    N = len(layers_H)
    layers_S_l = [None] * N if layers_S is None else layers_S
    coup_S_l = [None] * (N - 1) if coup_S is None else coup_S
    out = np.empty((len(E_list), sum(len(h) for h in layers_H)))
    for iE, E in enumerate(E_list):
        G, offs, _, _ = device_greens(
            E, layers_H, coup_H, lead_H00, lead_H01, layers_S, coup_S,
            lead_S00, lead_S01, eta, sigma_int, attach_leads)
        if layers_S is None and coup_S is None:
            GS = G
        else:
            # the full device overlap, inter-layer blocks included
            Sfull = np.eye(offs[-1], dtype=complex)
            for i in range(N):
                if layers_S_l[i] is not None:
                    Sfull[offs[i]:offs[i + 1], offs[i]:offs[i + 1]] = \
                        layers_S_l[i]
                if i < N - 1 and coup_S_l[i] is not None:
                    Sc = np.asarray(coup_S_l[i], dtype=complex)
                    Sfull[offs[i]:offs[i + 1], offs[i + 1]:offs[i + 2]] = Sc
                    Sfull[offs[i + 1]:offs[i + 2], offs[i]:offs[i + 1]] = \
                        Sc.conj().T
            GS = G @ Sfull
        out[iE] = -np.imag(np.diag(GS)) / np.pi
    return out


def bond_currents(E, layers_H, coup_H, lead_H00, lead_H01, eta=1e-9,
                  sigma_int=None):
    """Energy-resolved bond-current map at zero temperature for
    left-lead injection, in transmission units (Paulsson and
    Brandbyge, Phys. Rev. B 76, 115117 (2007), orthogonal basis):

        J_ij(E) = 2 Im[ H_ji (G GamL G^dag)_ij ],

    antisymmetric by construction, oriented so that positive J_ij is
    net flow from orbital i to orbital j. Three exact statements are
    asserted in the tests rather than stated: the net current out of
    every interior orbital vanishes (Kirchhoff, up to the eta leakage,
    which is why the default eta is small here); the summed current
    through EVERY inter-layer cut equals the Caroli transmission T(E)
    computed by the independent `transmission` code path; and the
    first/last layers inject/drain exactly +T/-T.

    Orthogonal bases only: the function takes no overlap arguments,
    because the bond-current partition with overlap requires
    energy-dependent bond operators it does not implement.

    Returns (J (n, n) antisymmetric, offs).
    """
    G, offs, GamL, _ = device_greens(
        E, layers_H, coup_H, lead_H00, lead_H01, None, None, None, None,
        eta, sigma_int, attach_leads=True)
    n = offs[-1]
    N = len(layers_H)
    Hfull = np.zeros((n, n), dtype=complex)
    for i in range(N):
        Hfull[offs[i]:offs[i + 1], offs[i]:offs[i + 1]] = layers_H[i]
        if i < N - 1:
            Hfull[offs[i]:offs[i + 1], offs[i + 1]:offs[i + 2]] = coup_H[i]
            Hfull[offs[i + 1]:offs[i + 2], offs[i]:offs[i + 1]] = \
                coup_H[i].conj().T
    AL = G @ GamL @ G.conj().T
    J = -2.0 * np.imag(Hfull * AL.T)
    return J, offs
