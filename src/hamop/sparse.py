"""Sparse assembly and large-system solvers.

Everything dense in the package has a small-matrix bias; this module is
the large-system path.  ``bloch_sparse`` assembles exactly the same
Bloch matrices as :meth:`TightBindingModel.bloch`, as CSR sparse
matrices (the test suite asserts element-for-element agreement with the
dense assembly).  ``lowest_bands`` diagonalizes only the low-energy
window with the Lanczos solver ``scipy.sparse.linalg.eigsh``,
generalized eigenproblem included.  ``kpm_dos`` is the kernel
polynomial method for the density of states -- Chebyshev moments with
the Jackson kernel, following A. Weisse, G. Wellein, A. Alvermann and
H. Fehske, Rev. Mod. Phys. 78, 275 (2006) -- with either an exact
(deterministic) trace or the standard stochastic estimator.

``kpm_sigma`` extends the same machinery to the Kubo-Greenwood optical
conductivity through the double Chebyshev expansion of the
velocity-velocity spectral density (same reference, Sec. V), and
``bloch_derivative_sparse`` supplies the sparse velocity assembly --
with intra-atomic dipole blocks entering as the exact sparse operator
i(HX - XH).  Sparse topology lives in the berry module
(solver="sparse" and the real-space Chern marker) and sparse transport
in the negf module (transmission_sparse).

Deliberate scope, stated plainly: kpm_sigma covers orthogonal bases
and the longitudinal response only.  A KPM *Hall* conductivity of a
finite open system is not offered for a reason the tests prove rather
than assert: in the site-diagonal position formulation the DC Hall
response of any bounded system vanishes identically
(Im Tr[P x Q y] = 0 for Hermitian P and real diagonal x, y), so such
a routine could only ever return broadening artifacts.  The honest
real-space Hall observable for finite systems is the local Chern
marker, :func:`hamop.chern_marker`.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

__all__ = ["bloch_sparse", "bloch_derivative_sparse", "lowest_bands",
           "kpm_dos", "kpm_sigma"]


def _velocity_sparse(model, direction):
    """Sparse velocity operator (times hbar) of a finite orthogonal
    model at k = 0: dH/dk in the site-diagonal position convention,
    plus -- when the model carries dipole blocks -- the exact operator
    form of the intra-atomic term, i (H X - X H), built by sparse
    products.  This is the same physics as the dense eigenbasis route
    i (E_n - E_m) X_nm, expressed without eigenpairs."""
    from scipy.sparse import csr_matrix
    v, _ = bloch_derivative_sparse(model, None, direction)
    if model.has_dipoles():
        H, _ = bloch_sparse(model, None)
        X = csr_matrix(model.dipole_matrix(direction))
        v = v + 1j * (H @ X - X @ H)
    return v


def _vv_moments(H, va, vb, a, b, M, n, n_random, seed):
    """Complex double Chebyshev moments mu_mn = Tr[T_m(H~) va T_n(H~) vb]
    of the rescaled Hamiltonian H~ = (H - b)/a, deterministic
    (n_random=None) or stochastic trace."""

    def cheb_block(V0):
        out = np.empty((M,) + V0.shape, dtype=complex)
        v_prev = V0.astype(complex)
        v_cur = (H @ v_prev - b * v_prev) / a
        out[0] = v_prev
        if M > 1:
            out[1] = v_cur
        for m in range(2, M):
            v_next = 2.0 * ((H @ v_cur) - b * v_cur) / a - v_prev
            out[m] = v_next
            v_prev, v_cur = v_cur, v_next
        return out

    if n_random is None:
        starts = [np.eye(n)]
        norm = 1.0
    else:
        rng = np.random.default_rng(seed)
        starts = [rng.choice([-1.0, 1.0], size=(n, int(n_random)))]
        norm = 1.0 / int(n_random)
    mu_mn = np.zeros((M, M), dtype=complex)
    for V0 in starts:
        C = cheb_block(V0)                       # T_m |r>
        A2 = cheb_block(np.asarray(vb @ V0))     # T_n vb |r>
        VA = np.stack([np.asarray(va @ A2[m]) for m in range(M)])
        Cf = C.reshape(M, -1)
        Vf = VA.reshape(M, -1)
        mu_mn += norm * (Cf.conj() @ Vf.T)
    return mu_mn


def bloch_sparse(model, k=None):
    """H(k), S(k) as CSR sparse matrices; S is None for an orthogonal
    model (identity implied).

    Same assembly, phases and Hermitization as the dense
    :meth:`TightBindingModel.bloch`; agreement is asserted in the
    tests to machine precision.
    """
    k = model._kvec(k)
    n = model.nao
    overlap = model.has_overlap()
    rows, cols, hv, sv = [], [], [], []
    for oi, oj, d, Hb, Sb in model._terms():
        ph = np.exp(1j * float(k @ d))
        ni, nj = Hb.shape
        r, c = np.meshgrid(np.arange(oi, oi + ni),
                           np.arange(oj, oj + nj), indexing="ij")
        rows.append(r.ravel())
        cols.append(c.ravel())
        hv.append((ph * Hb).ravel())
        sv.append((ph * Sb).ravel())
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    H = coo_matrix((np.concatenate(hv), (rows, cols)),
                   shape=(n, n)).tocsr()
    H = 0.5 * (H + H.conj().T)
    if not overlap:
        return H, None
    S = coo_matrix((np.concatenate(sv), (rows, cols)),
                   shape=(n, n)).tocsr()
    S = 0.5 * (S + S.conj().T)
    return H, S


def bloch_derivative_sparse(model, k=None, direction=0):
    """dH/dk, dS/dk along a Cartesian direction as CSR sparse matrices
    (dS is None for an orthogonal model) -- the sparse counterpart of
    :meth:`TightBindingModel.bloch_derivative`, identical element for
    element (asserted in the tests).  At k = 0 for a finite model this
    is the velocity operator (times hbar) in the site-diagonal position
    convention."""
    k = model._kvec(k)
    n = model.nao
    overlap = model.has_overlap()
    rows, cols, hv, sv = [], [], [], []
    for oi, oj, d, Hb, Sb in model._terms():
        ph = 1j * d[direction] * np.exp(1j * float(k @ d))
        ni, nj = Hb.shape
        r, c = np.meshgrid(np.arange(oi, oi + ni),
                           np.arange(oj, oj + nj), indexing="ij")
        rows.append(r.ravel())
        cols.append(c.ravel())
        hv.append((ph * Hb).ravel())
        sv.append((ph * Sb).ravel())
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    dH = coo_matrix((np.concatenate(hv), (rows, cols)),
                    shape=(n, n)).tocsr()
    dH = 0.5 * (dH + dH.conj().T)
    if not overlap:
        return dH, None
    dS = coo_matrix((np.concatenate(sv), (rows, cols)),
                    shape=(n, n)).tocsr()
    dS = 0.5 * (dS + dS.conj().T)
    return dH, dS


def lowest_bands(model, kpts, n_bands, sigma=None, tol=0.0):
    """The ``n_bands`` lowest eigenvalues at each k, by sparse Lanczos.

    kpts: list of Cartesian k-points ([None] for a finite system).
    sigma: optional shift for shift-invert mode (fastest when the
    states of interest sit near a known energy, e.g. a band edge);
    without it the smallest-algebraic mode is used.  Nonorthogonal
    models solve the generalized problem H c = e S c directly.
    Returns an (nk, n_bands) array, each row sorted ascending.

    This is the large-system counterpart of :func:`hamop.bands`; the
    tests pin it to the dense solver and to the closed-form spectrum
    2 t cos(pi j / (N + 1)) of the open chain.
    """
    n_bands = int(n_bands)
    if not 0 < n_bands < model.nao:
        raise ValueError("need 0 < n_bands < number of orbitals "
                         "(Lanczos cannot return the full spectrum)")
    out = np.empty((len(kpts), n_bands))
    for ik, k in enumerate(kpts):
        H, S = bloch_sparse(model, k)
        if sigma is None:
            e = eigsh(H, k=n_bands, M=S, which="SA", tol=tol,
                      return_eigenvectors=False)
        else:
            e = eigsh(H, k=n_bands, M=S, sigma=sigma, which="LM", tol=tol,
                      return_eigenvectors=False)
        out[ik] = np.sort(np.real(e))
    return out


def _spectral_bounds(H, S=None):
    """(E_min, E_max) of the (generalized) spectrum: Lanczos at both
    ends, or dense for matrices too small for ARPACK."""
    n = H.shape[0]
    if n < 4:
        from scipy.linalg import eigh as dense_eigh
        w = dense_eigh(H.toarray(),
                       None if S is None else S.toarray(),
                       eigvals_only=True)
        return float(w.min()), float(w.max())
    e_lo = eigsh(H, k=1, M=S, which="SA", return_eigenvectors=False,
                 tol=1e-6)[0]
    e_hi = eigsh(H, k=1, M=S, which="LA", return_eigenvectors=False,
                 tol=1e-6)[0]
    return float(e_lo), float(e_hi)


def _jackson(M):
    """Jackson kernel damping factors g_0 .. g_{M-1} (Weisse et al.,
    Rev. Mod. Phys. 78, 275 (2006), Eq. (71))."""
    m = np.arange(M)
    q = np.pi / (M + 1.0)
    return ((M - m + 1.0) * np.cos(q * m)
            + np.sin(q * m) / np.tan(q)) / (M + 1.0)


def kpm_dos(model, energies, n_moments=200, n_random=None, seed=0,
            bounds=None, margin=0.01):
    """Kernel-polynomial density of states of a large *finite* model
    (states / eV, integrating to the total orbital count).

    Chebyshev moments of the spectral density with Jackson-kernel
    damping (Weisse et al., Rev. Mod. Phys. 78, 275 (2006)).
    n_random=None computes the exact trace by running the recurrence on
    the full identity block (deterministic, affordable up to ~10^4
    orbitals); an integer uses that many random +/-1 vectors -- the
    standard stochastic trace estimator, whose relative error shrinks
    as 1/sqrt(n_random * nao).

    bounds: optional (E_min, E_max) enclosing the whole spectrum; by
    default the extreme eigenvalues are found with two Lanczos runs and
    widened by ``margin`` of the span (moments diverge if the true
    spectrum leaks outside the scaling window).

    The energy resolution is roughly pi * span / (2 n_moments); finer
    structure than that is smoothed, exactly like a broadened
    :func:`hamop.dos`.  Nonorthogonal bases are supported: the
    recurrence runs on B = S^-1 H through a sparse LU factorization of
    S -- B has the (real) generalized spectrum, and the trace of a
    polynomial in B is basis-independent, so the same moments give the
    density of generalized eigenvalues.  Finite models only; for a
    periodic model, build the corresponding supercell.
    """
    if model.cell is not None:
        raise ValueError("kpm_dos works on finite models; build a "
                        "supercell of a periodic model first")
    energies = np.asarray(energies, dtype=float)
    H, S = bloch_sparse(model, None)
    H = H.real.tocsr() if np.abs(H.imag).max() == 0.0 else H
    n = model.nao
    M = int(n_moments)
    if S is None:
        def applyB(X):
            return H @ X
    else:
        # nonorthogonal: iterate B = S^-1 H, whose (real) spectrum is
        # the generalized one; Tr of a polynomial in B is basis-free
        from scipy.sparse.linalg import splu
        S = S.real.tocsc() if np.abs(S.imag).max() == 0.0 else S.tocsc()
        lu = splu(S)

        def applyB(X):
            return lu.solve(np.asarray(H @ X))
    if bounds is None:
        e_lo, e_hi = _spectral_bounds(H, S)
    else:
        e_lo, e_hi = bounds
    span = float(e_hi - e_lo)
    a = 0.5 * span * (1.0 + margin)
    b = 0.5 * float(e_hi + e_lo)
    if n_random is None:
        V0 = np.eye(n, dtype=H.dtype)
        norm = 1.0
    else:
        rng = np.random.default_rng(seed)
        V0 = rng.choice([-1.0, 1.0], size=(n, int(n_random)))
        V0 = V0.astype(H.dtype)
        norm = 1.0 / int(n_random)
    mu = np.zeros(M)
    v_prev = V0
    v_cur = (applyB(V0) - b * V0) / a
    mu[0] = norm * float(np.real(np.sum(V0.conj() * v_prev)))
    if M > 1:
        mu[1] = norm * float(np.real(np.sum(V0.conj() * v_cur)))
    for m in range(2, M):
        v_next = 2.0 * (applyB(v_cur) - b * v_cur) / a - v_prev
        mu[m] = norm * float(np.real(np.sum(V0.conj() * v_next)))
        v_prev, v_cur = v_cur, v_next
    g = _jackson(M)
    x = (energies - b) / a
    inside = np.abs(x) < 1.0
    rho = np.zeros_like(energies)
    xc = x[inside]
    theta = np.arccos(xc)
    series = g[0] * mu[0] * np.ones_like(xc)
    for m in range(1, M):
        series += 2.0 * g[m] * mu[m] * np.cos(m * theta)
    rho[inside] = series / (np.pi * np.sqrt(1.0 - xc ** 2) * a)
    return rho


def kpm_sigma(model, omega, mu, direction=0, n_moments=128,
              n_random=None, seed=0, T=300.0, spin=2, bounds=None,
              margin=0.01, n_energy=400):
    """Kernel-polynomial Kubo-Greenwood optical conductivity of a large
    *finite* model (real part, units of e^2/(4 hbar), same convention
    as :func:`hamop.sigma_optical` with area = 1).

    The velocity-velocity spectral density
    rho_v(E, E') = Tr[ v delta(E - H) v delta(E' - H) ] is expanded in
    a double Chebyshev series with Jackson damping (Weisse et al.,
    Rev. Mod. Phys. 78, 275 (2006), Sec. V) and the conductivity is

        Re sigma(omega) = (4 pi spin) * Integral dE
                          [f(E) - f(E + omega)] rho_v(E, E + omega)
                          / omega,

    integrated on ``n_energy`` points.  The broadening is set by the
    Jackson kernel (resolution ~ pi * span / n_moments), not by an eta
    parameter.  n_random as in :func:`kpm_dos` (None = exact trace).

    Anchored in the tests on the two-site molecule, whose *integrated*
    peak weight has the kernel-independent closed form
    spin 4 pi (a t)^2 / (2 |t|), and against the dense Kubo route on a
    dimerized chain.  Orthogonal bases and finite models only, refused
    explicitly otherwise.  Intra-atomic dipole blocks are included
    through the exact sparse operator i(HX - XH), anchored on the same
    atomic line as the dense route.
    """
    if model.cell is not None:
        raise ValueError("kpm_sigma works on finite models; build a "
                         "supercell of a periodic model first")
    if model.has_overlap():
        raise ValueError("kpm_sigma supports orthogonal bases only")
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    if np.any(omega <= 0):
        raise ValueError("omega must be positive photon energies")
    KB = 8.617333262e-5  # eV / K (CODATA 2018)
    H, _ = bloch_sparse(model, None)
    v = _velocity_sparse(model, direction)
    n = model.nao
    M = int(n_moments)
    if bounds is None:
        e_lo, e_hi = _spectral_bounds(H)
    else:
        e_lo, e_hi = bounds
    span = float(e_hi - e_lo)
    a = 0.5 * span * (1.0 + margin)
    b = 0.5 * float(e_hi + e_lo)
    mu_mn = np.real(_vv_moments(H, v, v, a, b, M, n, n_random, seed))
    g = _jackson(M)
    pref = (2.0 - (np.arange(M) == 0)) * g
    mu_t = mu_mn * pref[:, None] * pref[None, :]

    sig = np.zeros(len(omega))
    for iw, hw in enumerate(omega):
        E = np.linspace(b - a + 1e-6 * span,
                        b + a - hw - 1e-6 * span, int(n_energy))
        if len(E) < 2 or E[-1] <= E[0]:
            continue
        x = (E - b) / a
        xp = (E + hw - b) / a
        ok = (np.abs(x) < 1.0) & (np.abs(xp) < 1.0)
        if not np.any(ok):
            continue
        x, xp, Eo = x[ok], xp[ok], E[ok]
        th, thp = np.arccos(x), np.arccos(xp)
        m = np.arange(M)[:, None]
        P = np.cos(m * th[None, :]) / (np.pi * np.sqrt(1.0 - x ** 2) * a)
        Pp = np.cos(m * thp[None, :]) \
            / (np.pi * np.sqrt(1.0 - xp ** 2) * a)
        rho = np.sum(P * (mu_t @ Pp), axis=0)
        xE = np.clip((Eo - mu) / (KB * T), -60.0, 60.0)
        xEp = np.clip((Eo + hw - mu) / (KB * T), -60.0, 60.0)
        df = 1.0 / (1.0 + np.exp(xE)) - 1.0 / (1.0 + np.exp(xEp))
        integ = getattr(np, "trapezoid", getattr(np, "trapz", None))
        sig[iw] = 4.0 * np.pi * spin * integ(df * rho, Eo) / hw
    return sig

