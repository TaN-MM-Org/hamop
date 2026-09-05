"""Fourier (Wannier-style) interpolation of the Bloch Hamiltonian.

The backbone of Wannier interpolation without the Wannierization:
sample H(k) (and S(k)) in the periodic gauge on a uniform grid,
inverse-transform to real-space matrices H(R) on the grid's torus with
R folded into the centered window, and evaluate H(k) = sum_R e^{ikR}
H(R) at arbitrary k.  For a tight-binding model whose hopping range
fits strictly inside half the sampling window this is *exact* -- the
interpolated bands equal the directly computed bands to machine
precision, which is what the tests assert.  When the range does not
fit, the real-space matrices alias; ``max_residual`` measures that by
re-evaluating on a shifted grid, so an undersampled model is detected
rather than silently interpolated wrong.

What this is not, stated plainly: no maximally localized Wannier
functions are constructed (Marzari and Vanderbilt, Phys. Rev. B 56,
12847 (1997), is not implemented) -- this is the exact-arithmetic
Fourier step those methods build on, useful on its own for
interpolating Hamiltonians sampled from external (e.g. ab initio)
sources through :class:`TightBindingModel` grids.
"""
from __future__ import annotations

import numpy as np

from .berry import _bloch_periodic
from .eigsolve import gen_eigh

__all__ = ["fourier_interpolation", "FourierInterpolator"]


class FourierInterpolator:
    """Callable Bloch interpolant; see :func:`fourier_interpolation`."""

    def __init__(self, R_list, H_R, S_R, recip):
        self._R = R_list
        self._H = H_R
        self._S = S_R
        self._recip = recip

    def bloch(self, k):
        """Interpolated H(k), S(k) in the periodic gauge (same
        eigenvalues as the atomic gauge; the gauge only reshuffles
        eigenvector phases)."""
        k = np.atleast_1d(np.asarray(k, dtype=float))
        ph = np.exp(1j * (self._R @ k))
        H = np.tensordot(ph, self._H, axes=(0, 0))
        H = 0.5 * (H + H.conj().T)
        if self._S is None:
            return H, np.eye(H.shape[0], dtype=complex)
        S = np.tensordot(ph, self._S, axes=(0, 0))
        return H, 0.5 * (S + S.conj().T)

    def bands(self, kpts, thresh=1e-10):
        """Eigenvalues along a k-list, shape (nk, nao)."""
        out = []
        for k in kpts:
            H, S = self.bloch(k)
            out.append(gen_eigh(H, S, thresh=thresh))
        return np.array(out)


def fourier_interpolation(model, mesh):
    """Build a Fourier interpolant of ``model`` from an N^dim periodic-
    gauge sample of its Bloch matrices.

    mesh: points per reciprocal direction (int or sequence).  Exact
    whenever every hopping image lies strictly inside the centered
    window (-mesh/2, mesh/2] per direction; the tests pin that
    exactness to 1e-12 and use :meth:`max_residual` to detect
    undersampling.
    """
    if model.cell is None:
        raise ValueError("finite system: nothing to interpolate in k")
    dim = model.cell.shape[0]
    if np.isscalar(mesh):
        mesh = (int(mesh),) * dim
    recip = 2.0 * np.pi * np.linalg.inv(model.cell).T
    grids = [np.arange(n) for n in mesh]
    idx = np.stack(np.meshgrid(*grids, indexing="ij"),
                   axis=-1).reshape(-1, dim)
    frac = idx / np.array(mesh, dtype=float)
    kpts = frac @ recip
    samples = [_bloch_periodic(model, k) for k in kpts]
    Hk = np.array([s[0] for s in samples])
    Sk = None if not model.has_overlap() \
        else np.array([s[1] for s in samples])
    # centered real-space images
    imgs = []
    for row in idx:
        imgs.append([int(i) if i <= n // 2 else int(i) - n
                     for i, n in zip(row, mesh)])
    imgs = np.array(imgs, dtype=float)
    R_cart = imgs @ model.cell
    ntot = len(kpts)
    phases = np.exp(-1j * (kpts @ R_cart.T))          # (nk, nR)
    H_R = np.tensordot(phases.T, Hk, axes=(1, 0)) / ntot
    S_R = None if Sk is None \
        else np.tensordot(phases.T, Sk, axes=(1, 0)) / ntot
    interp = FourierInterpolator(R_cart, H_R, S_R, recip)

    def max_residual(n_test=5, seed=0):
        """Largest eigenvalue mismatch between the interpolant and the
        model at random off-grid k -- machine-small iff the hopping
        range fits the sampling window."""
        rng = np.random.default_rng(seed)
        worst = 0.0
        for _ in range(int(n_test)):
            k = rng.uniform(-1.0, 1.0, dim) @ recip
            e1 = interp.bands([k])[0]
            e2 = gen_eigh(*model.bloch(k))
            worst = max(worst, float(np.abs(e1 - e2).max()))
        return worst

    interp.max_residual = max_residual
    return interp
