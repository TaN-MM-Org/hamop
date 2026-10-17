"""Generalized eigensolver with canonical orthogonalization.

A nonorthogonal basis can be mildly overcomplete, so the overlap matrix
S has eigenvalues close to zero.  Errors in H that live in that
near-null space are amplified enormously by a naive generalized
eigensolver.  Canonical orthogonalization is the standard remedy used
inside electronic-structure codes (Szabo and Ostlund, *Modern Quantum
Chemistry*, sec. 3.4.5): diagonalize S, drop directions whose overlap
eigenvalue falls below a threshold, and solve H in the remaining
well-conditioned subspace.

For a well-conditioned S this reduces to the ordinary generalized
eigenproblem; the test suite asserts agreement with
``scipy.linalg.eigh(H, S)`` to near machine precision in that case.
"""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigh

__all__ = ["gen_eigh"]


def gen_eigh(H: ArrayLike, S: ArrayLike, thresh: float = 1e-10,
             eigvals_only: bool = True
             ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Eigenvalues (and vectors) of H c = e S c, canonically orthogonalized.

    thresh: overlap eigenvalues below this are dropped.  Returned
    eigenvectors are columns in the original basis, S-orthonormal within
    the kept subspace.
    """
    H = 0.5 * (H + np.conj(H).T)
    S = 0.5 * (S + np.conj(S).T)
    s, U = eigh(S)
    keep = s > thresh
    if not np.any(keep):
        raise ValueError("no overlap eigenvalue above the threshold")
    X = U[:, keep] / np.sqrt(s[keep])
    Hp = X.conj().T @ H @ X
    if eigvals_only:
        return eigh(Hp, eigvals_only=True)
    w, Vp = eigh(Hp)
    return w, X @ Vp


# ---------------- many k-points at once (new in 0.13.0) ----------------

def _eigh_batch(H, S, thresh, vectors):
    """Batched counterpart of `gen_eigh` for a stack of matrices
    (shape (nk, n, n)); S None means the identity.  Returns (e, c) with
    c None unless ``vectors``, or None when some k-point would drop an
    overlap direction (canonical orthogonalization keeps a different
    number of states there, so the caller falls back to `gen_eigh` one
    k-point at a time)."""
    if S is None:
        if not thresh < 1.0:
            return None
        if vectors:
            return np.linalg.eigh(H)
        return np.linalg.eigvalsh(H), None
    s, U = np.linalg.eigh(S)
    if not np.all(s > thresh):
        return None
    X = U / np.sqrt(s)[:, None, :]
    Hp = np.conj(np.swapaxes(X, 1, 2)) @ H @ X
    if not vectors:
        return np.linalg.eigvalsh(Hp), None
    w, Vp = np.linalg.eigh(Hp)
    return w, X @ Vp


def _eig_chunks(model, kpts, weights, thresh=1e-10, vectors=False,
                directions=(), chunk=None):
    """Eigenpairs over a k-point list, a block of k-points at a time.

    Yields (w, e, c, derivs): the weights (m,), eigenvalues (m, nkept),
    S-normalized eigenvectors (m, nao, nkept) or None, and a dict
    direction -> (dH, dS) of k-derivatives (m, nao, nao) (dS None for an
    orthogonal model).  Where canonical orthogonalization drops overlap
    directions the block is split into single k-points solved by
    `gen_eigh`, so the result is the same as the per-k-point loop."""
    K = model._karray(kpts)
    w = np.asarray(weights, dtype=float).ravel()
    n = model.nao
    size = int(chunk) if chunk else max(1, (1 << 20) // max(1, n * n))
    for start in range(0, K.shape[0], size):
        Kc = K[start:start + size]
        wc = w[start:start + size]
        H, S = model._bloch_many(Kc)
        derivs = {a: model._bloch_many(Kc, a) for a in directions}
        res = _eigh_batch(H, S, thresh, vectors)
        if res is not None:
            yield wc, res[0], res[1], derivs
            continue
        eye = np.eye(n, dtype=complex)
        for i in range(Kc.shape[0]):
            Si = eye if S is None else S[i]
            if vectors:
                e_i, c_i = gen_eigh(H[i], Si, thresh=thresh,
                                    eigvals_only=False)
                c_i = c_i[None]
            else:
                e_i, c_i = gen_eigh(H[i], Si, thresh=thresh), None
            d_i = {a: (dH[i:i + 1], None if dS is None else dS[i:i + 1])
                   for a, (dH, dS) in derivs.items()}
            yield wc[i:i + 1], np.asarray(e_i)[None], c_i, d_i
