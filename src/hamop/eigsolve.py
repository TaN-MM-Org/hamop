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

import numpy as np
from scipy.linalg import eigh

__all__ = ["gen_eigh"]


def gen_eigh(H, S, thresh=1e-10, eigvals_only=True):
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
