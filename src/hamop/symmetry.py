"""Point-group folding of the Monkhorst-Pack grid, self-verified.

``symmetry_fold`` reduces a uniform k-grid to one representative per
orbit of a point group the *user supplies* (there is no automatic
space-group detection here, stated plainly).  Two checks make a wrong
group impossible to use silently:

1. each operation must map the reciprocal lattice to itself (its
   matrix in the fractional basis must be integer to 1e-9), and
2. each operation must actually leave the spectrum invariant --
   eigenvalues at rotated and unrotated random k-points are compared
   and the fold is refused if they differ.

The folded grid is exact for *spectral* observables (DOS, chemical
potentials, band edges, carrier counts), which depend on eigenvalues
only.  It is NOT valid for direction-resolved quantities such as
sigma_xx or the Drude weight, whose k-integrands are not invariant
under a rotation that mixes Cartesian axes; use the plain grid or the
time-reversal fold (which is valid for them) there.  The docstring
says so and the README repeats it.
"""
from __future__ import annotations

import numpy as np

from .eigsolve import gen_eigh

__all__ = ["symmetry_fold"]


def _group_closure(mats, max_order=48):
    """Close a set of integer matrices under multiplication."""
    def key(M):
        return tuple(int(x) for x in np.rint(M).ravel())
    dim = mats[0].shape[0]
    group = {key(np.eye(dim)): np.eye(dim)}
    frontier = [np.eye(dim)] + [np.rint(m) for m in mats]
    for m in frontier:
        group[key(m)] = m
    changed = True
    while changed:
        changed = False
        items = list(group.values())
        for a in items:
            for b in items:
                c = a @ b
                if key(c) not in group:
                    group[key(c)] = c
                    changed = True
                    if len(group) > max_order:
                        raise ValueError(
                            "the supplied operations generate more than "
                            f"{max_order} elements; not a crystallographic "
                            "point group")
    return list(group.values())


def symmetry_fold(model, mesh, ops, time_reversal=False, n_check=4,
                  tol=1e-9, seed=0, thresh=1e-10):
    """Fold a Monkhorst-Pack grid by a point group; returns
    (kpts_cart, weights) usable wherever explicit k-points are accepted.

    ops: iterable of dim x dim Cartesian orthogonal matrices (rotations
    / reflections about Gamma).  The group they generate is closed
    automatically.  time_reversal=True additionally folds k with -k;
    unlike the real-block gate in ``monkhorst_pack``, spectral
    invariance under k -> -k is verified *numerically* here (it can
    hold for other reasons, e.g. inversion symmetry in a
    time-reversal-broken model), and the fold is refused if the check
    fails.

    Every operation is verified before folding: its fractional-basis
    matrix must be integer (it must map the reciprocal lattice to
    itself), and eigenvalues at ``n_check`` random k-points must equal
    those at the mapped points to ``tol``.  A non-symmetry is refused,
    never silently averaged.

    Valid for spectral observables only (dos, fermi_level, band_edges,
    carrier_count); see the module docstring.
    """
    if model.cell is None:
        raise ValueError("finite system has no Brillouin zone")
    dim = model.cell.shape[0]
    if np.isscalar(mesh):
        mesh = (int(mesh),) * dim
    recip = 2.0 * np.pi * np.linalg.inv(model.cell).T
    Binv = np.linalg.inv(recip)

    frac_mats = []
    for R in ops:
        R = np.asarray(R, dtype=float)
        if R.shape != (dim, dim):
            raise ValueError(f"operation must be {dim} x {dim}")
        M = recip @ R.T @ Binv            # action on fractional rows
        if np.abs(M - np.rint(M)).max() > 1e-9:
            raise ValueError(
                "operation does not map the reciprocal lattice to "
                "itself (non-integer fractional matrix); it is not a "
                "symmetry of this lattice")
        frac_mats.append(np.rint(M))

    # spectral verification at random k, for every supplied op (and -k)
    rng = np.random.default_rng(seed)
    ktest = rng.uniform(-1.0, 1.0, size=(n_check, dim)) @ recip
    checks = [np.asarray(R, dtype=float) for R in ops]
    if time_reversal:
        checks.append(-np.eye(dim))
    for R in checks:
        for k in ktest:
            e1 = gen_eigh(*model.bloch(k), thresh=thresh)
            e2 = gen_eigh(*model.bloch(k @ R.T), thresh=thresh)
            if np.abs(e1 - e2).max() > tol:
                raise ValueError(
                    "operation is not a symmetry of the Hamiltonian: "
                    f"eigenvalues differ by {np.abs(e1 - e2).max():.2e} "
                    "at a test k-point; refusing to fold")

    group = _group_closure(frac_mats)
    if time_reversal:
        group = group + [-g for g in group]

    ntot = int(np.prod(mesh))
    mesh = tuple(int(n) for n in mesh)
    seen = {}
    order = []
    for flat in range(ntot):
        idx, rest = [], flat
        for n in reversed(mesh):
            idx.append(rest % n)
            rest //= n
        idx = tuple(reversed(idx))
        # orbit representative: smallest image under the group
        images = []
        v = np.array(idx, dtype=float)
        ok = True
        for g in group:
            w = v @ g
            # images must land back on the grid
            wi = tuple(int(np.rint(w[d])) % mesh[d] for d in range(dim))
            if np.abs(np.array([w[d] % mesh[d] for d in range(dim)])
                      - np.array(wi)).max() > 1e-6:
                ok = False
                break
            images.append(wi)
        if not ok:
            raise ValueError("group operation moves grid points off the "
                             "grid; choose a mesh compatible with the "
                             "symmetry")
        rep = min(images)
        if rep in seen:
            seen[rep] += 1
        else:
            seen[rep] = 1
            order.append(rep)
    frac = np.array([[i / n for i, n in zip(rep, mesh)] for rep in order])
    kpts = frac @ recip
    weights = np.array([seen[rep] for rep in order], dtype=float) / ntot
    return kpts, weights
