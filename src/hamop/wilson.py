"""Wilson loops, hybrid Wannier centers and the Z2 invariant (new in
0.15.0).

The Chern number of :func:`hamop.chern_number` is zero for every
crystal with time-reversal symmetry, so it cannot tell a quantum spin
Hall insulator from an ordinary one.  The invariant that does is the
Z2 index of Kane and Mele (C. L. Kane and E. J. Mele, Phys. Rev. Lett.
95, 146802 (2005)).  This module computes it from the Wilson loop of
the occupied bands, a method that needs no inversion symmetry and no
gauge fixing (A. A. Soluyanov and D. Vanderbilt, Phys. Rev. B 83,
235401 (2011); R. Yu, X. L. Qi, A. Bernevig, Z. Fang and X. Dai, Phys.
Rev. B 84, 075119 (2011)).

Hybrid Wannier centers.  Fix the wave-vector component along one
reciprocal vector, b_other, at the fraction s, and let k run once along
the other, b_axis.  The product of the overlap matrices of the occupied
states between neighboring k-points is the Wilson loop W(s), a unitary
n_occ x n_occ matrix (each overlap matrix is replaced by its unitary
part, the polar factor of its singular value decomposition, which
transforms like the overlap matrix under a change of gauge).  Its
eigenvalues exp(i theta_n) do not depend on the gauge, and

    x_n(s) = -theta_n / (2 pi)   (mod 1)

are the centers of the hybrid Wannier functions (localized along
a_axis, Bloch-like along the other direction) in units of the lattice
vector a_axis.  The minus sign makes x the position: theta follows the
sign convention of :func:`hamop.berry_phase`, which is minus Berry's
phase.  The sum of the theta_n equals ``berry_phase`` along the same
loop (mod 2 pi), and x(s) winds C times as s runs once across the
zone, C the Chern number (both are tests).

Frames: ``frame="lowdin"`` (the default, as in the Berry functions) is
the periodic gauge, in which the centers are measured as if every
orbital sat at the origin of the cell; ``frame="atomic"`` is the
atomic gauge, in which they are positions in the site-diagonal position
convention of the optics (a single band on one orbital has its center
exactly on that orbital).  The Z2 invariant is the same in both, and
the tests check that.

Z2.  With time-reversal symmetry (Theta^2 = -1, a spinful model) the
centers at s and -s coincide, and at the two time-reversal-invariant
lines s = 0 and s = 1/2 they come in degenerate (Kramers) pairs.  The
Z2 invariant is the parity of the number of times the centers cross a
reference line as s goes from 0 to 1/2.  Following Soluyanov and
Vanderbilt, the reference is the middle of the largest gap between the
centers on each line, and a crossing is counted for every center at
line j+1 that lies between the gap middles of lines j and j+1.  Lines
are added between neighbors until (a) no center at one line lies
within ``gap_tol`` times the largest gap of the other line from that
line's gap middle and (b) no center moves by more than ``move_tol``
times the smaller of the two largest gaps.  Checks of this kind are
used by the Z2Pack code (D. Gresch et al., Phys. Rev. B 95, 075146
(2017)); the form and the thresholds here are this package's own.  If the checks cannot be met within
``max_lines`` lines (a gap of the bands or of the centers that closes),
the calculation is refused.

Time reversal is checked, not assumed: the Kramers pairs at s = 0 and
1/2, and the coincidence of the centers at s and 1 - s on the starting
lines, must hold to ``tr_tol``.  A model without time-reversal
symmetry, for example two spin copies of a Chern insulator, is refused.
The occupied bands must be separated from the empty ones by a direct
gap of at least ``gap_min`` at every sampled k-point.

Units: k Cartesian in 1/Angstrom, as everywhere; the centers are
fractions of the lattice vector a_axis.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike

from .berry import _orbital_positions, _states_many

if TYPE_CHECKING:
    from .model import TightBindingModel

__all__ = ["hybrid_wannier_centers", "z2_invariant"]


def _check(model, n_occ, n_k, axis, frame, need_2d):
    if model.cell is None:
        raise ValueError("Wannier centers need a periodic model")
    p = model.cell.shape[0]
    if need_2d and p != 2:
        raise ValueError("z2_invariant needs a model with two lattice "
                         "vectors (a 2D crystal or a sheet in 3D space); "
                         f"this one has {p}")
    if p not in (1, 2):
        raise ValueError("hybrid_wannier_centers handles models with one "
                         f"or two lattice vectors; this one has {p}")
    if frame not in ("lowdin", "atomic"):
        raise ValueError("frame must be 'lowdin' or 'atomic'")
    n_occ = int(n_occ)
    if not 1 <= n_occ < model.nao:
        raise ValueError(f"n_occ must lie in [1, {model.nao - 1}]: at "
                         "least one band must stay empty")
    n_k = int(n_k)
    if n_k < 3:
        raise ValueError("n_k (k-points along each loop) must be >= 3")
    axis = int(axis)
    if axis not in range(p):
        raise ValueError(f"axis must be one of {list(range(p))}")
    return n_occ, n_k, axis


def _centers(model, n_occ, s, n_k, axis, frame, gap_min):
    """Sorted hybrid Wannier centers (len(s), n_occ) in [0, 1) for the
    lines at the fractions s of the other reciprocal vector (s ignored,
    one line, for a model with one lattice vector)."""
    B = model.reciprocal_vectors
    G = B[axis]
    t = np.arange(n_k) / n_k
    if B.shape[0] == 1:
        K = (t[None, :, None] * G).astype(float)
    else:
        other = B[1 - axis]
        K = t[None, :, None] * G + np.asarray(s, float)[:, None, None] * other
    L, dim = K.shape[0], K.shape[2]
    st, gap = _states_many(model, K.reshape(-1, dim), n_occ, frame,
                           gaps=True)
    bad = int(np.argmin(gap))
    if not gap[bad] >= gap_min:
        raise ValueError(
            f"the direct gap above the {n_occ} occupied bands is "
            f"{gap[bad]:.3g} (< gap_min = {gap_min:g}) at k = "
            f"{K.reshape(-1, dim)[bad].tolist()}: the occupied states "
            "are not separated from the empty ones there, so their "
            "Wannier centers are undefined")
    st = st.reshape(L, n_k, model.nao, n_occ)
    # the end point k0 + G is built from k0 (see berry._closing_states)
    if frame == "lowdin":
        end = st[:, 0]
    else:
        end = np.exp(-1j * (_orbital_positions(model) @ G))[None, :, None] \
            * st[:, 0]
    nxt = np.concatenate([st[:, 1:], end[:, None]], axis=1)
    sh = np.conj(np.swapaxes(st, -1, -2))
    if frame == "atomic" and model.has_overlap():
        Kn = np.concatenate([K[:, 1:], (K[:, 0] + G)[:, None]], axis=1)
        _, Smid = model._bloch_many((0.5 * (K + Kn)).reshape(-1, dim))
        M = sh @ Smid.reshape(L, n_k, model.nao, model.nao) @ nxt
    else:
        M = sh @ nxt
    U, sv, Vh = np.linalg.svd(M)
    if sv.min() < 1e-8:
        raise ValueError("vanishing overlap between the occupied states "
                         "of neighboring k-points; increase n_k")
    Mu = U @ Vh                              # unitary part of each link
    W = Mu[:, 0]
    for i in range(1, n_k):
        W = W @ Mu[:, i]
    theta = np.angle(np.linalg.eigvals(W))
    x = np.mod(-theta / (2.0 * np.pi), 1.0)
    x[x >= 1.0] -= 1.0
    return np.sort(x, axis=1)


def hybrid_wannier_centers(model: TightBindingModel, n_occ: int,
                           s: Optional[ArrayLike] = None, n_k: int = 48,
                           axis: int = 0, frame: str = "lowdin",
                           gap_min: float = 1e-6) -> np.ndarray:
    """Centers of the hybrid Wannier functions of the lowest ``n_occ``
    bands, from the eigenvalues of the Wilson loop (see the module
    docstring).

    model : a periodic model with one or two lattice vectors.
    n_occ : number of occupied bands (at least one band must stay
        empty, and the occupied ones must be separated from it by a
        direct gap of at least ``gap_min`` at every k used).
    s : for two lattice vectors, the lines: fractions of the other
        reciprocal vector at which the loops run (a number or a 1D
        array); must be None for one lattice vector.
    n_k : k-points along each loop (the loop is closed exactly; the
        result converges as n_k grows, and should be checked by
        raising it).
    axis : the lattice vector along which the Wannier functions are
        localized (0 or 1); the loops run along reciprocal vector
        ``axis``.
    frame : "lowdin" (periodic gauge) or "atomic"; see the module
        docstring.

    Returns the centers in units of the lattice vector a_axis, each in
    [0, 1) and sorted: shape (n_occ,) for one lattice vector, (len(s),
    n_occ) for two (one row per line).
    """
    n_occ, n_k, axis = _check(model, n_occ, n_k, axis, frame, False)
    if model.cell.shape[0] == 1:
        if s is not None:
            raise ValueError("a model with one lattice vector has a single "
                             "loop; leave s as None")
        return _centers(model, n_occ, [0.0], n_k, axis, frame, gap_min)[0]
    if s is None:
        raise ValueError("give s, the fractions of the other reciprocal "
                         "vector at which the loops run")
    sv = np.atleast_1d(np.asarray(s, dtype=float))
    if sv.ndim != 1 or sv.size == 0 or not np.all(np.isfinite(sv)):
        raise ValueError("s must be a finite number or a 1D array")
    return _centers(model, n_occ, sv, n_k, axis, frame, gap_min)


def _circ(d):
    """Signed distance on the unit circle, in [-1/2, 1/2)."""
    return np.mod(np.asarray(d) + 0.5, 1.0) - 0.5


def _largest_gap(x):
    """Middle and width of the largest gap between sorted centers x on
    the unit circle."""
    gaps = np.diff(np.concatenate([x, [x[0] + 1.0]]))
    i = int(np.argmax(gaps))
    return float(np.mod(x[i] + 0.5 * gaps[i], 1.0)), float(gaps[i])


def _set_distance(a, b):
    """Largest distance on the circle between two sorted sets of centers
    of equal size, after the best cyclic matching."""
    n = len(a)
    best = np.inf
    for r in range(n):
        best = min(best, float(np.abs(_circ(a - np.roll(b, r))).max()))
    return best


def _kramers_distance(x):
    """How far the sorted centers x are from forming degenerate pairs."""
    pairs = [np.abs(_circ(x[0::2] - x[1::2])).max()]
    y = np.roll(x, 1)                          # pairs across 0 / 1
    pairs.append(np.abs(_circ(y[0::2] - y[1::2])).max())
    return float(min(pairs))


def _step(xa, xb, gap_tol, move_tol):
    """(ok, crossings) for the step between two neighboring lines."""
    ga, wa = _largest_gap(xa)
    gb, wb = _largest_gap(xb)
    ok = (np.abs(_circ(xb - ga)).min() >= gap_tol * wa
          and np.abs(_circ(xa - gb)).min() >= gap_tol * wb)
    move = max(np.abs(_circ(xb[:, None] - xa[None, :])).min(axis=1).max(),
               np.abs(_circ(xa[:, None] - xb[None, :])).min(axis=1).max())
    ok = ok and move <= move_tol * min(wa, wb)
    d = float(_circ(gb - ga))                  # the gap middle moves by d
    rel = np.mod(xb - ga, 1.0) if d >= 0 else np.mod(ga - xb, 1.0)
    crossings = int(np.count_nonzero((rel > 0.0) & (rel < abs(d))))
    return ok, crossings


def z2_invariant(model: TightBindingModel, n_occ: int, n_k: int = 48,
                 n_lines: int = 11, axis: int = 0, frame: str = "lowdin",
                 max_lines: int = 1025, gap_tol: float = 0.3,
                 move_tol: float = 0.3, tr_tol: float = 1e-6,
                 gap_min: float = 1e-6) -> Dict[str, Any]:
    """Z2 topological invariant of the lowest ``n_occ`` bands of a
    time-reversal-symmetric 2D model (Kane and Mele, PRL 95, 146802
    (2005)), from the flow of the hybrid Wannier centers over half the
    Brillouin zone (see the module docstring for the method and the
    checks).

    model : a spinful periodic model with two lattice vectors and
        time-reversal symmetry (Theta^2 = -1), for example
        :func:`hamop.kane_mele` or a model made with :func:`hamop.with_spin`
        plus spin-orbit terms.
    n_occ : number of occupied bands, even (Kramers pairs); at least one
        band must stay empty.
    n_k : k-points along each Wilson loop.
    n_lines : lines on the half zone 0 <= s <= 1/2 to start with; more
        are added where the centers move too fast (up to ``max_lines``).
    axis : lattice vector along which the centers are taken (0 or 1);
        the result does not depend on it.
    frame : "lowdin" or "atomic"; the result does not depend on it.
    gap_tol, move_tol : the convergence checks of the module docstring.
    tr_tol : tolerance (fractions of the cell) of the time-reversal
        checks.
    gap_min : smallest direct band gap accepted above the occupied
        bands.

    Returns dict(z2 (0 or 1), crossings (the number counted; z2 is its
    parity), s (the lines used, in [0, 1/2]), centers ((len(s), n_occ),
    sorted, in [0, 1)), gap_centers (the middle of the largest gap on
    each line)).  z2 = 1 is a quantum spin Hall (topological)
    insulator, z2 = 0 an ordinary one.
    """
    n_occ, n_k, axis = _check(model, n_occ, n_k, axis, frame, True)
    if n_occ % 2:
        raise ValueError("n_occ must be even: with time-reversal symmetry "
                         "the occupied bands come in Kramers pairs")
    n_lines = int(n_lines)
    if n_lines < 2 or int(max_lines) < n_lines:
        raise ValueError("need n_lines >= 2 and max_lines >= n_lines")
    if not (0.0 < gap_tol < 0.5 and 0.0 < move_tol < 1.0 and tr_tol > 0):
        raise ValueError("need 0 < gap_tol < 1/2, 0 < move_tol < 1 and "
                         "tr_tol > 0")

    def run(svals):
        return _centers(model, n_occ, np.asarray(svals, float), n_k, axis,
                        frame, gap_min)

    s = list(np.linspace(0.0, 0.5, n_lines))
    X = dict(zip(s, run(s)))
    # time reversal: Kramers pairs on the invariant lines, and the
    # centers at s and 1 - s coincide
    for s0 in (0.0, 0.5):
        dk = _kramers_distance(X[s0])
        if dk > tr_tol:
            raise ValueError(
                f"the Wannier centers at s = {s0:g} do not form Kramers "
                f"pairs (off by {dk:.3g} > tr_tol = {tr_tol:g}): the model "
                "is not time-reversal symmetric with Theta^2 = -1, so it "
                "has no Z2 invariant")
    inner = s[1:-1]
    if inner:
        here = [X[x] for x in inner]
    else:                                      # n_lines = 2
        inner = [0.25]
        here = list(run(inner))
    mirror = run([1.0 - x for x in inner])
    for x, xs, xm in zip(inner, here, mirror):
        dm = _set_distance(xs, xm)
        if dm > tr_tol:
            raise ValueError(
                f"the Wannier centers at s = {x:.6g} and s = {1 - x:.6g} "
                f"differ by {dm:.3g} (> tr_tol = {tr_tol:g}); time "
                "reversal maps one line onto the other, so the model is "
                "not time-reversal symmetric and has no Z2 invariant")
    while True:
        lines = sorted(X)
        steps = [_step(X[a], X[b], gap_tol, move_tol)
                 for a, b in zip(lines[:-1], lines[1:])]
        new = [0.5 * (a + b) for (a, b), (ok, _) in
               zip(zip(lines[:-1], lines[1:]), steps) if not ok]
        if not new:
            break
        if len(X) + len(new) > int(max_lines):
            raise ValueError(
                f"the hybrid Wannier centers could not be followed with "
                f"{max_lines} lines: they move too fast or the largest gap "
                "between them closes (the band gap may be closing, or "
                "n_k may be too small); raise max_lines or n_k, or check "
                "the band gap")
        X.update(zip(new, run(new)))
    crossings = sum(c for _, c in steps)
    lines = sorted(X)
    return {"z2": int(crossings % 2), "crossings": int(crossings),
            "s": np.array(lines),
            "centers": np.array([X[x] for x in lines]),
            "gap_centers": np.array([_largest_gap(X[x])[0] for x in lines])}
