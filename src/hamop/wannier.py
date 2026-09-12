"""Wannier90 interoperability: real materials in, one line (new in v0.7).

Hand-built lattices are fine for models with closed-form anchors; a
real material arrives as a Wannier90 ``seedname_hr.dat`` file -- the
de-facto interchange format for first-principles-derived tight-binding
Hamiltonians (A. A. Mostofi et al., Comput. Phys. Commun. 185, 2309
(2014)). This module reads that documented format and builds a
`TightBindingModel` from it, so every observable in the package --
bands, DOS, Kubo optics, Berry/Chern topology, NEGF transport --
applies to the imported material with no re-typing of matrix elements.

The format, parsed strictly: a comment line; ``num_wann``; ``nrpts``;
``nrpts`` degeneracy integers (15 per line); then one line per matrix
element, ``R1 R2 R3 m n Re Im`` with 1-based orbital indices and
H_mn(R) = <m0|H|nR> in eV. Each stored element is divided by its
R-vector's degeneracy weight, per the Wannier90 convention. The parser
REFUSES malformed files -- wrong counts, duplicate or missing
elements, a truncated degeneracy list -- and refuses a Hamiltonian
that is not Hermitian in real space (H_mn(R) must equal
conj(H_nm(-R))), instead of symmetrizing it silently.

What the file does not carry, the caller must supply, stated rather
than guessed: the lattice vectors (`cell`, in Angstrom -- hr.dat has
no units or cell block), and optionally the Wannier centres
(`centers`, from ``seedname_centres.xyz``). Without centres all
orbitals sit at the origin; every eigenvalue-derived quantity (bands,
DOS, transmission, Chern numbers) is exactly independent of that
choice (the atomic and R gauges differ by a unitary), but optical
matrix elements use the site-diagonal position operator, so give the
true centres when the optics matter.

Anchors asserted in the tests rather than stated: a graphene hr.dat
written independently of the package reproduces the closed-form
anchored `lattices.graphene` bands at random k to machine precision
(two constructions of the same physics agreeing); the save/load round
trip preserves H(k) exactly; degeneracy weights divide as the
convention requires; and each refusal fires on a deliberately
corrupted file.
"""
from __future__ import annotations

import numpy as np

from .model import TightBindingModel

__all__ = ["load_wannier90_hr", "save_wannier90_hr", "from_wannier90"]


def load_wannier90_hr(path, hermitian_tol=1e-6):
    """Parse a Wannier90 ``seedname_hr.dat`` file.

    Returns (H_R, num_wann): H_R is a dict mapping the integer
    R-vector tuple to its (num_wann, num_wann) complex block, already
    divided by the degeneracy weight. Refuses malformed files and
    real-space Hermiticity violations beyond `hermitian_tol`.
    """
    with open(path) as fh:
        tokens_lines = [ln.split() for ln in fh]
    lines = [ln for ln in tokens_lines if ln]
    if len(lines) < 3:
        raise ValueError("hr.dat: file too short for header")
    try:
        num_wann = int(lines[1][0])
        nrpts = int(lines[2][0])
    except (ValueError, IndexError):
        raise ValueError("hr.dat: lines 2 and 3 must carry num_wann "
                         "and nrpts") from None
    if num_wann < 1 or nrpts < 1:
        raise ValueError("hr.dat: num_wann and nrpts must be positive")
    # degeneracy list: nrpts integers, 15 per line
    deg = []
    idx = 3
    while len(deg) < nrpts:
        if idx >= len(lines):
            raise ValueError("hr.dat: degeneracy list is truncated")
        row = lines[idx]
        if len(deg) + len(row) > nrpts + 14:
            raise ValueError("hr.dat: degeneracy list longer than "
                             "nrpts")
        try:
            deg.extend(int(x) for x in row)
        except ValueError:
            raise ValueError("hr.dat: non-integer degeneracy "
                             "entry") from None
        idx += 1
    if len(deg) != nrpts:
        raise ValueError(f"hr.dat: {len(deg)} degeneracy entries for "
                         f"nrpts = {nrpts}")
    if any(d < 1 for d in deg):
        raise ValueError("hr.dat: degeneracies must be positive")

    expected = nrpts * num_wann * num_wann
    data = lines[idx:]
    if len(data) != expected:
        raise ValueError(
            f"hr.dat: {len(data)} matrix-element lines where "
            f"nrpts * num_wann^2 = {expected} are required")
    H_R = {}
    seen_order = []
    filled = {}
    for ln_no, ln in enumerate(data):
        if len(ln) != 7:
            raise ValueError(f"hr.dat: matrix-element line "
                             f"{ln_no + 1} has {len(ln)} fields, "
                             "expected 7 (R1 R2 R3 m n Re Im)")
        R = (int(ln[0]), int(ln[1]), int(ln[2]))
        m, n = int(ln[3]), int(ln[4])
        if not (1 <= m <= num_wann and 1 <= n <= num_wann):
            raise ValueError(f"hr.dat: orbital index out of range on "
                             f"matrix-element line {ln_no + 1}")
        if R not in H_R:
            H_R[R] = np.zeros((num_wann, num_wann), dtype=complex)
            filled[R] = np.zeros((num_wann, num_wann), dtype=bool)
            seen_order.append(R)
        if filled[R][m - 1, n - 1]:
            raise ValueError(f"hr.dat: duplicate element for R={R}, "
                             f"(m, n)=({m}, {n})")
        H_R[R][m - 1, n - 1] = float(ln[5]) + 1j * float(ln[6])
        filled[R][m - 1, n - 1] = True
    if len(seen_order) != nrpts:
        raise ValueError(f"hr.dat: {len(seen_order)} distinct "
                         f"R-vectors where nrpts = {nrpts}")
    for R, mask in filled.items():
        if not mask.all():
            raise ValueError(f"hr.dat: R={R} is missing matrix "
                             "elements")
    for R, d in zip(seen_order, deg):
        H_R[R] = H_R[R] / float(d)
    # real-space Hermiticity: H(R) = H(-R)^dagger
    for R, blk in H_R.items():
        negR = tuple(-x for x in R)
        if negR not in H_R:
            raise ValueError(
                f"hr.dat: R={R} present but -R={negR} absent; the "
                "Hamiltonian cannot be Hermitian")
        err = float(np.abs(blk - H_R[negR].conj().T).max())
        if err > hermitian_tol:
            raise ValueError(
                f"hr.dat: H(R) != H(-R)^dagger at R={R} "
                f"(max deviation {err:.3g} eV > {hermitian_tol:g}); "
                "refusing to symmetrize a broken Hamiltonian "
                "silently")
    return H_R, num_wann


def save_wannier90_hr(path, H_R, comment="written by hamop"):
    """Write an R-space Hamiltonian in the hr.dat format (unit
    degeneracies). Blocks must satisfy H(R) = H(-R)^dagger; the exact
    load round trip is asserted in the tests."""
    Rs = sorted(H_R.keys())
    if not Rs:
        raise ValueError("H_R is empty")
    nw = np.asarray(H_R[Rs[0]]).shape[0]
    with open(path, "w") as fh:
        fh.write(f"{comment}\n{nw}\n{len(Rs)}\n")
        for start in range(0, len(Rs), 15):
            fh.write(" ".join("1" for _ in Rs[start:start + 15]) + "\n")
        for R in Rs:
            blk = np.asarray(H_R[R], dtype=complex)
            if blk.shape != (nw, nw):
                raise ValueError("all blocks must share one shape")
            Rpad = tuple(R) + (0,) * (3 - len(R))
            for n in range(nw):          # column outer, row inner --
                for m in range(nw):      # the wannier90 line order
                    v = blk[m, n]
                    fh.write(f"{Rpad[0]:5d}{Rpad[1]:5d}{Rpad[2]:5d}"
                             f"{m + 1:5d}{n + 1:5d}"
                             f"{v.real:22.10f}{v.imag:22.10f}\n")


def from_wannier90(path, cell, centers=None,
                   hermitian_tol=1e-6) -> TightBindingModel:
    """Build a `TightBindingModel` from a ``seedname_hr.dat`` file.

    cell : (dim, dim) lattice vectors as rows, in Angstrom -- hr.dat
        carries no cell, so it must come from you (your structure
        file or the Wannier90 ``.win``). ``dim`` is the number of
        R-vector components actually used (trailing all-zero
        components of every R are dropped to match, so a 2D material
        computed in a 3D code imports as 2D).
    centers : optional (num_wann, dim) Wannier centres in Angstrom
        (from ``seedname_centres.xyz``). Default: all orbitals at the
        origin -- exactly irrelevant for eigenvalue-derived
        quantities, but the optical matrix elements use the
        site-diagonal position operator, so supply the true centres
        when the optics matter (see module docstring).

    One site per Wannier orbital; each nonzero H_mn(R) becomes one
    directed hopping block, with the Hermitian partner implied by the
    model (each conjugate pair enters exactly once).
    """
    H_R, nw = load_wannier90_hr(path, hermitian_tol=hermitian_tol)
    cell = np.atleast_2d(np.asarray(cell, dtype=float))
    dim = cell.shape[0]
    if cell.shape != (dim, dim):
        raise ValueError("cell must be square (dim x dim)")
    # drop trailing R components that are zero for every R
    ncomp = len(next(iter(H_R)))
    for R in H_R:
        if len(R) != ncomp:
            raise ValueError("inconsistent R-vector lengths")
    used = ncomp
    while used > dim and all(R[used - 1] == 0 for R in H_R):
        used -= 1
    if used != dim:
        raise ValueError(
            f"hr.dat uses {used} R components but cell is {dim}D; "
            "supply the matching cell")
    if centers is None:
        pos = np.zeros((nw, dim))
    else:
        pos = np.atleast_2d(np.asarray(centers, dtype=float))
        if pos.shape != (nw, dim):
            raise ValueError(f"centers must be ({nw}, {dim})")
    model = TightBindingModel(positions=pos, norb=1, cell=cell)
    for R, blk in H_R.items():
        Rd = tuple(R[:dim])
        negRd = tuple(-x for x in Rd)
        for m in range(nw):
            for n in range(nw):
                v = blk[m, n]
                if v == 0.0:
                    continue
                key = (Rd, m, n)
                partner = (negRd, n, m)
                if key < partner:
                    model.add_hop(m, n, Rd, [[v]])
                elif key == partner:      # onsite diagonal, R = 0
                    model.add_hop(m, n, Rd, [[v.real]])
    return model
