"""The tight-binding model container: real-space blocks in, Bloch matrices out.

A model is a set of sites, each carrying ``norb`` orbitals, and a set of
directed hoppings between sites.  Each hopping is stored once, as
``(i, j, image, H_block, S_block)``: the block couples the orbitals of
site ``i`` in the home cell to the orbitals of site ``j`` in the cell
displaced by the integer lattice vector ``image``.  The Hermitian
partner (``j`` back to ``i`` in the opposite image) is implied and added
by the assembly, so a bond is never double-counted.  On-site blocks
(``i == j`` and zero image) must be Hermitian themselves.

Assembly follows the standard atomic-gauge convention: the Bloch phase
of a block is ``exp(i k . d)`` with ``d`` the *Cartesian* displacement
from site ``i`` to site ``j`` including the lattice vector, so the
k-derivative of the Hamiltonian carries a factor ``i d`` per block.
That derivative is exactly what the Kubo velocity operator needs, which
is why the two live in one class: the optics and the spectrum can never
drift out of sync with each other.

Units: energies in eV, positions in Angstrom, k in 1/Angstrom.  Finite
(non-periodic) systems are models with ``cell=None`` and only zero
images; every observable then works at the single "k-point" k = 0.
"""
from __future__ import annotations

import numpy as np

__all__ = ["TightBindingModel"]


class TightBindingModel:
    """Sites, orbitals and directed hopping blocks; assembles H(k), S(k).

    positions: (nsite, dim) Cartesian site positions (Angstrom).
    norb: int or sequence, orbitals per site.
    cell: (dim, dim) lattice vectors as rows (Angstrom), or None for a
    finite system.
    """

    def __init__(self, positions, norb, cell=None):
        self.positions = np.atleast_2d(np.asarray(positions, dtype=float))
        n_site = self.positions.shape[0]
        if np.isscalar(norb):
            self.norb = np.full(n_site, int(norb))
        else:
            self.norb = np.asarray(norb, dtype=int)
            if self.norb.shape != (n_site,):
                raise ValueError("norb must be scalar or one entry per site")
        self.cell = None if cell is None else np.atleast_2d(
            np.asarray(cell, dtype=float))
        self.offsets = np.concatenate([[0], np.cumsum(self.norb)])
        self.nao = int(self.offsets[-1])
        self._hops = []
        self._dipoles = {}

    # ------------------------------------------------------------------
    def add_hop(self, i, j, image, H_block, S_block=None):
        """Add one directed hopping block (the reverse partner is implied).

        image: integer lattice vector of the cell containing site j
        (all zeros for a finite system or an intra-cell bond).
        H_block: (norb_i, norb_j) array.  S_block defaults to zero for
        an inter-site block; identity is used automatically for on-site
        blocks when no overlap is given anywhere (orthogonal basis).
        """
        i, j = int(i), int(j)
        image = tuple(int(m) for m in np.atleast_1d(image))
        if self.cell is None and any(image):
            raise ValueError("finite system: image must be zero")
        H_block = np.asarray(H_block, dtype=complex)
        if H_block.shape != (self.norb[i], self.norb[j]):
            raise ValueError(
                f"H block shape {H_block.shape} != "
                f"({self.norb[i]}, {self.norb[j]}) for sites ({i}, {j})")
        onsite = (i == j) and not any(image)
        if onsite and not np.allclose(H_block, H_block.conj().T):
            raise ValueError("on-site block must be Hermitian")
        if S_block is not None:
            S_block = np.asarray(S_block, dtype=complex)
            if S_block.shape != H_block.shape:
                raise ValueError("S block shape must match H block shape")
            if onsite and not np.allclose(S_block, S_block.conj().T):
                raise ValueError("on-site overlap block must be Hermitian")
        self._hops.append((i, j, image, H_block, S_block))

    def has_overlap(self):
        return any(h[4] is not None for h in self._hops)

    # ------------------------------------------------------------------
    def set_dipole(self, i, X):
        """Intra-atomic dipole matrix elements of site ``i``.

        X: array of shape (norb_i, norb_i, dim) -- the on-site position
        matrix elements <alpha| r_a - tau_i |beta> in Angstrom, one
        Hermitian block per Cartesian direction.  The optics modules add
        the corresponding velocity contribution i (E_n - E_m) X_nm to
        the interband matrix element, restoring transitions that the
        site-diagonal position approximation leaves dark (e.g. s -> p
        on one atom).  Orthogonal bases only; the overlap generalization
        is not implemented and is refused in the optics.
        """
        i = int(i)
        X = np.asarray(X, dtype=complex)
        dim = self.positions.shape[1]
        if X.shape != (self.norb[i], self.norb[i], dim):
            raise ValueError(
                f"dipole block must have shape "
                f"({self.norb[i]}, {self.norb[i]}, {dim})")
        for a in range(dim):
            if not np.allclose(X[:, :, a], X[:, :, a].conj().T):
                raise ValueError("dipole blocks must be Hermitian "
                                 "per direction")
        self._dipoles[i] = X

    def has_dipoles(self):
        return bool(self._dipoles)

    def dipole_matrix(self, direction=0):
        """Block-diagonal intra-atomic dipole operator X_a (nao x nao),
        zero wherever no dipole block was set."""
        X = np.zeros((self.nao, self.nao), dtype=complex)
        for i, blk in self._dipoles.items():
            o = self.offsets[i]
            n = self.norb[i]
            X[o:o + n, o:o + n] = blk[:, :, direction]
        return X

    # ------------------------------------------------------------------
    def _displacement(self, i, j, image):
        d = self.positions[j] - self.positions[i]
        if self.cell is not None:
            d = d + np.asarray(image, dtype=float) @ self.cell
        return d

    def _terms(self):
        """Yield (oi, oj, d, Hb, Sb) for every block and its implied
        Hermitian partner, exactly once each."""
        overlap = self.has_overlap()
        seen_onsite_S = set()
        for i, j, image, Hb, Sb in self._hops:
            oi, oj = self.offsets[i], self.offsets[j]
            d = self._displacement(i, j, image)
            onsite = (i == j) and not any(image)
            if Sb is None:
                if onsite and overlap:
                    Sb = np.eye(self.norb[i], dtype=complex)
                else:
                    Sb = np.zeros_like(Hb)
            if onsite:
                seen_onsite_S.add(i)
                yield oi, oj, d, Hb, Sb
            else:
                yield oi, oj, d, Hb, Sb
                yield oj, oi, -d, Hb.conj().T, Sb.conj().T
        if overlap:
            # sites whose on-site block was never given still need S = 1
            given = {i for i, j, im, _, _ in self._hops
                     if i == j and not any(im)}
            for i in range(len(self.norb)):
                if i not in given:
                    oi = self.offsets[i]
                    yield (oi, oi, np.zeros(self.positions.shape[1]),
                           np.zeros((self.norb[i], self.norb[i]), complex),
                           np.eye(self.norb[i], dtype=complex))

    # ------------------------------------------------------------------
    def bloch(self, k=None):
        """H(k), S(k) as dense Hermitian matrices.

        k: Cartesian wave vector (1/Angstrom); None means k = 0.  For a
        finite system pass None.  S(k) is the identity when the model
        has no overlap blocks (orthogonal basis).
        """
        k = self._kvec(k)
        H = np.zeros((self.nao, self.nao), dtype=complex)
        S = np.zeros((self.nao, self.nao), dtype=complex)
        overlap = self.has_overlap()
        for oi, oj, d, Hb, Sb in self._terms():
            ph = np.exp(1j * float(k @ d))
            ni, nj = Hb.shape
            H[oi:oi + ni, oj:oj + nj] += ph * Hb
            S[oi:oi + ni, oj:oj + nj] += ph * Sb
        H = 0.5 * (H + H.conj().T)
        if overlap:
            S = 0.5 * (S + S.conj().T)
        else:
            S = np.eye(self.nao, dtype=complex)
        return H, S

    def bloch_derivative(self, k=None, direction=0):
        """dH/dk and dS/dk along a Cartesian direction, at wave vector k.

        These are the matrices the Kubo velocity operator is built from;
        they are Hermitian by construction because every block enters
        together with its reversed partner at -d.
        """
        k = self._kvec(k)
        dH = np.zeros((self.nao, self.nao), dtype=complex)
        dS = np.zeros((self.nao, self.nao), dtype=complex)
        for oi, oj, d, Hb, Sb in self._terms():
            ph = 1j * d[direction] * np.exp(1j * float(k @ d))
            ni, nj = Hb.shape
            dH[oi:oi + ni, oj:oj + nj] += ph * Hb
            dS[oi:oi + ni, oj:oj + nj] += ph * Sb
        dH = 0.5 * (dH + dH.conj().T)
        dS = 0.5 * (dS + dS.conj().T)
        return dH, dS

    def _kvec(self, k):
        dim = self.positions.shape[1]
        if k is None:
            return np.zeros(dim)
        k = np.atleast_1d(np.asarray(k, dtype=float))
        if k.shape != (dim,):
            raise ValueError(f"k must have dimension {dim}")
        return k

    # ------------------------------------------------------------------
    def all_blocks_real(self):
        """True when every stored H and S block is real, in which case
        H(-k) = conj(H(k)) and the spectrum is even in k (spinless
        time-reversal symmetry of the assembly)."""
        return all(np.all(Hb.imag == 0.0)
                   and (Sb is None or np.all(Sb.imag == 0.0))
                   for _, _, _, Hb, Sb in self._hops)

    def monkhorst_pack(self, mesh, time_reversal=False):
        """Uniform Gamma-centered k-grid over the Brillouin zone.

        mesh: number of points per reciprocal direction (int or sequence
        matching the cell dimension).  Returns (kpts_cart, weights) with
        weights summing to one.

        time_reversal=True folds k and -k into one point of doubled
        weight (roughly halving the work), which is exact for k-even
        observables -- eigenvalue sums, the DOS, sigma_xx, the Drude
        weight.  The fold is guaranteed only when every real-space block
        is real, so that H(-k) = conj(H(k)); a model with complex blocks
        (e.g. Haldane) is refused rather than silently averaged.
        """
        if self.cell is None:
            raise ValueError("finite system has no Brillouin zone")
        dim = self.cell.shape[0]
        if np.isscalar(mesh):
            mesh = (int(mesh),) * dim
        recip = 2.0 * np.pi * np.linalg.inv(self.cell).T
        if not time_reversal:
            grids = [np.arange(n) / n for n in mesh]
            frac = np.stack(np.meshgrid(*grids, indexing="ij"),
                            axis=-1).reshape(-1, dim)
            kpts = frac @ recip
            w = np.full(len(kpts), 1.0 / len(kpts))
            return kpts, w
        if not self.all_blocks_real():
            raise ValueError(
                "time_reversal=True needs every real-space block real "
                "(H(-k) = conj(H(k))); this model has complex blocks, "
                "so the k <-> -k fold is not guaranteed exact")
        ntot = int(np.prod(mesh))
        frac_rows, weights = [], []
        for flat in range(ntot):
            idx, rest = [], flat
            for n in reversed(mesh):
                idx.append(rest % n)
                rest //= n
            idx = tuple(reversed(idx))
            partner = tuple((-i) % n for i, n in zip(idx, mesh))
            if idx > partner:
                continue                      # its partner represents it
            frac_rows.append([i / n for i, n in zip(idx, mesh)])
            weights.append((1.0 if idx == partner else 2.0) / ntot)
        kpts = np.asarray(frac_rows) @ recip
        return kpts, np.asarray(weights)

    @property
    def cell_volume(self):
        """Length / area / volume of the unit cell (Angstrom^dim)."""
        if self.cell is None:
            raise ValueError("finite system has no cell")
        c = self.cell
        if c.shape == (1, 1):
            return float(abs(c[0, 0]))
        if c.shape == (2, 2):
            return float(abs(np.linalg.det(c)))
        return float(abs(np.linalg.det(c)))
