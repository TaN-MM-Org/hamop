"""v0.7 real-material anchors: a graphene hr.dat written independently
of the parser reproduces the closed-form-anchored lattice builder's
bands to machine precision, the save/load round trip preserves H(k)
exactly, degeneracy weights divide per the convention, and every
refusal fires on a deliberately corrupted file."""
import numpy as np
import pytest

import hamop


def _graphene_hr_text(t=-2.7):
    """A nearest-neighbour graphene hr.dat, written BY HAND from the
    R-space Hamiltonian (A at 0, B at (a1+a2)/3): H_AB(R) = t for
    R in {(0,0), (-1,0), (0,-1)} -- an independent construction of
    the same physics as hamop.lattices.graphene."""
    HAB = {(0, 0): t, (-1, 0): t, (0, -1): t}
    Rs = sorted({(0, 0), (-1, 0), (0, -1), (1, 0), (0, 1)})
    lines = ["hand-written graphene", "2", str(len(Rs))]
    lines.append(" ".join("1" for _ in Rs))
    for R in Rs:
        blk = np.zeros((2, 2))
        if R in HAB:
            blk[0, 1] = HAB[R]
        negR = (-R[0], -R[1])
        if negR in HAB:
            blk[1, 0] = HAB[negR]
        for n in range(2):
            for m in range(2):
                lines.append(f"{R[0]} {R[1]} 0 {m + 1} {n + 1} "
                             f"{blk[m, n]:.10f} 0.0")
    return "\n".join(lines) + "\n"


def test_hand_written_graphene_hr_matches_the_lattice_builder(tmp_path):
    t, a = -2.7, 2.46
    p = tmp_path / "graphene_hr.dat"
    p.write_text(_graphene_hr_text(t))
    cell = a * np.array([[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
    pos = np.array([[0.0, 0.0], (cell[0] + cell[1]) / 3.0])
    m_w = hamop.from_wannier90(p, cell, centers=pos)
    m_ref = hamop.graphene(t=t, a=a)
    rng = np.random.default_rng(0)
    ks = rng.uniform(-2.0, 2.0, (20, 2))
    e_w = hamop.bands(m_w, ks)
    e_r = hamop.bands(m_ref, ks)
    assert np.abs(e_w - e_r).max() < 1e-12
    # the Dirac point: gap closes at K
    K = np.array([4.0 * np.pi / (3.0 * a), 0.0])
    eK = hamop.bands(m_w, [K])[0]
    assert abs(eK[1] - eK[0]) < 1e-9
    # centers change nothing eigenvalue-derived (gauge invariance)
    m_w0 = hamop.from_wannier90(p, cell)
    assert np.abs(hamop.bands(m_w0, ks) - e_r).max() < 1e-12


def test_save_load_round_trip_preserves_H_of_k(tmp_path):
    rng = np.random.default_rng(1)
    nw = 3
    Rs = [(0, 0), (1, 0), (0, 1), (1, 1), (1, -1)]
    H_R = {}
    for R in Rs:
        H_R[R] = rng.normal(size=(nw, nw)) + 1j * rng.normal(size=(nw, nw))
        H_R[tuple(-x for x in R)] = H_R[R].conj().T
    H_R[(0, 0)] = 0.5 * (H_R[(0, 0)] + H_R[(0, 0)].conj().T)
    p = tmp_path / "rand_hr.dat"
    hamop.save_wannier90_hr(p, H_R)
    H2, nw2 = hamop.load_wannier90_hr(p)
    assert nw2 == nw
    assert set(H2) == {tuple(R) + (0,) for R in H_R}
    for R, blk in H_R.items():
        # blocks agree to the 1e-10 the fixed-point text format keeps
        assert np.abs(H2[tuple(R) + (0,)] - blk).max() < 1e-9
    # and through the model: H(k) eigenvalues agree at random k
    cell = np.array([[1.0, 0.0], [0.0, 1.3]])
    m1 = hamop.from_wannier90(p, cell)
    ks = rng.uniform(-3.0, 3.0, (10, 2))
    e1 = hamop.bands(m1, ks)
    ref = []
    for k in ks:                      # independent direct Bloch sum
        Hk = np.zeros((nw, nw), dtype=complex)
        for R, blk in H_R.items():
            Hk += np.exp(1j * k @ (np.asarray(R) @ cell)) * blk
        ref.append(np.linalg.eigvalsh(Hk))
    assert np.abs(e1 - np.asarray(ref)).max() < 1e-8


def test_degeneracy_weights_divide(tmp_path):
    """An R with degeneracy 2 must enter at half weight."""
    body = ["deg test", "1", "3", "2 2 2",
            "0 0 0 1 1 1.0 0.0",
            "1 0 0 1 1 0.5 0.0",
            "-1 0 0 1 1 0.5 0.0"]
    p = tmp_path / "deg_hr.dat"
    p.write_text("\n".join(body) + "\n")
    H_R, nw = hamop.load_wannier90_hr(p)
    assert H_R[(0, 0, 0)][0, 0] == 0.5          # 1.0 / degeneracy 2
    assert H_R[(1, 0, 0)][0, 0] == 0.25         # 0.5 / degeneracy 2
    assert H_R[(-1, 0, 0)][0, 0] == 0.25


def test_refusals_on_corrupted_files(tmp_path):
    good = _graphene_hr_text()
    lines = good.splitlines()

    def write(ls):
        q = tmp_path / "bad_hr.dat"
        q.write_text("\n".join(ls) + "\n")
        return q

    with pytest.raises(ValueError, match="matrix-element lines"):
        hamop.load_wannier90_hr(write(lines[:-1]))       # one line short
    dup = list(lines)
    dup[-1] = dup[-2]                                    # same count, repeated
    with pytest.raises(ValueError, match="duplicate"):
        hamop.load_wannier90_hr(write(dup))
    bad = list(lines)
    bad[3] = "1 1"                                       # deg list short
    with pytest.raises(ValueError, match="degeneracy"):
        hamop.load_wannier90_hr(write(bad))
    # break real-space Hermiticity: H_AB(0) != conj(H_BA(0))
    broken = [ln if not ln.startswith("0 0 0 1 2") else
              "0 0 0 1 2 9.9000000000 0.0" for ln in lines]
    with pytest.raises(ValueError, match="Hermitian|dagger"):
        hamop.load_wannier90_hr(write(broken))
    # a 2D cell with 3 used R components is refused
    p = tmp_path / "ok_hr.dat"
    p.write_text(good)
    with pytest.raises(ValueError, match="cell"):
        hamop.from_wannier90(p, np.eye(2)[:1, :1])


def test_imported_material_runs_the_full_stack(tmp_path):
    """The imported model is a first-class citizen: DOS integrates to
    the orbital count and the Kubo optics of imported graphene shows
    the universal interband plateau, exactly as the builder's model
    does (same numbers, same code path downstream)."""
    t, a = -2.7, 2.46
    p = tmp_path / "graphene_hr.dat"
    p.write_text(_graphene_hr_text(t))
    cell = a * np.array([[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
    pos = np.array([[0.0, 0.0], (cell[0] + cell[1]) / 3.0])
    m_w = hamop.from_wannier90(p, cell, centers=pos)
    m_ref = hamop.graphene(t=t, a=a)
    ks, wts = m_w.monkhorst_pack((24, 24))
    w = np.array([1.5])
    s_w = hamop.sigma_optical(m_w, w, mu=0.0, kpts=ks, weights=wts,
                              eta=0.15)
    s_r = hamop.sigma_optical(m_ref, w, mu=0.0, kpts=ks, weights=wts,
                              eta=0.15)
    assert np.abs(s_w - s_r).max() < 1e-10
