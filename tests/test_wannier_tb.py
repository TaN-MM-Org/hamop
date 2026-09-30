"""Wannier90 seedname_tb.dat import (new in 0.11.0).  A graphene tb.dat
is written by hand in the layout of the Wannier90 writer
(hamiltonian_write_tb: '(15I5)' degeneracies, '(/,3I5)' R lines,
'(2I5,3x,2(E15.8,1x))' Hamiltonian and '(2I5,3x,6(E15.8,1x))' position
lines), independently of the parser.  The imported model must equal
the lattice builder in its bands AND its optical conductivity; the
optics depend on the Wannier centres, so they check that the centres
were read from the position block."""
import numpy as np
import pytest

import hamop


def _e15_8(x):
    """Fortran E15.8: 0.dddddddd E+ee, right-justified in 15 columns."""
    if x == 0.0:
        return " 0.00000000E+00"
    e = int(np.floor(np.log10(abs(x)))) + 1
    mant = x / 10.0 ** e
    if abs(round(mant, 8)) >= 1.0:          # rounding pushed it to 1.0
        e += 1
        mant = x / 10.0 ** e
    return f"{mant:.8f}E{e:+03d}".rjust(15)


def _graphene_tb_text(t=-2.7, a=2.46, c=15.0, z=(1.7, 1.7), deg2=True,
                      drop_position=False):
    cell = np.array([[a, 0.0, 0.0], [0.5 * a, 0.5 * np.sqrt(3.0) * a, 0.0],
                     [0.0, 0.0, c]])
    cent = np.array([[0.0, 0.0, z[0]],
                     list((cell[0] + cell[1])[:2] / 3.0) + [z[1]]])
    HAB = {(0, 0, 0): t, (-1, 0, 0): t, (0, -1, 0): t}
    Rs = sorted({(0, 0, 0), (-1, 0, 0), (0, -1, 0), (1, 0, 0), (0, 1, 0),
                 (1, -1, 0), (-1, 1, 0)})
    # degeneracy 2 on the two (+-1, -+1) vectors: every value stored
    # there is multiplied by 2, so the reader must divide it back
    deg = {R: (2 if deg2 and R in {(1, -1, 0), (-1, 1, 0)} else 1)
           for R in Rs}
    out = [" written on 30Sep2026 at 12:00:00"]
    for v in cell:
        out.append("  " + "  ".join(f"{x:.16f}" for x in v))
    out.append(f"{2:12d}")
    out.append(f"{len(Rs):12d}")
    out.append("".join(f"{deg[R]:5d}" for R in Rs))
    for R in Rs:
        blk = np.zeros((2, 2), dtype=complex)
        if R in HAB:
            blk[0, 1] = HAB[R]
        negR = tuple(-x for x in R)
        if negR in HAB:
            blk[1, 0] = HAB[negR]
        if R in {(1, -1, 0), (-1, 1, 0)}:
            blk[0, 0] = blk[1, 1] = 0.05        # a small real 3rd-shell hop
        out.append("")
        out.append("".join(f"{x:5d}" for x in R))
        for i in range(2):                     # i outer, j inner:
            for j in range(2):                 # write(j, i, ham_r(j, i))
                v = blk[j, i] * deg[R]
                out.append(f"{j + 1:5d}{i + 1:5d}   {_e15_8(v.real)} "
                           f"{_e15_8(v.imag)} ")
    if not drop_position:
        rng = np.random.default_rng(7)
        for R in Rs:
            out.append("")
            out.append("".join(f"{x:5d}" for x in R))
            for i in range(2):
                for j in range(2):
                    if R == (0, 0, 0) and i == j:
                        p = cent[i].astype(complex)
                    else:           # off-diagonal elements: not used
                        p = rng.normal(size=3) * 0.01 + 0j
                    p = p * deg[R]
                    fields = " ".join(f"{_e15_8(q.real)} {_e15_8(q.imag)}"
                                      for q in p)
                    out.append(f"{j + 1:5d}{i + 1:5d}   {fields} ")
    return "\n".join(out) + "\n", cell, cent


def _reference(t=-2.7, a=2.46):
    m = hamop.graphene(t=t, a=a)
    # the (+-1, -+1) hop of 0.05 eV on both sublattices
    m.add_hop(0, 0, (1, -1), [[0.05]])
    m.add_hop(1, 1, (1, -1), [[0.05]])
    return m


def test_e15_8_formatter_is_fortran_like():
    assert _e15_8(-2.7) == "-0.27000000E+01"
    assert _e15_8(0.05) == " 0.50000000E-01"
    assert float(_e15_8(1.4201)) == 1.4201


def test_tb_dat_reproduces_bands_and_optics_of_graphene(tmp_path):
    text, cell, cent = _graphene_tb_text()
    p = tmp_path / "graphene_tb.dat"
    p.write_text(text)
    data = hamop.load_wannier90_tb(p)
    assert np.abs(data["cell"] - cell).max() < 1e-15
    assert abs(data["H_R"][(1, -1, 0)][0, 0] - 0.05) < 1e-15   # divided
    m = hamop.from_wannier90_tb(p)                 # 2D, cell from file
    assert m.cell.shape == (2, 2)
    assert np.abs(m.positions - cent[:, :2]).max() < 1e-8
    ref = _reference()
    ks = np.random.default_rng(0).uniform(-2.0, 2.0, (20, 2))
    assert np.abs(hamop.bands(m, ks) - hamop.bands(ref, ks)).max() < 1e-12
    om = np.array([1.0, 2.0, 4.0])
    # the file stores 8 significant digits (E15.8), so the centres are
    # exact to ~1e-9 Angstrom and the optics to 1e-7 relative
    for d in (0, 1):
        s_tb = hamop.sigma_optical(m, om, 0.0, mesh=24, eta=0.2, T=300.0,
                                   direction=d)
        s_ref = hamop.sigma_optical(ref, om, 0.0, mesh=24, eta=0.2,
                                    T=300.0, direction=d)
        assert np.abs(s_tb - s_ref).max() < 1e-7 * s_ref.max()
    # the check is sensitive to the centres: all orbitals at the origin
    # (what from_wannier90 assumes without centres) changes the optics
    m0 = hamop.from_wannier90_tb(p)
    m0.positions[:] = 0.0
    s0 = hamop.sigma_optical(m0, om, 0.0, mesh=24, eta=0.2, T=300.0)
    s_ref = hamop.sigma_optical(ref, om, 0.0, mesh=24, eta=0.2, T=300.0)
    assert np.abs(s0 - s_ref).max() > 1e-2 * s_ref.max()
    # and the tb.dat route equals the hr.dat route given the same centres
    H_R = data["H_R"]
    q = tmp_path / "graphene_hr.dat"
    hamop.save_wannier90_hr(q, H_R)
    m_hr = hamop.from_wannier90(q, cell, centers=cent)       # 3D cell
    k3 = np.c_[ks, np.zeros(len(ks))]
    assert np.abs(hamop.bands(m_hr, k3) - hamop.bands(m, ks)).max() < 1e-9


def test_tb_dat_refusals(tmp_path):
    def write(text, name="bad_tb.dat"):
        q = tmp_path / name
        q.write_text(text)
        return q

    text, _, _ = _graphene_tb_text(drop_position=True)
    with pytest.raises(ValueError, match="position sections"):
        hamop.load_wannier90_tb(write(text))
    text, _, _ = _graphene_tb_text()
    lines = text.splitlines()
    with pytest.raises(ValueError, match="lines after the degeneracies"):
        hamop.load_wannier90_tb(write("\n".join(lines[:-1]) + "\n"))
    # a Hamiltonian line with a broken value -> non-Hermitian
    k = next(i for i, ln in enumerate(lines)
             if ln.startswith("    1    2") and "0.27000000E+01" in ln)
    broken = list(lines)
    broken[k] = broken[k].replace("-0.27000000E+01", "-0.26000000E+01")
    with pytest.raises(ValueError, match="H\\(-R\\)"):
        hamop.load_wannier90_tb(write("\n".join(broken) + "\n"))
    # a tilted cell cannot be cut to 2D without changing the geometry
    tilted = list(lines)
    tilted[1] = "  2.46  0.0  0.3"
    with pytest.raises(ValueError, match="Cartesian components"):
        hamop.from_wannier90_tb(write("\n".join(tilted) + "\n", "t.dat"))
    # but as a 3D model it imports
    m3 = hamop.from_wannier90_tb(write("\n".join(tilted) + "\n", "t3.dat"),
                                 dim=3)
    assert m3.cell.shape == (3, 3)
