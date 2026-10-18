"""Regression tests for the bugs fixed in 0.14.0 and for periodic models
with fewer lattice vectors than Cartesian dimensions (ribbons).  Each
regression test fails on 0.13.0 (the failing numbers are quoted) and
passes now."""
import numpy as np
import pytest

from hamop import (TightBindingModel, band_edges, bands,
                   boltzmann_transport, carrier_count, carrier_density,
                   chemical_potential_from_density, dos, drude_weight,
                   fermi_level, find_point_group, fourier_interpolation,
                   gen_eigh, graphene, linear_chain, sigma_optical,
                   symmetry_fold)
from hamop.symmetry import _fold_grid, _group_closure


def _strained_graphene():
    g = graphene(t=-2.7, a=2.46)
    m = TightBindingModel(positions=g.positions, norb=1, cell=g.cell)
    m.add_hop(0, 1, (0, 0), [[-3.2]])
    m.add_hop(0, 1, (-1, 0), [[-2.7]])
    m.add_hop(0, 1, (0, -1), [[-2.7]])
    return m


def _c3():
    c, s = np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)
    return np.array([[c, -s], [s, c]])


def test_symmetry_fold_checks_operations_given_as_a_generator():
    """A strained bond breaks the three-fold rotation.  Given as a list,
    C3 was refused; given as a generator, the first loop used it up, the
    spectral check saw no operation, and 0.13.0 folded the grid: DOS
    (eta = 0.1 eV, mesh 12) at -3, 1 and 2.5 eV of 0.2636, 0.0492,
    0.2212 against 0.2393, 0.0657, 0.2078 on the full grid.  Now both
    forms are refused."""
    m = _strained_graphene()
    with pytest.raises(ValueError, match="not a symmetry"):
        symmetry_fold(m, 12, [_c3()])
    with pytest.raises(ValueError, match="not a symmetry"):
        symmetry_fold(m, 12, (R for R in [_c3()]))
    g = graphene(t=-2.7, a=2.46)          # unstrained: C3 is a symmetry
    k1, w1 = symmetry_fold(g, 12, [_c3()])
    k2, w2 = symmetry_fold(g, 12, iter([_c3()]))
    assert np.array_equal(k1, k2) and np.array_equal(w1, w2)


def _square(t2=-0.3):
    sq = TightBindingModel(positions=[[0.0, 0.0]], norb=1, cell=np.eye(2))
    sq.add_hop(0, 0, (1, 0), [[-1.0]])
    sq.add_hop(0, 0, (0, 1), [[-1.0]])
    sq.add_hop(0, 0, (1, 1), [[t2]])
    sq.add_hop(0, 0, (1, -1), [[t2]])
    return sq


def test_symmetry_fold_with_an_uneven_mesh():
    """A grid point idx / mesh maps to (idx / mesh) g; 0.13.0 used
    idx g, which is the same only for an even mesh.  For the square
    lattice on an (8, 4) mesh, C4 maps points off the grid, but the fold
    was accepted: DOS (eta = 0.2 eV) at 0.5 and 2 eV of 0.0422 and
    0.1310 against 0.0729 and 0.2574 on the full grid; graphene on a
    (6, 3) mesh with its 12 operations gave 0.0000 at 0.5 eV against
    0.0737.  Now these folds are refused, and a group that does fit the
    mesh (the two axis mirrors) reproduces the full grid to 1e-12."""
    sq = _square()
    ops = find_point_group(sq)
    E = np.array([-3.0, -1.0, 0.5, 2.0])
    for mesh in [(8, 4), (6, 3)]:
        with pytest.raises(ValueError, match="off the grid"):
            symmetry_fold(sq, mesh, ops)
        mirrors = [R for R in ops if abs(R[0, 1]) < 1e-9]
        k, w = symmetry_fold(sq, mesh, mirrors)
        assert len(k) < np.prod(mesh)
        full = dos(sq, E, mesh=mesh, eta=0.2)
        assert np.abs(dos(sq, E, kpts=k, weights=w, eta=0.2)
                      - full).max() < 1e-12
    g = graphene()
    with pytest.raises(ValueError, match="off the grid"):
        symmetry_fold(g, (6, 3), find_point_group(g))


def _old_fold_grid(group, mesh, recip):
    """The 0.13.0 orbit loop, verbatim in substance (valid for an even
    mesh)."""
    dim = len(mesh)
    ntot = int(np.prod(mesh))
    seen, order = {}, []
    for flat in range(ntot):
        idx, rest = [], flat
        for n in reversed(mesh):
            idx.append(rest % n)
            rest //= n
        idx = tuple(reversed(idx))
        v = np.array(idx, dtype=float)
        images = []
        for g in group:
            w = v @ g
            images.append(tuple(int(np.rint(w[d])) % mesh[d]
                                for d in range(dim)))
        rep = min(images)
        if rep in seen:
            seen[rep] += 1
        else:
            seen[rep] = 1
            order.append(rep)
    frac = np.array([[i / n for i, n in zip(rep, mesh)] for rep in order])
    return frac @ recip, np.array([seen[r] for r in order],
                                  dtype=float) / ntot


def _old_time_reversal_grid(model, mesh):
    """The 0.13.0 loop of monkhorst_pack(time_reversal=True)."""
    recip = 2.0 * np.pi * np.linalg.inv(model.cell).T
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
            continue
        frac_rows.append([i / n for i, n in zip(idx, mesh)])
        weights.append((1.0 if idx == partner else 2.0) / ntot)
    return np.asarray(frac_rows) @ recip, np.asarray(weights)


def _same_grid(a, b):
    """Same weights (integer counts over the same divisor: compared
    exactly) and the same k-points in the same order.  The k-points are
    the same fractions times the reciprocal vectors; they are compared
    to 1e-13 of the largest component, not exactly, because the product
    with the reciprocal vectors runs through BLAS."""
    assert np.array_equal(a[1], b[1])
    assert a[0].shape == b[0].shape
    assert np.abs(a[0] - b[0]).max() <= 1e-13 * max(np.abs(a[0]).max(), 1.0)


def test_vectorized_grid_folds_equal_the_old_loops():
    """The fold of symmetry_fold and of monkhorst_pack(time_reversal=True)
    is now done with integer array operations.  The kept points, their
    order and the weights are integer logic and the same divisions, so
    on even meshes they equal the 0.13.0 loops (see _same_grid)."""
    for m in (graphene(), _square()):
        recip = 2.0 * np.pi * np.linalg.inv(m.cell).T
        M = [np.rint(recip @ R.T @ np.linalg.inv(recip))
             for R in find_point_group(m)]
        group = _group_closure(M)
        for tr in (False, True):
            G = group + ([-x for x in group] if tr else [])
            for n in (1, 2, 3, 6, 13):
                _same_grid(_old_fold_grid(G, (n, n), recip),
                           _fold_grid(G, (n, n), recip))
    cube = TightBindingModel([[0, 0, 0]], 1, cell=[[1, 0.2, 0],
                                                   [0.1, 1.3, 0],
                                                   [0, 0.3, 2]])
    for m, meshes in [(graphene(), [(1, 1), (2, 2), (7, 7), (5, 8),
                                    (6, 1)]),
                      (linear_chain(), [(1,), (2,), (9,), (40,)]),
                      (cube, [(3, 3, 3), (2, 3, 5), (4, 4, 1)])]:
        for mesh in meshes:
            _same_grid(_old_time_reversal_grid(m, mesh),
                       m.monkhorst_pack(mesh, time_reversal=True))


def _dropping_model():
    """One site with two orbitals whose overlap matrix
    S(k) = [[1, c], [c*, 1]], c = (1 + e^{ik}) / 2, is singular at k = 0
    only, so canonical orthogonalization keeps one state there and two
    elsewhere."""
    m = TightBindingModel(positions=[[0.0]], norb=2, cell=[[1.0]])
    m.add_hop(0, 0, (0,), [[0.0, -0.5], [-0.5, 1.0]],
              [[1.0, 0.5], [0.5, 1.0]])
    m.add_hop(0, 0, (1,), [[-0.3, -0.2], [0.0, -0.3]],
              [[0.0, 0.5], [0.0, 0.0]])
    return m


def test_band_count_that_changes_with_k():
    """dos, carrier_count and sigma_optical handled a model whose band
    count changes with k, but fermi_level, band_edges and
    carrier_density raised NumPy's "inhomogeneous shape" ValueError in
    0.13.0 (and with NumPy < 1.24 failed on an object array).  Now the
    states are counted one by one: fermi_level is the root of
    carrier_count (1e-9), band_edges equals a k-point-by-k-point loop
    over gen_eigh, carrier_density equals carrier_count per length, and
    bands, which must return an (nk, nbands) array, is refused with an
    explanation."""
    m = _dropping_model()
    k, w = m.monkhorst_pack(24)
    per_k = [gen_eigh(*m.bloch(kk)) for kk in k]
    assert len(per_k[0]) == 1 and len(per_k[1]) == 2
    mu = fermi_level(m, 0.5, mesh=24, T=300.0)
    assert abs(carrier_count(m, mu, mesh=24, T=300.0, spin=1) - 0.5) < 1e-9
    allE = np.concatenate(per_k)
    for mu_e in (-0.2, 0.4):
        vbm, cbm, gap = band_edges(m, mu_e, mesh=24)
        assert vbm == allE[allE <= mu_e].max()
        assert cbm == allE[allE > mu_e].min()
    n = carrier_density(m, 0.4, 1.0, mesh=24)
    assert abs(n - (carrier_count(m, 0.4, mesh=24) - 1.0) * 1e8) \
        < 1e-9 * abs(n)
    r = chemical_potential_from_density(m, n, 1.0, mesh=400, T=600.0)
    assert abs(carrier_density(m, r["mu"], 1.0, mesh=400, T=600.0) / n
               - 1.0) < 1e-8
    with pytest.raises(ValueError, match="number of bands changes"):
        bands(m, k)


# ---------------- ribbons: fewer lattice vectors than dimensions --------

def _chain_in_plane(theta, a=1.7, e0=0.1, t=-1.0):
    u = np.array([np.cos(theta), np.sin(theta)])
    m = TightBindingModel(positions=[[0.3, -0.2]], norb=1, cell=[a * u])
    m.add_hop(0, 0, (0,), [[e0]])
    m.add_hop(0, 0, (1,), [[t]])
    return m


def test_chain_in_a_plane_equals_the_one_dimensional_chain():
    """A chain along a direction at 30 degrees in the plane (cell shape
    (1, 2)).  0.13.0 accepted the model, but monkhorst_pack, cell_volume
    and every function given mesh= raised LinAlgError ("Last 2
    dimensions of the array must be square").  Now: the DOS, the
    chemical potential and the carrier density equal those of
    linear_chain to 1e-12, and the Drude weight and the Boltzmann
    conductivity along x and y are the chain's times cos^2 and sin^2 of
    the angle (the velocity is along the chain); the Seebeck coefficient
    along the chain equals the chain's."""
    th, a = np.pi / 6, 1.7
    m, c = _chain_in_plane(th, a), linear_chain(t=-1.0, e0=0.1, a=a)
    assert m.cell_volume == pytest.approx(a, rel=1e-15)
    b = m.reciprocal_vectors
    assert abs(b @ m.cell.T - 2 * np.pi).max() < 1e-14
    E = np.array([-1.5, 0.3, 1.2])
    assert np.abs(dos(m, E, mesh=400) - dos(c, E, mesh=400)).max() < 1e-12
    assert abs(fermi_level(m, 0.3, mesh=400)
               - fermi_level(c, 0.3, mesh=400)) < 1e-12
    n_m = carrier_density(m, 0.2, 1.0, mesh=400)
    assert abs(n_m / carrier_density(c, 0.2, 1.0, mesh=400) - 1) < 1e-12
    proj = (np.cos(th) ** 2, np.sin(th) ** 2)
    Dc = drude_weight(c, 0.2, mesh=400)
    bc = boltzmann_transport(c, 0.2, 300.0, mesh=4000)
    bm = boltzmann_transport(m, 0.2, 300.0, mesh=4000)
    for d in (0, 1):
        assert abs(drude_weight(m, 0.2, mesh=400, direction=d)
                   / (proj[d] * Dc) - 1) < 1e-12
        assert abs(bm["sigma"][d, d] / (proj[d] * bc["sigma"][0, 0]) - 1) \
            < 1e-12
    u = m.cell[0] / np.linalg.norm(m.cell[0])        # along the chain
    assert abs(u @ bm["seebeck"] @ u / bc["seebeck"][0, 0] - 1) < 1e-12
    k, w = m.monkhorst_pack(10, time_reversal=True)
    assert np.isclose(w.sum(), 1.0) and k.shape == (6, 2)
    assert fourier_interpolation(m, 6).max_residual() < 1e-12
    with pytest.raises(ValueError, match="as many lattice vectors"):
        symmetry_fold(m, 10, [np.eye(2)])


def test_zak_phase_of_an_ssh_chain_lying_in_a_plane():
    """berry_phase on the SSH chain placed along a direction at 0.4 rad
    in a plane (cell shape (1, 2)) equals the 1D chain's Zak phase, 0 or
    pi, to 1e-12."""
    from hamop import berry_phase, ssh
    u = np.array([np.cos(0.4), np.sin(0.4)])
    b = 2 * np.pi / 2.0
    for t1, t2 in ((-1.0, -0.6), (-0.6, -1.0)):
        m = TightBindingModel(positions=[[0.0, 0.0], list(1.0 * u)], norb=1,
                              cell=[2.0 * u])
        m.add_hop(0, 1, (0,), [[t1]])
        m.add_hop(1, 0, (1,), [[t2]])
        z2 = berry_phase(m, [b * i / 60 * u for i in range(60)])
        z1 = berry_phase(ssh(t1, t2, a=2.0), [[b * i / 60] for i in range(60)])
        assert abs(np.exp(1j * z2) - np.exp(1j * z1)) < 1e-12


def test_square_lattice_strip_closed_form_and_vacuum_cell():
    """A strip of W = 3 rows of the square lattice, periodic along x
    only.  Its bands are -2t cos(kx) + 2t cos(p pi / (W + 1)), p = 1..W
    (standing waves across the strip), to 1e-12; and the same strip
    built as a 2D model with a 40 Angstrom vacuum cell along y (no
    hopping across it, mesh (N, 1)) gives the same DOS and Boltzmann
    S, and a sigma_yy (transitions across the strip) and conductivity
    40 times smaller per area, to 1e-12."""
    W, t, L = 3, -1.0, 40.0
    pos = [[0.0, float(y)] for y in range(W)]
    rib = TightBindingModel(positions=pos, norb=1, cell=[[1.0, 0.0]])
    vac = TightBindingModel(positions=pos, norb=1,
                            cell=[[1.0, 0.0], [0.0, L]])
    for m, img in ((rib, (1,)), (vac, (1, 0))):
        for y in range(W):
            m.add_hop(y, y, img, [[t]])
            if y + 1 < W:
                m.add_hop(y, y + 1, (0,) * len(img), [[t]])
    kx = np.linspace(-np.pi, np.pi, 9)[:, None] * np.array([[1.0, 0.0]])
    exact = np.sort(np.array([[2 * t * np.cos(k[0])
                               + 2 * t * np.cos(p * np.pi / (W + 1))
                               for p in range(1, W + 1)] for k in kx]),
                    axis=1)
    assert np.abs(bands(rib, kx) - exact).max() < 1e-12
    E = np.linspace(-3.5, 3.5, 8)
    assert np.abs(dos(rib, E, mesh=300) - dos(vac, E, mesh=(300, 1))
                  ).max() < 1e-12
    s_r = sigma_optical(rib, [1.0, 2.0], 0.3, mesh=300, direction=1)
    s_v = sigma_optical(vac, [1.0, 2.0], 0.3, mesh=(300, 1), direction=1)
    assert s_r.min() > 0.0
    assert np.abs(s_r - L * s_v).max() < 1e-12 * s_r.max()
    b_r = boltzmann_transport(rib, 0.3, 300.0, mesh=3000)
    b_v = boltzmann_transport(vac, 0.3, 300.0, mesh=(3000, 1))
    # S m for the ribbon, S per square for the 2D cell of width L
    assert abs(b_r["sigma"][0, 0] / (L * 1e-10 * b_v["sigma"][0, 0])
               - 1) < 1e-12
    assert abs(b_r["seebeck"][0, 0] / b_v["seebeck"][0, 0] - 1) < 1e-12
    # nothing conducts across the strip (rounding only)
    assert abs(b_r["sigma"][1, 1]) < 1e-12 * b_r["sigma"][0, 0]
    assert abs(b_r["seebeck"][1, 1]) < 1e-12 * abs(b_r["seebeck"][0, 0])
    assert b_r["units"]["sigma"] == "S m"


def test_cells_that_are_not_lattices_are_refused():
    """Before 0.14.0 these were accepted and failed later with a NumPy
    LinAlgError (or "Singular matrix") at the first k-grid."""
    for pos, cell in (([[0, 0]], [[1, 0], [2, 0]]), ([[0]], [[0.0]]),
                      ([[0, 0]], [[1, 0], [0, 1], [1, 1]]),
                      ([[0, 0]], [[np.nan, 0]])):
        with pytest.raises(ValueError):
            TightBindingModel(pos, 1, cell=cell)
