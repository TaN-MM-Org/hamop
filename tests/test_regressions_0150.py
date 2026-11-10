"""Regression tests for the bugs fixed in 0.15.0.  Each one fails on
0.14.0 (the failing numbers are quoted in the docstrings and in the
CHANGELOG) and passes now.

Two of them concern the order of the lattice vectors.  A crystal can be
described with (a1, a2) or with (a2, a1); the second order is
left-handed (a1 x a2 < 0).  Physical results such as the Chern number,
the Berry curvature dipole or the direction of a magnetic field must not
depend on that choice."""
import numpy as np
import pytest

from hamop import (TightBindingModel, berry_dipole, berry_phase,
                   chern_number, haldane, magnetic_supercell,
                   quantum_weight, sigma_tensor, with_peierls)


def _relisted(model):
    """The same crystal with its two lattice vectors in the other order:
    identical sites and bonds, the image of every bond swapped to
    match."""
    m = TightBindingModel(positions=model.positions, norb=model.norb,
                          cell=model.cell[::-1])
    for i, j, img, Hb, Sb in model._hops:
        m.add_hop(i, j, img[::-1], Hb, Sb)
    return m


def test_chern_number_does_not_depend_on_the_order_of_the_lattice_vectors():
    """Haldane model, lattice vectors listed as (a2, a1).  0.14.0 took
    every plaquette loop b1 first and b2 second, which is clockwise for
    a left-handed cell: chern_number gave -1.000000 (both frames, both
    solvers) while the Kubo sigma_xy(0) of the same model is
    +1.000000 e^2/h, so the TKNN relation failed.  Now the loops run
    counterclockwise in Cartesian k-space."""
    h = haldane(t1=-1.0, t2=0.1)
    hl = _relisted(h)
    assert np.linalg.det(h.cell) > 0 > np.linalg.det(hl.cell)
    sxy = sigma_tensor(hl, [0.0], mu=0.0, directions=(0, 1), mesh=36,
                       T=10.0, eta=1e-4, spin=1)[0].real * np.pi / 2
    assert abs(sxy - 1.0) < 1e-6                  # Kubo: Cartesian, +1
    for frame in ("lowdin", "atomic"):
        assert abs(chern_number(hl, mesh=18, frame=frame) - 1.0) < 1e-12
    # the sparse solver (the point-by-point loop) on a three-cell
    # supercell of the same model, whose three lowest bands carry C = 1
    sc = _relisted(magnetic_supercell(h, 0, 3))
    assert np.linalg.det(sc.cell) < 0
    assert abs(chern_number(sc, mesh=12, n_occ=3, solver="sparse")
               - 1.0) < 1e-12
    # the quantum-geometry Chern number was never affected; it agrees
    assert abs(quantum_weight(hl, mesh=18)["chern"] - 1.0) < 0.02
    # and the trivial phase stays 0
    assert abs(chern_number(_relisted(haldane(m_ab=0.9)), mesh=18)) < 1e-12


def test_plaquette_fluxes_of_a_relisted_cell_are_the_same_set():
    """The plaquettes of the two descriptions are the same squares of
    k-space, traversed the same way now: the flux arrays are transposes
    of each other (0.14.0: minus the transpose)."""
    from hamop import berry_curvature
    h = haldane(t1=-1.0, t2=0.1, m_ab=0.2)
    F = berry_curvature(h, 12)
    Fl = berry_curvature(_relisted(h), 12)
    assert np.abs(Fl - F.T).max() < 1e-12


def _gapped_graphene(mass=0.5, t=-2.7, bond_scale=1.3, a=2.46):
    """Gapped graphene with one strained bond (the model of
    tests/test_nonlinear.py): a nonzero Berry curvature dipole."""
    cell = a * np.array([[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
    pos = np.array([np.zeros(2), (cell[0] + cell[1]) / 3.0])
    m = TightBindingModel(positions=pos, norb=1, cell=cell)
    m.add_hop(0, 0, (0, 0), [[+mass]])
    m.add_hop(1, 1, (0, 0), [[-mass]])
    m.add_hop(0, 1, (0, 0), [[t * bond_scale]])
    m.add_hop(0, 1, (-1, 0), [[t]])
    m.add_hop(0, 1, (0, -1), [[t]])
    return m


def test_berry_dipole_does_not_depend_on_the_order_of_the_lattice_vectors():
    """The dipole is a Cartesian vector of the crystal.  0.14.0 returned
    D = (-0.005340, +0.009250) Angstrom (method="grad", mesh 36, mu =
    0.9 eV, kT = 0.3 eV) for the cell (a1, a2) and the opposite vector,
    (+0.005340, -0.009250), for (a2, a1); method="fermi" likewise gave
    (-0.005866, +0.010160) and its negative.  Now the two cells agree,
    for both integral forms."""
    m = _gapped_graphene()
    ml = _relisted(m)
    for method in ("grad", "fermi"):
        d = berry_dipole(m, 36, mu=0.9, kT=0.3, method=method)["D"]
        dl = berry_dipole(ml, 36, mu=0.9, kT=0.3, method=method)["D"]
        assert np.abs(d).max() > 1e-3                 # genuinely nonzero
        assert np.abs(dl - d).max() < 1e-10 * np.abs(d).max()


def _square(t=-1.0):
    m = TightBindingModel([[0.0, 0.0]], 1, cell=[[1.0, 0.0], [0.0, 1.0]])
    m.add_hop(0, 0, (1, 0), [[t]])
    m.add_hop(0, 0, (0, 1), [[t]])
    return m


def _element(model, i, j, image):
    """<i, 0| H |j, image> of a one-orbital model, read off its blocks."""
    image = tuple(int(x) for x in image)
    v = 0.0j
    for a, b, im, Hb, _ in model._hops:
        im = tuple(im)
        if (a, b, im) == (i, j, image):
            v += Hb[0, 0]
        if (b, a, tuple(-x for x in im)) == (i, j, image) and \
                not (a == b and not any(im)):
            v += np.conj(Hb[0, 0])
    return v


def _loop_phase(model, corners):
    """Phase of the product of hoppings around a loop of (site, image)
    corners (the hoppings of the bare lattice are real, so this is the
    Peierls phase)."""
    prod = 1.0 + 0.0j
    for (i, Ri), (j, Rj) in zip(corners, corners[1:] + corners[:1]):
        prod *= _element(model, i, j, np.subtract(Rj, Ri))
    return np.angle(prod)


def test_magnetic_supercell_field_direction_does_not_depend_on_the_cell_order():
    """Square lattice at flux 1/3.  For the cell (a1, a2) the phase
    around a unit square traversed counterclockwise in the plane is
    +2 pi/3, the direction with_peierls uses for phi > 0.  0.14.0 gave
    -2 pi/3 for the cell (a2, a1): the field was reversed, and sigma_xy
    with mu in the lowest gap came out +1.0000 e^2/h instead of -1.0000
    (while chern_number, wrong in the opposite way, gave -1)."""
    p, q = 1, 3
    # with_peierls on a finite square: the reference direction
    flake = TightBindingModel([[x, y] for x in range(2) for y in range(2)],
                              1)
    flake.add_hop(0, 2, (0, 0), [[-1.0]])        # (0,0) -> (1,0)
    flake.add_hop(2, 3, (0, 0), [[-1.0]])        # (1,0) -> (1,1)
    flake.add_hop(3, 1, (0, 0), [[-1.0]])        # (1,1) -> (0,1)
    flake.add_hop(1, 0, (0, 0), [[-1.0]])        # (0,1) -> (0,0)
    fp = with_peierls(flake, p / q)
    ccw = [(i, (0, 0)) for i in (0, 2, 3, 1)]
    want = 2.0 * np.pi * p / q
    assert abs(_loop_phase(fp, ccw) - want) < 1e-12

    right = magnetic_supercell(_square(), p, q)
    # sites n at (n, 0); cell [(3, 0), (0, 1)]
    loop_r = [(0, (0, 0)), (1, (0, 0)), (1, (0, 1)), (0, (0, 1))]
    assert abs(_loop_phase(right, loop_r) - want) < 1e-12
    left = magnetic_supercell(_relisted(_square()), p, q)
    # sites n at (0, n); cell [(0, 3), (1, 0)]
    loop_l = [(0, (0, 0)), (0, (0, 1)), (1, (0, 1)), (1, (0, 0))]
    assert abs(_loop_phase(left, loop_l) - want) < 1e-12

    # the Kubo Hall conductivity, in Cartesian axes, now agrees, and the
    # TKNN relation holds for both cells
    s = []
    for sc in (right, left):
        sxy = sigma_tensor(sc, [0.0], mu=-1.3, directions=(0, 1), mesh=36,
                           T=10.0, eta=1e-4, spin=1)[0].real * np.pi / 2
        C = chern_number(sc, mesh=15, n_occ=1)
        assert abs(C + 1.0) < 1e-12
        assert abs(sxy - C) < 1e-3
        s.append(sxy)
    assert abs(s[0] - s[1]) < 1e-3


def _loop(B, s0, s2, n=60):
    return [(s0 + i / n) * B[0] + s2 * B[1] for i in range(n)]


def test_berry_phase_closure_does_not_depend_on_the_solver_phase():
    """With closure=G, 0.14.0 solved for the states at k0 + G again
    instead of building them from those at k0, so the phase of the
    solver's eigenvector entered the result.  Haldane model (m_ab =
    0.2), loop along b1 at s2 = 0: in the atomic frame the loop started
    at 0 and at b1/2 gave -1.211349 and +1.930244 (pi apart); in the
    Loewdin frame closure=b1 gave +3.088952 where the same loop without
    closure gives -0.052641.  Now every start point gives the same
    phase, equal to an independent construction from periodic-gauge
    eigenvectors."""
    h = haldane(t1=-1.0, t2=0.1, m_ab=0.2)
    B = h.reciprocal_vectors
    tau = np.repeat(h.positions, h.norb, axis=0)

    def by_hand(loop, G):
        # atomic-gauge links built from periodic-gauge eigenvectors:
        # c_p(k)^dag diag(exp(-i dk . tau)) c_p(k'), closing with c_p(k0)
        from hamop.berry import _bloch_periodic
        cs = [np.linalg.eigh(_bloch_periodic(h, k)[0])[1][:, :1]
              for k in loop]
        ks = list(loop) + [loop[0] + G]
        cs.append(cs[0])
        p = 1.0 + 0.0j
        for i in range(len(loop)):
            d = np.exp(-1j * (tau @ (ks[i + 1] - ks[i])))
            p *= np.linalg.det(cs[i].conj().T @ (d[:, None] * cs[i + 1]))
        return np.angle(p)

    for s2 in (0.0, 0.13):
        ref = by_hand(_loop(B, 0.0, s2), B[0])
        free = berry_phase(h, _loop(B, 0.0, s2))
        for s0 in (0.0, 0.21, 0.5):
            loop = _loop(B, s0, s2)
            za = berry_phase(h, loop, frame="atomic", closure=B[0])
            zl = berry_phase(h, loop, frame="lowdin", closure=B[0])
            assert abs(np.angle(np.exp(1j * (za - ref)))) < 1e-10
            assert abs(np.angle(np.exp(1j * (zl - free)))) < 1e-10


def test_closure_must_be_a_reciprocal_lattice_vector():
    h = haldane()
    B = h.reciprocal_vectors
    with pytest.raises(ValueError, match="reciprocal lattice vector"):
        berry_phase(h, _loop(B, 0.0, 0.1), frame="atomic",
                    closure=0.5 * B[0])
    # an integer combination is accepted
    berry_phase(h, _loop(B, 0.0, 0.1), frame="atomic", closure=B[0] - B[1])


def test_berry_phase_sign_convention_spin_half():
    """Documented in 0.15.0, not changed: berry_phase is the phase of
    the product of the link determinants, minus Berry's gamma.  For the
    lower state of H = -n.sigma with n turning once counterclockwise
    about z at polar angle theta, Berry's gamma is -pi (1 - cos theta)
    (half the solid angle), so berry_phase gives +pi (1 - cos theta).
    The uniform loop of N points has the exact discrete value
    N arg(cos^2(theta/2) + sin^2(theta/2) exp(2 pi i / N))."""
    for theta in (0.4, 0.7, 1.3):
        m = TightBindingModel(positions=[[0.0]], norb=2, cell=[[1.0]])
        m.add_hop(0, 0, (0,), [[-np.cos(theta), 0.0], [0.0, np.cos(theta)]])
        # n_x sigma_x + n_y sigma_y has (0, 1) element sin(theta) e^{-ik}
        m.add_hop(0, 0, (-1,), [[0.0, -np.sin(theta)], [0.0, 0.0]])
        N = 400
        g = berry_phase(m, [[2 * np.pi * i / N] for i in range(N)])
        c2, s2 = np.cos(theta / 2) ** 2, np.sin(theta / 2) ** 2
        exact = N * np.angle(c2 + s2 * np.exp(2j * np.pi / N))
        assert abs(g - exact) < 1e-10
        assert abs(g - np.pi * (1 - np.cos(theta))) < 1e-4
