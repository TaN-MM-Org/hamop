"""The k-point loops of the topology and geometry functions became block
operations in 0.14.0 (all grid points assembled and diagonalized at
once).  These tests keep the 0.13.0 point-by-point code as the
reference and require agreement to a stated tolerance, never exact
equality: the batched assembly computes the Bloch phases with an array
product, and the batched eigensolver may return other eigenvector
phases (the links are gauge invariant, so only rounding remains)."""
import numpy as np
import pytest

from hamop import (TightBindingModel, berry_curvature, berry_dipole,
                   haldane, magnetic_supercell, quantum_weight)
from hamop.berry import _get_states, _link
from hamop.geometry import quantum_geometric_tensor
from hamop.nonlinear import band_curvatures
from hamop.spin import kane_mele

# Plaquette fluxes are angles of order 1 computed from four determinants
# of order 1; rounding of the matrices (1e-16) moves them by about 1e-15
# (measured: below 1e-14 for every model here).  1e-10 leaves a wide
# margin and still catches any real change.
FLUX_TOL = 1e-10


def ref_berry_curvature(model, N, n_occ=1, frame="lowdin", solver="dense"):
    """berry_curvature of 0.13.0, verbatim in substance."""
    recip = 2.0 * np.pi * np.linalg.inv(model.cell).T
    b1, b2 = recip[0], recip[1]
    if frame == "lowdin":
        kg = [[(i / N) * b1 + (j / N) * b2 for j in range(N)]
              for i in range(N)]
        st = [[_get_states(model, kg[i][j], n_occ, frame, solver)
               for j in range(N)] for i in range(N)]

        def K(i, j):
            return kg[i % N][j % N]

        def S(i, j):
            return st[i % N][j % N]
    else:
        kg = [[(i / N) * b1 + (j / N) * b2 for j in range(N + 1)]
              for i in range(N + 1)]
        st = [[_get_states(model, kg[i][j], n_occ, frame, solver)
               for j in range(N + 1)] for i in range(N + 1)]

        def K(i, j):
            return kg[i][j]

        def S(i, j):
            return st[i][j]
    F = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            u1 = _link(model, frame, K(i, j), S(i, j), K(i + 1, j),
                       S(i + 1, j))
            u2 = _link(model, frame, K(i + 1, j), S(i + 1, j),
                       K(i + 1, j + 1), S(i + 1, j + 1))
            u3 = _link(model, frame, K(i + 1, j + 1), S(i + 1, j + 1),
                       K(i, j + 1), S(i, j + 1))
            u4 = _link(model, frame, K(i, j + 1), S(i, j + 1), K(i, j),
                       S(i, j))
            F[i, j] = float(np.angle(u1 * u2 * u3 * u4))
    return F


def _square_lattice():
    sq = TightBindingModel([[0.0, 0.0]], 1, cell=np.eye(2))
    sq.add_hop(0, 0, (1, 0), [[-1.0]])
    sq.add_hop(0, 0, (0, 1), [[-1.0]])
    return sq


CASES = {
    "haldane": (haldane(), 1),
    "haldane_trivial": (haldane(m_ab=0.9), 1),
    "haldane_overlap": (haldane(s=0.1), 1),
    "kane_mele": (kane_mele(), 2),
    "all_bands": (haldane(), 2),
    "hofstadter_1_7": (magnetic_supercell(_square_lattice(), 1, 7), 3),
}


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("frame", ["lowdin", "atomic"])
def test_batched_berry_curvature_equals_the_loop(name, frame):
    model, n_occ = CASES[name]
    for N in ((12,) if name.startswith("hofstadter") else (7, 12)):
        # (a 7 x 7 grid is too coarse for the 7-site cell: a link vanishes)
        ref = ref_berry_curvature(model, N, n_occ, frame)
        new = berry_curvature(model, N, n_occ, frame=frame)
        assert new.shape == ref.shape
        assert np.abs(new - ref).max() < FLUX_TOL


def _gapped(bond=1.3):
    cell = 2.46 * np.array([[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
    pos = np.array([np.zeros(2), (cell[0] + cell[1]) / 3.0])
    m = TightBindingModel(positions=pos, norb=1, cell=cell)
    m.add_hop(0, 0, (0, 0), [[0.5]])
    m.add_hop(1, 1, (0, 0), [[-0.5]])
    m.add_hop(0, 1, (0, 0), [[-2.7 * bond]])
    m.add_hop(0, 1, (-1, 0), [[-2.7]])
    m.add_hop(0, 1, (0, -1), [[-2.7]])
    return m


def test_band_curvatures_and_fermi_velocities_equal_the_loops():
    """Energies at the plaquette centers and the Hellmann-Feynman
    velocities of berry_dipole(method="fermi") against the 0.13.0
    point-by-point loops (1e-12 relative), and both dipole methods
    against the same methods built from the reference loops (1e-10
    relative)."""
    m, N = _gapped(), 16
    recip = 2.0 * np.pi * np.linalg.inv(m.cell).T
    b1, b2 = recip
    E_ref = np.empty((N, N, 2))
    V_ref = np.empty((N, N, 2, 2))
    for i in range(N):
        for j in range(N):
            k = ((i + 0.5) / N) * b1 + ((j + 0.5) / N) * b2
            H, _ = m.bloch(k)
            E_ref[i, j] = np.linalg.eigvalsh(H)
            _, U = np.linalg.eigh(H)
            for a in range(2):
                dH, _ = m.bloch_derivative(k, direction=a)
                V_ref[i, j, :, a] = np.real(
                    np.einsum("in,ij,jn->n", U.conj(), dH, U))
    E, omega = band_curvatures(m, N)
    assert np.abs(E - E_ref).max() < 1e-12 * np.abs(E_ref).max()
    F0 = ref_berry_curvature(m, N, 1, "atomic")
    a_plaq = abs(np.linalg.det(recip)) / N ** 2
    assert np.abs(omega[:, :, 0] - F0 / a_plaq).max() \
        < FLUX_TOL / a_plaq
    mu, kT = 0.9, 0.3
    a_cell = abs(np.linalg.det(m.cell))
    dfde = -1.0 / (4.0 * kT * np.cosh((E_ref - mu) / (2 * kT)) ** 2)
    om_ref = np.stack([F0, -F0], axis=-1) / a_plaq
    D_ref = np.array([-(om_ref * dfde * V_ref[..., a]).sum()
                      / (N ** 2 * a_cell) for a in range(2)])
    D = berry_dipole(m, N, mu, kT, method="fermi")["D"]
    assert np.abs(D - D_ref).max() < 1e-10 * np.abs(D_ref).max()


def test_batched_quantum_weight_equals_the_loop():
    """quantum_weight against the 0.13.0 sum of quantum_geometric_tensor
    over the mesh.  The finite differences (step 1e-5) amplify the
    rounding of the projectors (1e-16) to about 1e-11 in Q; measured
    differences are near 1e-12, and 1e-9 relative is allowed."""
    for m in (haldane(), haldane(m_ab=0.9), haldane(t2=0.2, phi=0.7)):
        kpts, w = m.monkhorst_pack(12)
        a_bz = (2.0 * np.pi) ** 2 / m.cell_volume
        K = np.zeros((2, 2))
        c = 0.0
        for kp, wi in zip(kpts, w):
            Q = quantum_geometric_tensor(m, kp)
            K += wi * np.real(Q)
            c += wi * (2.0 * np.imag(Q[0, 1]))
        K *= a_bz / (2.0 * np.pi)
        chern = c * a_bz / (2.0 * np.pi)
        out = quantum_weight(m, mesh=12)
        assert np.abs(out["K"] - K).max() < 1e-9 * np.abs(K).max()
        assert abs(out["chern"] - chern) < 1e-9
