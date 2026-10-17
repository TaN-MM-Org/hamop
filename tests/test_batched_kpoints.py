"""The k-point loops became block operations in 0.13.0 (many k-points
assembled and diagonalized at once).  These tests keep the 0.12.1
one-k-point-at-a-time code as the reference, verbatim in substance, and
require the new routines to agree with it to 1e-12 relative (the
difference is the order of floating-point sums, about 1e-15), for
orthogonal, nonorthogonal, complex (Haldane), dipole-carrying, finite
and overlap-dropping (canonically orthogonalized) models."""
import numpy as np
import pytest

from hamop import (TightBindingModel, bands, carrier_count, dos,
                   drude_weight, fermi_level, gen_eigh, graphene, haldane,
                   linear_chain, sigma_optical, sigma_tensor, two_site,
                   with_spin)
from hamop.model import KB, fermi_dirac

TOL = 1e-12


# ---------------- the 0.12.1 reference loops ---------------------------

def _grid(model, mesh):
    if mesh is None:
        return [None], [1.0], 1.0
    k, w = model.monkhorst_pack(mesh)
    return k, w, model.cell_volume


def _dip(model, c, e, a):
    if not model.has_dipoles():
        return None
    return 1j * (e[:, None] - e[None, :]) * (
        c.conj().T @ model.dipole_matrix(a) @ c)


def ref_sigma_optical(model, omega, mu, mesh, T, eta, direction=0,
                      lineshape="gaussian", spin=2):
    kpts, weights, area = _grid(model, mesh)
    sig = np.zeros(len(omega))
    for k, w in zip(kpts, weights):
        H, S = model.bloch(k)
        dH, dS = model.bloch_derivative(k, direction)
        e, c = gen_eigh(H, S, eigvals_only=False)
        f = fermi_dirac(e, mu, T)
        M = c.conj().T @ dH @ c
        M = M - 0.5 * (e[:, None] + e[None, :]) * (c.conj().T @ dS @ c)
        Xt = _dip(model, c, e, direction)
        if Xt is not None:
            M = M + Xt
        dE = e[None, :] - e[:, None]
        df = f[:, None] - f[None, :]
        A2 = np.abs(M) ** 2
        mask = dE > 1e-3
        for iw, hw in enumerate(omega):
            if lineshape == "gaussian":
                g = np.exp(-0.5 * ((dE - hw) / eta) ** 2) \
                    / (eta * np.sqrt(2.0 * np.pi))
            else:
                g = (eta / np.pi) / ((dE - hw) ** 2 + eta ** 2)
            sig[iw] += w * (df * A2 * g / np.where(mask, dE, 1.0))[mask].sum()
    return sig * spin * 4.0 * np.pi / area


def ref_sigma_tensor(model, omega, mu, a, b, mesh, T, eta, spin=2):
    kpts, weights, area = _grid(model, mesh)
    sig = np.zeros(len(omega), dtype=complex)
    for k, w in zip(kpts, weights):
        H, S = model.bloch(k)
        e, c = gen_eigh(H, S, eigvals_only=False)
        Ms = []
        for d in (a, b):
            dH, dS = model.bloch_derivative(k, d)
            M = c.conj().T @ dH @ c \
                - 0.5 * (e[:, None] + e[None, :]) * (c.conj().T @ dS @ c)
            X = _dip(model, c, e, d)
            Ms.append(M if X is None else M + X)
        f = fermi_dirac(e, mu, T)
        dE = e[None, :] - e[:, None]
        df = f[:, None] - f[None, :]
        num = df * Ms[0] * Ms[1].T
        mask = np.abs(dE) > 1e-9
        for iw, hw in enumerate(omega):
            sig[iw] += w * (num[mask] / (dE * (dE - hw - 1j * eta))[mask]).sum()
    return sig * (-4j) * spin / area


def ref_eigs(model, kpts):
    return np.array([gen_eigh(*model.bloch(k)) for k in kpts])


def ref_dos(model, E, mesh, eta):
    kpts, weights, _ = _grid(model, mesh)
    rho = np.zeros_like(E)
    for k, w in zip(kpts, weights):
        for ei in gen_eigh(*model.bloch(k)):
            rho += w * np.exp(-0.5 * ((E - ei) / eta) ** 2) \
                / (eta * np.sqrt(2.0 * np.pi))
    return rho


def ref_carrier_count(model, mu, mesh, T, spin=2):
    kpts, weights, _ = _grid(model, mesh)
    return sum(w * spin * float(fermi_dirac(gen_eigh(*model.bloch(k)),
                                            mu, T).sum())
               for k, w in zip(kpts, weights))


def ref_drude(model, mu, mesh, T, direction=0, spin=2):
    kpts, weights, area = _grid(model, mesh)
    D = 0.0
    for k, w in zip(kpts, weights):
        H, S = model.bloch(k)
        dH, dS = model.bloch_derivative(k, direction)
        e, c = gen_eigh(H, S, eigvals_only=False)
        x = np.clip((e - mu) / (KB * T), -60.0, 60.0)
        mdfde = 1.0 / (4.0 * KB * T * np.cosh(0.5 * x) ** 2)
        v = np.real(np.einsum("in,ij,jn->n", c.conj(), dH, c)
                    - e * np.einsum("in,ij,jn->n", c.conj(), dS, c))
        D += w * float((v ** 2 * mdfde).sum())
    return D * spin * 4.0 * np.pi / area


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


# ---------------- models -----------------------------------------------

def _dipole_chain():
    """Two orbitals per site (s and p), a real dipole block, periodic."""
    m = TightBindingModel(positions=[[0.0]], norb=2, cell=[[2.0]])
    m.add_hop(0, 0, (0,), [[0.0, 0.0], [0.0, 1.5]])
    m.add_hop(0, 0, (1,), [[-0.7, 0.2], [-0.2, 0.4]])
    m.set_dipole(0, np.array([[0.0, 0.4], [0.4, 0.0]])[:, :, None])
    return m


def _overcomplete_chain():
    """Two orbitals on one site that overlap completely (S has a zero
    eigenvalue at every k), so canonical orthogonalization keeps one
    state per k-point: the per-k-point fallback path."""
    m = TightBindingModel(positions=[[0.0]], norb=2, cell=[[1.0]])
    m.add_hop(0, 0, (0,), [[0.0, 0.0], [0.0, 0.0]],
              [[1.0, 1.0], [1.0, 1.0]])
    m.add_hop(0, 0, (1,), [[-1.0, -1.0], [-1.0, -1.0]],
              [[0.1, 0.1], [0.1, 0.1]])
    return m


MODELS = {
    "graphene": (graphene(t=-2.7, a=2.46), 24, 0.4),
    "haldane": (haldane(t1=-1.0, t2=0.15, phi=0.7, m_ab=0.2), 18, 0.1),
    "overlap chain": (linear_chain(t=-1.0, e0=0.3, s=0.2), 101, 0.2),
    "spinful overlap chain": (with_spin(linear_chain(t=-1.0, s=0.15)),
                              60, 0.1),
    "dipole chain": (_dipole_chain(), 80, 0.5),
    "overcomplete chain": (_overcomplete_chain(), 50, 0.0),
}


@pytest.mark.parametrize("name", sorted(MODELS))
def test_batched_routines_equal_the_per_k_loops(name):
    m, mesh, mu = MODELS[name]
    omega = np.array([0.7, 1.9, 3.1])
    T = 150.0
    for ls in ("gaussian", "lorentzian"):
        new = sigma_optical(m, omega, mu, mesh=mesh, T=T, eta=0.1,
                            lineshape=ls)
        ref = ref_sigma_optical(m, omega, mu, mesh, T, 0.1, lineshape=ls)
        assert _rel(new, ref) < TOL
    dirs = [(0, 0)] if m.cell.shape[0] == 1 else [(0, 0), (0, 1), (1, 0)]
    scale = None
    for a, b in dirs:
        new = sigma_tensor(m, np.array([0.0, 1.1]), mu, directions=(a, b),
                           mesh=mesh, T=T, eta=0.05)
        ref = ref_sigma_tensor(m, np.array([0.0, 1.1]), mu, a, b, mesh, T,
                               0.05)
        # an off-diagonal component may vanish by symmetry (graphene):
        # compare on the scale of the diagonal one
        scale = np.abs(ref).max() if scale is None else scale
        assert np.abs(new - ref).max() <= TOL * scale
    E = np.linspace(-4.0, 4.0, 41)
    assert _rel(dos(m, E, mesh=mesh, eta=0.2), ref_dos(m, E, mesh, 0.2)) < TOL
    assert abs(carrier_count(m, mu, mesh=mesh, T=T)
               / ref_carrier_count(m, mu, mesh, T) - 1.0) < TOL
    kpts, _ = m.monkhorst_pack(mesh)
    assert _rel(bands(m, kpts), ref_eigs(m, kpts)) < TOL
    assert _rel(bands(m, list(kpts)), ref_eigs(m, kpts)) < TOL


@pytest.mark.parametrize("name", ["graphene", "overlap chain",
                                  "dipole chain", "haldane"])
def test_batched_drude_weight_equals_the_per_k_loop(name):
    """Meshes without band degeneracies, where the 0.12.1 diagonal sum
    is the right answer (24 and 18 avoid K for graphene and Haldane
    only when not multiples of 3, so 25 and 19 are used there)."""
    m, mesh, mu = MODELS[name]
    if m.cell.shape[0] == 2:
        mesh += 1
    for d in range(m.cell.shape[0]):
        new = drude_weight(m, mu, mesh=mesh, T=300.0, direction=d)
        ref = ref_drude(m, mu, mesh, 300.0, direction=d)
        assert abs(new / ref - 1.0) < TOL


def test_finite_models_and_fermi_level():
    m = two_site(t=-1.0, s=0.1)
    new = sigma_optical(m, [1.5, 2.2], 0.0, T=10.0, eta=0.05)
    ref = ref_sigma_optical(m, np.array([1.5, 2.2]), 0.0, None, 10.0, 0.05)
    assert _rel(new, ref) < TOL
    g = graphene(t=-2.7, a=2.46)
    kpts, w = g.monkhorst_pack(30)
    eig = ref_eigs(g, kpts)

    def count(mu):
        return float((fermi_dirac(eig, mu, 300.0) * w[:, None]).sum())
    mu = fermi_level(g, 1.1, mesh=30, T=300.0)
    assert abs(count(mu) - 1.1) < 1e-9


def _ref_kpm_moments(Hs, a, b, M, V0, norm):
    """The 0.12.1 Chebyshev recurrence of kpm_dos (orthogonal model)."""
    mu = np.zeros(M)
    v_prev = V0
    v_cur = (Hs @ V0 - b * V0) / a
    mu[0] = norm * float(np.real(np.sum(V0.conj() * v_prev)))
    mu[1] = norm * float(np.real(np.sum(V0.conj() * v_cur)))
    for m in range(2, M):
        v_next = 2.0 * (Hs @ v_cur - b * v_cur) / a - v_prev
        mu[m] = norm * float(np.real(np.sum(V0.conj() * v_next)))
        v_prev, v_cur = v_cur, v_next
    return mu


def test_kpm_dos_in_place_recurrence_equals_the_old_one():
    """kpm_dos now updates the Chebyshev vectors in place and takes the
    exact trace as a trace (2.5 times faster for a 1000-site chain).
    Rebuild the DOS from the 0.12.1 moments and compare: equal to 1e-12
    for the exact trace, bit for bit for the stochastic one."""
    from hamop import bloch_sparse, kpm_dos
    from hamop.sparse import _jackson
    n, M = 150, 80
    m = TightBindingModel(positions=[[float(i)] for i in range(n)], norb=1)
    rng = np.random.default_rng(4)
    for i in range(n):
        m.add_hop(i, i, (0,), [[0.3 * rng.standard_normal()]])
    for i in range(n - 1):
        m.add_hop(i, i + 1, (0,), [[-1.0]])
    E = np.linspace(-2.0, 2.0, 61)
    lo, hi = -3.0, 3.0
    a, b = 0.5 * (hi - lo) * 1.01, 0.5 * (hi + lo)
    Hs, _ = bloch_sparse(m, None)
    Hs = Hs.real.tocsr()

    def rebuild(mu):
        g = _jackson(M)
        th = np.arccos((E - b) / a)
        ser = g[0] * mu[0] + sum(2 * g[j] * mu[j] * np.cos(j * th)
                                 for j in range(1, M))
        return ser / (np.pi * np.sqrt(1 - ((E - b) / a) ** 2) * a)
    ref = rebuild(_ref_kpm_moments(Hs, a, b, M, np.eye(n), 1.0))
    new = kpm_dos(m, E, n_moments=M, bounds=(lo, hi))
    assert _rel(new, ref) < TOL
    V0 = np.random.default_rng(7).choice([-1.0, 1.0], size=(n, 5))
    ref = rebuild(_ref_kpm_moments(Hs, a, b, M, V0, 1.0 / 5))
    new = kpm_dos(m, E, n_moments=M, n_random=5, seed=7, bounds=(lo, hi))
    assert _rel(new, ref) < TOL
