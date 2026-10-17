"""Carrier density, chemical potential from a measured density, and the
quantum capacitance (new in 0.13.0), checked against the exact integral
of the chain's density of states, an independent closed-form graphene
dispersion on the same k-grid, and the Dirac-cone closed forms
n = E_F^2 / (pi (hbar v_F)^2) and C_Q = 2 e^2 |E_F| / (pi (hbar v_F)^2)
(graphene's quantum capacitance was measured by J. Xia, F. Chen, J. Li
and N. Tao, Nature Nanotechnology 4, 505 (2009))."""
import numpy as np
import pytest
from scipy.integrate import quad

from hamop import (carrier_density, chemical_potential_from_density,
                   graphene, linear_chain, two_site)
from hamop.model import E_CHARGE, KB

T = 300.0


def test_elementary_charge_is_the_exact_si_value():
    from scipy.constants import e
    assert E_CHARGE == 1.602176634e-19
    assert E_CHARGE == e


def _chain_exact_density(mu, t, a, T, n_neutral, spin=2):
    """Electrons per cm of the infinite chain E = 2t cos(theta):
    spin/(pi a) * integral_0^pi f(2t cos theta) dtheta, minus the
    neutral filling, by adaptive quadrature."""
    kT = KB * T

    def f(th):
        x = (2 * t * np.cos(th) - mu) / kT
        return 0.5 * (1.0 - np.tanh(0.5 * x))
    occ = quad(f, 0.0, np.pi, epsabs=1e-14, epsrel=1e-13, limit=400)[0]
    return (spin * occ / np.pi - n_neutral) / a * 1e8


@pytest.mark.parametrize("mu", [-1.7, -0.6, 0.0, 0.25, 1.3])
def test_chain_density_matches_the_exact_integral(mu):
    t, a = -1.0, 2.0
    m = linear_chain(t=t, a=a)
    n = carrier_density(m, mu, n_neutral=1.0, mesh=4000, T=T)
    exact = _chain_exact_density(mu, t, a, T, 1.0)
    assert abs(n - exact) <= 1e-9 * max(abs(exact), 1e6)
    if mu == 0.0:
        assert abs(n) < 1e-3          # half filling: neutral (per cm)


def test_chain_inverse_interval_and_dn_dmu():
    t, a = -1.0, 2.0
    m = linear_chain(t=t, a=a)
    mu0, mesh = 0.4, 4000
    n = carrier_density(m, mu0, 1.0, mesh=mesh, T=T)
    err = 1e-3 * abs(n)
    r = chemical_potential_from_density(m, n, 1.0, density_err=err,
                                        mesh=mesh, T=T)
    assert abs(r["mu"] - mu0) < 1e-9
    lo, hi = r["interval"]
    assert abs(carrier_density(m, lo, 1.0, mesh=mesh, T=T) - (n - err)) \
        < 1e-6 * err
    assert abs(carrier_density(m, hi, 1.0, mesh=mesh, T=T) - (n + err)) \
        < 1e-6 * err
    # dn/dmu against a central difference of carrier_density
    h = 1e-5
    fd = (carrier_density(m, mu0 + h, 1.0, mesh=mesh, T=T)
          - carrier_density(m, mu0 - h, 1.0, mesh=mesh, T=T)) / (2 * h)
    assert abs(r["dn_dmu"] / fd - 1.0) < 1e-6
    # at low T the chain's dn/dmu is spin / (pi a sqrt(4 t^2 - mu^2))
    assert abs(r["dn_dmu"] / (2 / (np.pi * a * np.sqrt(4 * t * t - mu0 ** 2))
                              * 1e8) - 1.0) < 1e-3
    # small error bar: the interval is the linear one
    assert abs(r["mu_err"] / (err / r["dn_dmu"]) - 1.0) < 1e-3
    assert r["unit"] == "cm^-1"
    assert abs(r["quantum_capacitance"] - E_CHARGE * r["dn_dmu"]) == 0.0


def _graphene_closed_form_density(mu, t, a, mesh, T):
    """Same Monkhorst-Pack grid, the textbook dispersion
    E = +/- |t| |1 + exp(i k.a1) + exp(i k.a2)| written out directly."""
    cell = a * np.array([[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
    s = np.arange(mesh) / mesh
    s1, s2 = np.meshgrid(s, s, indexing="ij")
    f = np.abs(1 + np.exp(2j * np.pi * s1) + np.exp(2j * np.pi * s2))
    E = np.stack([-abs(t) * f, abs(t) * f])
    occ = 0.5 * (1.0 - np.tanh(0.5 * (E - mu) / (KB * T)))
    return (2 * occ.mean() * 2 - 2.0) / abs(np.linalg.det(cell)) * 1e16


def test_graphene_density_equals_the_closed_form_dispersion_on_the_grid():
    t, a, mesh = -2.7, 2.46, 120
    g = graphene(t=t, a=a)
    for mu in (-0.5, 0.0, 0.3, 1.0):
        n = carrier_density(g, mu, 2.0, mesh=mesh, T=T)
        ref = _graphene_closed_form_density(mu, t, a, mesh, T)
        # 1e3 cm^-2 is 5e-14 electrons per cell: rounding of the sums
        assert abs(n - ref) <= 1e-10 * abs(ref) + 1e3
    # electron-hole symmetry of nearest-neighbor graphene
    assert abs(carrier_density(g, 0.4, 2.0, mesh=mesh, T=T)
               + carrier_density(g, -0.4, 2.0, mesh=mesh, T=T)) < 1e-10 * \
        abs(carrier_density(g, 0.4, 2.0, mesh=mesh, T=T))


def test_graphene_dirac_closed_forms_and_quantum_capacitance():
    """Near the Dirac point n = mu^2 / (pi (hbar v_F)^2) and
    C_Q = 2 e^2 mu / (pi (hbar v_F)^2) with hbar v_F = (sqrt 3 / 2)|t| a.
    The lattice bands curve away from the cone (trigonal warping), by
    0.1 % at 0.2 eV and 0.4 % at 0.4 eV in this model (mesh 300); 1 %
    is required.  The finite-temperature cone values are computed by
    quadrature, independently of the package."""
    t, a, mesh = -2.7, 2.46, 300
    g = graphene(t=t, a=a)
    hvf = np.sqrt(3.0) / 2.0 * abs(t) * a              # eV Angstrom
    dos = 2.0 / (np.pi * hvf ** 2)                     # per eV per A^2
    kT = KB * T
    for mu in (0.2, 0.4):
        def f(E):
            return 0.5 * (1.0 - np.tanh(0.5 * (E - mu) / kT))
        ne = quad(lambda E: dos * E * f(E), 0, mu + 60 * kT)[0]
        nh = quad(lambda E: dos * E * (1 - f(-E)), 0, 60 * kT)[0]
        n_cone = (ne - nh) * 1e16
        r = chemical_potential_from_density(g, n_cone, 2.0, mesh=mesh, T=T)
        assert abs(r["mu"] - mu) < 0.01 * mu
        n_lat = carrier_density(g, mu, 2.0, mesh=mesh, T=T)
        assert abs(n_lat / n_cone - 1.0) < 0.01
        # (at T = 0 the cone gives mu^2 / (pi (hbar v_F)^2); at 300 K the
        # thermal term pi^2 (kT)^2 / 3 adds 5.5 % at 0.2 eV, which the
        # quadrature above includes)
        r = chemical_potential_from_density(g, n_lat, 2.0, mesh=mesh, T=T)
        cq_xia = 2 * E_CHARGE ** 2 * mu / (np.pi * hvf ** 2) \
            / E_CHARGE * 1e16                          # F/cm^2
        assert abs(r["quantum_capacitance"] / cq_xia - 1.0) < 0.01
        assert r["unit"] == "cm^-2"
    # hole doping: negative density, mirrored chemical potential
    r = chemical_potential_from_density(g, -carrier_density(
        g, 0.3, 2.0, mesh=mesh, T=T), 2.0, mesh=mesh, T=T)
    assert abs(r["mu"] + 0.3) < 1e-8


def test_carrier_refusals():
    g = graphene()
    with pytest.raises(ValueError):          # finite model
        carrier_density(two_site(), 0.0, 1.0)
    with pytest.raises(ValueError):          # no grid
        carrier_density(g, 0.0, 2.0)
    with pytest.raises(ValueError):          # n_neutral beyond the bands
        carrier_density(g, 0.0, 5.0, mesh=10)
    with pytest.raises(ValueError):          # T = 0 in the inverse
        chemical_potential_from_density(g, 1e12, 2.0, mesh=30, T=0.0)
    with pytest.raises(ValueError):          # more than the bands hold
        chemical_potential_from_density(g, 1e17, 2.0, mesh=30)
    with pytest.raises(ValueError):          # negative error bar
        chemical_potential_from_density(g, 1e12, 2.0, density_err=-1.0,
                                        mesh=30)
    with pytest.raises(ValueError, match="too few"):   # coarse mesh, cold
        chemical_potential_from_density(g, 1e12, 2.0, mesh=30, T=10.0)
