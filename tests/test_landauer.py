"""Finite-temperature Landauer conductance, current and thermoelectric
coefficients (new in 0.11.0).  Every anchor is a closed form:
Sommerfeld integrals that are exact for constant or linear T(E), the
exact current through an energy-independent transmission, and the
single-impurity chain."""
import numpy as np
import pytest

from hamop import (chain_lead_blocks, landauer_conductance,
                   landauer_current, thermoelectric, transmission)
from hamop.model import KB, fermi_dirac

LORENZ_SOMMERFELD = np.pi ** 2 / 3.0 * KB ** 2      # (pi^2/3)(k_B/e)^2


def _grid(lo=-3.0, hi=3.1, step=5e-4):
    return np.arange(lo, hi, step)


def test_constant_transmission_gives_exact_sommerfeld_values():
    """T(E) = tau: G = spin tau, S = 0 and kappa / (G T) is exactly
    (pi^2/3)(k_B/e)^2, at every temperature."""
    E = _grid()
    tau = 0.7
    for T in (30.0, 300.0, 900.0):
        r = thermoelectric(E, np.full_like(E, tau), 0.05, T)
        assert abs(r["G"] - 2 * tau) < 1e-10
        assert abs(r["S"]) < 1e-12
        assert abs(r["lorenz"] / LORENZ_SOMMERFELD - 1.0) < 1e-9
        assert abs(r["kappa"] - r["lorenz"] * r["G"] * T) < 1e-15
    assert abs(LORENZ_SOMMERFELD - 2.443e-8) < 1e-11
    g1 = landauer_conductance(E, np.full_like(E, tau), 0.05, 300.0,
                              spin=1)
    assert abs(g1 - tau) < 1e-10


def test_linear_transmission_gives_the_exact_mott_seebeck():
    """T(E) = a + b E: L0 = a + b mu, L1 = b (pi^2/3) kT^2 and
    L2 = (a + b mu)(pi^2/3) kT^2 exactly, so S is the Mott form
    -(pi^2/3) k_B^2 T b / (a + b mu) and the Lorenz ratio has a closed
    form as well."""
    E = _grid()
    a, b, mu, T = 0.4, 0.3, 0.05, 300.0
    kT = KB * T
    r = thermoelectric(E, a + b * E, mu, T)
    a_mu = a + b * mu
    s_mott = -(np.pi ** 2 / 3) * KB ** 2 * T * b / a_mu
    assert r["S"] < 0          # T(E) rising with E: electron-like, S < 0
    assert abs(r["S"] / s_mott - 1.0) < 1e-9
    assert abs(r["G"] - 2 * a_mu) < 1e-10
    c = np.pi ** 2 / 3 * kT ** 2
    lorenz = (a_mu * c - (b * c) ** 2 / a_mu) / (a_mu * T ** 2)
    assert abs(r["lorenz"] / lorenz - 1.0) < 1e-9
    # a falling T(E) reverses the sign (hole-like)
    assert thermoelectric(E, a - b * E, mu, T)["S"] > 0


def test_current_through_a_constant_transmission_is_exact():
    """integral [f_L - f_R] dE = mu_L - mu_R at any temperature, so
    I = spin tau V; the current is odd in the bias."""
    E = _grid()
    tau = 0.35
    for T in (4.0, 300.0):
        E_T = E if T > 100 else np.arange(-0.2, 0.3, 1e-4)
        I = landauer_current(E_T, np.full_like(E_T, tau), 0.1, -0.05, T)
        assert abs(I - 2 * tau * 0.15) < 1e-10
        Ir = landauer_current(E_T, np.full_like(E_T, tau), -0.05, 0.1, T)
        assert abs(I + Ir) < 1e-14


def test_single_impurity_chain_low_temperature_and_small_bias():
    """Conductance of the impurity chain from `transmission`: at 30 K
    it equals spin * T(mu) of the closed form within the Sommerfeld
    correction (kT)^2 T''/6 (below 1e-5 here), and the current at a
    tiny bias V divided by V equals G (linear response) to 1e-6."""
    t, eps, mu, T = -1.0, 0.8, 0.3, 30.0
    kT = KB * T
    H00, H01 = chain_lead_blocks(t=t)
    E = np.arange(mu - 32 * kT, mu + 32 * kT, 0.5 * kT)
    Tr = transmission(E, [H00, H00 + eps, H00], [H01, H01], H00, H01,
                      eta=1e-9)
    exact = (4 - E ** 2) / ((4 - E ** 2) + eps ** 2)
    assert np.abs(Tr - exact).max() < 1e-6
    G = landauer_conductance(E, Tr, mu, T)
    T_mu = (4 - mu ** 2) / ((4 - mu ** 2) + eps ** 2)
    assert abs(G - 2 * T_mu) < 1e-5
    V = 1e-4
    I = landauer_current(E, Tr, mu + V / 2, mu - V / 2, T)
    assert abs(I / V - G) < 1e-6 * G


def test_band_edge_step_matches_the_fermi_function_closed_form():
    """A clean chain transmits exactly 1 on (-2|t|, 2|t|): with mu near
    the upper edge the conductance is spin [f(-2|t|) - f(2|t|)].  The
    step converges linearly in the grid spacing; on a 1e-5 eV grid the
    closed form is met to 1e-4."""
    mu, T = 1.95, 300.0
    E = np.arange(1.0, 3.0, 1e-5)
    step = (np.abs(E) < 2.0).astype(float)
    G = landauer_conductance(E, step, mu, T)
    exact = 2 * float(fermi_dirac(-2.0, mu, T) - fermi_dirac(2.0, mu, T))
    assert abs(G - exact) < 1e-4


def test_landauer_refusals():
    E = _grid()
    one = np.ones_like(E)
    with pytest.raises(ValueError, match="Fermi window"):
        landauer_conductance(E, one, 2.9, 300.0)      # window leaves grid
    with pytest.raises(ValueError, match="refine"):
        landauer_conductance(np.linspace(-1, 1, 30), np.ones(30), 0.0,
                             300.0)                  # step > kT
    with pytest.raises(ValueError, match="T must be > 0"):
        landauer_conductance(E, one, 0.0, 0.0)
    with pytest.raises(ValueError, match="increasing"):
        landauer_conductance(E[::-1], one, 0.0, 300.0)
    with pytest.raises(ValueError, match="transmission values"):
        landauer_conductance(E, one[:-1], 0.0, 300.0)
    with pytest.raises(ValueError, match="L0 = 0"):
        thermoelectric(E, np.zeros_like(E), 0.0, 300.0)
    with pytest.raises(ValueError, match="Fermi window"):
        landauer_current(E, one, 0.0, 2.9, 300.0)
