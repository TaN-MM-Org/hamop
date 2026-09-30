"""Regression tests for the silent failures fixed in 0.11.0.  Each fix
is pinned by a check that failed (or returned a wrong number without
an error) on 0.10.1; the 0.5 and 1.0 eV surface cases are sanity
checks that passed before as well."""
import warnings

import numpy as np
import pytest

from hamop import (TightBindingModel, bands, carrier_count,
                   band_edges, landauer_conductance,
                   chain_lead_blocks, dos, drude_weight, fermi_level,
                   linear_chain, sancho_rubio, sigma_optical, sigma_tensor,
                   transmission)


def _two_level_atom():
    """One site, levels 0 and 1 eV exactly (diagonal H), coupled by an
    intra-atomic dipole so the 0 -> 1 line is bright."""
    m = TightBindingModel([[0.0]], norb=2, cell=None)
    m.add_hop(0, 0, (0,), [[0.0, 0.0], [0.0, 1.0]])
    X = np.zeros((2, 2, 1), dtype=complex)
    X[0, 1, 0] = X[1, 0, 0] = 0.7
    m.set_dipole(0, X)
    return m


def test_zero_temperature_with_a_level_at_mu_is_finite():
    """At T = 0 a level exactly at mu gave (e - mu)/(kB T) = 0/0 = NaN.
    It now has occupation 1/2 -- the value f(mu) = 1/2 that the Fermi
    function has at every T > 0 -- so the T = 0 result equals the
    T = 1 K result (the upper level's occupation there is 1e-26) and is
    exactly half the result with mu in the gap."""
    m = _two_level_atom()
    assert bands(m, [None])[0][0] == 0.0          # the level is at mu
    om = np.array([0.9, 1.0, 1.1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")            # no 0/0 warnings
        s0 = sigma_optical(m, om, mu=0.0, T=0.0, eta=0.05)
        s_gap = sigma_optical(m, om, mu=0.5, T=0.0, eta=0.05)
        t0 = sigma_tensor(m, om, mu=0.0, T=0.0, eta=0.05,
                          directions=(0, 0))
        n0 = carrier_count(m, mu=0.0, T=0.0)
    s1 = sigma_optical(m, om, mu=0.0, T=1.0, eta=0.05)
    assert np.all(np.isfinite(s0)) and np.all(np.isfinite(t0))
    assert np.abs(s0 - s1).max() < 1e-12 * s1.max()
    assert np.abs(s0 - 0.5 * s_gap).max() < 1e-12 * s_gap.max()
    assert abs(n0 - 1.0) < 1e-15                  # spin 2 x 1/2
    t1 = sigma_tensor(m, om, mu=0.0, T=1.0, eta=0.05, directions=(0, 0))
    assert np.abs(t0 - t1).max() < 1e-12 * np.abs(t1).max()


def test_k_weights_that_do_not_match_are_refused():
    """A weight list of the wrong length was silently truncated by zip;
    unnormalized or negative weights silently rescaled the result."""
    m = linear_chain()
    k, w = m.monkhorst_pack(10)
    E = np.array([0.0])
    ref = dos(m, E, mesh=10)
    assert np.abs(dos(m, E, kpts=k, weights=w) - ref).max() < 1e-14
    assert np.abs(dos(m, E, kpts=k) - ref).max() < 1e-14
    for bad, msg in [(w[:3], "weights for"), (2 * w, "sum to"),
                     (np.r_[-w[0], w[1:] + 2 * w[0] / 9], ">= 0")]:
        with pytest.raises(ValueError, match=msg):
            dos(m, E, kpts=k, weights=bad)
        with pytest.raises(ValueError, match=msg):
            sigma_optical(m, [1.0], 0.0, kpts=k, weights=bad)
        with pytest.raises(ValueError, match=msg):
            drude_weight(m, 0.0, kpts=k, weights=bad)
        with pytest.raises(ValueError, match=msg):
            carrier_count(m, 0.0, kpts=k, weights=bad)
        with pytest.raises(ValueError, match=msg):
            fermi_level(m, 0.5, kpts=k, weights=bad)
    # weights stored in single precision (sum 1 +- 1e-7) are accepted
    E2 = np.array([2.0])
    w32 = w.astype(np.float32)
    ref2 = dos(m, E2, kpts=k, weights=w)
    assert abs(dos(m, E2, kpts=k, weights=w32)[0] / ref2[0] - 1) < 1e-6
    w3 = np.full(3, 1 / 3, dtype=np.float32)          # sums to 1 + 3e-8
    dos(m, E2, kpts=k[:3], weights=w3)
    # band_edges does not use weights, so it does not check them
    assert band_edges(m, 0.0, kpts=k, weights=w[:3]) == \
        band_edges(m, 0.0, kpts=k)


def test_sancho_rubio_refuses_to_return_an_unconverged_result():
    """At eta = 0 inside the band the decimation never converges; 0.10.1
    returned a real surface Green function there (0.92 instead of
    (E - i sqrt(4 - E^2))/2 at E = 0.5) and T = 0 inside the band.
    Outside the band eta = 0 converges and matches the real closed form
    (E - sign(E) sqrt(E^2 - 4))/2; inside, any eta > 0 works."""
    H00, H01 = chain_lead_blocks(t=-1.0)
    with pytest.raises(RuntimeError, match="eta > 0"):
        sancho_rubio(0.5, H00, H01, eta=0.0)
    with pytest.raises(RuntimeError, match="eta > 0"):
        transmission([0.5], [H00], [], H00, H01, eta=0.0)
    for E in (2.5, -3.0):
        g = sancho_rubio(E, H00, H01, eta=0.0)[0, 0]
        assert abs(g - (E - np.sign(E) * np.sqrt(E * E - 4)) / 2) < 1e-12
    g = sancho_rubio(0.5, H00, H01, eta=1e-12)[0, 0]
    assert abs(g - (0.5 - 1j * np.sqrt(4 - 0.25)) / 2) < 1e-10


def _chain_surface(z, t=-1.0, s=0.0):
    """Closed-form surface Green function of the chain lead at complex
    z: on-site z, coupling tau = z s - t both ways, g = 1/(z - tau^2 g);
    of the two roots the retarded one is the decaying one, |tau g| < 1."""
    tau = z * s - t
    r = np.sqrt(z * z - 4 * tau * tau + 0j)
    roots = [(z + r) / (2 * tau * tau), (z - r) / (2 * tau * tau)]
    return min(roots, key=lambda g: abs(tau * g))


@pytest.mark.parametrize("E", [0.0, 8.9e-16, 1e-10, 0.5, 1.0])
def test_surface_green_function_at_a_lead_resonance_with_small_eta(E):
    """When E equals an eigenvalue of the lead layer (E = 0 for the
    chain), the decimation divides by eta in its first steps; with
    eta = 1e-8 it reached a wrong fixed point (-6.7e7 i at E = 0, -0.40 i
    at E = 8.9e-16, instead of -i) and passed its own convergence test.
    The Dyson check now sends such cases to the mode-matching route.
    Anchors: the closed form at the same complex z, for the orthogonal
    chain, the nonorthogonal chain (s = 0.2), and a two-site principal
    layer (singular inter-layer coupling), all to 1e-8 -- below the
    shift of order eta that the broadening itself causes (the old
    errors were 0.6 and 6.7e7)."""
    eta = 1e-8
    z = E + 1j * eta
    H00, H01 = chain_lead_blocks(t=-1.0)
    g = sancho_rubio(E, H00, H01, eta=eta)[0, 0]
    assert abs(g - _chain_surface(z)) < 1e-8
    S00, S01 = np.eye(1), np.array([[0.2]])
    g = sancho_rubio(E, H00, H01, S00, S01, eta=eta)[0, 0]
    assert abs(g - _chain_surface(z, s=0.2)) < 1e-8
    H00b, H01b = chain_lead_blocks(t=-1.0, per_layer=2)
    g = sancho_rubio(E, H00b, H01b, eta=eta)[0, 0]    # surface site
    assert abs(g - _chain_surface(z)) < 1e-8


def test_transmission_and_conductance_through_the_band_centre():
    """The same failure reached transmission (T(0) = 1.4e-15 and
    T(8.9e-16) = 0.9745 for the impurity chain, exact 0.862) and the
    conductance on any grid containing E = 0 (1.5574 instead of
    1.7241 at 30 K).  Now T(E) matches the closed form to 1e-7 at and
    around the band centre (the eta = 1e-8 broadening itself shifts it
    by 3e-8), and the conductance on np.linspace(-1, 1, 2001) equals
    the one from the closed-form T(E) to 1e-7."""
    eps = 0.8
    H00, H01 = chain_lead_blocks(t=-1.0)
    args = ([H00, H00 + eps, H00], [H01, H01], H00, H01)
    E0 = np.array([0.0, 8.9e-16, 1e-10])
    exact0 = (4 - E0 ** 2) / ((4 - E0 ** 2) + eps ** 2)
    assert np.abs(transmission(E0, *args, eta=1e-8) - exact0).max() < 1e-7
    E = np.linspace(-1.0, 1.0, 2001)
    Tr = transmission(E, *args, eta=1e-8)
    exact = (4 - E ** 2) / ((4 - E ** 2) + eps ** 2)
    assert np.abs(Tr - exact).max() < 1e-7
    G = landauer_conductance(E, Tr, 0.0, 30.0)
    G_exact = landauer_conductance(E, exact, 0.0, 30.0)
    assert abs(G - G_exact) < 1e-7
    assert abs(G_exact - 2 * 4 / (4 + eps ** 2)) < 1e-4
