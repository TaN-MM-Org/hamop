"""A 2D sheet in an optical measurement (new in 0.12.0).

The sheet formulas are checked against an independent calculation: the
textbook thin-film (Airy) transmission and reflection of a film of
thickness d and permittivity 1 + i s / (k0 d), which carries the same
sheet conductance, in the limit k0 d -> 0.  The fine-structure constant
is checked against SciPy's CODATA table, free-standing graphene against
the absorption of about pi alpha = 2.3 % reported by Nair et al.,
Science 320, 1308 (2008), and the inverse and its error bar against the
Airy film and a seeded Monte Carlo."""
import numpy as np
import pytest
import scipy.constants

from hamop import (graphene, sheet_conductivity_from_transmittance,
                   sheet_optics, sigma_optical)
from hamop.model import ALPHA

PA = np.pi * ALPHA


def _airy_film(sigma, n_in, n_out, k0d=1e-8):
    """T, R of a film between two half-spaces at normal incidence
    (characteristic Fabry-Perot sum, exp(-i omega t) convention), with
    the film's permittivity chosen so that its sheet conductance equals
    sigma (units e^2/(4 hbar)): eps = 1 + i pi alpha sigma / (k0 d)."""
    s = PA * sigma
    n_f = np.sqrt(1.0 + 1j * s / k0d)
    beta = k0d * n_f
    r12 = (n_in - n_f) / (n_in + n_f)
    r23 = (n_f - n_out) / (n_f + n_out)
    t12 = 2.0 * n_in / (n_in + n_f)
    t23 = 2.0 * n_f / (n_f + n_out)
    ph = np.exp(2j * beta)
    r = (r12 + r23 * ph) / (1.0 + r12 * r23 * ph)
    t = t12 * t23 * np.exp(1j * beta) / (1.0 + r12 * r23 * ph)
    return (n_out / n_in) * abs(t) ** 2, abs(r) ** 2


def test_fine_structure_constant_matches_scipy():
    assert abs(ALPHA / scipy.constants.fine_structure - 1.0) < 1e-8


@pytest.mark.parametrize("sigma, n_in, n_out", [
    (1.0, 1.0, 1.0), (0.3 + 0.8j, 1.0, 1.5), (2.5 - 0.4j, 1.33, 1.0),
    (40.0 + 10.0j, 1.0, 3.4), (0.0, 1.0, 1.5)])
def test_sheet_formulas_match_the_zero_thickness_film(sigma, n_in, n_out):
    out = sheet_optics(sigma, n_in, n_out)
    T, R = _airy_film(sigma, n_in, n_out)
    assert abs(out["T"] - T) < 1e-6 * T
    assert abs(out["R"] - R) < 1e-6 * max(R, 1e-3)
    assert abs(out["A"] - (1.0 - T - R)) < 1e-6
    assert abs(out["T"] + out["R"] + out["A"] - 1.0) < 1e-14
    T0, _ = _airy_film(0.0, n_in, n_out)          # the bare interface
    assert abs(out["T_rel"] - T / T0) < 1e-6


def test_free_standing_graphene_absorbs_pi_alpha():
    """On its plateau (sigma = 1) a free-standing sheet transmits
    (1 + pi alpha / 2)^-2; to first order it absorbs pi alpha = 2.3 %
    (Nair et al. 2008).  The plateau value from the graphene model
    itself gives the same within the model's 5 % tolerance."""
    out = sheet_optics(1.0)
    assert abs(out["T"] - (1.0 + PA / 2.0) ** -2) < 1e-15
    assert round(100.0 * (1.0 - out["T"]), 1) == 2.3
    assert abs((1.0 - out["T"]) - PA) < 0.02 * PA
    s = sigma_optical(graphene(), [1.0, 1.3], mu=0.0, mesh=120,
                      eta=0.12, T=10.0)
    T_model = sheet_optics(s)["T"]
    assert np.all(np.abs((1.0 - T_model) / PA - 1.0) < 0.06)


def test_inverse_recovers_the_film_conductivity():
    for sigma, n_out in [(1.0, 1.0), (0.4 + 0.3j, 1.5), (7.0 - 2.0j, 3.4)]:
        T, _ = _airy_film(sigma, 1.0, n_out)
        T0, _ = _airy_film(0.0, 1.0, n_out)
        got = sheet_conductivity_from_transmittance(
            T / T0, 1.0, n_out, sigma_imag=np.imag(sigma))
        assert abs(got["sigma"] - np.real(sigma)) < 1e-5 * max(
            1.0, abs(sigma))
        assert got["sigma_err"] is None


def test_inverse_error_bar_matches_monte_carlo():
    sigma, n_out, err = 0.8, 1.5, 2e-3
    T_rel = sheet_optics(sigma, 1.0, n_out)["T_rel"]
    got = sheet_conductivity_from_transmittance(T_rel, 1.0, n_out,
                                                T_rel_err=err)
    rng = np.random.default_rng(11)
    draws = sheet_conductivity_from_transmittance(
        T_rel + err * rng.standard_normal(20000), 1.0, n_out,
        T_rel_err=err)["sigma"]
    assert abs(np.std(draws, ddof=1) / got["sigma_err"] - 1.0) < 0.03
    assert abs(got["sigma"] - sigma) < 1e-12
    # arrays in, arrays out
    arr = sheet_conductivity_from_transmittance(
        np.array([T_rel, T_rel]), 1.0, n_out, T_rel_err=err)
    assert arr["sigma"].shape == (2,) and arr["sigma_err"].shape == (2,)


def test_sheet_refusals():
    with pytest.raises(ValueError, match="gain"):
        sheet_optics(-0.1)
    with pytest.raises(ValueError, match="refractive index"):
        sheet_optics(1.0, n_out=1.5 + 0.1j)
    with pytest.raises(ValueError, match="refractive index"):
        sheet_optics(1.0, n_in=0.0)
    with pytest.raises(ValueError, match="finite"):
        sheet_optics(np.nan)
    with pytest.raises(ValueError, match="passive"):
        sheet_conductivity_from_transmittance(1.001)
    with pytest.raises(ValueError, match="> 0"):
        sheet_conductivity_from_transmittance(0.0)
    with pytest.raises(ValueError, match="3 T_rel_err"):
        sheet_conductivity_from_transmittance(1.01, T_rel_err=1e-3)
    # within 3 standard errors of the bound: noise, the estimate is kept
    got = sheet_conductivity_from_transmittance(1.001, T_rel_err=1e-3)
    assert got["sigma"] < 0.0 and got["sigma_err"] > 0.0
