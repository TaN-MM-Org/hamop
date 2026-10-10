"""Band-fitting anchors: the linear chain E(k) = e0 + 2t cos(ka) makes
the fit exactly linear, so the covariance has a CLOSED FORM
((X^T W X)^-1 with X = [1, 2 cos(ka)]) that the fit and the planner
must both hit; noiseless fits recover the truth; seeded Monte Carlo
matches the reported error bars; a same-cos(ka) design is exactly
rank-one and refused by planner and fit alike; and the greedy k-point
design reproduces its own rule and never loses to a random subset."""
import numpy as np
import pytest

from math import factorial

from hamop import (band_information, bands, design_kpoints, fit_bands,
                   linear_chain, propagate_uncertainty)

E0, T = 0.2, -1.0


def _build(theta):
    t, e0 = theta
    return linear_chain(t=t, e0=e0)


K = np.linspace(0.1, 3.0, 12)[:, None]


def _truth(k):
    return E0 + 2.0 * T * np.cos(k[:, 0])


def test_forward_model_matches_closed_form():
    """bands() on the chain is the textbook dispersion, exactly."""
    e = bands(_build([T, E0]), K)[:, 0]
    assert np.allclose(e, _truth(K), rtol=0, atol=1e-12)


def test_noiseless_fit_and_closed_form_covariance():
    """Linear-in-parameters model: the reported covariance must equal
    sigma^2 (X^T X)^-1 with the exact design matrix X = [2 cos(ka), 1]
    (parameter order (t, e0))."""
    y = _truth(K)
    sigma = 0.03
    fit = fit_bands(_build, [-1.3, 0.5], K, y, sigmas=sigma)
    assert abs(fit.theta[0] - T) < 1e-7
    assert abs(fit.theta[1] - E0) < 1e-7
    assert fit.chi2 < 1e-10
    X = np.column_stack([2.0 * np.cos(K[:, 0]), np.ones(K.shape[0])])
    want = sigma ** 2 * np.linalg.inv(X.T @ X)
    assert np.allclose(fit.cov, want, rtol=1e-6, atol=1e-15)
    info = band_information(_build, [T, E0], K, sigmas=sigma)
    assert info["identifiable"]
    assert np.allclose(info["sigma"], np.sqrt(np.diag(want)),
                       rtol=1e-6)


def test_error_bars_agree_with_monte_carlo():
    y = _truth(K)
    sigma = 0.05
    rng = np.random.default_rng(13)
    draws = []
    reported = None
    for _ in range(300):
        fit = fit_bands(_build, [-1.2, 0.4], K,
                        y + sigma * rng.standard_normal(y.size),
                        sigmas=sigma)
        draws.append(fit.theta)
        reported = fit.sigma
    emp = np.std(np.array(draws), axis=0, ddof=1)
    assert np.allclose(emp, reported, rtol=0.15)


def test_same_cos_design_exactly_rank_one():
    """k and 2 pi - k share cos(ka): the design matrix column
    [2 cos(ka)] is constant up to the intercept... with only such
    pairs at ONE cosine value, [1, 2 cos] are collinear -- exact rank
    one, reported by the planner and refused by the fit."""
    q = 1.1
    kk = np.array([[q], [2.0 * np.pi - q], [q], [2.0 * np.pi - q]])
    info = band_information(_build, [T, E0], kk, sigmas=0.02)
    assert not info["identifiable"]
    fi = info["fisher"]
    d = np.sqrt(np.diag(fi))
    sv = np.linalg.svd(fi / np.outer(d, d), compute_uv=False)
    assert sv[-1] < 1e-10 * sv[0]
    y = _truth(kk)
    with pytest.raises(ValueError, match="cannot tell"):
        fit_bands(_build, [-1.2, 0.4], kk, y, sigmas=0.02)
    with pytest.raises(ValueError, match="identifiable"):
        design_kpoints(_build, [T, E0], kk, 2, sigmas=0.02)


def test_design_greedy_invariant_and_quality():
    out = design_kpoints(_build, [T, E0], K, 4, sigmas=0.02)
    idx = out["indices"]
    assert len(idx) == 4 and len(set(idx)) == 4
    assert out["sigma"] is not None

    def logdet_of(subset):
        info = band_information(_build, [T, E0], K[list(subset)],
                                sigmas=0.02)
        s, d = np.linalg.slogdet(info["fisher"])
        return d if s > 0 else -np.inf

    best = logdet_of(idx)
    rng = np.random.default_rng(7)
    for _ in range(30):
        assert best >= logdet_of(rng.choice(K.shape[0], 4,
                                            replace=False)) - 1e-9


def test_input_refusals():
    y = _truth(K)
    with pytest.raises(ValueError, match="callable"):
        fit_bands(None, [1.0], K, y)
    with pytest.raises(ValueError, match="positive"):
        fit_bands(_build, [-1.0, 0.0], K, y, sigmas=-1.0)
    with pytest.raises(ValueError, match="determine"):
        fit_bands(_build, [-1.0, 0.0], K[:1], y[:1], sigmas=0.1)
    with pytest.raises(ValueError, match="band index"):
        fit_bands(_build, [-1.0, 0.0], K, y, band_index=5, sigmas=0.1)
    with pytest.raises(ValueError, match="n_pick"):
        design_kpoints(_build, [T, E0], K, 1, sigmas=0.1)


# ---------------- propagate_uncertainty and p_value (new in 0.12.0) ----

def _chi2_sf_even(x, dof):
    """Closed-form chi-squared survival function for an even number of
    degrees of freedom: exp(-x/2) sum_{i < dof/2} (x/2)^i / i!."""
    return np.exp(-x / 2.0) * sum((x / 2.0) ** i / factorial(i)
                                  for i in range(dof // 2))


def _band_energy(theta, kx=0.7):
    return bands(_build(theta), [[kx]])[0, 0]


def test_linear_propagation_matches_the_closed_form():
    """E(k) = e0 + 2 t cos(k) is linear in (t, e0), so its variance is
    exactly g^T C g with g = (2 cos k, 1)."""
    y = _truth(K) + 0.03 * np.random.default_rng(3).standard_normal(12)
    fit = fit_bands(_build, [-1.3, 0.5], K, y, sigmas=0.03)
    for kx in (0.7, 2.2, 4.0):
        out = propagate_uncertainty(fit, lambda th: _band_energy(th, kx))
        g = np.array([2.0 * np.cos(kx), 1.0])
        want = np.sqrt(g @ fit.cov @ g)
        assert abs(out["sigma"] - want) < 1e-6 * want
        assert np.allclose(out["jacobian"][0], g, rtol=1e-6, atol=1e-9)
        assert abs(out["value"] - (fit.theta[1] + 2.0 * fit.theta[0]
                                   * np.cos(kx))) < 1e-12
    # vector quantity: the band width W = E(pi) - E(0) = -4 t and the
    # band center E(pi/2) = e0, with their closed-form covariance
    out = propagate_uncertainty(
        fit, lambda th: bands(_build(th), [[np.pi], [0.0], [np.pi / 2]])
        [:, 0] @ np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]))
    G = np.array([[-4.0, 0.0], [0.0, 1.0]])
    assert np.allclose(out["cov"], G @ fit.cov @ G.T, rtol=1e-6,
                       atol=1e-14)


def test_propagated_error_bar_matches_simulated_experiments():
    """200 seeded simulated experiments: the scatter of the fitted band
    width 4|t| matches the propagated error bar, and the p-values of
    these correct-model fits are uniform (about 10 % below 0.1)."""
    y = _truth(K)
    sigma = 0.05
    rng = np.random.default_rng(29)
    widths, pvals, reported = [], [], []
    for _ in range(200):
        fit = fit_bands(_build, [-1.2, 0.4], K,
                        y + sigma * rng.standard_normal(y.size),
                        sigmas=sigma)
        out = propagate_uncertainty(fit, lambda th: 4.0 * abs(th[0]))
        widths.append(out["value"])
        reported.append(out["sigma"])
        pvals.append(fit.p_value)
    emp = np.std(widths, ddof=1)
    assert abs(emp - np.mean(reported)) < 0.15 * emp
    assert abs(np.mean(widths) - 4.0) < 3.0 * emp / np.sqrt(200)
    frac = np.mean(np.array(pvals) < 0.1)
    assert 0.04 < frac < 0.16


def test_p_value_matches_the_closed_form_survival_function():
    y = _truth(K) + 0.03 * np.random.default_rng(13).standard_normal(12)
    fit = fit_bands(_build, [-1.3, 0.5], K, y, sigmas=0.03)
    assert fit.chi2_dof == 10
    assert abs(fit.p_value - _chi2_sf_even(fit.chi2, 10)) < 1e-12
    k4 = K[:4]
    fit4 = fit_bands(_build, [-1.3, 0.5], k4, _truth(k4) + 0.01,
                     sigmas=0.02)
    assert fit4.chi2_dof == 2
    assert abs(fit4.p_value - np.exp(-fit4.chi2 / 2.0)) < 1e-12
    # no measurement errors: no absolute chi-squared, no p-value
    assert fit_bands(_build, [-1.3, 0.5], K, y).p_value is None


def test_monte_carlo_agrees_with_linear_and_handles_a_nonlinear_case():
    """For a linear quantity the two methods agree; for q = (e0 - e0_fit)^2
    (zero slope at the best fit) the linear method is refused and the
    Monte Carlo spread matches the chi-squared closed form: mean
    sigma_e0^2 and standard deviation sqrt(2) sigma_e0^2."""
    fit = fit_bands(_build, [-1.3, 0.5], K, _truth(K), sigmas=0.03)
    lin = propagate_uncertainty(fit, _band_energy)
    mc = propagate_uncertainty(fit, _band_energy, method="monte_carlo",
                               n_samples=4000, seed=5)
    assert abs(mc["sigma"] - lin["sigma"]) < 0.05 * lin["sigma"]
    assert abs(mc["value"] - lin["value"]) < 1e-15
    lo, hi = mc["interval"]
    assert lo < mc["mean"] < hi

    e0_fit = fit.theta[1]

    def q(th):
        return (th[1] - e0_fit) ** 2

    with pytest.raises(ValueError, match="not linear enough"):
        propagate_uncertainty(fit, q)
    s2 = fit.cov[1, 1]
    mc = propagate_uncertainty(fit, q, method="monte_carlo",
                               n_samples=4000, seed=5)
    assert abs(mc["mean"] - s2) < 0.06 * s2
    assert abs(mc["sigma"] - np.sqrt(2.0) * s2) < 0.1 * s2


def test_propagation_refusals():
    fit = fit_bands(_build, [-1.3, 0.5], K, _truth(K), sigmas=0.03)
    with pytest.raises(ValueError, match="BandFit"):
        propagate_uncertainty({"theta": [1.0]}, _band_energy)
    with pytest.raises(ValueError, match="callable"):
        propagate_uncertainty(fit, 3.0)
    with pytest.raises(ValueError, match="method"):
        propagate_uncertainty(fit, _band_energy, method="bootstrap")
    with pytest.raises(ValueError, match="n_samples"):
        propagate_uncertainty(fit, _band_energy, method="monte_carlo",
                              n_samples=10)
    with pytest.raises(ValueError, match="non-finite"):
        propagate_uncertainty(fit, lambda th: np.nan)
