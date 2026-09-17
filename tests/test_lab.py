"""Band-fitting anchors: the linear chain E(k) = e0 + 2t cos(ka) makes
the fit exactly linear, so the covariance has a CLOSED FORM
((X^T W X)^-1 with X = [1, 2 cos(ka)]) that the fit and the planner
must both hit; noiseless fits recover the truth; seeded Monte Carlo
matches the reported error bars; a same-cos(ka) design is exactly
rank-one and refused by planner and fit alike; and the greedy k-point
design reproduces its own rule and never loses to a random subset."""
import numpy as np
import pytest

from hamop import (band_information, bands, design_kpoints, fit_bands,
                   linear_chain)

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
