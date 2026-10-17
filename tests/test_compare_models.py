"""compare_models (new in 0.13.0): the chi-squared (likelihood-ratio)
and F tests for nested band models.  For a single-band chain the
energies e0 + 2 t cos k + 2 t2 cos 2k are linear in the parameters, so
both statistics have closed forms (least-squares projections, computed
here with numpy.linalg.lstsq independently of the package) and exact
distributions, chi2(1) and F(1, n - 3), which seeded simulated
experiments check."""
import numpy as np
import pytest
from scipy.stats import chi2 as chi2_dist
from scipy.stats import f as f_dist

from hamop import TightBindingModel, compare_models, linear_chain

K = np.linspace(0.1, 3.0, 12)[:, None]
SIG = 0.03


def simple(th):                    # (t, e0)
    return linear_chain(t=th[0], e0=th[1])


def extended(th):                  # (t, e0, t2)
    m = TightBindingModel(positions=[[0.0]], norb=1, cell=[[1.0]])
    m.add_hop(0, 0, (0,), [[th[1]]])
    m.add_hop(0, 0, (1,), [[th[0]]])
    m.add_hop(0, 0, (2,), [[th[2]]])
    return m


def embed(th):
    return [th[0], th[1], 0.0]


def _data(seed, t2=0.0):
    rng = np.random.default_rng(seed)
    k = K[:, 0]
    return (0.2 - 2.0 * np.cos(k) + 2 * t2 * np.cos(2 * k)
            + SIG * rng.standard_normal(k.size))


def _rss(X, y):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(np.sum((y - X @ beta) ** 2))


def _closed_forms(y):
    k = K[:, 0]
    X1 = np.column_stack([2 * np.cos(k), np.ones_like(k)])
    X2 = np.column_stack([X1, 2 * np.cos(2 * k)])
    r1, r2 = _rss(X1, y), _rss(X2, y)
    return (r1 - r2) / SIG ** 2, (r1 - r2) / (r2 / (k.size - 3))


def test_statistics_equal_the_least_squares_closed_forms():
    y = _data(13)
    d_chi2, F = _closed_forms(y)
    r = compare_models(simple, [-1.3, 0.5], extended, None, K, y,
                       sigmas=SIG, embed=embed)
    assert r["test"] == "chi2" and r["dof"] == 1 and r["nested_checked"]
    assert abs(r["statistic"] / d_chi2 - 1.0) < 1e-6
    assert abs(r["p_value"] - chi2_dist.sf(d_chi2, 1)) < 1e-6
    r = compare_models(simple, [-1.3, 0.5], extended, [-1.0, 0.0, 0.05],
                       K, y)
    assert r["test"] == "F" and r["dof"] == (1, 9)
    assert not r["nested_checked"]
    assert abs(r["statistic"] / F - 1.0) < 1e-6
    assert abs(r["p_value"] - f_dist.sf(F, 1, 9)) < 1e-6


def test_null_distribution_and_power_in_simulated_experiments():
    """60 experiments from the simple model: the statistic follows
    chi2(1) (mean 1; p < 0.1 in about 10 % of cases), each one equal to
    its closed form.  With a real second-neighbor hopping of 0.1 eV the
    test rejects the simple model overwhelmingly."""
    stats, small = [], 0
    for seed in range(60):
        y = _data(1000 + seed)
        r = compare_models(simple, [-1.0, 0.2], extended, None, K, y,
                           sigmas=SIG, embed=embed)
        d_chi2, _ = _closed_forms(y)
        assert abs(r["statistic"] - d_chi2) < 1e-6 * max(d_chi2, 1.0)
        stats.append(r["statistic"])
        small += r["p_value"] < 0.1
    assert abs(np.mean(stats) - 1.0) < 0.45      # sd of the mean: 0.18
    assert 1 <= small <= 13                      # binomial(60, 0.1)
    r = compare_models(simple, [-1.0, 0.2], extended, None, K,
                       _data(5, t2=0.1), sigmas=SIG, embed=embed)
    assert r["p_value"] < 1e-10
    assert abs(r["extended"].theta[2] - 0.1) < 4 * r["extended"].sigma[2]


def test_compare_models_refusals():
    y = _data(3)
    with pytest.raises(ValueError, match="does not place"):
        compare_models(simple, [-1.0, 0.2], extended, None, K, y,
                       sigmas=SIG, embed=lambda th: [th[0], th[1], 0.3])
    with pytest.raises(ValueError, match="not more"):
        compare_models(simple, [-1.0, 0.2], simple, [-1.0, 0.2], K, y,
                       sigmas=SIG)
    with pytest.raises(ValueError, match="theta0_extended"):
        compare_models(simple, [-1.0, 0.2], extended, None, K, y)
    with pytest.raises(ValueError, match="degrees of freedom"):
        compare_models(simple, [-1.0, 0.2], extended, None, K[:3], y[:3],
                       sigmas=SIG, embed=embed)

    def not_nested(th):        # on-site energy pinned at 3 eV
        m = TightBindingModel(positions=[[0.0]], norb=1, cell=[[1.0]])
        m.add_hop(0, 0, (0,), [[3.0]])
        for j in range(3):
            m.add_hop(0, 0, (j + 1,), [[th[j]]])
        return m
    with pytest.raises(ValueError, match="larger residual"):
        compare_models(simple, [-1.0, 0.2], not_nested, [-1.0, 0.0, 0.0],
                       K, y, sigmas=SIG)
