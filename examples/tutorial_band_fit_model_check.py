"""Tutorial: fit measured band energies, check the model, report error bars.

A typical ARPES-style analysis.  The "measured" data are synthetic: a
chain with hopping t = -1.0 eV, on-site energy e0 = 0.2 eV and a small
second-neighbor hopping t2 = 0.08 eV, sampled at 20 k-points with
Gaussian noise of 0.03 eV (fixed seed).  The analysis does not know
the truth.  It

1. fits the nearest-neighbor model and looks at its p-value,
2. asks with `compare_models` whether a second-neighbor hopping is
   needed (likelihood-ratio test),
3. fits the extended model, and
4. propagates the fit covariance to quantities that were not measured
   directly: the band width and the band top at k = pi (1/Angstrom,
   lattice constant 1 Angstrom), for both models.

Run time: about 2 s.  Printed output (hamop 0.13.0):

step 1, nearest-neighbor fit: chi2 = 245.8 for 18 dof, p = 5.8e-42
step 2, is t2 needed? Delta chi2 = 236.5 for 1 dof, p = 2.3e-53
step 3, extended fit: chi2 = 9.3 for 17 dof, p = 0.93
   t  = -0.9991 +/- 0.0049 eV   (truth -1)
   e0 = +0.1966 +/- 0.0067 eV   (truth +0.2)
   t2 = +0.0749 +/- 0.0049 eV   (truth +0.08)
step 4, quantities not measured directly:
   band width 4|t|      = 3.9962 +/- 0.0194 eV   (truth 4)
   band top, k = pi     = 2.3446 +/- 0.0156 eV   (truth 2.36)
   same, nearest-neighbor model only: 2.1886 +/- 0.0118 eV
"""
import numpy as np

from hamop import (TightBindingModel, bands, compare_models, fit_bands,
                   propagate_uncertainty)

# ---- synthetic measurement (replace with your own k-points and energies)
rng = np.random.default_rng(2026)
k = np.linspace(0.15, 3.0, 20)[:, None]          # 1/Angstrom
sigma = 0.03                                     # eV, measurement error
t_true, e0_true, t2_true = -1.0, 0.2, 0.08
truth = (e0_true + 2 * t_true * np.cos(k[:, 0])
         + 2 * t2_true * np.cos(2 * k[:, 0]))
measured = truth + sigma * rng.standard_normal(k.shape[0])


# ---- the two parameterizations
def chain(th):
    """th = (t, e0, t2): first- and second-neighbor hopping."""
    m = TightBindingModel(positions=[[0.0]], norb=1, cell=[[1.0]])
    m.add_hop(0, 0, (0,), [[th[1]]])
    m.add_hop(0, 0, (1,), [[th[0]]])
    m.add_hop(0, 0, (2,), [[th[2]]])
    return m


def nn_chain(th):
    """th = (t, e0): the same chain without t2."""
    return chain([th[0], th[1], 0.0])


# ---- 1. the simple model
simple = fit_bands(nn_chain, [-1.2, 0.0], k, measured, sigmas=sigma)
print(f"step 1, nearest-neighbor fit: chi2 = {simple.chi2:.1f} for "
      f"{simple.chi2_dof} dof, p = {simple.p_value:.1e}")

# ---- 2. is the extra parameter needed?
r = compare_models(nn_chain, [-1.2, 0.0], chain, None, k, measured,
                   sigmas=sigma, embed=lambda th: [th[0], th[1], 0.0])
print(f"step 2, is t2 needed? Delta chi2 = {r['statistic']:.1f} for "
      f"{r['dof']} dof, p = {r['p_value']:.1e}")

# ---- 3. the extended model
fit = r["extended"]
print(f"step 3, extended fit: chi2 = {fit.chi2:.1f} for {fit.chi2_dof} dof, "
      f"p = {fit.p_value:.2f}")
for name, v, s, tr in zip(("t", "e0", "t2"), fit.theta, fit.sigma,
                          (t_true, e0_true, t2_true)):
    print(f"   {name:2s} = {v:+.4f} +/- {s:.4f} eV   (truth {tr:+g})")

# ---- 4. error bars of derived quantities
k_dense = np.linspace(0.0, np.pi, 401)[:, None]


def width(th):
    return float(np.ptp(bands(chain(th), k_dense)))


def top(th):
    return float(bands(chain(th), [[np.pi]])[0, 0])


w = propagate_uncertainty(fit, width)
tp = propagate_uncertainty(fit, top)
tp_nn = propagate_uncertainty(simple, lambda th: top([th[0], th[1], 0.0]))
print("step 4, quantities not measured directly:")
print(f"   band width 4|t|      = {w['value']:.4f} +/- {w['sigma']:.4f} eV"
      f"   (truth {4 * abs(t_true):g})")
print(f"   band top, k = pi     = {tp['value']:.4f} +/- {tp['sigma']:.4f} eV"
      f"   (truth {e0_true - 2 * t_true + 2 * t2_true:g})")
print(f"   same, nearest-neighbor model only: {tp_nn['value']:.4f} +/- "
      f"{tp_nn['sigma']:.4f} eV")
