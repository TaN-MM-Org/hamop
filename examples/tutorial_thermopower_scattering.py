"""Tutorial: what scatters the electrons?  The answer from the thermopower of gated graphene.

A graphene sheet is gated to six electron densities.  At each density
the "measurement" gives a Hall density (2 % error) and a Seebeck
coefficient S (error 1.5 microvolt/K) at 300 K.  The data are
synthetic, made in this script with a fixed seed from a model whose
relaxation time grows with energy as tau ~ |E|^p with p = 1 (replace
them with your own).

In the Boltzmann picture S depends on how tau varies with energy but
not on its size, so S measured against the density tells p without
knowing tau itself.  p = 0 is a constant tau; p = 1 makes the
conductivity of the Dirac cone grow linearly with the density.  The
analysis

1. turns each Hall density into the chemical potential mu with
   `chemical_potential_from_density`, with the interval its error maps
   to;
2. computes S(mu) with `boltzmann_transport` for tau ~ |E|^p, and adds
   the spread of S over the mu interval to the error of each point;
3. fits p by weighted least squares (error bar from the curvature of
   chi-squared) and reports chi-squared and its p-value;
4. tests the constant-tau model (p = 0) on the same data.

Nearest-neighbor graphene with t = -2.7 eV and a = 2.46 Angstrom on a
240 x 240 mesh; tau ~ max(|E|, 1 meV)^p with E measured from the Dirac
point (the 1 meV floor keeps tau positive at the Dirac point itself).
Run time: about 8 s of CPU time.  Printed output (hamop 0.14.0):

n (1e12/cm^2)   mu (eV)   S measured     S fit   S, constant tau
   2.00      0.1364    -94.0 +/- 1.5    -94.9        -50.3
   3.08      0.1727    -81.3 +/- 1.5    -80.2        -43.1
   4.10      0.2008    -67.8 +/- 1.5    -68.2        -35.3
   4.95      0.2218    -66.4 +/- 1.5    -63.9        -33.2
   6.46      0.2546    -52.5 +/- 1.5    -55.1        -28.4
   7.92      0.2827    -50.8 +/- 1.5    -50.8        -26.0
(S in microvolt/K; negative: electrons)
scattering exponent p = 1.00 +/- 0.02   (tau ~ |E|^p)
fit: chi2 = 5.8 for 5 dof, p-value 0.33
constant tau (p = 0): chi2 = 2779 for 6 dof, p-value 10^-597
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_dist

from hamop import (boltzmann_transport, chemical_potential_from_density,
                   graphene)

T, MESH = 300.0, 240
g = graphene(t=-2.7, a=2.46)


def tau(p):
    """Relaxation time ~ |E|^p (seconds; the scale drops out of S)."""
    return lambda E: 1e-14 * (np.maximum(np.abs(E), 1e-3) / 0.1) ** p


def seebeck(mu, p):
    """S_xx in microvolt/K at the chemical potentials mu (eV)."""
    r = boltzmann_transport(g, mu, T, mesh=MESH, tau=tau(p))
    return r["seebeck"][:, 0, 0] * 1e6


# ---- synthetic measurement (replace with your own data)
rng = np.random.default_rng(11)
n_true = np.array([2.0, 3.0, 4.0, 5.0, 6.5, 8.0]) * 1e12        # cm^-2
mu_true = np.array([chemical_potential_from_density(
    g, n, 2.0, mesh=MESH, T=T)["mu"] for n in n_true])
n_meas = n_true * (1.0 + 0.02 * rng.standard_normal(6))
n_err = 0.02 * n_meas
S_err = 1.5                                                     # microvolt/K
S_meas = seebeck(mu_true, 1.0) + S_err * rng.standard_normal(6)

# ---- 1. chemical potential and its interval from each Hall density
mu, mu_lo, mu_hi = np.empty(6), np.empty(6), np.empty(6)
for i in range(6):
    r = chemical_potential_from_density(g, n_meas[i], 2.0,
                                        density_err=n_err[i], mesh=MESH, T=T)
    mu[i], (mu_lo[i], mu_hi[i]) = r["mu"], r["interval"]


# ---- 2. and 3. weighted least squares for p
def model(p):
    """S at mu, and the error of each point: the measurement error and
    half the spread of S over the mu interval, added in quadrature."""
    s = seebeck(np.concatenate([mu, mu_lo, mu_hi]), p)
    err = np.sqrt(S_err ** 2 + (0.5 * (s[6:12] - s[12:])) ** 2)
    return s[:6], err


def residuals(theta):
    s, err = model(theta[0])
    return (S_meas - s) / err


fit = least_squares(residuals, [0.5], diff_step=1e-3, xtol=1e-10)
p_fit = fit.x[0]
p_err = 1.0 / np.sqrt(float(fit.jac[:, 0] @ fit.jac[:, 0]))
chi2_p = float(fit.fun @ fit.fun)
chi2_0 = float(residuals([0.0]) @ residuals([0.0]))
S_fit, _ = model(p_fit)
S_const, _ = model(0.0)

print("n (1e12/cm^2)   mu (eV)   S measured     S fit   S, constant tau")
for i in range(6):
    print(f"  {n_meas[i] / 1e12:5.2f}      {mu[i]:.4f}   {S_meas[i]:6.1f} +/- "
          f"{S_err:.1f}   {S_fit[i]:6.1f}       {S_const[i]:6.1f}")
print("(S in microvolt/K; negative: electrons)")
print(f"scattering exponent p = {p_fit:.2f} +/- {p_err:.2f}   (tau ~ |E|^p)")
print(f"fit: chi2 = {chi2_p:.1f} for 5 dof, p-value {chi2_dist.sf(chi2_p, 5):.2f}")
# the p-value underflows in double precision; for 6 dof the chi-squared
# tail is exactly exp(-x/2) (1 + x/2 + (x/2)^2 / 2), so take its log
h = 0.5 * chi2_0
log10_p0 = (-h + np.log1p(h + 0.5 * h * h)) / np.log(10.0)
print(f"constant tau (p = 0): chi2 = {chi2_0:.0f} for 6 dof, p-value "
      f"10^{log10_p0:.0f}")
