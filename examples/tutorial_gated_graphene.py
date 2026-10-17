"""Tutorial: from a Hall density to the optical absorption edge of gated graphene.

A gated graphene sheet on a substrate (refractive index 1.5).  The
"measurement" is synthetic: eight Hall-effect readings of the electron
density, scattered by 3 % around 6.0e12 cm^-2 (fixed seed).  The
analysis

1. averages them into a density with a standard error,
2. turns the density into the chemical potential mu with
   `chemical_potential_from_density` (the error interval comes from
   solving for the density -/+ its error), and reports the quantum
   capacitance e^2 dn/dmu,
3. computes the optical sheet conductivity at photon energies around
   the Pauli-blocking edge 2 mu (transitions below 2 mu are blocked
   because their final states are already filled), at mu and at both
   ends of its interval, and
4. turns it into the transmittance relative to the bare substrate,
   the number an infrared transmission measurement reports.

Nearest-neighbor graphene with t = -2.7 eV and a = 2.46 Angstrom, at
295 K; mesh 480 x 480 and a Gaussian broadening of 30 meV.  Run time:
about 3 s.  Printed output (hamop 0.13.0):

Hall density: n = (5.980 +/- 0.047) x 1e12 cm^-2
chemical potential: mu = 0.2448 eV, interval [0.2438, 0.2458] eV
Pauli-blocking edge: 2 mu = 0.490 +/- 0.002 eV
quantum capacitance: 7.57 microfarad/cm^2
photon (eV)  Re sigma / (e^2/4 hbar)   T/T_bare (substrate n = 1.5)
   0.30      0.028 [0.027, 0.029]           0.9995
   0.45      0.326 [0.318, 0.334]           0.9940
   0.60      0.888 [0.884, 0.892]           0.9839
   0.90      1.014 [1.014, 1.014]           0.9817
"""
import numpy as np

from hamop import (chemical_potential_from_density, graphene, sheet_optics,
                   sigma_optical)

T = 295.0                                  # K
g = graphene(t=-2.7, a=2.46)

# ---- 1. synthetic Hall readings (replace with your own)
rng = np.random.default_rng(7)
hall = 6.0e12 * (1.0 + 0.03 * rng.standard_normal(8))      # cm^-2
n, n_err = hall.mean(), hall.std(ddof=1) / np.sqrt(hall.size)
print(f"Hall density: n = ({n / 1e12:.3f} +/- {n_err / 1e12:.3f}) x 1e12 "
      "cm^-2")

# ---- 2. chemical potential and quantum capacitance
r = chemical_potential_from_density(g, n, n_neutral=2.0, density_err=n_err,
                                    mesh=300, T=T)
mu, (mu_lo, mu_hi) = r["mu"], r["interval"]
print(f"chemical potential: mu = {mu:.4f} eV, interval [{mu_lo:.4f}, "
      f"{mu_hi:.4f}] eV")
print(f"Pauli-blocking edge: 2 mu = {2 * mu:.3f} +/- {2 * r['mu_err']:.3f} eV")
print(f"quantum capacitance: {r['quantum_capacitance'] * 1e6:.2f} "
      "microfarad/cm^2")

# ---- 3. and 4. optical conductivity and transmittance across the edge
omega = np.array([0.30, 0.45, 0.60, 0.90])               # photon energies, eV
sig = {m: sigma_optical(g, omega, m, mesh=480, T=T, eta=0.03)
       for m in (mu, mu_lo, mu_hi)}
print("photon (eV)  Re sigma / (e^2/4 hbar)   T/T_bare (substrate n = 1.5)")
for i, w in enumerate(omega):
    s_lo, s_hi = sorted((sig[mu_lo][i], sig[mu_hi][i]))
    tr = sheet_optics(sig[mu][i], n_in=1.0, n_out=1.5)["T_rel"]
    print(f"   {w:.2f}      {sig[mu][i]:.3f} [{s_lo:.3f}, {s_hi:.3f}]"
          f"           {tr:.4f}")
