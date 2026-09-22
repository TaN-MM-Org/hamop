# hamop

[![Tests](https://github.com/TaN-MM-Org/hamop/actions/workflows/ci.yml/badge.svg)](https://github.com/TaN-MM-Org/hamop/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/hamop?label=PyPI&color=blue&cacheSeconds=3600)](https://pypi.org/project/hamop/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22311381-blue)](https://doi.org/10.5281/zenodo.22311381)

`hamop` is a Python package for computing what electrons do in a
material or molecule described by a **tight-binding model**: a short
list of orbitals and the energies that let an electron hop between
them. You build the model once. From that one model the package
computes:

- the allowed electron energies (the **band structure**) and how many
  states sit at each energy (the **density of states**);
- how strongly the material absorbs light at each photon energy (the
  **optical conductivity**);
- how easily electrons pass through a small device placed between two
  wires (the **transmission**);
- topological numbers such as the **Chern number**, and related
  geometric quantities of the electron states.

The point of the package is that all of these come from the *same*
matrices. When the optics and the energy levels of one model are
computed by different codes, small differences in conventions (how
the velocity is defined, how overlapping orbitals are treated, how
peaks are broadened) can make the results disagree, for example an
absorption edge that no longer sits at the band gap. Here the
spectrum and the optics are computed from the same Bloch matrices
through the same eigensolver, the topology functions work from the
same hopping blocks, and `principal_layers` cuts the same model into
the layer blocks that the transport functions take.

The package also works when neighbouring orbitals overlap (a
**nonorthogonal basis**, as in LCAO calculations, which build electron
states from atomic orbitals), and it can read Hamiltonians exported by
Wannier90, a widely used program that turns first-principles
(quantum-mechanical, parameter-free) calculations into tight-binding
models. It ships no material data: the numbers of a real
material come from your own calculation or measurement.

## Contents

- [A short guide to the words used here](#a-short-guide-to-the-words-used-here)
- [Install, units and conventions](#install-units-and-conventions)
- [Examples](#examples) (each with the output it prints)
- [What is in the package](#what-is-in-the-package)
- [When it refuses, and why](#when-it-refuses-and-why)
- [How the results are checked](#how-the-results-are-checked)
- [Bugs fixed in 0.10.1](#bugs-fixed-in-0101)
- [Limits](#limits)
- [Relation to existing tools](#relation-to-existing-tools)
- [Where it comes from](#where-it-comes-from)
- [Citing, support and license](#citing-support-and-license)

## A short guide to the words used here

- **Site, orbital** -- an atom (site) carries one or more orbitals, the
  states an electron can occupy there.
- **Hopping** -- the energy that couples an orbital on one site to an
  orbital on another. You give each coupling once, in one direction;
  the reverse direction is added for you.
- **Hermitian** -- a matrix equal to its own conjugate transpose. An
  energy matrix must be Hermitian so that its energies are real; the
  reverse hopping added for you is the "Hermitian partner" of the one
  you give.
- **Overlap, nonorthogonal basis** -- in some models neighbouring
  orbitals are not independent: they overlap by an amount `s`. The
  overlap matrix is called `S`. If you give no overlap, `S` is the
  identity (an **orthogonal basis**).
- **Periodic model, unit cell** -- a crystal repeats one unit cell.
  The package handles periodicity in any number of directions, and
  also **finite** models (a molecule, a flake, a device) with no
  repetition (`cell=None`).
- **Wave vector k, Brillouin zone, k-mesh** -- in a crystal each
  electron state carries a wave vector `k`. All distinct `k` fill one
  cell of "k-space" called the Brillouin zone. Many results are sums
  over an even grid of `k` points, the **k-mesh** (`mesh=` in the
  functions).
- **Bloch Hamiltonian H(k)** -- the model's energy matrix at one `k`.
  Its eigenvalues are the band energies at that `k`.
- **Band structure, band gap** -- the band energies as `k` varies. A
  gap is an energy range with no states.
- **Density of states (DOS)** -- how many states lie near each energy.
- **Chemical potential `mu`, filling** -- states below `mu` are
  occupied (smoothly, at temperature `T`); the filling is the number
  of occupied states per cell.
- **Optical conductivity** -- how strongly light of photon energy
  `hbar omega` is absorbed. It is computed with the Kubo-Greenwood
  formula (the standard weak-field response formula) and returned in
  units of `e^2/(4 hbar)`. That unit is the measured absorption
  plateau of graphene, so graphene gives about 1.
- **Broadening `eta`** -- a small energy width given to each sharp
  line, so that sums over a finite k-mesh give smooth curves.
- **Drude weight** -- the part of the conductivity carried by freely
  moving electrons in a metal (the "intraband" part).
- **Transmission T(E)** -- the number of electron channels that pass
  through a device at energy `E`; in the Landauer picture the
  conductance is this number times `e^2/h` per spin. The device sits
  between two semi-infinite **leads** (wires), and is cut into
  **principal layers** that couple only to their nearest neighbours.
  The method is the **nonequilibrium Green function (NEGF)** method.
- **Green function** -- the matrix `(zS - H)^-1` at the complex
  energy `z = E + i eta`. Its imaginary part gives the local density
  of states (the DOS on each orbital, **LDOS**), and the transmission
  is built from it.
- **Self-energy** -- a term added to a layer's Hamiltonian that
  describes what the rest of the system (a lead, disorder, a probe)
  does to it.
- **Zeeman term, spin-orbit coupling** -- energy terms that act on
  the electron's spin: the Zeeman term comes from a magnetic field and
  splits each level by `2B` (`B` in eV); spin-orbit coupling comes
  from the electron's own motion.
- **Berry phase, Chern number** -- properties of how the electron
  states change as `k` moves. The Chern number is a whole number that
  cannot change unless a band gap closes. It sets the quantized Hall
  conductivity `C e^2/h`.
- **Quantum metric** -- how fast the occupied electron states change
  as `k` moves; the Berry curvature is its partner.
- **Kernel polynomial method (KPM)** -- a way to get the DOS and the
  conductivity of very large finite models without computing every
  eigenvalue.

## Install, units and conventions

```
pip install hamop
```

It needs Python 3.9 or newer, NumPy 1.22 or newer and SciPy 1.8 or
newer, and nothing else.

Units and conventions, stated once:

- Energies in eV, positions in Angstrom, `k` in 1/Angstrom, all
  Cartesian. Temperatures `T` are in kelvin, except `berry_dipole`,
  which takes `kT` in energy units.
- Each directed hopping block is added once; its Hermitian partner is
  implied.
- The optical conductivity is the real sheet conductivity in units of
  `e^2/(4 hbar)`. Spin degeneracy enters as an explicit factor
  `spin` (default 2). For a model made spinful with `with_spin`, pass
  `spin=1`. For a finite model the result is the conductivity times
  the system area; divide by your geometric area.
- `dos` counts states per unit cell per eV without spin; `fermi_level`
  takes the filling without spin; `carrier_count` includes spin.
- The velocity operator uses the usual tight-binding position
  convention: the position operator is diagonal at the sites. Extra
  on-site position blocks (`model.set_dipole`) add the intra-atomic
  term on top.
- Magnetic flux for `with_peierls` is given per square Angstrom, in
  units of the flux quantum; `magnetic_supercell` takes the flux
  `p/q` per unit cell.

## Examples

Each example below runs as written, and the output shown is what it
printed with hamop 0.10.1. The model parameters are illustrative
values, not fitted to any material, unless the text says otherwise.

### 1. Graphene: bands and the absorption plateau

```python
import numpy as np
from hamop import graphene, bands, sigma_optical

g = graphene(t=-2.7, a=2.46)      # hopping t in eV, lattice constant a in Angstrom

b = 2 * np.pi * np.linalg.inv(g.cell).T   # reciprocal lattice vectors (rows)
K = (2 * b[0] + b[1]) / 3                 # a corner of the Brillouin zone
e_gamma = bands(g, [np.zeros(2)])[0]
e_K = bands(g, [K])[0]
print(f"band energies at the zone centre: {e_gamma[0]:+.4f} {e_gamma[1]:+.4f} eV")
print("gap at the zone corner K below 1e-9 eV:", abs(e_K[1] - e_K[0]) < 1e-9)

omega = np.array([1.0, 1.3])              # photon energies, eV
sigma = sigma_optical(g, omega, mu=0.0, mesh=120, eta=0.12, T=10.0)
for w, s in zip(omega, sigma):
    print(f"photon {w:.1f} eV: sigma = {s:.3f} x e^2/(4 hbar)")
```

```
band energies at the zone centre: -8.1000 +8.1000 eV
gap at the zone corner K below 1e-9 eV: True
photon 1.0 eV: sigma = 1.020 x e^2/(4 hbar)
photon 1.3 eV: sigma = 1.032 x e^2/(4 hbar)
```

The two bands sit at `-/+ 3|t|` at the zone centre and touch at the
corner K (the "Dirac point"). The absorption is close to 1 in units of
`e^2/(4 hbar)`, the universal optical conductivity of graphene
(Kuzmenko et al., Phys. Rev. Lett. 100, 117401 (2008)). The test suite
checks these two photon energies to within 5 %. The script
`examples/graphene_universal_conductivity.py` prints the same quantity
from 0.6 to 1.8 eV.

### 2. Build your own model: a dimerized chain

```python
import numpy as np
from hamop import TightBindingModel, band_edges, berry_phase

def ssh_chain(t1, t2, a=2.0):
    """Two sites per cell, at 0 and a/2; hopping t1 inside the cell,
    t2 to the next cell."""
    m = TightBindingModel(positions=[[0.0], [0.5 * a]], norb=1, cell=[[a]])
    m.add_hop(0, 1, (0,), [[t1]])        # site 0 -> site 1, same cell
    m.add_hop(1, 0, (1,), [[t2]])        # site 1 -> site 0, next cell
    return m

for t1, t2 in [(-1.0, -0.6), (-0.6, -1.0)]:
    m = ssh_chain(t1, t2)
    vbm, cbm, gap = band_edges(m, mu=0.0, mesh=2001)
    b = 2 * np.pi / 2.0                              # reciprocal period
    loop = [[b * i / 60] for i in range(60)]         # one pass through k
    zak = berry_phase(m, loop, n_occ=1)
    print(f"t1={t1:+.1f} t2={t2:+.1f}: gap {gap:.4f} eV, "
          f"Zak phase {zak / np.pi:+.4f} pi")
```

```
t1=-1.0 t2=-0.6: gap 0.8000 eV, Zak phase +0.0000 pi
t1=-0.6 t2=-1.0: gap 0.8000 eV, Zak phase -1.0000 pi
```

This is the Su-Schrieffer-Heeger (SSH) chain. Both orderings of the
bonds have the same gap, `2 | |t1| - |t2| |`. They differ in the Berry
phase of the filled band (here called the Zak phase): the two values
differ by pi. Which of the two is 0 depends on where the unit cell
starts; the difference of pi does not.

### 3. Topology: Chern number and the quantized Hall conductivity

```python
import numpy as np
from hamop import haldane, chern_number, sigma_tensor

for mass in (0.0, 0.9):
    h = haldane(t1=-1.0, t2=0.1, phi=np.pi / 2, m_ab=mass)
    C = chern_number(h, mesh=18)
    sxy = sigma_tensor(h, [0.0], mu=0.0, directions=(0, 1), mesh=48,
                       T=10.0, eta=1e-4, spin=1)[0]
    hall = sxy.real * np.pi / 2       # e^2/h is 2/pi in the unit e^2/(4 hbar)
    print(f"mass {mass}: Chern number {round(C, 6) + 0.0:+.6f}, "
          f"sigma_xy(0) = {round(hall, 6) + 0.0:+.6f} e^2/h")
```

```
mass 0.0: Chern number +1.000000, sigma_xy(0) = +1.000000 e^2/h
mass 0.9: Chern number +0.000000, sigma_xy(0) = +0.000000 e^2/h
```

The Haldane model (Haldane, Phys. Rev. Lett. 61, 2015 (1988)) has a
topological phase (Chern number 1) and, when the sublattice energy
difference `m_ab` is large enough, an ordinary one (Chern number 0).
Two separate calculations agree: the Chern number from the band
states, and the Hall conductivity at zero frequency from the Kubo
formula. Their agreement is the TKNN relation (Thouless, Kohmoto,
Nightingale and den Nijs, Phys. Rev. Lett. 49, 405 (1982)).

### 4. Transport: one impurity in a chain

```python
import numpy as np
from hamop import chain_lead_blocks, transmission

t, eps = -1.0, 0.8                     # hopping and impurity energy, eV
H00, H01 = chain_lead_blocks(t=t)      # one lead layer and its coupling
layers = [H00, H00 + eps, H00]         # three device layers, impurity in the middle
coup = [H01, H01]                      # couplings between device layers

E = np.array([-1.2, 0.3, 1.5, 2.5])
T = transmission(E, layers, coup, H00, H01, eta=1e-8)
exact = np.where(np.abs(E) < 2, (4 - E**2) / ((4 - E**2) + eps**2), 0.0)
for e, a, b in zip(E, T, exact):
    print(f"E = {e:+.1f} eV: T = {a:.6f}   closed form {b:.6f}")
```

```
E = -1.2 eV: T = 0.800000   closed form 0.800000
E = +0.3 eV: T = 0.859341   closed form 0.859341
E = +1.5 eV: T = 0.732218   closed form 0.732218
E = +2.5 eV: T = 0.000000   closed form 0.000000
```

A clean chain passes one channel (T = 1) at energies inside its band,
between -2|t| and +2|t|, and none outside. A single impurity of energy
`eps` reflects part of the wave; the closed form for this chain is
`T = (4t^2 - E^2) / ((4t^2 - E^2) + eps^2)`. At 2.5 eV the energy is
outside the band, so nothing passes.

### 5. Overlapping orbitals: moving the energy zero changes nothing

```python
from hamop import linear_chain, drude_weight

# A chain whose neighbouring orbitals overlap (s = 0.2), half filled.
m1 = linear_chain(t=-1.0, e0=0.0, s=0.2)
D1 = drude_weight(m1, mu=0.0, mesh=800, T=100.0)

# The same chain with the energy zero moved by c = 5 eV.  With overlap
# that shift is H -> H + c S, so the hopping changes by c * s as well.
c = 5.0
m2 = linear_chain(t=-1.0 + c * 0.2, e0=c, s=0.2)
D2 = drude_weight(m2, mu=c, mesh=800, T=100.0)
print(f"Drude weight, original zero: {D1:.8f}")
print(f"Drude weight, shifted zero:  {D2:.8f}")
print("difference below 1e-10:", abs(D1 - D2) < 1e-10)
```

```
Drude weight, original zero: 16.01338983
Drude weight, shifted zero:  16.01338983
difference below 1e-10: True
```

Where you put the zero of energy is a free choice, so no physical
result may depend on it. With overlapping orbitals, that only holds if
the velocity includes the overlap term
`v = dH/dk - (E_n + E_m)/2 dS/dk`, which the package uses for the Drude
weight, the optical conductivity and the conductivity tensor.

### 6. Fit a model to measured bands, and plan the measurement first

```python
import numpy as np
from hamop import linear_chain, band_information, fit_bands

def build(theta):                      # theta = (t, e0), both in eV
    return linear_chain(t=theta[0], e0=theta[1])

k = np.linspace(0.1, 3.0, 12)[:, None] # 12 measured k-points (1/Angstrom)
sigma = 0.03                           # measurement error, eV

plan = band_information(build, [-1.0, 0.2], k, sigmas=sigma)
print("identifiable:", plan["identifiable"])
print("predicted error bars (eV):", np.round(plan["sigma"], 5))

# Illustrative "measured" data: the chain with t = -1.0, e0 = 0.2,
# plus seeded random noise of size sigma.
rng = np.random.default_rng(13)
measured = 0.2 + 2 * (-1.0) * np.cos(k[:, 0]) + sigma * rng.standard_normal(12)
fit = fit_bands(build, [-1.3, 0.5], k, measured, sigmas=sigma)
print("fitted (t, e0):", np.round(fit.theta, 4))
print("error bars:", np.round(fit.sigma, 5))
print(f"chi-squared {fit.chi2:.2f} for {fit.chi2_dof} degrees of freedom")
```

```
identifiable: True
predicted error bars (eV): [0.0061  0.00866]
fitted (t, e0): [-1.0002  0.2087]
error bars: [0.0061  0.00866]
chi-squared 18.87 for 10 degrees of freedom
```

`build` is your own parameterization: any function that turns a list
of numbers into a model. Before measuring, `band_information` says
whether the planned k-points can tell the parameters apart and how
large the error bars will be; `design_kpoints` picks the most useful
k-points from a candidate list. `fit_bands` then fits the measured
energies and reports error bars from the standard least-squares
covariance.

### 7. Quantum geometry of the occupied band

```python
import numpy as np
from hamop import haldane, quantum_geometric_tensor, quantum_weight

m = haldane(t1=-1.0, t2=0.1)                 # lattice constant 1 Angstrom
b = 2 * np.pi * np.linalg.inv(m.cell).T
K = (2 * b[0] + b[1]) / 3
Q = quantum_geometric_tensor(m, K)           # at the zone corner
g = Q.real                                   # quantum metric, Angstrom^2
omega = 2 * Q[0, 1].imag                     # Berry curvature, Angstrom^2
print(f"tr g = {np.trace(g):.4f}, 2 sqrt(det g) = "
      f"{2 * np.sqrt(np.linalg.det(g)):.4f}, |Omega| = {abs(omega):.4f}")

out = quantum_weight(m, mesh=18)             # integrated over the zone
print(f"tr K = {out['tr_K']:.4f}, Chern number from the same data = {out['chern']:.4f}")
```

```
tr g = 1.3889, 2 sqrt(det g) = 1.3889, |Omega| = 1.3889
tr K = 1.1827, Chern number from the same data = 1.0000
```

At every `k`, `tr g >= 2 sqrt(det g) >= |Omega|` must hold; at this
point all three are equal to four decimals. Integrated over the zone
the same chain gives `tr K >= |C|` (Onishi and Fu, PRX 14, 011052
(2024)): here 1.1827 against a Chern number of 1.

### 8. Reading a Wannier90 file

```python
import os, tempfile
import numpy as np
import hamop

# Write graphene's real-space Hamiltonian in the Wannier90 hr.dat format,
# read it back, and compare with the built-in graphene model.
t, a = -2.7, 2.46
H_R = {(0, 0): [[0, t], [t, 0]],
       (1, 0): [[0, 0], [t, 0]], (-1, 0): [[0, t], [0, 0]],
       (0, 1): [[0, 0], [t, 0]], (0, -1): [[0, t], [0, 0]]}
path = os.path.join(tempfile.mkdtemp(), "graphene_hr.dat")
hamop.save_wannier90_hr(path, H_R)

cell = a * np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]])
centres = np.array([[0.0, 0.0], (cell[0] + cell[1]) / 3])
m = hamop.from_wannier90(path, cell, centers=centres)

k = np.random.default_rng(0).uniform(-2, 2, (20, 2))
diff = np.abs(hamop.bands(m, k) - hamop.bands(hamop.graphene(t, a), k)).max()
print("bands agree to 1e-12 eV at 20 random k-points:", diff < 1e-12)
```

```
bands agree to 1e-12 eV at 20 random k-points: True
```

A `seedname_hr.dat` file (Mostofi et al., Comput. Phys. Commun. 185,
2309 (2014)) holds the real-space Hamiltonian but not the lattice
vectors or the orbital centres. You supply the lattice vectors
(`cell`). The centres are optional: band energies do not depend on
them, but optical matrix elements do, so give the true centres when
you need optics.

## What is in the package

Each function's docstring (`help(hamop.sigma_optical)`, for example)
gives its inputs, units and conventions.

**The model**

- `TightBindingModel(positions, norb, cell=None)` -- sites, orbitals
  per site and lattice vectors. `add_hop(i, j, image, H_block,
  S_block=None)` adds a hopping (and optional overlap) block;
  `set_dipole(i, X)` adds on-site position blocks; `bloch(k)` and
  `bloch_derivative(k, direction)` return `H(k), S(k)` and their
  exact `k`-derivatives; `monkhorst_pack(mesh, time_reversal=False)`
  returns an even k-grid with weights.
- `gen_eigh(H, S)` -- eigenvalues of `H c = E S c`. It drops overlap
  directions below a threshold ("canonical orthogonalization"; Szabo
  and Ostlund, *Modern Quantum Chemistry*, sec. 3.4.5), so a nearly
  redundant set of orbitals does not produce huge wrong eigenvalues.
- Ready-made models: `linear_chain`, `two_site` (a two-atom
  molecule), `graphene`, `ssh`, `haldane`, and `chain_lead_blocks`
  (lead blocks of a chain, for transport).

**Bands and density of states**

- `bands`, `k_path` (a path through k-space for a band plot), `dos`
  (Gaussian-broadened DOS), `fermi_level` (chemical potential for a
  given filling), `band_edges` (valence-band top, conduction-band
  bottom and gap around `mu`).

**Optics**

- `sigma_optical` -- the real optical conductivity (Gaussian or
  Lorentzian broadening).
- `sigma_tensor` -- the complex conductivity tensor component
  `sigma_ab(omega)` between bands, including the Hall component
  `sigma_xy`.
- `drude_weight` -- the intraband (Drude) weight.
- `carrier_count` -- occupied states per cell, spin included.

**Topology and band geometry**

- `berry_phase` (along a closed loop in k-space), `berry_curvature`
  and `chern_number` (on a 2D k-grid, by the lattice method of Fukui,
  Hatsugai and Suzuki, J. Phys. Soc. Jpn. 74, 1674 (2005)). With
  overlap, two conventions are offered (`frame="lowdin"` or
  `"atomic"`); a sparse solver (`solver="sparse"`) serves large cells
  with few occupied bands.
- `chern_marker` -- the local Chern marker of a finite 2D model
  (Bianco and Resta, Phys. Rev. B 84, 241106(R) (2011)).
- `berry_dipole`, `band_curvatures`, `fermi_occupation` -- the Berry
  curvature dipole behind the nonlinear Hall effect (Sodemann and Fu,
  Phys. Rev. Lett. 115, 216806 (2015)), computed in two independent
  ways (`method="grad"` or `"fermi"`), the per-band curvature, and the
  Fermi occupation factor.
- `quantum_geometric_tensor`, `quantum_metric`, `quantum_weight` --
  the quantum metric and Berry curvature of the occupied states
  (Provost and Vallee, Commun. Math. Phys. 76, 289 (1980); Peotta and
  Torma, Nat. Commun. 6, 8944 (2015); Onishi and Fu, PRX 14, 011052
  (2024)).

**Spin and magnetic fields**

- `with_spin` -- a spinful copy of a model; `PAULI` -- the Pauli
  matrices, for adding Zeeman or spin-orbit terms; `kane_mele` -- the
  Kane-Mele model (Kane and Mele, Phys. Rev. Lett. 95, 226801 (2005)).
- `with_peierls` -- a uniform magnetic field on a finite model, by
  Peierls substitution: each hopping gets a phase set by the magnetic
  flux (Peierls, Z. Phys. 80, 763 (1933)).
- `magnetic_supercell` -- a periodic 2D model at rational flux `p/q`
  per cell: the unit cell is repeated `q` times so that the field fits
  it (a Hofstadter magnetic cell).

**Transport**

- `transmission` -- T(E) by a recursive sweep through the layers;
  `transmission_direct` and `transmission_sparse` compute the same
  quantity by dense and by sparse inversion. `sancho_rubio` gives
  the surface Green function of a lead (Lopez Sancho, Lopez Sancho
  and Rubio, J. Phys. F 15, 851 (1985)).
- `principal_layers` -- cuts a finite model into layers along an axis
  and checks that no coupling skips a layer.
- `buttiker_transmission` -- one dephasing probe, an imaginary
  contact that scrambles the phase of the electrons passing it but
  draws no net current (Buttiker, Phys. Rev. B 33, 3020 (1986)); `multiprobe_transmission` -- probes on many
  layers (D'Amato and Pastawski, Phys. Rev. B 41, 7411 (1990));
  `scba_transmission` -- averaged on-site disorder, solved
  self-consistently (the self-consistent Born approximation, SCBA).
- `device_greens`, `device_ldos`, `bond_currents` -- the device Green
  function, the local density of states per orbital, and the map of
  currents between orbitals (Paulsson and Brandbyge, Phys. Rev. B 76,
  115117 (2007)).
- `transmission`, `transmission_direct` and `transmission_sparse` also
  accept your own self-energy per layer (`sigma_int`).

**Large systems**

- `bloch_sparse`, `bloch_derivative_sparse` -- the same matrices as
  `bloch` and `bloch_derivative`, stored sparse.
- `lowest_bands` -- only the lowest few eigenvalues (Lanczos method,
  an iterative solver that never forms the full eigenvector set).
- `kpm_dos`, `kpm_sigma` -- DOS and optical conductivity of large
  finite models by the kernel polynomial method (Weisse, Wellein,
  Alvermann and Fehske, Rev. Mod. Phys. 78, 275 (2006)).

**Fewer k-points**

- `monkhorst_pack(..., time_reversal=True)` pairs `k` with `-k`.
- `symmetry_fold` folds a k-grid by a point group you supply;
  `find_point_group` finds that group. Both check every operation on
  the model first.

**Interpolation and file exchange**

- `fourier_interpolation`, `FourierInterpolator` -- sample `H(k)` on a
  grid, transform to real space, and evaluate at any `k`.
- `from_wannier90`, `load_wannier90_hr`, `save_wannier90_hr` -- read
  and write the Wannier90 `seedname_hr.dat` format.

**Fitting to measurements**

- `fit_bands`, `BandFit`, `band_information`, `design_kpoints` --
  example 6; `design_kpoints` uses greedy D-optimal selection, which
  adds, one at a time, the k-point that most shrinks the joint
  uncertainty of the parameters (Pukelsheim, Optimal Design of
  Experiments, SIAM (2006)).

## When it refuses, and why

`hamop` raises an error with an explanation, instead of returning a
doubtful number, when:

- a periodic model is given to `dos`, `fermi_level`, `band_edges`,
  `sigma_optical`, `sigma_tensor`, `drude_weight` or `carrier_count`
  without `mesh` or `kpts`;
- a function for finite models (`with_peierls`, `principal_layers`,
  `kpm_dos`, `kpm_sigma`, `chern_marker`) gets a periodic one, or a
  function that needs a k-grid (`monkhorst_pack`, `symmetry_fold`,
  `find_point_group`, `fourier_interpolation`) gets a finite one;
  `berry_curvature`, `chern_number`, `band_curvatures`,
  `berry_dipole`, `quantum_weight` and `magnetic_supercell` need a
  periodic 2D model;
- a function that does not handle overlap gets a model with overlap:
  `kpm_sigma`, `chern_marker`, the quantum-geometry functions,
  `band_curvatures` / `berry_dipole`, and `solver="sparse"` in the
  Berry functions;
- `monkhorst_pack(time_reversal=True)` is asked of a model with
  complex hopping blocks, where pairing `k` with `-k` is not
  guaranteed;
- an operation given to `symmetry_fold` does not map the lattice to
  itself, changes the eigenvalues at a random test `k`, or moves grid
  points off the grid;
- `magnetic_supercell` cannot represent the flux in its gauge; the
  message names an equivalent flux that works;
- `principal_layers` finds a coupling that skips a layer (the layers
  are thinner than the hopping range);
- the quantum geometry is asked at a `k` where the occupied and empty
  bands are closer than `gap_min` (the geometry is undefined there);
- a Berry link between neighbouring k-points vanishes ("refine the
  grid"), or the overlap matrix is not positive definite in the
  `lowdin` frame;
- a Wannier90 file is malformed (wrong counts, duplicate or missing
  elements, a short degeneracy list, an orbital index out of range),
  its Hamiltonian is not Hermitian, or the cell does not match the
  file;
- a planned or measured set of band energies cannot tell the fit
  parameters apart, or there are fewer measurements than parameters;
- a block has the wrong shape, an on-site or dipole block is not
  Hermitian, or a finite model is given a non-zero lattice image;
- photon energies given to `sigma_optical` or `kpm_sigma` are zero or
  negative (`sigma_tensor` accepts `omega = 0`, the static Hall
  conductivity of example 3), or the lineshape is unknown;
- the temperature is negative; `drude_weight` also refuses `T = 0`,
  and `berry_dipole(method="fermi")` needs `kT > 0`;
- `lowest_bands` is asked for the full spectrum, a filling or `mu`
  lies outside the spectrum, or the SCBA iteration does not converge.

## How the results are checked

153 automated tests run on every change, on Python 3.9, 3.10, 3.11,
3.12, 3.13 and 3.14, and once more on Python 3.10 with the oldest
NumPy (1.22.0) and SciPy (1.8.0) the package allows. The numerical
tests compare the package with a closed-form result, a symmetry, or a
second calculation done a different way; none compares against a
number stored from an earlier run. The main checks, with the
tolerances the tests use:

**Bands, DOS, optics**

- The chain's bands equal `e0 + 2t cos ka` to 1e-12; with overlap `s`
  they equal `2t cos ka / (1 + 2s cos ka)` to 1e-12.
- The chain's DOS at the band centre equals `1/(2 pi |t|)` to 5e-4,
  and the DOS integrates to the orbital count to 1e-6.
- Graphene's bands touch at K to 1e-9 and sit at `-/+ 3|t|` at the zone
  centre to 1e-12. Its optical conductivity at 1.0 and 1.3 eV is
  within 5 % of `e^2/(4 hbar)`.
- The two-site molecule absorbs at `2|t|` (within 2e-3 eV) with the
  hand-derived peak height to 1e-3 relative; the Lorentzian lineshape
  also gives its hand-derived peak height to 1e-3.
- Shifting the energy zero with overlap (`H -> H + cS`) changes the
  conductivity, the tensor and the Drude weight by less than 1e-10
  (relative, for the Drude weight).
- The Drude weight of the half-filled chain equals `8 spin |t| a` to
  1e-3 relative, and is below 1e-12 for an empty or full band.
- An on-site transition is exactly dark without a dipole block; with
  one, its peak matches the hand-derived value to 1e-3 relative, in
  the dense route. The same holds in a nonorthogonal basis (two
  uncoupled atoms give exactly twice one atom).

**Topology and geometry**

- The Haldane model's Chern number is 1 in the topological phase and 0
  in the trivial phase, to 1e-12; it flips sign with the flux; all
  bands together give 0. The same integers come out with overlap and
  in both frames; the sparse solver gives 2 for two stacked copies,
  like the dense one.
- `sigma_xy(0)` equals the Chern number times `e^2/h` to 1e-6, sign
  included, and the tensor is antisymmetric to 1e-12.
- The SSH chain's Zak phases are 0 or pi to 1e-9 and differ by pi,
  with and without overlap.
- The Kane-Mele model equals two Haldane copies to 1e-12, and its gap
  at K is `6 sqrt(3) lambda_so` to 1e-9.
- The Chern marker of a 10 x 10 flake sums to zero to 1e-8, and its
  bulk average is within 0.05 of the Chern number.
- The Berry curvature dipole is below 1e-10 with inversion symmetry
  and below 1e-12 for completely filled bands. For gapped graphene it
  at least halves when the mesh doubles, and one strained bond makes
  it more than 10 times larger. With one mirror line, the component
  along the mirror is below 1e-8 times the other one. The two
  integral forms agree within 10 %.
- The quantum metric and curvature match the two-band closed forms to
  1e-4 relative; `tr g >= 2 sqrt(det g) >= |Omega|` holds at 25 random
  points per phase; the integrated Chern number is within 0.02 of the
  lattice one; `tr K >= |C|` holds in both phases.

**Magnetic fields and spin**

- A flux-threaded ring matches `2t cos((2 pi j + Theta)/N)` to 1e-12;
  two gauges give the same spectrum to 1e-12; the flux through one
  square is exact to 1e-12; the spectrum repeats after one flux
  quantum to 1e-12.
- The lowest Landau level of the square lattice sits at
  `-4|t| + hbar omega_c / 2` within 3 %.
- At zero flux the magnetic supercell equals folded bands to 1e-12;
  at half a flux quantum the square lattice matches
  `+/- 2|t| sqrt(cos^2 kx + cos^2 ky)` to 1e-12; at 1/3 the lowest
  band's Chern number matches `sigma_xy` to 1e-4.
- Spin doubling gives doublets to 1e-12, and a Zeeman term splits
  them by `2B` to 1e-12. With overlap, the Zeeman-split bands equal
  `(2t cos ka -/+ B)/(1 + 2s cos ka)` to 1e-12.

**Transport**

- The lead surface Green function matches its closed form to 1e-6.
- A clean chain transmits 1 inside the band (to 1e-4) and 0 outside
  (to 1e-8); with overlap it still transmits 1 inside the band; two
  uncoupled chains transmit 2.
- One impurity matches the closed form to 1e-5.
- The recursive sweep equals direct inversion to 1e-10 (1e-12 with
  self-energies, and with overlap at `eta = 0.05`); sparse equals
  dense to 1e-12.
- With overlap, the lead surface Green function matches its closed
  form to 1e-12 at `eta = 0.05`, the Green function of the closed
  device (leads detached) equals `(zS - H)^-1` to 1e-12, and `i(G - G^dag) = G(GamL + GamR + 2 eta S)G^dag` holds to
  1e-12.
- The Buttiker probe at zero coupling equals the coherent result to
  1e-12 and matches the single-site closed form to 1e-8; a probe on
  every layer gives a resistance linear in length (R^2 > 0.9999);
  current is conserved to 1e-12.
- SCBA with zero disorder equals the coherent result to 1e-12; its
  fixed point is converged below 1e-10 with `Im Sigma <= 0`; the
  central layer of a long chain matches the bulk SCBA equation to
  1e-3.
- The closed-device LDOS equals the eigenvalue sum to 1e-13 (and to
  1e-12 with overlap); bond currents obey Kirchhoff's law to 1e-7
  and every cut carries T to 1e-6.
- `principal_layers` reproduces hand-built blocks exactly and refuses
  layers that are too thin.

**Large systems, k-grids, interpolation, files, fitting**

- Sparse and dense matrices are identical; Lanczos matches the open
  chain's closed form to 1e-8 (with overlap too).
- KPM DOS: band-centre value within 1 %, integral within 0.5 %, zero
  outside the band with overlap. KPM conductivity: molecular line
  weight within 3 % of the closed form, and within 2 % of the dense
  route on a dimerized chain.
- The time-reversal fold reproduces DOS and conductivity to 1e-12 and
  the Drude weight to 1e-10; the six-fold fold of graphene reproduces
  the DOS to 1e-12 and the chemical potential to 1e-9, and wrong
  operations are refused. `find_point_group` finds 12, 8 and 2
  operations for the hexagonal, square and chain lattices.
- Fourier interpolation reproduces the bands to 1e-12 for five models
  and flags a too-coarse grid.
- A hand-written graphene `hr.dat` reproduces `graphene()` bands to
  1e-12 and its optical conductivity to 1e-10; a save/load round trip
  keeps each block to 1e-9 (the file stores 10 decimals); four kinds
  of corrupted file (too few lines, a duplicate element, a short
  degeneracy list, a non-Hermitian Hamiltonian) and a mismatched cell
  are refused.
- On the chain the fit covariance matches its closed form to 1e-6
  relative; 300 seeded simulated experiments match the reported error
  bars within 15 %; a design that cannot separate the parameters is
  refused.

## Bugs fixed in 0.10.1

**0.10.1 fixed six problems.**

- *Overlap doubled by on-site terms.* In a model with overlap, every
  on-site block added without its own `S_block` added another identity
  to `S`. Adding an on-site energy or a Zeeman term in a second
  `add_hop` call (as the `with_spin` docstring suggests) therefore
  changed `S` and gave wrong bands. Now each site's on-site overlap is
  the identity unless you give it explicitly.
- *Transport with overlap at finite `eta`.* The NEGF functions built
  the reverse coupling block as the conjugate transpose of `zS - H`,
  which is `conj(z) S^dag - H^dag` rather than `z S^dag - H^dag`. The
  error is of order `eta * S`: negligible at the default `eta = 1e-6`
  for transmission, but visible in `device_greens` and `device_ldos`
  at the larger `eta` often used for a local DOS.
- *`device_ldos` ignored the inter-layer overlap* (`coup_S`). With
  overlap the LDOS is `-Im (G S)_ii / pi` (the Mulliken convention,
  which shares the overlap between the orbitals involved), so leaving
  out part of `S` made the orbital LDOS not add up to the total DOS.
- *`with_spin` dropped dipole blocks*, so a spinful copy lost the
  on-site transitions set with `set_dipole`.
- *`carrier_count` without `mesh` or `kpts`* summed a periodic model at
  `k = 0` only (2.0 instead of 1.0 for the half-filled chain). It now
  refuses, like the other functions.
- *Temperature.* `drude_weight` returned NaN at `T = 0` and a negative
  weight at `T < 0`; the other functions accepted `T < 0`. Both are now
  refused.

Earlier README and CHANGELOG text also overstated some checks; the
0.10.1 entry of [CHANGELOG.md](CHANGELOG.md) lists the corrections.
The full history is in the CHANGELOG.

## Limits

These are deliberate choices, not oversights:

- The velocity uses the site-diagonal position approximation; the
  intra-atomic part enters only through the dipole blocks you supply.
- `sigma_optical` sums transitions between bands only (pairs closer
  than 1 meV are left out); add `drude_weight` for metals.
- `symmetry_fold` is valid for results that depend on eigenvalues
  only (DOS, chemical potential, band edges, carrier count), not for
  direction-dependent ones such as `sigma_xx` or the Drude weight.
- The SCBA is elastic (disorder only) and has no vertex corrections
  (the next-order correction to the disorder-averaged conductance);
  inelastic electron-phonon scattering is not implemented.
  Electron-electron interaction enters only as a self-energy you
  supply; there is no mean-field or GW self-consistency.
- `bond_currents` works in an orthogonal basis only (it takes no
  overlap arguments).
- `kpm_sigma` covers orthogonal, finite models and the longitudinal
  response only. A KPM Hall conductivity of a finite system is not
  offered: in this position convention it is zero for any bounded
  system (`Im Tr[PxQy] = 0`, itself a test), so `chern_marker` is the
  finite-system alternative.
- `fourier_interpolation` is the Fourier step of Wannier
  interpolation; it does not build maximally localized Wannier
  functions (Marzari and Vanderbilt, Phys. Rev. B 56, 12847 (1997), is
  not implemented).
- The quantum-geometry functions and the Berry curvature dipole
  handle orthogonal bases only.
- The package ships no material constants. The only physical constant
  in the code is the Boltzmann constant (CODATA 2018).

## Relation to existing tools

Excellent tools cover parts of this space: [PythTB](https://www.physics.rutgers.edu/pythtb/) and [pybinding](https://docs.pybinding.site/) build tight-binding models and their spectra, and [Kwant](https://kwant-project.org/) is the standard for quantum transport. hamop does not replace any of them, and for their core use cases they are more capable. Its niche is the combination they leave open: overlap matrices handled across the observables listed above, optics computed from the same Bloch matrices as the spectrum, and a small NumPy/SciPy-only core checked against closed forms.

## Where it comes from

Methodological basis:

> "Learning the quantum Hamiltonian of defective monolayer MoS2
> reveals collective vacancy brightness decoupled from defect count";
> code for the paper:
> https://github.com/Tanvir-Mahmud-Mahim/mos2-vacancy-optics

That study computes the optics, the electronic structure and the
transport of vacancy-disordered MoS2 supercells from one
density-functional Hamiltonian, so that a defect configuration's
optical and electronic signatures are consistent, and its conclusions
depend on that consistency. This package is the general-purpose
engine distilled from that pipeline: the same observables for any
Hamiltonian a user supplies, with the material-specific machinery
(DFT extraction, machine-learned Hamiltonians, MoS2 structures) left
in the paper repository.

## Citing, support and license

If `hamop` helps your work, please cite it with the concept DOI
[10.5281/zenodo.22311381](https://doi.org/10.5281/zenodo.22311381),
which always resolves to the latest release; every release is
archived on Zenodo. Citation metadata is in
[CITATION.cff](CITATION.cff).

The package is written and maintained by Tanvir Mahmud Mahim
(Department of Electrical and Electronic Engineering, BRAC University),
who reviews every change and takes the final decision on scope and
releases. There is no separate governance body; design questions are
discussed in the open in issues and pull requests, and the standing
rule of [CONTRIBUTING.md](CONTRIBUTING.md) binds the maintainer exactly
as it binds contributors: a change that touches physics arrives with a
test, and a constant arrives with its source.

Support runs through the issue tracker at
https://github.com/TaN-MM-Org/hamop/issues. Usage questions are welcome
there alongside bug reports; a docstring that left a unit or a sign
convention unclear is treated as a documentation bug, not as user
error. The maintainer aims to respond within a week.

While the version is below 1.0 the API may still move between minor
versions; such changes are called out in the release notes.

Licensed under Apache-2.0 (see [LICENSE](LICENSE)).
