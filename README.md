# hamop

[![Tests](https://github.com/TaN-MM-Org/hamop/actions/workflows/ci.yml/badge.svg)](https://github.com/TaN-MM-Org/hamop/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/hamop?label=PyPI&color=blue&cacheSeconds=3600)](https://pypi.org/project/hamop/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22311381-blue)](https://doi.org/10.5281/zenodo.22311381)

**One tight-binding Hamiltonian, every observable, strictly
consistent.** Build a Hamiltonian once, as real-space blocks in an
orthogonal or nonorthogonal basis, and compute its band structure,
density of states, Kubo-Greenwood optical conductivity and Landauer
(NEGF) transmission from the same matrices.

The point of the package is the consistency, not any single solver.
When the optics of a model and its spectrum are computed by different
codes with different conventions, they drift: a different gauge for the
velocity operator, a different treatment of the overlap matrix, a
different broadening, and suddenly the absorption edge no longer sits
at the band gap. Here every observable diagonalizes the same Bloch
matrices through the same canonically orthogonalized solver, and the
Kubo velocity operator is built from the exact k-derivative of the same
assembly, so spectral, optical and transport statements about one model
cannot disagree with each other.

## What it does

- **`TightBindingModel`**: sites with any number of orbitals, directed
  hopping blocks with automatic Hermitian completion, optional overlap
  blocks (LCAO-style nonorthogonal bases), periodic in any dimension or
  finite. Assembles H(k), S(k) and their exact k-derivatives in the
  atomic gauge.
- **Spectrum** (`bands`, `dos`, `fermi_level`, `band_edges`,
  `k_path`): band structures along arbitrary k-lists or interpolated
  high-symmetry paths, Gaussian-broadened densities of states, chemical
  potential at a given filling by bisection, band edges and gap about a
  chemical potential.
- **Optics** (`sigma_optical`, `drude_weight`): Kubo-Greenwood real
  sheet conductivity in units of e²/(4ℏ) with Gaussian or Lorentzian
  broadening, plus the intraband (Drude) weight — both built on the
  nonorthogonal velocity correction
  `v = dH/dk − (eₙ+eₘ)/2 dS/dk` that makes them exactly invariant
  under a shift of the energy zero.
- **Topology** (`berry_phase`, `berry_curvature`, `chern_number`):
  Wilson-loop Berry phases and the gauge-invariant lattice field
  strength of Fukui, Hatsugai and Suzuki (J. Phys. Soc. Jpn. 74, 1674
  (2005)), whose Brillouin-zone sum is an exact integer — the Chern
  number. Orthogonal bases only for now, refused explicitly otherwise.
- **Transport** (`sancho_rubio`, `transmission`,
  `transmission_direct`, `principal_layers`): two-probe Landauer
  transmission with Sancho-Rubio lead surface Green functions and a
  recursive Green function sweep, nonorthogonal bases included, plus a
  dense direct-inversion reference implementation of the same quantity
  — and automatic partitioning of a finite model into principal
  layers, which *verifies* that no coupling skips a layer instead of
  silently truncating it.
- **`gen_eigh`**: generalized eigensolver with canonical
  orthogonalization (Szabo and Ostlund, *Modern Quantum Chemistry*,
  sec. 3.4.5), so mildly overcomplete overlaps cannot blow up the
  spectrum — the standard remedy used inside electronic-structure
  codes.

Dependencies: NumPy and SciPy. Nothing else.

## Validation against closed forms

Every physical claim in the package is pinned by a test against an
exact result, not a stored number:

- the single-orbital chain reproduces E(k) = e₀ + 2t cos ka to machine
  precision, and its nonorthogonal variant reproduces
  E(k) = 2t cos ka / (1 + 2s cos ka);
- the chain density of states matches 1/(π√(4t² − E²)) and integrates
  to the orbital count;
- graphene's nearest-neighbour model gives Dirac-point closure at K
  exactly, ±3|t| at Γ exactly, and the **universal optical sheet
  conductivity e²/(4ℏ)** on the interband plateau (Kuzmenko et al.,
  Phys. Rev. Lett. 100, 117401 (2008)) — which is also the absolute
  anchor for the package's conductivity unit;
- the two-site molecule absorbs at exactly 2|t| with the hand-derived
  velocity matrix element |M| = |a t|;
- σ(ω) is invariant to 10⁻¹⁰ under H → H + cS with μ → μ + c, which
  pins the nonorthogonal velocity term;
- the chain's lead surface Green function matches its closed form
  (E − i√(4t² − E²))/(2t²); a pristine chain transmits exactly one
  channel inside the band and nothing outside; two decoupled chains
  transmit two; an on-site impurity ε reproduces
  T = (4t² − E²)/((4t² − E²) + ε²);
- the recursive Green function sweep agrees with dense direct inversion
  to machine precision, disorder and overlap included;
- the Haldane model returns its known phase diagram (Haldane, Phys.
  Rev. Lett. 61, 2015 (1988)) with the Chern number an **exact integer
  to 10⁻¹²**: ±1 inside the topological phase, 0 outside, sign
  reversal with the flux direction, and zero total over all bands;
- the SSH chain's Zak phase is quantized to 0 or π and the two
  dimerizations differ by exactly π — the convention-free statement;
- the Drude weight of the half-filled chain reproduces its closed form
  8·spin·|t|·a and is exactly invariant under a shift of the energy
  zero in a nonorthogonal basis;
- the automatic principal-layer partition reproduces hand-built blocks
  exactly, reproduces the single-impurity closed form end to end, and
  refuses a layer width smaller than the interaction range.

Run them yourself: `pip install -e .[test]` then `pytest`.

## Install and use

```
pip install hamop
```

```python
import numpy as np
from hamop import graphene, bands, dos, sigma_optical

g = graphene(t=-2.7, a=2.46)          # eV, Angstrom
omega = np.linspace(0.5, 2.0, 60)
sigma = sigma_optical(g, omega, mu=0.0, mesh=120, eta=0.12)
# sigma is ~1.0 on the plateau: the universal e^2/(4 hbar)
```

Building your own model:

```python
from hamop import TightBindingModel, band_edges

m = TightBindingModel(positions=[[0.0], [0.7]], norb=1, cell=[[2.0]])
m.add_hop(0, 1, (0,), [[-1.0]])       # intra-cell bond
m.add_hop(1, 0, (1,), [[-0.6]])       # inter-cell bond
print(band_edges(m, mu=0.0, mesh=2001))   # the SSH gap, 2|t1 - t2|
```

Conventions, stated once: energies in eV, positions in Angstrom, k in
1/Angstrom, Cartesian. Each directed hopping block is added once and
its Hermitian partner is implied. Optical conductivity is the real
sheet conductivity in units of e²/(4ℏ) with spin degeneracy as an
explicit factor (default 2). The velocity operator uses the standard
atomistic position gauge (position operator diagonal at the sites);
the intra-atomic dipole contribution is neglected, the common
approximation in tight-binding optics.

## Relation to existing tools

Excellent tools cover parts of this space: [PythTB](https://www.physics.rutgers.edu/pythtb/) and [pybinding](https://docs.pybinding.site/) build tight-binding models and their spectra, and [Kwant](https://kwant-project.org/) is the standard for quantum transport. hamop does not replace any of them, and for their core use cases they are more capable. Its niche is the combination they leave open: nonorthogonal (LCAO-style) overlap matrices as first-class citizens across *all* observables, optics and transport computed from the same Bloch assembly as the spectrum so the three can never disagree, and a deliberately small NumPy/SciPy-only core validated line by line against closed forms -- the shape of engine an LCAO electronic-structure pipeline exports its Hamiltonians into.

## Status

v0.2.0 (alpha). Implemented and tested: the model container with exact
k-derivatives, canonical-orthogonalization eigensolver, band
structures and k-paths, densities of states, filling-resolved chemical
potentials, band edges, Kubo-Greenwood optical conductivity (Gaussian
or Lorentzian broadening) and the intraband Drude weight for periodic
and finite systems, Wilson-loop Berry phases, lattice Berry curvature
and Chern numbers, Sancho-Rubio surface Green functions, recursive
plus direct-inversion Landauer transmission, and verified automatic
principal-layer partitioning.

Not yet implemented, stated plainly: k-space symmetry reduction (grids
are full Monkhorst-Pack), Berry phases in nonorthogonal bases (refused
with an explicit error), the finite-frequency Hall conductivity
σ_xy(ω) (the Chern number is implemented; the full Hall spectrum is
not), spin-orbit-coupled blocks as a first-class convention (complex
blocks work, but no helper), and interaction self-energies in the
transport module. Sparse or very large models are out of scope for
now: matrices are dense.

## Where it comes from

Methodological basis:

> "Learning the quantum Hamiltonian of defective monolayer MoS2
> reveals collective vacancy brightness decoupled from defect count";
> code for the paper:
> https://github.com/Tanvir-Mahmud-Mahim/mos2-vacancy-optics

That study computes the optics, the electronic structure and the
transport of vacancy-disordered MoS2 supercells from one
density-functional Hamiltonian, so that a defect configuration's
optical and electronic signatures are strictly consistent — and its
conclusions depend on that consistency. This package is the
general-purpose engine distilled from that pipeline: the same
observables for any Hamiltonian a user supplies, with the
material-specific machinery (DFT extraction, machine-learned
Hamiltonians, MoS2 structures) left in the paper repository.

## Support and governance

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
versions; such changes are called out in the release notes. The
limitations named under Status are deliberate scope, recorded there
precisely so that a user can tell a designed-out feature from an
oversight.

## License

Apache-2.0 (see [LICENSE](LICENSE)). Citation metadata is in
[CITATION.cff](CITATION.cff); every release is archived on Zenodo
under the concept DOI
[10.5281/zenodo.22311381](https://doi.org/10.5281/zenodo.22311381),
which always resolves to the latest version.
