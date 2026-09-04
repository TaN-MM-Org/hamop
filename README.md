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
- **Optics** (`sigma_optical`, `sigma_tensor`, `drude_weight`):
  Kubo-Greenwood real sheet conductivity in units of e²/(4ℏ) with
  Gaussian or Lorentzian broadening, the full complex interband
  conductivity tensor σ_ab(ω) — including the finite-frequency Hall
  component σ_xy(ω), whose ω → 0 limit reproduces the TKNN
  quantization σ_xy = C e²/h against the package's own Chern number,
  sign included — plus the intraband (Drude) weight. All built on the
  nonorthogonal velocity correction
  `v = dH/dk − (eₙ+eₘ)/2 dS/dk` that makes them exactly invariant
  under a shift of the energy zero. Intra-atomic dipole blocks
  (`model.set_dipole`) add the on-site velocity contribution
  i(Eₙ−Eₘ)X_nm, restoring transitions the site-diagonal position
  approximation leaves dark — validated against a hand-derived atomic
  s→p line (orthogonal bases only, refused otherwise).
- **Topology** (`berry_phase`, `berry_curvature`, `chern_number`):
  Wilson-loop Berry phases and the gauge-invariant lattice field
  strength of Fukui, Hatsugai and Suzuki (J. Phys. Soc. Jpn. 74, 1674
  (2005)), whose Brillouin-zone sum is an exact integer — the Chern
  number. Nonorthogonal bases are handled in *two* conventions: the
  smooth Löwdin frame d = S(k)^½ c (a bundle isomorphism under which
  the Chern number is invariant), and the atomic frame, whose links
  carry the midpoint overlap metric implied by the same site-diagonal
  position operator the velocity uses. Berry-phase *values* are
  convention-dependent; the integers must agree between frames and are
  asserted to. Large cells with few occupied bands can use
  solver="sparse".
- **Magnetic fields** (`with_peierls`): uniform out-of-plane fields on
  finite models by Peierls substitution (Peierls, Z. Phys. 80, 763
  (1933)); the midpoint line integral is exact for linear gauges, so
  gauge invariance, ring flux spectra, plaquette fluxes and flux-
  quantum periodicity all hold to machine precision, not
  approximately.
- **Spin** (`with_spin`, `PAULI`, `kane_mele`): spin doubling as a
  stated convention (spin innermost, blocks tensored with Pauli
  matrices), so Zeeman and intrinsic spin-orbit terms are ordinary
  hopping blocks; the Kane-Mele model ships as the canonical
  spin-orbit anchor.
- **Transport** (`sancho_rubio`, `transmission`,
  `transmission_direct`, `transmission_sparse`, `principal_layers`,
  `buttiker_transmission`, `scba_transmission`): two-probe Landauer
  transmission with Sancho-Rubio lead surface Green functions and a
  recursive Green function sweep, nonorthogonal bases included, plus
  dense and sparse-LU reference implementations of the same quantity —
  and automatic partitioning of a finite model into principal layers,
  which *verifies* that no coupling skips a layer instead of silently
  truncating it. User-supplied retarded interaction self-energies
  Σ(E) can be attached per layer; a current-conserving Büttiker
  dephasing probe (Phys. Rev. B 33, 3020 (1986)) is built in; and
  `scba_transmission` iterates the elastic self-consistent Born
  self-energy for uncorrelated on-site disorder,
  Σᵢ = W² diag(Gᵢᵢ), to a verified fixed point — cross-checked
  against the independent bulk scalar SCBA equation of the chain.
- **Sparse / large systems** (`bloch_sparse`,
  `bloch_derivative_sparse`, `lowest_bands`, `kpm_dos`, `kpm_sigma`):
  CSR assembly of the identical Bloch matrices and their
  k-derivatives, Lanczos diagonalization of just the low-energy window
  (generalized eigenproblem included), the kernel polynomial method
  for the density of states — nonorthogonal bases included, via a
  sparse LU of S — and the KPM Kubo-Greenwood optical conductivity
  from the double Chebyshev expansion of the velocity-velocity
  spectral density (Weisse et al., Rev. Mod. Phys. 78, 275 (2006)),
  deterministic or stochastic trace.
- **k-mesh reduction**: `monkhorst_pack(mesh, time_reversal=True)`
  folds k with −k for k-even observables, roughly halving the work
  (offered only when every real-space block is real, refused
  otherwise); `symmetry_fold(model, mesh, ops)` folds by a
  user-supplied point group, verifying *before* folding both that each
  operation maps the reciprocal lattice to itself and that it actually
  leaves the spectrum invariant at random test k-points — a
  non-symmetry is refused, never silently averaged. Valid for
  spectral observables (DOS, fillings, band edges), stated plainly.
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
  refuses a layer width smaller than the interaction range;
- σ_xy(0) of the gapped Haldane model equals its Chern number times
  e²/h (TKNN; Phys. Rev. Lett. 49, 405 (1982)) to 10⁻⁶, **sign
  included**, computed by two independent routes through the package
  (Kubo tensor vs. lattice field strength); it vanishes in the trivial
  phase, and the tensor is antisymmetric to machine precision;
- the Chern number survives a nonorthogonal deformation of the basis
  unchanged (the Löwdin frame is a bundle isomorphism), and the
  overlap-SSH chain keeps its quantized Zak phases with the exact π
  difference;
- the Kane-Mele model equals two Haldane copies to machine precision,
  its spin-orbit gap at K is exactly 6√3 λ_so, its total Chern number
  vanishes and its spin sectors carry ±1 (Kane and Mele, Phys. Rev.
  Lett. 95, 226801 (2005)); spin doubling is an exact double
  degeneracy, and a Zeeman term splits it by exactly 2B;
- a constant self-energy on one layer reproduces the impurity closed
  form; the recursive sweep with complex Σ(E) agrees with direct
  inversion to machine precision; the Büttiker probe at γ = 0 is the
  coherent result exactly, matches the hand-written scalar closed form
  on a single-site device, and suppresses the double-barrier
  resonance;
- the time-reversal-folded k-mesh reproduces full-grid DOS, σ(ω) and
  Drude weight to 10⁻¹² with roughly half the points, and refuses
  complex-block models;
- the sparse assembly equals the dense assembly element for element;
  the Lanczos window reproduces the open chain's closed form
  2t cos(πj/(N+1)) (nonorthogonal variant included); the KPM density
  of states matches the chain's closed form at the band center and
  integrates to the orbital count — in the nonorthogonal chain too,
  where the band-center DOS is again 1/(2π|t|) and the band edges sit
  at 2t/(1±2s), outside of which the KPM DOS vanishes identically;
- the atomic-frame links return the same exact Chern integers as the
  Löwdin frame (orthogonal and overlap Haldane, both phases), the
  atomic-frame Zak phases of the orthogonal SSH chain are ∓π/2 with
  the exact π difference, and inversion antisymmetry of the Zak pair
  survives the overlap; the sparse Berry solver returns the same
  integers as the dense one (C = 2 on stacked Haldane copies);
- a flux-threaded ring reproduces 2t cos((2πj + Θ)/N) to machine
  precision, the Landau and symmetric gauges give identical spectra to
  10⁻¹², the plaquette-flux product is exactly e^{2πiφ}, the spectrum
  is exactly periodic in the flux quantum, and the lowest Landau level
  of the square lattice sits at −4|t| + ħω_c/2 with ħω_c = 4π|t|φ to
  3%, macroscopically degenerate;
- the dark on-site s→p transition acquires exactly the hand-derived
  peak spin·4π(Δd)²/(η√(2π)Δ) once the dipole block is set, and a
  dipole that commutes with H changes nothing identically;
- SCBA: W² = 0 is the coherent result exactly; the converged Σ
  satisfies its own equation below 10⁻¹⁰ with Im Σ ≤ 0; the central
  layer of a long chain reproduces the independent bulk scalar SCBA
  fixed point to 10⁻³;
- the C6-folded graphene grid (≈ ×6 fewer points) reproduces
  full-grid DOS and chemical potentials to 10⁻¹²; the same fold works
  on the time-reversal-broken Haldane model (C6 is still a spectral
  symmetry, verified not assumed); a 90° rotation on the hexagonal
  lattice and a C6 request on a bond-stretched model are both refused;
- the KPM conductivity's integrated molecular line weight matches the
  kernel-independent closed form spin·4π(at)²/(2|t|) to 3%, and the
  KPM route agrees with the dense eigenpair Kubo route on a dimerized
  chain to 2%; `transmission_sparse` equals dense direct inversion to
  machine precision, overlap and complex Σ(E) included.

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

v0.4.0 (alpha). Implemented and tested: the model container with exact
k-derivatives and intra-atomic dipole blocks,
canonical-orthogonalization eigensolver, band structures and k-paths,
densities of states, filling-resolved chemical potentials, band edges;
Kubo-Greenwood optical conductivity (Gaussian or Lorentzian
broadening, dipole term included), the complex interband conductivity
tensor σ_ab(ω) including the finite-frequency Hall component, and the
intraband Drude weight; Wilson-loop Berry phases, lattice Berry
curvature and Chern numbers in orthogonal and nonorthogonal bases,
in two frame conventions (Löwdin and atomic), dense or sparse solver;
spin doubling, Pauli-block spin-orbit terms and the Kane-Mele
builder; uniform magnetic fields on finite models by Peierls
substitution; Sancho-Rubio surface Green functions, recursive,
direct-inversion and sparse-LU Landauer transmission, verified
automatic principal-layer partitioning, per-layer interaction
self-energies, the Büttiker dephasing probe and the elastic
self-consistent Born (SCBA) disorder self-energy; time-reversal and
verified point-group k-mesh folding; sparse Bloch and velocity
assembly, Lanczos low-energy bands, KPM densities of states
(nonorthogonal included) and the KPM optical conductivity.

Not yet implemented, stated plainly: magnetic fields in *periodic*
systems (magnetic unit cells / Hofstadter physics — `with_peierls` is
finite-only and refuses otherwise); inelastic (Keldysh)
electron-phonon SCBA — the SCBA here is elastic, disorder-type, and
the disorder-averaged conductance carries no vertex corrections;
automatic space-group detection — `symmetry_fold` verifies
user-supplied operations, it does not find them, and its folded grids
serve spectral observables only; the Hall component and intra-atomic
dipoles in the *sparse* (KPM) conductivity; intra-atomic dipoles in
nonorthogonal bases (refused explicitly); and Wannier interpolation
of any kind.

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
