# Changelog

Every physical claim added in any release is pinned by a test against
an exact result; the release notes on GitHub carry the full anchor
lists. Versions below 1.0 may move the API between minor versions;
such changes are called out here and in the release notes.

## v0.11.0 - 2026-09-30

Finite-temperature transport, a one-file Wannier90 import, and three
silent wrong answers turned into errors or correct numbers.

### Added

- `landauer_conductance`, `landauer_current`, `thermoelectric` (new
  module `hamop.landauer`): conductance (units of e^2/h), current at a
  finite bias (units of e^2/h times 1 V), Seebeck coefficient (V/K),
  electronic heat conductance at zero current ((e^2/h) V^2/K) and
  Lorenz ratio (V^2/K^2) from a transmission T(E) sampled on an
  energy grid, by trapezoid sums of the Landauer integrals (Datta,
  Electronic Transport in Mesoscopic Systems (1995), ch. 2; Sivan and
  Imry, Phys. Rev. B 33, 551 (1986)). The grid is checked: it must
  reach 30 kT beyond every chemical potential and have steps of at
  most kT there. No constant beyond the Boltzmann constant is used.
- `load_wannier90_tb`, `from_wannier90_tb`: read the Wannier90
  `seedname_tb.dat` file (lattice vectors, H(R) and position matrix
  elements; layout from the Wannier90 user guide and the writer
  `hamiltonian_write_tb`). The model takes the cell and the orbital
  centres (diagonal position elements at R = 0) from the file; both
  H(R) and r(R) are divided by the degeneracy of R. The off-diagonal
  position elements are returned by the loader but not used (the
  package's optics use the site-diagonal position operator).
  Trailing lattice directions without hopping are dropped (a 2D
  material from a 3D code imports as 2D), and a cell that cannot be
  cut that way without changing its geometry is refused.
- `hamop.model.fermi_dirac` and `hamop.model.check_kgrid`: the one
  Fermi function and the one k-weight check that the spectrum, Kubo
  and KPM modules now share.

### Fixed

- `sancho_rubio` could return a wrong surface Green function without
  warning. (1) It returned the last iterate when the decimation had
  not converged after `maxiter` steps, which always happens at
  `eta = 0` inside a lead band. (2) When E equals an eigenvalue of the
  lead layer (E = 0 for a chain with zero on-site energy) the first
  decimation steps divide by a number of size eta; with a small eta
  the decimation loses up to log10(1/eta) digits and can pass its own
  convergence test at a wrong answer. Every result is now checked
  against the lead's Dyson equation g = (A - a g b)^-1 (relative
  residual <= `check_tol` = 1e-10, new keyword) and for retardedness
  (i (g - g^dag) has no negative eigenvalues); a decimation that fails
  is recomputed from the lead's decaying modes (generalized
  eigenproblem of b + A lam + a lam^2 = 0, mode-matching form of Lee
  and Joannopoulos, Phys. Rev. B 23, 4988 and 4997 (1981)), which
  must pass the same checks. If neither route passes, RuntimeError is
  raised.
- At `T = 0` a level exactly at `mu` gave (e - mu)/(kB T) = 0/0 = NaN
  in `sigma_optical`, `sigma_tensor` and `carrier_count`, with
  divide-by-zero warnings at every `T = 0` call; `fermi_level` and
  `kpm_sigma` used the same expression. `T = 0` is now the exact step, with occupation 1/2 at `mu`.
- Explicit k-point weights were used unchecked: a weight list shorter
  than the k-point list was truncated by `zip`, and weights not
  summing to 1 rescaled every result. `dos`, `fermi_level`,
  `band_edges`, `sigma_optical`, `sigma_tensor`, `drude_weight` and
  `carrier_count` now refuse wrong-length, negative, non-finite or
  unnormalized (|sum - 1| > 1e-6, loose enough for single-precision
  weights) weights. `band_edges` ignores weights, so it does not
  check them.

### Behaviour changes

- `sancho_rubio(0.5, *chain_lead_blocks(t=-1.0), eta=0.0)`: before
  0.9236 + 0i (exact: 0.25 - 0.9682i), and
  `transmission([0.5], ..., eta=0.0)` gave 0.0 inside the band; now
  RuntimeError.
- Chain lead at the band centre E = 0: with `eta = 1e-8` the surface
  Green function was -6.7e7 i, now -0.999999995 i (closed form at the
  same complex energy: -(1 - eta/2) i); with the default `eta = 1e-6`
  it was -0.99996589 i, now -0.99999950 i. The impurity chain of
  README example 4 (eps = 0.8): T(0) at `eta = 1e-8` was 1.4e-15 and
  T(8.9e-16) was 0.9745, both now 0.86206894 (eta -> 0 limit
  0.86206897); at `eta = 1e-6` T(0) was 0.86207437, now 0.86206638.
  Where the decimation passes the Dyson check (every other energy in
  the test suite) results are unchanged.
- Two-level atom with a level at `mu`, `T = 0` (the test model):
  `carrier_count` before NaN, now 1.0; `sigma_optical` at 1.0 eV
  before NaN, now 49.13 (equal to the `T = 1 K` value to 1e-12).
  At `T = 0` with no level exactly at `mu`, occupations change from
  1/(1 + e^60) (about 9e-27) to exactly 0 or 1; nothing else changes
  at `T > 0` (the same formula is used).
- `dos(linear_chain(), [2.0], kpts=k, weights=w[:3])` with the
  10-point grid: before 0.0 (correct value 0.798), now ValueError;
  with `weights=2*w`: before 1.596, now ValueError.

### Tests

- New `tests/test_landauer.py` (6 tests), `tests/test_wannier_tb.py`
  (3 tests) and `tests/test_regressions_0110.py` (9 tests; each fix is
  pinned by a check that fails on 0.10.1). 171 tests in total (153
  before).

## v0.10.1 - 2026-09-22

Bug fixes, a rewritten README, and CI coverage of every supported
Python version.

### Fixed

- Overlap models: an on-site block added without its own `S_block`
  added another identity to the on-site overlap each time. Adding an
  on-site energy or a Zeeman term in a second `add_hop` call (for
  example after `with_spin`) therefore changed S and gave wrong bands.
  Each site's on-site overlap is now the identity unless an on-site
  `S_block` is given for it (`TightBindingModel` and the
  periodic-gauge assembly used by the Berry functions).
- NEGF with overlap: the reverse coupling block of z S - H was built as
  the conjugate transpose of z S - H, i.e. with conj(z). It is now
  z S^dag - H^dag in `sancho_rubio`, the lead self-energies,
  `transmission`, `transmission_direct`, `transmission_sparse`,
  `buttiker_transmission`, `scba_transmission`,
  `multiprobe_transmission`, `device_greens` and `device_ldos`. The
  error was of order eta * S (negligible at the default eta = 1e-6,
  visible at the finite eta used for local DOS). Orthogonal models
  are unaffected.
- `device_ldos` ignored the inter-layer overlap blocks (`coup_S`) in
  its Mulliken LDOS; they are now included.
- `with_spin` dropped intra-atomic dipole blocks; they are now copied
  as kron(X, identity) per direction.
- `carrier_count` on a periodic model without `mesh` or `kpts`
  silently summed k = 0 only; it now refuses, like `dos` and
  `sigma_optical`.
- `drude_weight` returned NaN at T = 0 and a negative weight at
  T < 0; it now refuses T <= 0. `sigma_optical`, `sigma_tensor`,
  `carrier_count`, `fermi_level` and `kpm_sigma` now refuse T < 0.
- Stale text: the `with_peierls` error message and module docstring
  said magnetic unit cells were not implemented (they are:
  `magnetic_supercell`); the `set_dipole` docstring said overlap was
  refused in the optics (it is supported); the `bond_currents`
  docstring said overlap was "refused" (the function simply takes no
  overlap arguments).

### Tests

- New `tests/test_regressions.py` with six tests, each failing on
  0.10.0: `test_onsite_blocks_in_two_calls_do_not_double_the_overlap`,
  `test_with_spin_keeps_the_dipole_blocks`,
  `test_device_ldos_uses_the_inter_layer_overlap`,
  `test_carrier_count_refuses_a_periodic_model_without_a_grid`,
  `test_temperatures_the_fermi_factors_cannot_use_are_refused`,
  `test_negf_with_overlap_builds_z_s_minus_h_at_finite_eta`.
  153 tests in total (147 before).
- CI now runs Python 3.10 as well (3.9 to 3.14), plus an
  `oldest-dependencies` job with NumPy 1.22.0 and SciPy 1.8.0 on
  Python 3.10.

### Changed

- README rewritten for non-specialists: a word guide, eight examples
  with their exact output (checked by running them), every public
  name, the refusals, and the checks with the tolerances the tests
  actually use.

### Corrections to earlier notes

- The 0.10.0 README said "147 tests across Python 3.9 through 3.13";
  CI then ran 3.9 and 3.11-3.14 (not 3.10). Its Status section still
  said "v0.8.0" and "136 tests".
- Earlier text called several checks "exact" or "machine precision"
  where the tests use tolerances: the Wannier90 save/load round trip
  is checked to 1e-9 per block (the file stores 10 decimals), not
  exactly (0.7.0); the imported model's optical conductivity to 1e-10
  (0.7.0); the chain fit covariance to 1e-6 relative, not "exactly"
  (0.9.0); the C6-folded chemical potential to 1e-9, not 1e-12, and
  the time-reversal-folded Drude weight to 1e-10, not 1e-12.
- 0.7.0 said "every refusal fires on a deliberately corrupted file";
  the tests cover four kinds of corrupted file and a mismatched cell.
- 0.8.0 said the along-bond dipole component "cancels to 1e-8"; the
  test checks it below 1e-8 times the perpendicular component.
- 0.6.0 said nonorthogonal bases are "refused" by `bond_currents`; no
  error is raised, the function has no overlap arguments.

## v0.10.0 - 2026-09-18

Quantum geometry, and a future-proofing pass.

- `geometry.quantum_geometric_tensor` / `quantum_metric` /
  `quantum_weight`: the gauge-invariant quantum geometric tensor of
  the occupied subspace via projector finite differences -- metric,
  Berry curvature (in the package's own plaquette sign convention,
  fixed by an independent-path identity), and the integrated
  quantum-weight tensor with the bound tr K >= |Chern| (Provost &
  Vallee (1980); Peotta & Torma, Nat. Commun. 6, 8944 (2015);
  Onishi & Fu, PRX 14, 011052 (2024)).
- CI now also runs on Python 3.14.
- Anchors: three independent code paths agree (projector QGT,
  two-band Bloch-sphere closed forms, plaquette Chern number); the
  exact Gram-matrix chain tr g >= 2 sqrt(det g) >= |Omega| asserted
  pointwise; the integrated bound in both Haldane phases; metric
  symmetry and positive semidefiniteness; refusals at band
  crossings and for overlap models.

## v0.9.0 - 2026-09-17

Lab adaptability: fit the model to measured bands, and plan the
measurement first.

- `lab.fit_bands`: weighted least-squares fit of any user
  parameterization (a builder theta -> `TightBindingModel`) to
  measured band energies at known k-points, with the standard
  (J^T W J)^-1 covariance, chi-squared when measurement errors are
  supplied, and refusals of non-identifiable designs (scale-invariant
  rank test, so mixed units cannot fake a degeneracy).
- `lab.band_information`: the same matrix before any data exist --
  predicted error bars and an identifiability verdict for a planned
  (k-point, band) set.
- `lab.design_kpoints`: greedy D-optimal choice of the most
  informative k-points (Pukelsheim, Optimal Design of Experiments,
  SIAM (2006)).
- Anchors: on the linear chain the model is linear in (t, e0), so the
  reported covariance equals the textbook closed form
  sigma^2 (X^T X)^-1 with X = [2 cos(ka), 1], exactly; noiseless fits
  recover the truth; 300 seeded Monte-Carlo experiments match the
  reported error bars; a same-cos(ka) design is exactly rank-one and
  refused by planner and fit alike; the greedy design never loses to
  a random subset.

## v0.8.0 - 2026-09-13

Physics upgrade: the Berry curvature dipole -- the band-geometric
generator of the nonlinear Hall effect in time-reversal-symmetric
crystals (Sodemann and Fu, Phys. Rev. Lett. 115, 216806 (2015)).

- `berry_dipole`: D_alpha = sum_n int d^2k/(2 pi)^2 f_n dOmega_n/dk_alpha
  by two genuinely independent routes -- method="grad" (central
  differences of the band-resolved plaquette-flux field, works at
  kT = 0) and method="fermi" (the Fermi-surface form with the
  analytic Fermi-window derivative and Hellmann-Feynman velocities)
  -- with per-band contributions reported. Units: Angstrom (2D).
  Turning D into a nonlinear Hall voltage needs a scattering time
  and device geometry that are deliberately not guessed.
- `band_curvatures`: per-band lattice curvature density and energies
  on the mesh, via exact abelian differences of the tested
  Fukui-Hatsugai-Suzuki fluxes; the complete band set sums to zero
  identically, asserted.
- `fermi_occupation`: the occupation factor, exact step at kT = 0.
- Anchors, symmetry doing the verifying: inversion symmetry
  (Haldane at zero mass) cancels D exactly on the mesh; completely
  filled bands carry exactly zero dipole whatever the symmetry (the
  BZ average of a gradient); time reversal + C3 (unstrained gapped
  graphene) forces D -> 0 with mesh refinement while one scaled bond
  (broken C3) converges to a nonzero dipole an order of magnitude
  above the trigonal residual -- the strain-switch that makes the
  nonlinear Hall effect a strain probe; a single mirror line pins D
  perpendicular to itself (component along the bond cancels to 1e-8);
  and the two integral forms agree where the answer is nonzero.
- Honest limits: orthogonal bases only (refused for overlap models
  rather than dropping the dS terms), and the curvature field is
  computed in the atomic frame deliberately -- the Loewdin frame's
  periodic gauge identification preserves Chern totals but
  redistributes flux locally, which a first moment feels; the C3
  anchor is what caught this, and the docstring records it.

## v0.7.0 - 2026-09-12

Real-materials release: the standard interchange format for
first-principles-derived tight-binding Hamiltonians now imports in
one line.

- `from_wannier90` / `load_wannier90_hr` / `save_wannier90_hr`:
  strict parsing of the Wannier90 ``seedname_hr.dat`` format
  (degeneracy weights divided per the convention), model construction
  with user-supplied lattice vectors and optional Wannier centres
  (hr.dat carries neither; they come from you, stated rather than
  guessed -- centres are exactly irrelevant for eigenvalue-derived
  quantities and matter only for the optical matrix elements), and
  an exact write/read round trip. Refusals with explanations: wrong
  element counts, duplicate or missing elements, truncated
  degeneracy lists, a cell dimension that does not match the
  R-vectors, and real-space Hermiticity violations
  (H(R) != H(-R)^dagger), which are never symmetrized silently.
- Anchors: a graphene hr.dat written independently of the parser
  reproduces the closed-form-anchored `lattices.graphene` bands at
  random k to 1e-12, closes the gap at the Dirac point, and gives
  the identical Kubo optical conductivity through the imported
  model; centres shift changes no eigenvalue (gauge invariance,
  asserted); the round trip preserves H(k) against an independent
  direct Bloch sum; degeneracy division is checked entry by entry;
  and every refusal fires on a deliberately corrupted file.
- README: a "Real materials in" section; status counts updated.

## v0.6.0 - 2026-09-10

Local spectroscopy and current imaging: the NEGF device becomes
spatially resolvable, with every new observable anchored to an exact
identity.

- `device_greens`: full retarded device Green function with embedded
  lead broadenings; held to the exact finite-eta spectral identity
  i(G - G^dag) = G (GamL + GamR + 2 eta) G^dag at ANY eta (machine
  precision, asserted).
- `device_ldos`: orbital-resolved LDOS of the open (or closed)
  device; the closed-device case equals the exact Lorentzian
  eigen-sum to 1e-13, the clean-ribbon case is translation-uniform,
  and the open case is positive.
- `bond_currents`: zero-temperature bond-current map for left-lead
  injection (Paulsson and Brandbyge, PRB 76, 115117 (2007)),
  antisymmetric and oriented; Kirchhoff holds at every interior
  orbital, EVERY inter-layer cut sums to the independently computed
  Caroli transmission, the contact layers inject/drain exactly +T/-T,
  and the map vanishes outside the lead band. Nonorthogonal bases
  are refused rather than approximated.

## v0.5.0 - 2026-09-05

- Hofstadter magnetic supercells for periodic 2D models at rational
  flux (`magnetic_supercell`), with a self-validating gauge check;
  anchored on exact zero-flux band folding, the pi-flux square-lattice
  closed form, and the TKNN consistency of the lowest Hofstadter
  band's Chern number with `sigma_tensor` on the same magnetic cell.
- Automatic point-group detection (`find_point_group`) by exact
  lattice-automorphism enumeration filtered through the spectral
  check; hexagonal (12), square (8) and chain (2) group orders pinned.
- Multi-probe dephasing network (`multiprobe_transmission`,
  D'Amato-Pastawski): exact current conservation, exact single-probe
  reduction to the Buttiker formula, Ohmic length scaling.
- Real-space topology for finite systems: the Bianco-Resta local
  Chern marker (`chern_marker`), whose whole-system total vanishes
  identically and whose bulk average reproduces the periodic Chern
  number. A finite-system KPM Hall conductivity is deliberately NOT
  offered: the tests prove Im Tr[PxQy] = 0 for any bounded system in
  the site-diagonal position formulation, so such a routine could
  only return broadening artifacts.
- Intra-atomic dipoles in nonorthogonal bases (the eigenstate
  operator identity makes the orthogonal expression exact there too),
  and in `kpm_sigma` through the exact sparse operator i(HX - XH).
- Fourier (Wannier-style) band interpolation
  (`fourier_interpolation`): exact to machine precision when the
  hopping range fits the sampling window, undersampling detected by a
  residual check.
- CI now tests Python 3.9, 3.11, 3.12 and 3.13.

## v0.4.0 - 2026-09-04

- Uniform magnetic fields on finite models by Peierls substitution
  (`with_peierls`): machine-precision ring spectra, gauge invariance,
  plaquette flux and flux-quantum periodicity; Landau-level anchor.
- Verified point-group k-mesh folding (`symmetry_fold`) for spectral
  observables; refuses unverified operations.
- Elastic self-consistent Born disorder self-energy
  (`scba_transmission`), cross-checked against the independent bulk
  scalar SCBA equation of the chain.
- Atomic-frame Berry phases (frame="atomic"): a second link
  convention; Chern integers agree exactly with the Loewdin frame.
- KPM for nonorthogonal bases (sparse LU of S) and the KPM
  Kubo-Greenwood conductivity (`kpm_sigma`); sparse Berry solver;
  sparse-LU device transmission (`transmission_sparse`).
- Intra-atomic dipole velocity term (`set_dipole`), anchored on the
  hand-derived atomic s->p line.

## v0.3.0 - 2026-09-04

- Complex interband conductivity tensor `sigma_tensor` including the
  finite-frequency Hall component; sigma_xy(0) = C e^2/h against the
  package's own Chern number, sign included (TKNN).
- Nonorthogonal topology through the smooth Loewdin frame.
- Spin as a first-class convention (`with_spin`, `PAULI`,
  `kane_mele`).
- Per-layer interaction self-energies and the current-conserving
  Buttiker dephasing probe.
- Sparse assembly, Lanczos low-energy bands, KPM density of states.
- Time-reversal k-mesh folding.

## v0.2.0 - 2026-09-04

- Wilson-loop Berry phases, lattice Berry curvature and Chern numbers
  (Fukui-Hatsugai-Suzuki); Haldane phase diagram and SSH Zak anchors.
- Intraband Drude weight; Lorentzian lineshape option.
- Verified automatic principal-layer partitioning
  (`principal_layers`); `k_path`; `ssh` and `haldane` builders.

## v0.1.1 - 2026-09-04

- Version bump for the first Zenodo-archived release (concept DOI
  10.5281/zenodo.22311381).

## v0.1.0 - 2026-09-04

- Model container with exact k-derivatives (atomic gauge),
  canonical-orthogonalization eigensolver, band structures, DOS,
  chemical potentials, Kubo-Greenwood optical conductivity in units
  of e^2/(4 hbar), Sancho-Rubio surface Green functions, recursive and
  direct-inversion Landauer transmission; 23 closed-form tests.
