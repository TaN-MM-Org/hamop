# Changelog

Every physical claim added in any release is pinned by a test against
an exact result; the release notes on GitHub carry the full anchor
lists. Versions below 1.0 may move the API between minor versions;
such changes are called out here and in the release notes.

## 0.13.0 (2026-10-17)

Two tools for a lab's own data, two bugs fixed, the k-point loops made
many times faster, and two tutorials. No public function was removed or
renamed; results of valid calls are unchanged except where a bug is
fixed below (and by floating-point rounding, at most 1e-12 relative,
where sums are now taken in a different order).

### Added

- `carrier_density(model, mu, n_neutral, ...)` and
  `chemical_potential_from_density(model, density, n_neutral,
  density_err=None, ...)` (new module `hamop.carriers`): the net carrier
  density per cm^dim at a chemical potential, counted from a stated
  neutral filling, and the inverse, with the interval that the
  density's error maps to (solved exactly at density -/+ error),
  `dn/dmu` and the quantum capacitance `e^2 dn/dmu` in F/cm^dim.
  Refused: finite models, densities the bands cannot hold, `T <= 0` in
  the inverse, and a k-mesh too coarse for the temperature (one state
  carrying more than 5 % of `dn/dmu`). Tests: the chain against the
  exact integral of its density of states by adaptive quadrature
  (1e-9, five chemical potentials); graphene against its closed-form
  dispersion summed on the same grid (1e-10) and electron-hole symmetry
  (1e-10); the inverse (mu to 1e-9; interval ends reproduce the density
  -/+ error to 1e-6 of the error); `dn/dmu` against a finite difference
  (1e-6) and the chain's closed form (1e-3); graphene at 0.2 and 0.4 eV
  against the Dirac-cone density (quadrature at 300 K) and the
  Dirac-cone quantum capacitance `2 e^2 mu / (pi (hbar v_F)^2)` (graphene's
  quantum capacitance was measured by J. Xia, F. Chen, J. Li and N. Tao,
  Nature Nanotechnology 4, 505 (2009)), within 1 % (the
  lattice deviates from the cone by 0.1 % and 0.4 %).
- `hamop.model.E_CHARGE = 1.602176634e-19` C, exact in the SI (CODATA
  2018); the test suite checks it against `scipy.constants.e`.
- `compare_models(build_simple, theta0_simple, build_extended,
  theta0_extended, kpts, energies, ..., embed=None)` (module
  `hamop.lab`): fits nested models to the same band energies and tests
  whether the extra parameters are needed, by the likelihood-ratio
  test (Delta chi-squared against chi-squared(Delta p); S. S. Wilks,
  Ann. Math. Statist. 9, 60 (1938)) with measurement errors, or by the
  F test of linear regression without them. `embed` verifies the
  nesting and starts the extended fit from the simple one. Refused:
  no extra parameters, no degrees of freedom, an `embed` that does not
  reproduce the simple model, and an extended fit that ends worse.
  Tests: both statistics against least-squares closed forms computed
  with `numpy.linalg.lstsq` (1e-6) and the p-values against the
  chi-squared and F tails (1e-6); 60 seeded experiments under the
  simple model (mean statistic 1.25, within 0.45 of 1; p < 0.1 in 11 of
  60, allowed 1 to 13); power against a real `t2 = 0.1 eV`
  (p < 1e-10).
- `drude_weight(..., degen_tol=1e-9)`: the energy window (eV) within
  which two states count as degenerate (see Fixed).

### Fixed

- `drude_weight` summed |M_nn|^2 over the diagonal only. At a k-point
  where two bands are degenerate that diagonal depends on the
  eigenvectors the solver happens to return inside the degenerate
  pair. Graphene (t = -2.7 eV, a = 2.46 Angstrom, mu = 0, 300 K): with
  only the Dirac point K as k-point, D was 3068.84 along x and 0.0
  along y (the eigenvector-free value is 3068.84 for both); on a
  30 x 30 mesh through K and K', D_xx = 3.5382 and D_yy = 3.2815, 7 %
  apart where the hexagonal symmetry makes them equal. The sum now runs
  over all pairs with |e_n - e_m| <= `degen_tol`, the zero-frequency
  limit (f_n - f_m)/(e_m - e_n) -> -df/de of the Kubo formula, which is
  basis independent: 3068.84 in both directions, and D_xx = D_yy =
  6.8196 on the mesh. Without degeneracies the result is unchanged
  (1e-12). Tests:
  `test_drude_weight_at_a_degenerate_k_point_is_basis_independent`,
  `test_drude_weight_away_from_degeneracies_is_unchanged`.
- `principal_layers` assigned sites to layers with
  floor((x - x0) / width), which falls one layer short when the
  division lands just below an integer (15 * 1.42 / 1.42 =
  14.999999999999998). A 40-site chain of spacing 1.42 Angstrom cut
  with width 1.42: before ValueError ("layers 14 and 16 couple"), now
  40 one-site layers and the impurity chain's closed-form transmission
  to 1e-5; with width 2.84: before layers of 3 and 1 sites, now twenty
  of 2. Positions within 1e-9 layer widths below a boundary count on
  it. Test: `test_principal_layers_with_a_non_integer_spacing`.

### Changed

- `kpm_dos` and `kpm_sigma` check their Chebyshev moments against the
  bound |mu_m| <= mu_0 that holds when the spectrum lies inside the
  energy window (a factor 10 is allowed for the stochastic trace and
  the nonorthogonal recurrence) and refuse `bounds` that cut the
  spectrum. Before: a 200-site chain with bounds (-1.9, 1.9) gave a DOS
  of -4.7e7 at E = -1 eV, and with (-1, 1) 5e52. Test:
  `test_kpm_refuses_bounds_that_cut_the_spectrum`.
- `principal_layers` refuses a layer width that is not a finite
  positive length (before: division by zero).
- Documentation: `sigma_tensor` and `monkhorst_pack` now state that a
  grid folded by time reversal is not valid for the Hall component
  (a != b). The fold keeps one of k and -k, whose Hall integrands are
  complex conjugates, so their imaginary parts double instead of
  cancelling: gapped graphene (`haldane(t2=0.0, m_ab=0.3)`, mesh 30)
  gives sigma_xy(0) = -0.912 on the folded grid and 0 on the full grid.
  This is a documentation change; a grid passed as `kpts` cannot be
  recognized as folded.
- Type hints on the new and changed functions.

### Performance

- New private helpers `TightBindingModel._bloch_many` and
  `hamop.eigsolve._eig_chunks` assemble and diagonalize H(k), S(k) and
  their k-derivatives for blocks of k-points with NumPy array
  operations; where canonical orthogonalization drops overlap
  directions the block falls back to `gen_eigh` one k-point at a time.
  `bands`, `dos`, `fermi_level`, `band_edges`, `carrier_count`,
  `sigma_optical`, `sigma_tensor` and `drude_weight` use them. Measured
  speedups (one thread, median of 5 runs, lower of two runs on a
  shared 2-CPU machine): `sigma_optical` graphene mesh 120, two photon
  energies, 140x (19.5 s to 0.14 s); `sigma_tensor` Haldane mesh 48,
  270x; `dos` graphene mesh 100 at 200 energies, 26x; `bands` graphene
  at 2000 k-points, 100x; `fermi_level` Kane-Mele mesh 60, 73x;
  `carrier_count` graphene mesh 100, 110x; `drude_weight` overlap chain
  mesh 800, 470x; `fit_bands` on README example 6, 2.5x. Equality is
  checked in `tests/test_batched_kpoints.py` against the 0.12.1
  per-k-point loops, kept there verbatim as the reference, to 1e-12
  relative for six models (orthogonal, overlap, complex Haldane,
  spinful, dipole blocks, and an overcomplete basis that exercises the
  fallback); measured differences are 1e-14 or below.
- `kpm_dos` evaluates its Chebyshev recurrence in place with the same
  operations in the same order, and takes the exact trace as a trace:
  2.5x faster for a 1000-site chain (19.3 s to 7.7 s; the
  nonorthogonal case is dominated by the sparse solves and unchanged).
  Same moments to 1e-15 (exact trace) and bit for bit (stochastic
  trace); test `test_kpm_dos_in_place_recurrence_equals_the_old_one`.
- The full test suite runs in about 110 to 150 s instead of 376 s on
  the same machine, with 32 more tests.

### Tutorials

- `examples/tutorial_gated_graphene.py`: Hall readings of gated
  graphene, averaged, turned into mu with an error interval and the
  quantum capacitance, then into the optical conductivity and the
  transmittance across the Pauli-blocking edge at 2 mu (about 3 s).
- `examples/tutorial_band_fit_model_check.py`: a band fit that fails
  its p-value, `compare_models` showing that a second-neighbor hopping
  is needed, the extended fit, and error bars of unmeasured quantities
  from both models (about 2 s).
- Both outputs are quoted in the scripts and in the README;
  `tests/test_tutorials.py` runs the scripts and checks every printed
  number to the last digit.

### Tests

- 222 tests (190 before): new `tests/test_regressions_0130.py` (4),
  `tests/test_batched_kpoints.py` (12 including parametrized cases),
  `tests/test_carriers.py` (10), `tests/test_compare_models.py` (3) and
  `tests/test_tutorials.py` (3). The new tests take about 35 s in all.

## 0.12.1 (2026-10-10)

Citation correction and packaging.

### Fixed (documentation)

- `sparse.py` (`kpm_sigma` and the module docstring): the double
  Chebyshev expansion and the finite-temperature optical conductivity
  were credited to Sec. V of Weisse et al., Rev. Mod. Phys. 78, 275
  (2006). That section compares the method with other numerical
  approaches; the expansion is in Sec. II.E and the conductivity in
  Sec. III.D, which are now cited.
- Packaging: the license is now given as the SPDX expression
  `license = "Apache-2.0"` with `license-files = ["LICENSE"]`, and the
  old license table and license classifier are gone. Building no longer
  prints setuptools' deprecation warnings (their support ends on
  2027-02-18). Building from source now needs setuptools 77 or newer;
  installing the package is unchanged.

## 0.12.0 (2026-10-10)

Two tools for a lab's own measurements, three bugs fixed, and input
checks and type hints across the public functions. No public function
was removed or renamed, and results of valid calls are unchanged.

### Added

- `propagate_uncertainty(fit, quantity, method="linear")` (module
  `hamop.lab`): the error bar and covariance of any quantity computed
  from the parameters of a `BandFit` (a band energy at an unmeasured k,
  a band width, a gap). "linear" is first-order propagation through
  `fit.cov` along its principal axes; it is refused when the quantity
  deviates from its linear prediction one standard deviation out by
  more than `nonlinearity_tol` (default 0.1) times the propagated
  error bar. "monte_carlo" samples N(theta, cov) with a seeded
  generator and also reports the 15.87/84.13 percentiles. Tests: the
  closed form g^T C g for band energies of the chain (1e-6 relative),
  200 seeded simulated experiments (band-width scatter within 15 % of
  the propagated error bar), Monte Carlo against linear (5 %), and the
  chi-squared closed form for a zero-slope quantity.
- `BandFit.p_value` (a property; the dataclass fields are unchanged):
  the chi-squared tail probability of the fit, None without
  measurement errors. Tested against the closed-form tail for 2 and 10
  degrees of freedom (1e-12) and for uniformity over simulated
  correct-model fits.
- `sheet_optics(sigma, n_in, n_out)` and
  `sheet_conductivity_from_transmittance(T_rel, n_in, n_out,
  T_rel_err, sigma_imag)`: normal-incidence transmittance, reflectance
  and absorptance of a 2D sheet between two transparent media, from
  the sheet conductivity in units of e^2/(4 hbar), and the inverse with
  an error bar. They use Z0 e^2/(4 hbar) = pi alpha exactly. Tests: an
  independent thin-film (Airy) calculation in the limit of zero
  thickness (1e-6), T + R + A = 1, the absorption of about pi alpha =
  2.3 % of free-standing graphene reported by R. R. Nair et al.,
  Science 320, 1308 (2008), and a seeded Monte Carlo of the inverse's
  error bar (3 %).
- `hamop.model.ALPHA`, the fine-structure constant 7.2973525693e-3
  (CODATA 2018: E. Tiesinga, P. J. Mohr, D. B. Newell and B. N. Taylor,
  Rev. Mod. Phys. 93, 025010 (2021)); the test suite checks it against
  `scipy.constants.fine_structure` to 1e-8. The same reference is now
  given for the Boltzmann constant.
- README examples 11 and 12.

### Fixed

- `TightBindingModel.monkhorst_pack(time_reversal=True)` checked only
  the H and S blocks for being real. A complex dipole block
  (`set_dipole`) keeps the bands even in k but not the optical matrix
  elements, so the k <-> -k fold gave a wrong optical conductivity.
  For the test model (one site, two orbitals, dipole element 0.3i Angstrom,
  40 k-points, mu = 0.5 eV, eta = 0.1 eV), sigma_optical at
  1, 2, 3 eV was 1.9489, 2.6526, 1.3849 with the folded grid against
  2.0963, 3.0159, 1.5778 on the full grid (7 %, 12 % and 12 % low).
  Now: ValueError, pointing to `symmetry_fold(..., time_reversal=True)`
  for eigenvalue-only quantities. Real dipole blocks fold exactly as
  before. Test: `test_time_reversal_fold_refuses_complex_dipole_blocks`.
- `add_hop` and `set_dipole` accepted negative site indices. The shape
  check used `norb[-1]`, but the block was assembled at the orbital
  offset `offsets[-1] = nao`, outside H, so it vanished without an
  error. Two-site molecule with `add_hop(0, -1, (0,), [[-1.0]])`:
  levels before (0, 0), correct (-1, +1); now ValueError. Out-of-range
  indices also raise ValueError now (before: IndexError). Test:
  `test_negative_site_indices_are_refused`.
- `save_wannier90_hr` wrote the comment verbatim as the header. With
  `comment=""` the file began with a blank line, and a comment with a
  line break gave two header lines; `load_wannier90_hr` skips blank
  lines, so it could not read either file back (before: ValueError
  "non-integer degeneracy entry" and "lines 2 and 3 must carry
  num_wann and nrpts"; now the round trip is exact to the written
  precision). Line breaks are collapsed to spaces and an empty comment
  is written as "written by hamop". Test:
  `test_hr_dat_round_trip_with_an_empty_or_multiline_comment`.

### Changed

- Input checks with messages that name the problem, where the old code
  failed later with an unrelated error or returned a meaningless
  number: lattice images with the wrong number of components (before:
  a matmul error at the first `bloch` call), `norb < 1`, positions and
  lattice vectors of different Cartesian dimension, a `mesh` of the
  wrong length, `eta <= 0` in `dos` and `sigma_optical` (before: a
  negative or NaN DOS and conductivity) and `eta < 0` in
  `sigma_tensor`, `gamma < 0` in `buttiker_transmission` and
  `multiprobe_transmission`, `W2 < 0` or `mixing` outside (0, 1] in
  `scba_transmission`, `window <= 0` in the Landauer functions,
  `n_occ` outside [1, nao] in the Berry functions, fewer than two
  vertices in `k_path`, and R-vectors with more than three components
  in `save_wannier90_hr` (now checked before the file is opened).
  Test: `test_input_checks_added_in_0120`.
- `sigma_optical` accepts a scalar photon energy (before: TypeError)
  and checks `lineshape` before the k-loop.
- Type hints on the public functions and methods.
- Docstrings corrected where they contradicted the code: `chern_marker`
  returns one value per orbital, not per site; `lowest_bands` with
  `sigma` returns the eigenvalues closest to sigma, the lowest ones
  only when sigma lies at or below the bottom of the spectrum; the
  `hamop.kubo` module docstring now mentions the dipole blocks.
- 190 tests (171 before): new `tests/test_regressions_0120.py` (4),
  `tests/test_sheet_optics.py` (10 including parametrized cases), and
  5 new tests in `tests/test_lab.py`.

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
