# Changelog

Every physical claim added in any release is pinned by a test against
an exact result; the release notes on GitHub carry the full anchor
lists. Versions below 1.0 may move the API between minor versions;
such changes are called out here and in the release notes.

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
