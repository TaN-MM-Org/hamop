# Changelog

Every physical claim added in any release is pinned by a test against
an exact result; the release notes on GitHub carry the full anchor
lists. Versions below 1.0 may move the API between minor versions;
such changes are called out here and in the release notes.

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
