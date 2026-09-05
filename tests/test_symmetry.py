"""Time-reversal k-mesh reduction: the folded grid must reproduce the
full grid exactly for k-even observables, halve the work, and refuse
models where the fold is not guaranteed."""
import numpy as np
import pytest

from hamop import (dos, drude_weight, graphene, haldane, linear_chain,
                   sigma_optical)


def test_reduced_grid_reproduces_full_grid_dos_and_sigma():
    g = graphene()
    k1, w1 = g.monkhorst_pack(24)
    k2, w2 = g.monkhorst_pack(24, time_reversal=True)
    assert abs(w2.sum() - 1.0) < 1e-12
    E = np.array([0.5, 1.0, 2.0])
    assert np.abs(dos(g, E, kpts=k1, weights=w1)
                  - dos(g, E, kpts=k2, weights=w2)).max() < 1e-12
    om = np.array([1.0, 1.3])
    s1 = sigma_optical(g, om, 0.0, kpts=k1, weights=w1, eta=0.12, T=10.0)
    s2 = sigma_optical(g, om, 0.0, kpts=k2, weights=w2, eta=0.12, T=10.0)
    assert np.abs(s1 - s2).max() < 1e-12


def test_reduced_grid_reproduces_the_drude_weight_with_overlap():
    c = linear_chain(t=-1.0, s=0.2)
    k1, w1 = c.monkhorst_pack(200)
    k2, w2 = c.monkhorst_pack(200, time_reversal=True)
    D1 = drude_weight(c, 0.0, kpts=k1, weights=w1)
    D2 = drude_weight(c, 0.0, kpts=k2, weights=w2)
    assert abs(D1 - D2) < 1e-10


def test_reduction_roughly_halves_the_grid():
    g = graphene()
    k1, _ = g.monkhorst_pack(24)
    k2, _ = g.monkhorst_pack(24, time_reversal=True)
    assert len(k2) < 0.55 * len(k1)
    # odd mesh: only Gamma is self-paired
    k3, _ = g.monkhorst_pack(23, time_reversal=True)
    assert len(k3) == (23 * 23 - 1) // 2 + 1


def test_complex_blocks_are_refused():
    """Haldane has complex second-neighbour hops, so H(-k) != conj(H(k))
    and the fold must be refused, not silently averaged."""
    with pytest.raises(ValueError):
        haldane().monkhorst_pack(12, time_reversal=True)


# ----------------------------------------------------------------------
# point-group folding (self-verified)

def test_c6_fold_of_graphene_reproduces_spectral_observables():
    from hamop import fermi_level, symmetry_fold
    g = graphene()
    c, s = np.cos(np.pi / 3), np.sin(np.pi / 3)
    R6 = np.array([[c, -s], [s, c]])
    k1, w1 = g.monkhorst_pack(24)
    k2, w2 = symmetry_fold(g, 24, [R6], time_reversal=True)
    assert abs(w2.sum() - 1.0) < 1e-12
    assert len(k2) < len(k1) / 5          # ~ x6 reduction
    E = np.array([0.5, 1.0, 2.0])
    assert np.abs(dos(g, E, kpts=k1, weights=w1)
                  - dos(g, E, kpts=k2, weights=w2)).max() < 1e-12
    mu1 = fermi_level(g, 1.0, kpts=k1, weights=w1)
    mu2 = fermi_level(g, 1.0, kpts=k2, weights=w2)
    assert abs(mu1 - mu2) < 1e-9


def test_fold_works_for_the_time_reversal_broken_haldane_model():
    """Haldane breaks time reversal but keeps C6 spectral symmetry
    (sublattice-exchanging rotation); the spectral check must accept
    it and the folded DOS must be exact."""
    from hamop import symmetry_fold
    c, s = np.cos(np.pi / 3), np.sin(np.pi / 3)
    R6 = np.array([[c, -s], [s, c]])
    h = haldane()
    k, w = symmetry_fold(h, 12, [R6])
    E = np.array([0.5, 1.0, 2.0])
    assert np.abs(dos(h, E, mesh=12)
                  - dos(h, E, kpts=k, weights=w)).max() < 1e-12


def test_non_symmetry_operations_are_refused():
    from hamop import symmetry_fold
    g = graphene()
    R4 = np.array([[0.0, -1.0], [1.0, 0.0]])   # 90 deg: not hexagonal
    with pytest.raises(ValueError):
        symmetry_fold(g, 24, [R4])
    # a lattice symmetry that the Hamiltonian breaks must also be
    # refused: stretch one of the three graphene bonds (anisotropic
    # hopping), which the spectral check catches even though C6 still
    # maps the lattice to itself
    from hamop import TightBindingModel
    a = 2.46
    cell = a * np.array([[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
    pos = np.array([np.zeros(2), (cell[0] + cell[1]) / 3.0])
    m = TightBindingModel(positions=pos, norb=1, cell=cell)
    m.add_hop(0, 1, (0, 0), [[-3.4]])          # one bond different
    m.add_hop(0, 1, (-1, 0), [[-2.7]])
    m.add_hop(0, 1, (0, -1), [[-2.7]])
    c6, s6 = np.cos(np.pi / 3), np.sin(np.pi / 3)
    with pytest.raises(ValueError):
        symmetry_fold(m, 12, [np.array([[c6, -s6], [s6, c6]])])


# ----------------------------------------------------------------------
# automatic point-group detection

def test_detected_group_orders_match_the_lattices():
    """Hexagonal point group has order 12, square 8, the 1D chain 2 --
    counted here from exact lattice-automorphism enumeration filtered
    by the spectral check."""
    from hamop import TightBindingModel, find_point_group, linear_chain
    assert len(find_point_group(graphene())) == 12
    sq = TightBindingModel([[0.0, 0.0]], 1, cell=[[1.0, 0.0], [0.0, 1.0]])
    sq.add_hop(0, 0, (0, 0), [[0.0]])
    sq.add_hop(0, 0, (1, 0), [[-1.0]])
    sq.add_hop(0, 0, (0, 1), [[-1.0]])
    assert len(find_point_group(sq)) == 8
    assert len(find_point_group(linear_chain())) == 2


def test_broken_symmetry_shrinks_the_detected_group():
    from hamop import TightBindingModel, find_point_group
    a = 2.46
    cell = a * np.array([[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])
    pos = np.array([np.zeros(2), (cell[0] + cell[1]) / 3.0])
    m = TightBindingModel(positions=pos, norb=1, cell=cell)
    m.add_hop(0, 1, (0, 0), [[-3.4]])          # one stretched bond
    m.add_hop(0, 1, (-1, 0), [[-2.7]])
    m.add_hop(0, 1, (0, -1), [[-2.7]])
    ops = find_point_group(m)
    assert 1 < len(ops) < 12                   # subset, identity included


def test_detected_group_folds_the_dos_exactly():
    from hamop import fermi_level, find_point_group, symmetry_fold
    g = graphene()
    ops = find_point_group(g)
    k1, w1 = g.monkhorst_pack(24)
    k2, w2 = symmetry_fold(g, 24, ops)
    assert len(k2) < len(k1) / 5
    E = np.array([0.5, 1.0, 2.0])
    assert np.abs(dos(g, E, kpts=k1, weights=w1)
                  - dos(g, E, kpts=k2, weights=w2)).max() < 1e-12
