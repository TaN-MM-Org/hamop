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
