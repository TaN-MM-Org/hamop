"""Automatic principal-layer partitioning: exact agreement with
hand-built blocks, a working end-to-end device, and refusal of a layer
width smaller than the interaction range."""
import numpy as np
import pytest

from hamop import (TightBindingModel, chain_lead_blocks, principal_layers,
                   transmission)


def _finite_chain(n, t=-1.0, t2=None):
    m = TightBindingModel(positions=[[float(i)] for i in range(n)],
                          norb=1, cell=None)
    for i in range(n):
        m.add_hop(i, i, (0,), [[0.0]])
    for i in range(n - 1):
        m.add_hop(i, i + 1, (0,), [[t]])
    if t2 is not None:
        for i in range(n - 2):
            m.add_hop(i, i + 2, (0,), [[t2]])
    return m


def test_auto_partition_matches_hand_built_blocks_exactly():
    P = principal_layers(_finite_chain(8), layer_width=2.0)
    H00, H01 = chain_lead_blocks(t=-1.0, e0=0.0, per_layer=2)
    assert len(P["layers_H"]) == 4
    for blk in P["layers_H"]:
        assert np.abs(blk - H00).max() == 0.0
    for blk in P["coup_H"]:
        assert np.abs(blk - H01).max() == 0.0
    assert P["layers_S"] is None


def test_auto_partitioned_device_transmits_one_channel():
    P = principal_layers(_finite_chain(8), layer_width=2.0)
    H00, H01 = chain_lead_blocks(t=-1.0, e0=0.0, per_layer=2)
    T = transmission(np.array([-1.0, 0.5, 1.5]), P["layers_H"],
                     P["coup_H"], H00, H01)
    assert np.abs(T - 1.0).max() < 1e-4


def test_impurity_device_reproduces_the_closed_form():
    """One on-site impurity in the auto-partitioned chain gives
    T = (4 t^2 - E^2) / ((4 t^2 - E^2) + eps^2)."""
    eps = 0.8
    m = _finite_chain(7)
    # replace the middle on-site block via a fresh model
    m2 = TightBindingModel(positions=[[float(i)] for i in range(7)],
                           norb=1, cell=None)
    for i in range(7):
        m2.add_hop(i, i, (0,), [[eps if i == 3 else 0.0]])
    for i in range(6):
        m2.add_hop(i, i + 1, (0,), [[-1.0]])
    P = principal_layers(m2, layer_width=1.0)
    H00, H01 = chain_lead_blocks(t=-1.0, e0=0.0, per_layer=1)
    for E in (-1.2, 0.3, 0.7):
        T = transmission(np.array([E]), P["layers_H"], P["coup_H"],
                         H00, H01, eta=1e-8)[0]
        v2 = 4.0 - E ** 2
        assert abs(T - v2 / (v2 + eps ** 2)) < 1e-5


def test_undersized_layer_width_is_refused():
    """With second-neighbour hopping, one-site layers couple beyond
    nearest neighbours and the partition must refuse."""
    m = _finite_chain(8, t2=-0.3)
    with pytest.raises(ValueError):
        principal_layers(m, layer_width=1.0)
    P = principal_layers(m, layer_width=2.0)   # wide enough: fine
    assert len(P["layers_H"]) == 4


def test_periodic_model_is_refused():
    from hamop import linear_chain
    with pytest.raises(ValueError):
        principal_layers(linear_chain(), layer_width=1.0)
