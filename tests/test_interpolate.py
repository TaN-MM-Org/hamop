"""Fourier interpolation against its defining exactness statement: a
model whose hopping range fits the sampling window interpolates to the
directly computed bands at machine precision -- orthogonal,
nonorthogonal and complex-hopping models alike -- and an undersampled
model is detected by the residual check, not silently aliased."""
import numpy as np
import pytest

from hamop import (fourier_interpolation, graphene, haldane, linear_chain,
                   ssh)


def test_interpolation_is_exact_when_the_range_fits():
    rng = np.random.default_rng(7)
    for model, mesh in [(linear_chain(), 8),
                        (linear_chain(s=0.2), 8),
                        (graphene(), 6),
                        (ssh(), 8),
                        (haldane(), 8)]:
        itp = fourier_interpolation(model, mesh)
        recip = 2.0 * np.pi * np.linalg.inv(model.cell).T
        dim = model.cell.shape[0]
        for _ in range(3):
            k = rng.uniform(-1.0, 1.0, dim) @ recip
            e1 = itp.bands([k])[0]
            from hamop import gen_eigh
            e2 = gen_eigh(*model.bloch(k))
            assert np.abs(e1 - e2).max() < 1e-12


def test_undersampling_is_detected_by_the_residual():
    m = linear_chain()
    m.add_hop(0, 0, (2,), [[-0.3]])           # range 2
    assert fourier_interpolation(m, 3).max_residual() > 1e-2
    assert fourier_interpolation(m, 8).max_residual() < 1e-12


def test_finite_models_are_refused():
    from hamop import two_site
    with pytest.raises(ValueError):
        fourier_interpolation(two_site(), 4)
