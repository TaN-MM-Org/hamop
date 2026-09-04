"""Automatic principal-layer partitioning of a finite model.

The NEGF module takes explicit layer blocks; this module builds them
from a :class:`~hamop.model.TightBindingModel` directly.  Sites of a
finite model are sorted along the transport axis and grouped into
principal layers of a chosen width; the dense H (and S) are then cut
into on-layer and nearest-layer blocks.

The partition is *verified*, not assumed: if any Hamiltonian or overlap
matrix element couples layers further than nearest neighbours -- i.e.
the chosen layer width is smaller than the interaction range -- the
function raises with the magnitude of the offending element instead of
silently truncating physics.  The test suite asserts both directions:
a valid partition reproduces hand-built blocks exactly, and an
undersized layer width is refused.
"""
from __future__ import annotations

import numpy as np

__all__ = ["principal_layers"]


def principal_layers(model, layer_width, axis=0, tol=1e-12):
    """Partition a finite model into principal layers along an axis.

    layer_width: layer thickness in the model's length unit; must be at
    least the interaction range so only nearest layers couple.
    Returns a dict with 'layers_H', 'coup_H' (lists of dense blocks),
    'layers_S', 'coup_S' (None for an orthogonal model), 'layer_of'
    (layer index per site) and 'order' (site order used).
    """
    if model.cell is not None:
        raise ValueError("principal_layers takes a finite model "
                         "(cell=None); build a finite device supercell")
    x = model.positions[:, axis]
    x0 = float(x.min())
    layer_of = np.floor((x - x0) / float(layer_width)).astype(int)
    n_layers = int(layer_of.max()) + 1
    if n_layers < 2:
        raise ValueError("fewer than two layers; reduce layer_width")

    # site order: by layer, then by position within the layer
    order = np.lexsort((x, layer_of))
    # orbital permutation
    perm = np.concatenate([
        np.arange(model.offsets[s], model.offsets[s] + model.norb[s])
        for s in order])
    H, S = model.bloch(None)
    H = H[np.ix_(perm, perm)]
    overlap = model.has_overlap()
    S = S[np.ix_(perm, perm)] if overlap else None

    sizes = [int(model.norb[order[layer_of[order] == L]].sum())
             for L in range(n_layers)]
    offs = np.concatenate([[0], np.cumsum(sizes)])

    # refuse partitions whose layers couple beyond nearest neighbours
    for La in range(n_layers):
        for Lb in range(La + 2, n_layers):
            blk = H[offs[La]:offs[La + 1], offs[Lb]:offs[Lb + 1]]
            worst = float(np.abs(blk).max()) if blk.size else 0.0
            if overlap:
                sblk = S[offs[La]:offs[La + 1], offs[Lb]:offs[Lb + 1]]
                worst = max(worst, float(np.abs(sblk).max()))
            if worst > tol:
                raise ValueError(
                    f"layers {La} and {Lb} couple (|element| = "
                    f"{worst:.3e}); increase layer_width beyond the "
                    "interaction range")

    layers_H = [H[offs[L]:offs[L + 1], offs[L]:offs[L + 1]]
                for L in range(n_layers)]
    coup_H = [H[offs[L]:offs[L + 1], offs[L + 1]:offs[L + 2]]
              for L in range(n_layers - 1)]
    layers_S = coup_S = None
    if overlap:
        layers_S = [S[offs[L]:offs[L + 1], offs[L]:offs[L + 1]]
                    for L in range(n_layers)]
        coup_S = [S[offs[L]:offs[L + 1], offs[L + 1]:offs[L + 2]]
                  for L in range(n_layers - 1)]
    return {"layers_H": layers_H, "coup_H": coup_H,
            "layers_S": layers_S, "coup_S": coup_S,
            "layer_of": layer_of, "order": order}
