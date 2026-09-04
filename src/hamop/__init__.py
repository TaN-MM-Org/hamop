"""hamop: one tight-binding Hamiltonian, every observable, strictly consistent.

Build a Hamiltonian once, as real-space blocks in an orthogonal or
nonorthogonal basis, and compute its band structure, density of states,
Kubo-Greenwood optical conductivity and Landauer transmission from the
same matrices, so that spectral, optical and transport statements about
one model can never drift apart.

Methodological basis: the study "Learning the quantum Hamiltonian of
defective monolayer MoS2 reveals collective vacancy brightness
decoupled from defect count" (code:
https://github.com/Tanvir-Mahmud-Mahim/mos2-vacancy-optics); this
package is the general-purpose engine distilled from that pipeline.
"""
from .berry import berry_curvature, berry_phase, chern_number
from .device import principal_layers
from .eigsolve import gen_eigh
from .kubo import carrier_count, drude_weight, sigma_optical, sigma_tensor
from .lattices import (chain_lead_blocks, graphene, haldane,
                       linear_chain, ssh, two_site)
from .model import TightBindingModel
from .negf import (buttiker_transmission, sancho_rubio, scba_transmission,
                   transmission, transmission_direct, transmission_sparse)
from .peierls import with_peierls
from .sparse import (bloch_derivative_sparse, bloch_sparse, kpm_dos,
                     kpm_sigma, lowest_bands)
from .spectrum import band_edges, bands, dos, fermi_level, k_path
from .spin import PAULI, kane_mele, with_spin
from .symmetry import symmetry_fold

__version__ = "0.4.0"
__all__ = [
    "TightBindingModel", "gen_eigh",
    "bands", "dos", "fermi_level", "band_edges", "k_path",
    "sigma_optical", "sigma_tensor", "drude_weight", "carrier_count",
    "berry_phase", "berry_curvature", "chern_number",
    "sancho_rubio", "transmission", "transmission_direct",
    "transmission_sparse", "buttiker_transmission", "scba_transmission",
    "principal_layers",
    "bloch_sparse", "bloch_derivative_sparse", "lowest_bands",
    "kpm_dos", "kpm_sigma",
    "PAULI", "with_spin", "kane_mele", "with_peierls", "symmetry_fold",
    "linear_chain", "two_site", "graphene", "ssh", "haldane",
    "chain_lead_blocks",
]
