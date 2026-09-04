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
from .eigsolve import gen_eigh
from .kubo import carrier_count, sigma_optical
from .lattices import chain_lead_blocks, graphene, linear_chain, two_site
from .model import TightBindingModel
from .negf import sancho_rubio, transmission, transmission_direct
from .spectrum import band_edges, bands, dos, fermi_level

__version__ = "0.1.0"
__all__ = [
    "TightBindingModel", "gen_eigh",
    "bands", "dos", "fermi_level", "band_edges",
    "sigma_optical", "carrier_count",
    "sancho_rubio", "transmission", "transmission_direct",
    "linear_chain", "two_site", "graphene", "chain_lead_blocks",
]
