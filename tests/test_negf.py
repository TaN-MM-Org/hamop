"""Transport against closed forms: the chain surface Green function,
unit transmission across the band, the single-impurity formula, exact
RGF-vs-direct-inversion agreement, and multichannel counting."""
import numpy as np

from hamop import (chain_lead_blocks, sancho_rubio, transmission,
                   transmission_direct)


def test_surface_green_function_closed_form():
    """g(E) = (E - i sqrt(4 t^2 - E^2)) / (2 t^2) inside the band of
    the single-orbital chain (e0 = 0, retarded branch)."""
    H00, H01 = chain_lead_blocks(t=-1.0, e0=0.0)
    for E in (-1.5, -0.5, 0.5, 1.2):
        g = sancho_rubio(E, H00, H01, eta=1e-9)[0, 0]
        exact = (E - 1j * np.sqrt(4.0 - E ** 2)) / 2.0
        assert abs(g - exact) < 1e-6


def test_pristine_chain_transmits_one_channel():
    H00, H01 = chain_lead_blocks(t=-1.0, e0=0.0)
    E_in = np.array([-1.9, -1.0, 0.0, 1.0, 1.9])
    T = transmission(E_in, [H00] * 4, [H01] * 3, H00, H01)
    assert np.abs(T - 1.0).max() < 1e-4
    E_out = np.array([-2.5, 2.5, 3.0])
    T = transmission(E_out, [H00] * 4, [H01] * 3, H00, H01)
    assert np.abs(T).max() < 1e-8


def test_single_impurity_closed_form():
    """On-site impurity eps in the chain:
    T(E) = (4 t^2 - E^2) / ((4 t^2 - E^2) + eps^2)."""
    t, eps = -1.0, 0.8
    H00, H01 = chain_lead_blocks(t=t, e0=0.0)
    layers = [H00, H00 + eps * np.eye(1), H00]
    for E in (-1.2, 0.3, 0.7, 1.5):
        T = transmission(np.array([E]), layers, [H01] * 2, H00, H01,
                         eta=1e-8)[0]
        v2 = 4.0 * t ** 2 - E ** 2
        assert abs(T - v2 / (v2 + eps ** 2)) < 1e-5


def test_rgf_equals_direct_inversion_exactly():
    """The recursive sweep and the dense-inversion reference are the
    same object evaluated two ways; they must agree to machine
    precision, disorder and overlap included."""
    rng = np.random.default_rng(3)
    H00, H01 = chain_lead_blocks(t=-1.0, e0=0.0, per_layer=2)
    layers = [H00 + np.diag(rng.normal(scale=0.3, size=2)) for _ in range(5)]
    layers = [H00] + [0.5 * (h + h.conj().T) for h in layers] + [H00]
    coup = [H01] * (len(layers) - 1)
    E = np.linspace(-1.5, 1.5, 11)
    T1 = transmission(E, layers, coup, H00, H01)
    T2 = transmission_direct(E, layers, coup, H00, H01)
    assert np.abs(T1 - T2).max() < 1e-10


def test_two_decoupled_chains_transmit_two_channels():
    """Two identical uncoupled chains in one principal layer: T = 2
    inside the band -- transmission counts channels."""
    t = -1.0
    H00 = np.zeros((2, 2), complex)
    H01 = np.diag([t, t]).astype(complex)
    E = np.array([-1.0, 0.0, 1.0])
    T = transmission(E, [H00] * 3, [H01] * 2, H00, H01)
    assert np.abs(T - 2.0).max() < 1e-4


def test_nonorthogonal_chain_still_transmits_one_channel():
    """Overlap s on the chain bonds deforms the band but a pristine
    chain must still transmit exactly one channel inside it."""
    t, s = -1.0, 0.2
    H00 = np.array([[0.0]], complex)
    H01 = np.array([[t]], complex)
    S00 = np.array([[1.0]], complex)
    S01 = np.array([[s]], complex)
    # band edges of the nonorthogonal chain: E = 2t cos / (1 + 2s cos)
    E = np.array([2.0 * t / (1.0 + 2.0 * s) * 0.6, 0.0,
                  -2.0 * t / (1.0 - 2.0 * s) * 0.6])
    T = transmission(E, [H00] * 3, [H01] * 2, H00, H01,
                     layers_S=[S00] * 3, coup_S=[S01] * 2,
                     lead_S00=S00, lead_S01=S01)
    assert np.abs(T - 1.0).max() < 1e-4
