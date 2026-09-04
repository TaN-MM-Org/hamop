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


# ----------------------------------------------------------------------
# interaction self-energies and the current-conserving dephasing probe

def test_constant_self_energy_reproduces_the_impurity_closed_form():
    """A constant real Sigma on one layer of the pristine chain IS the
    on-site impurity, so T = (4 t^2 - E^2)/((4 t^2 - E^2) + eps^2)."""
    from hamop import chain_lead_blocks, transmission
    eps = 0.8
    H00, H01 = chain_lead_blocks(t=-1.0, e0=0.0, per_layer=1)
    Z = np.zeros((1, 1))
    layers = [Z] * 7
    coup = [np.array([[-1.0]])] * 6
    sig = {3: np.array([[eps]], complex)}
    for E in (-1.2, 0.3, 0.7):
        T = transmission(np.array([E]), layers, coup, H00, H01,
                         eta=1e-8, sigma_int=sig)[0]
        v2 = 4.0 - E ** 2
        assert abs(T - v2 / (v2 + eps ** 2)) < 1e-5


def test_rgf_equals_direct_with_complex_self_energies():
    from hamop import chain_lead_blocks, transmission, transmission_direct
    H00, H01 = chain_lead_blocks(t=-1.0, e0=0.0, per_layer=1)
    Z = np.zeros((1, 1))
    layers = [Z] * 7
    coup = [np.array([[-1.0]])] * 6
    sig = {2: lambda E: np.array([[0.3 - 0.15j]]),
           4: np.array([[0.2 + 0.05j]])}
    E = np.linspace(-1.8, 1.8, 9)
    T1 = transmission(E, layers, coup, H00, H01, sigma_int=sig)
    T2 = transmission_direct(E, layers, coup, H00, H01, sigma_int=sig)
    assert np.abs(T1 - T2).max() < 1e-12


def test_buttiker_probe_at_zero_coupling_is_the_coherent_result():
    from hamop import (buttiker_transmission, chain_lead_blocks,
                       transmission)
    H00, H01 = chain_lead_blocks(t=-1.0, e0=0.0, per_layer=1)
    Z = np.zeros((1, 1))
    layers = [Z] * 7
    coup = [np.array([[-1.0]])] * 6
    E = np.linspace(-1.8, 1.8, 9)
    Tc = transmission(E, layers, coup, H00, H01)
    Tb = buttiker_transmission(E, layers, coup, H00, H01,
                               probe_layer=3, gamma=0.0)
    assert np.abs(Tc - Tb).max() < 1e-12


def test_buttiker_probe_scalar_closed_form():
    """Single-site device: every Green function is a scalar built from
    the closed-form surface GF, so T_LR, T_Lp, T_pR and the composed
    T_eff = T_LR + T_Lp T_pR / (T_Lp + T_pR) (Buttiker, PRB 33, 3020
    (1986)) can be written by hand and must match the pipeline."""
    from hamop import buttiker_transmission, chain_lead_blocks
    H00, H01 = chain_lead_blocks(t=-1.0, e0=0.0, per_layer=1)
    Z = np.zeros((1, 1))
    gam = 0.4
    for E in (-1.1, 0.2, 0.9):
        parts = buttiker_transmission(
            np.array([E]), [Z], [], H00, H01, probe_layer=0, gamma=gam,
            eta=1e-10, return_parts=True)
        gs = (E - 1j * np.sqrt(4.0 - E ** 2)) / 2.0   # t^2 g_surf, t = -1
        G = 1.0 / (E - 2.0 * gs + 0.5j * gam)
        Gam = np.sqrt(4.0 - E ** 2)
        T_LR = Gam ** 2 * abs(G) ** 2
        T_Lp = Gam * gam * abs(G) ** 2
        T_eff = T_LR + 0.5 * T_Lp                     # T_Lp = T_pR here
        assert abs(parts["T_eff"][0] - T_eff) < 1e-8
        assert abs(parts["T_Lp"][0] - parts["T_pR"][0]) < 1e-10


def test_dephasing_suppresses_the_double_barrier_resonance():
    """The coherent double barrier transmits ~1 on resonance; a probe
    in the well destroys the interference and must lower it."""
    from hamop import buttiker_transmission, chain_lead_blocks, transmission
    H00, H01 = chain_lead_blocks(t=-1.0, e0=0.0, per_layer=1)
    Z = np.zeros((1, 1))
    lay = [np.array([[1.5]]), Z, Z, Z, np.array([[1.5]])]
    cp = [np.array([[-1.0]])] * 4
    E = np.linspace(-1.9, 1.9, 381)
    Tc = transmission(E, lay, cp, H00, H01)
    Td = buttiker_transmission(E, lay, cp, H00, H01,
                               probe_layer=2, gamma=0.3)
    ipk = int(np.argmax(Tc))
    assert Tc[ipk] > 0.99
    assert Td[ipk] < Tc[ipk] - 0.1
