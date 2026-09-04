"""Reproduce graphene's universal optical sheet conductivity.

The interband plateau of nearest-neighbour graphene equals e^2/(4 hbar)
(Kuzmenko et al., PRL 100, 117401 (2008)), which is exactly 1.0 in the
package units -- the same anchor the test suite asserts.
"""
import numpy as np

from hamop import graphene, sigma_optical

g = graphene(t=-2.7, a=2.46)
omega = np.linspace(0.6, 1.8, 25)
sigma = sigma_optical(g, omega, mu=0.0, mesh=120, eta=0.12, T=10.0)
for w, s in zip(omega, sigma):
    print(f"hw = {w:5.2f} eV   sigma = {s:6.3f}  (e^2 / 4 hbar)")
