#%% Modules setup

# Math
from numpy import pi
import numpy as np

# Managing classes
from dataclasses import dataclass, field

# Tracking time
import time

# Managing logging
import logging
import colorlog
from colorlog import ColoredFormatter

#%% Module


@Dataclass
class InfiniteNanowire_FuBerg:
    """ Infinite amorphous cross-section nanowire based on the crystalline Fu and Berg model"""

    # Lattice parameters
    Nx: int           # Number of lattice sites along x direction
    Ny: int           # Number of lattice sites along y direction
    d:  float         # Scale of the displacement away from the crystalline positions (between 0 and 1)

    # Electronic parameters
    eps:    float     # Onsite energy coupling different orbitals
    t:      float     # Isotropic hopping strength
    lamb:   float     # Spin orbit coupling in the cross-section
    lamb_z: float     # Spin orbit coupling along z

    # Class fields
    dimH:          int = field(init=False)   # Dimension of the single-particle Hilbert Space
    Nsites:        int = field(init=False)   # Number of sites in the cross-section
    sites:          {} = field(init=False)   # Dictionary of sites and positions
    H:      np.ndarray = field(init=False)   # Hamiltonian

    # Methods
    def build_lattice(self):
        self.Nsites = int(self.Nx * self.Ny)
        list_sites = np.arange(0, self.Nsites)
        self.sites = {k: {'x': list_sites[k] % self.Nx, 'y': list_sites[k] // self.Nx}
                      for k in list_sites}



    def get_Hamiltonian(self):
        self.dimH = self.Nx * self.Ny * 4
        self.H = np.zeros((self.dimH, self.dimH), dtype=np.complex128)
        for i in sites.keys():
            psi_i = np.zeros((self.dimH, ), dtype=np.complex128)
            psi_i_x = np.zeros((self.dimH, ), dtype=np.complex128)
            psi_i_y = np.zeros((self.dimH,), dtype=np.complex128)
            psi_i[i] = 1.
            psi_i_x[i + 1] = 1. if sites[i]['x'] != (self.Nx - 1) else 0.
            psi_i_y[i + Nx] = 1. if sites[i]['y'] != (self.Ny - 1) else 0.

