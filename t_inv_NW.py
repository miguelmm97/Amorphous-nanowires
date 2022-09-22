# DIII 3D Amorphous model: Band structure for closed and open boundaries

import numpy as np
import time
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from numpy import pi
from random import seed
from functions import GaussianPointSet2D, H_onsite, H_offdiag, spectrum

start_time = time.time()
# %%  Global definitions

# Parameters of the model
n_orb = 4                           # Number of orbitals per site
n_neighbours = 6                    # Number of neighbours
width = 0                         # Width of the gaussian for the WT model
M = 0                               # Mass parameter in units of t1
t1, t2, lamb = 1, 0, 1              # Hopping and spin-orbit coupling in WT model
mu = 0                              # Disorder strength
kz = np.linspace(-pi, pi, 101)      # Momentum space

# Lattice definition
L_x, L_y= 8, 8                             # In units of a (average bond length)
n_sites = int(L_x * L_y)                   # Number of sites in the lattice
n_states = n_sites * n_orb                 # Number of basis states
n_particles = int(n_states / 2)            # Number of filled states
sites = np.arange(0, L_x * L_y)            # Array with the number of each site
x = sites % L_x                            # x position of the sites
y = sites // L_x                           # y position of the sites
x, y = GaussianPointSet2D(x, y, width)     # Positions of the sites in the amorphous lattice


# Declarations
energy_PBC = np.zeros((n_states, len(kz)))
energy_OBC = np.zeros((n_states, len(kz)))
# %% Main

# Hamiltonians and band structures
H_offdiag_PBC = H_offdiag(n_sites, n_orb, n_neighbours, L_x, L_y, x, y, t1, t2, lamb, "Closed")
H_offdiag_OBC = H_offdiag(n_sites, n_orb, n_neighbours, L_x, L_y, x, y, t1, t2, lamb, "Open")
for j, k in enumerate(kz):
    print(str(j) + "/" + str(len(kz)))
    H_diag = np.kron(np.eye(n_sites), H_onsite(M, t1, t2, lamb, k))  # Onsite Hamiltonian
    H_PBC = H_diag + H_offdiag_PBC                                   # Off-diagonal Hamiltonian PBC
    H_OBC = H_diag + H_offdiag_OBC                                   # Off-diagonal Hamiltonian OBC
    energy_PBC[:, j] = spectrum(H_PBC)[0]                            # PBC bands
    energy_OBC[:, j] = spectrum(H_OBC)[0]                            # OBC bands

for j in range(n_states):
    # plt.plot(kz, energy_PBC[j, :], '.b', markersize=2)
    plt.plot(kz, energy_OBC[j, :], 'r', markersize=2)

plt.xlim(-pi, pi)
plt.xlabel("k")
plt.ylabel("Energy")
plt.legend(["OBC"])
plt.title(" M=" + str(M))
plt.show()
