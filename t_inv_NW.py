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
width = 0.1                           # Width of the gaussian for the WT model
M = 2                               # Mass parameter in units of t1
t1, t2, lamb = 1, 0, 1              # Hopping and spin-orbit coupling in WT model
flux = 0.5                            # Magnetic flux through the cross-section
kz = np.linspace(-pi, pi, 1001)      # Momentum space

# Lattice definition
L_x, L_y= 9, 9                             # In units of a (average bond length)
n_sites = int(L_x * L_y)                   # Number of sites in the lattice
n_states = n_sites * n_orb                 # Number of basis states
n_particles = int(n_states / 2)            # Number of filled states
sites = np.arange(0, L_x * L_y)            # Array with the number of each site
x = sites % L_x                            # x position of the sites
y = sites // L_x                           # y position of the sites
x, y = GaussianPointSet2D(x, y, width)     # Positions of the sites in the amorphous lattice
S = L_x * L_y                              # Surface of the cross-section
B = flux / S                               # Magnetic field through the cross-section

# Declarations
energy_OBC = np.zeros((n_states, len(kz)))
# %% Main

# Hamiltonians and band structures
H_offdiag_OBC = H_offdiag(n_sites, n_orb, L_x, L_y, x, y, t1, t2, lamb, B, "Open",  n_neighbours)
for j, k in enumerate(kz):
    print(str(j) + "/" + str(len(kz)))
    H_diag = np.kron(np.eye(n_sites), H_onsite(M, t1, t2, lamb, k))  # Onsite Hamiltonian
    H_OBC = H_diag + H_offdiag_OBC                                   # Off-diagonal Hamiltonian OBC
    energy_OBC[:, j] = spectrum(H_OBC)[0]                            # OBC bands




# %% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 13, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=20)

fig, ax = plt.subplots(figsize=(8, 6))
for j in range(n_states):
    ax.plot(kz, energy_OBC[j, :], '.b', markersize=0.05)

# Axis labels and limits
ax.set_ylabel("$E$", fontsize=25)
ax.set_xlabel("$ka$", fontsize=25)
ax.set_xlim(-pi, pi)
# ax.set_ylim(-3, 2)

plt.title(" $w=$ " + str(width) + "," + " $M=$ " + str(M) + ", $\phi / \phi_0=$ " + str(flux))
plt.show()



