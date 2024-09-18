# %% modules setup

# Math and plotting
from numpy import pi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# modules
from modules.functions import *
from modules.AmorphousLattice_2d import AmorphousLattice_2d
from modules.InfiniteNanowire import InfiniteNanowire_FuBerg


# Functions
def check_blocks(H, len, ax):
    ax.spy(H)
    for j in np.arange(0, len, 4):
        aux = j - 0.5
        ax.plot(aux * np.ones((len,)), np.arange(0, len, 1), 'b', linewidth=0.2)
        ax.plot(np.arange(0, len, 1), aux * np.ones((len,)), 'b', linewidth=0.2)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

def peierls_x(y):
    return np.exp(2 * pi * 1j * flux * y / area)


# %% Variables
"""
We compare the crystalline and amorphous model for an infinite nanowire.
"""

Nx, Ny = 6, 6      # Number of sites in the cross-section
width = 0.0000001  # Spread of the Gaussian distribution for the lattice sites
r = 1.3            # Nearest-neighbour cutoff distance
flux = 0.5         # Flux threaded through the cross-section (in units of flux quantum)
t = 1              # Hopping
eps = 4 * t        # Onsite orbital hopping (in units of t)
lamb = 1 * t       # Spin-orbit coupling in the cross-section (in units of t)
lamb_z = 1.8 * t   # Spin-orbit coupling along z direction

sigma_0 = np.eye(2, dtype=np.complex128)
sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
tau_0, tau_x, tau_y, tau_z = sigma_0, sigma_x, sigma_y, sigma_z

# %% Main: Crystalline model

# Lattice parameters
Nsites = int(Nx * Ny)
area = (Nx - 1) * (Ny - 1)
dimH = Nsites * 4
sites = np.arange(0, Nsites)
x = sites % Nx
y = sites // Nx

# Hamiltonian parameters
kz = np.linspace(-pi, pi, 1000)
H_offdiag = np.zeros((dimH, dimH), dtype=np.complex128)
H = np.zeros((len(kz), dimH, dimH), dtype=np.complex128)

# Off-diagonal Hamiltonian
for n in sites:
    # Jump operators along x and y direction
    state_n, state_nx, state_ny = np.zeros((Nsites,)), np.zeros((Nsites,)), np.zeros((Nsites,))
    state_n[n] = 1
    if x[n] != Nx - 1: state_nx[n + 1] = 1
    if y[n] != Ny - 1: state_ny[n + Nx] = 1
    jump_x, jump_y = np.outer(state_n, state_nx) * peierls_x(y[n]), np.outer(state_n, state_ny)

    # Off-diagonal Hamiltonian
    H_offdiag += -t * np.kron(jump_x, np.kron(sigma_x, tau_0)) - t * np.kron(jump_y, np.kron(sigma_x, tau_0)) + \
        1j * 0.5 * lamb * (np.kron(jump_x, np.kron(sigma_z, tau_y)) - np.kron(jump_y, np.kron(sigma_z, tau_x)))
H_offdiag += H_offdiag.T.conj()

# Full Hamiltonian
for i, k in enumerate(kz):
    H[i, :, :] = (eps - 2 * t * np.cos(k)) * np.kron(np.eye(Nsites), np.kron(sigma_x, tau_0)) + \
                 + lamb_z * np.sin(k) * np.kron(np.eye(Nsites), np.kron(sigma_y, tau_0)) + H_offdiag

# Band structure
energy_bands, eigenstates = {}, {}
aux_bands = np.zeros((len(kz), dimH))
aux_eigenstates = np.zeros((len(kz), dimH, dimH), dtype=np.complex128)
for j in range(len(kz)):
    bands_k, eigenstates_k = np.linalg.eigh(H[j, :, :])
    idx = bands_k.argsort()
    aux_bands[j, :], aux_eigenstates[j, :, :] = bands_k[idx], eigenstates_k[:, idx]

# Ordering bands
for i in range(dimH):
    energy_bands[i] = aux_bands[:, i]
    eigenstates[i] = aux_eigenstates[:, :, i]

# %% Main: Amorphous model

# Amorphous cross-section
cross_section = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=width, r=r)
cross_section.build_lattice()

# Infinite amorphous nanowire
wire = InfiniteNanowire_FuBerg(lattice=cross_section, t=t, eps=eps, lamb=lamb, lamb_z=lamb_z, flux=flux)
wire.get_bands()

# %% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125', '#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']

fig2 = plt.figure(figsize=(6, 6))
gs = GridSpec(2, 4, figure=fig2, wspace=0.8)
ax2_1 = fig2.add_subplot(gs[:, :2])
ax2_2 = fig2.add_subplot(gs[:, 2:])

for i in energy_bands.keys():
    ax2_1.plot(kz, energy_bands[i], color=color_list[8], linewidth=0.5)
    ax2_2.plot(kz, energy_bands[i], color=color_list[8], linewidth=0.5)

for i in wire.energy_bands.keys():
    ax2_1.plot(wire.kz[::5], wire.energy_bands[i][::5], '.', color=color_list[0], markersize=1)
    ax2_2.plot(wire.kz[::5], wire.energy_bands[i][::5], '.', color=color_list[0], markersize=1)
ax2_2.text(-pi + 0.5, 0.01, f'$E_g=$ {wire.get_gap():.2f}')

ax2_1.set_xlabel('$k/a$')
ax2_1.set_ylabel('$E(k)/t$')
ax2_1.set_xlim(-pi, pi)
ax2_1.tick_params(which='major', width=0.75, labelsize=10)
ax2_1.tick_params(which='major', length=6, labelsize=10)
ax2_1.set(xticks=[-pi, -pi / 2, 0, pi / 2, pi], xticklabels=['$-\pi$', '$-\pi/2$', '$0$', '$\pi/2$', '$\pi$'])

ax2_2.set_xlabel('$k/a$')
ax2_2.set_xlim(-pi, pi)
ax2_2.set_ylim(-0.5, 0.5)
ax2_2.tick_params(which='major', width=0.75, labelsize=10)
ax2_2.tick_params(which='major', length=6, labelsize=10)
ax2_2.set(xticks=[-pi, -pi / 2, 0, pi / 2, pi], xticklabels=['$-\pi$', '$-\pi/2$', '$0$', '$\pi/2$', '$\pi$'])
fig2.suptitle(
    f'$w=$ {width}, $r=$ {r}, $\phi/\phi_0=$ {flux}, $\epsilon=$ {eps}, $\lambda=$ {lamb}, $\lambda_z=$ {lamb_z}')

# fig3 = plt.figure(figsize=(6, 6))
# ax3 = fig3.gca()
# check_blocks(H_offdiag, dimH, ax3)
plt.show()
# fig2.savefig("crystalline-amorphous-comparison-2.pdf")
