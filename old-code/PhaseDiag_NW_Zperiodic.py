# DIII 3D Amorphous model: Band structure for closed and open boundaries
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from colorbar_functions import hex_to_rgb, rgb_to_dec, get_continuous_cmap  # import functions for colormap
from numpy import pi
from functions import GaussianPointSet2D, H_onsite, H_offdiag, spectrum, list_neighbours, physical_boundary

start_time = time.time()
# %%  Global definitions

# Parameters of the model
n_orb = 4  # Number of orbitals per site
n_neighbours = 4  # Number of neighbours
width_vec = np.linspace(0.02, 0.5, 500)  # Width of the gaussian for the WT model
M_vec = np.linspace(-3.5, 3.5, 700)  # Mass parameter in units of t1
t1, t2, lamb = 1, 0, 1  # Hopping and spin-orbit coupling in WT model
flux = 0.5  # Magnetic flux through the cross-section
r_cutoff = 2
N_samples = 100  # Number of lattice configurations to average over

# Lattice definition
L_x, L_y = 10, 10  # In units of a (average bond length)
n_sites = int(L_x * L_y)  # Number of sites in the lattice
n_states = n_sites * n_orb  # Number of basis states
n_particles = int(n_states / 2)  # Number of filled states
sites = np.arange(0, L_x * L_y)  # Array with the number of each site
x = sites % L_x  # x position of the sites
y = sites // L_x  # y position of the sites

# Declarations
gap = np.zeros((len(M_vec), len(width_vec), N_samples))
loops = 0
# %% Main

for M_index, M in enumerate(M_vec):
    for W_index, W in enumerate(width_vec):
        for sample in range(N_samples):

            loop = 1
            while loop == 1:

                print("M= " + str(M_index) + "/" + str(len(M_vec)) + ", W= " + str(W_index) + "/" + str(
                    len(width_vec)) + ", sample= " + str(sample) + ", " + str(loop))

                # Amorphous realisation of the lattice
                x, y = GaussianPointSet2D(x, y, W)
                neighbours, dist, phis, vecs_x, vecs_y = list_neighbours(x, y, L_x, L_y, "Open", "sorted")
                pts_boundary, sites_boundary, S, loop = physical_boundary(x, y, neighbours, dist, vecs_x, vecs_y, r_cutoff)
                B = flux / S

                # Gap at k=pi
                H_offdiag_OBC = H_offdiag(n_sites, n_orb, x, y, t1, t2, lamb, B, r_cutoff, neighbours, dist, phis)
                H_diag = np.kron(np.eye(n_sites), H_onsite(M, t1, t2, lamb, pi))  # Onsite Hamiltonian
                H_OBC = H_diag + H_offdiag_OBC  # Off-diagonal Hamiltonian OBC
                energy_pi = spectrum(H_OBC)[0]  # OBC bands at k=pi
                gap_pi = energy_pi[int(n_states / 2)] - energy_pi[int(n_states / 2) - 1]  # Gap at k=pi

                # Gap at k=0
                H_offdiag_OBC = H_offdiag(n_sites, n_orb, x, y, t1, t2, lamb, B, r_cutoff, neighbours, dist, phis)
                H_diag = np.kron(np.eye(n_sites), H_onsite(M, t1, t2, lamb, 0))  # Onsite Hamiltonian
                H_OBC = H_diag + H_offdiag_OBC  # Off-diagonal Hamiltonian OBC
                energy_0 = spectrum(H_OBC)[0]  # OBC bands at k=0
                gap_0 = energy_pi[int(n_states / 2)] - energy_0[int(n_states / 2) - 1]  # Gap at k=pi

                # Minimum gap
                gap[M_index, W_index, sample] = min(gap_pi, gap_0)
                loops = loops + loop


PhaseDiag = np.mean(gap, index=2)
print("Thrown away configurations: " + str(loops))

# %% Figures
y_axis = np.repeat(M_vec, len(width_vec))
x_axis = np.tile(width_vec, len(M_vec))

# Font format
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

divnorm = mcolors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=1)
hex_list = ['#ff416d', '#ff7192', '#ffa0b6', '#ffd0db', '#ffffff', '#cfdaff', '#9fb6ff', '#6f91ff', '#3f6cff']

# Phase diagram
fig, ax = plt.subplots(figsize=(8, 6))
scatters = ax.scatter(x_axis, y_axis, c=PhaseDiag, marker='s', norm=divnorm, cmap=get_continuous_cmap(hex_list),
                      linewidths=2.5)
cbar = plt.colorbar(scatters, ax=ax)

# Colorbar format
cbar.set_label(label='$\\nu$', size=35, labelpad=-15, y=0.5)
# cbar.set_label_coords(0.075, 0.5)
cbar.ax.tick_params(labelsize=25)
cbar.set_ticks([-2, -1, 0, 1, 2])
cbar.set_ticklabels([-2, -1, 0, 1, 2])

# Axis labels and limits
ax.set_ylabel("$M$", fontsize=35)
ax.set_xlabel("$w$", fontsize=35)
ax.set_xlim(0, 0.3)
ax.set_ylim(-3.5, 3.5)
ax.yaxis.set_label_coords(-0.09, 0.5)

# Axis ticks
ax.tick_params(which='major', width=0.75)
ax.tick_params(which='major', length=14)
ax.tick_params(which='minor', width=0.75)
ax.tick_params(which='minor', length=7)
majorsy = [-3, -2, -1, 0, 1, 2, 3]
minorsy = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
majorsx = [0, 0.1, 0.2, 0.3]
minorsx = [0.05, 0.15, 0.25]
ax.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
ax.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
ax.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
ax.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))








