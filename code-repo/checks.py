# %% modules set up

# Math and plotting
import numpy as np
import sys
from datetime import date
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Kwant
import kwant

# modules
from functions import *
from calculations import G_vs_L_fully_amorphous

#%% Main
Nx, Ny, L = 10, 10, [50, 100] # Number of sites in the cross-section
width = 0.1            # Spread of the Gaussian distribution for the lattice sites
flux = np.linspace(0, 2, 100)
r = 1.3                  # Nearest-neighbour cutoff distance
t = 1                    # Hopping
eps = 4 * t              # Onsite orbital hopping (in units of t)
lamb = 1 * t             # Spin-orbit coupling in the cross-section (in units of t)
eta = 1.8 * t         # Spin-orbit coupling along z direction
mu_leads = -1 * t        # Chemical potential at the leads
fermi = 0.
# Fermi energy
K_hopp = 0.
K_onsite = 0.
params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'eta': eta}

filename = 'try.h5'
datadir = '..'
G_vs_L_fully_amorphous(flux, width, fermi, Nx, Ny, L, K_onsite, t, eps, lamb, eta, r, mu_leads, filename, datadir)


# #%% Loading data
# file_list = ['try.h5']
# data_dict = load_my_data(file_list, '.')
# G0            = data_dict[file_list[0]]['Simulation']['G']
# flux         = data_dict[file_list[0]]['Simulation']['flux']
#
#
# fig1 = plt.figure(figsize=(8, 6))
# gs = GridSpec(1, 1, figure=fig1, wspace=0.2, hspace=0.1)
# ax1 = fig1.add_subplot(gs[0, 0])
# ax1.plot(flux, G0, '-')
# plt.show()

