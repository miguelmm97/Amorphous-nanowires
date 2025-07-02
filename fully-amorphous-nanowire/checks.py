# %% modules set up

# Math and plotting
import numpy as np
import sys
from datetime import date
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Kwant
import kwant

# modules
from functions import *
from calculations import DoS

#%% Main
Nx, Ny, L = 10, 10, 200 # Number of sites in the cross-section
width = 0.1           # Spread of the Gaussian distribution for the lattice sites
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

# filename = 'try.h5'
# datadir = '..'
# file_list = ['fig4-G-vs-flux.h5', 'fig4-DoS.h5']
# data_dict = load_my_data(file_list, 'data')
# flux          = data_dict[file_list[0]]['Simulation']['flux']
# G_array       = data_dict[file_list[0]]['Simulation']['G_array']
# width         = data_dict[file_list[0]]['Simulation']['width']
# x             = data_dict[file_list[0]]['Simulation']['x']
# y             = data_dict[file_list[0]]['Simulation']['y']
# z             = data_dict[file_list[0]]['Simulation']['z']
#
# flux_top = flux[39]
# w_top = width[2]
# index_top = 2
# disorder=np.zeros((len(x[index_top, :]), ))
#
# DoS(flux_top, w_top, fermi, Nx, Ny, L, x[index_top, :], y[index_top, :], z[index_top, :],
#     K_onsite, disorder, t, eps, lamb, eta, r, mu_leads, filename, datadir)


# #%% Loading data
file_list = ['try.h5']
data_dict = load_my_data(file_list, '.')
local_DoS          = data_dict[file_list[0]]['Simulation']['local_DoS']
bulk_pos          = data_dict[file_list[0]]['Simulation']['bulk_pos']
#
#
fig1 = plt.figure(figsize=(8, 6))
gs = GridSpec(1, 1, figure=fig1, wspace=0.2, hspace=0.1)
ax2 = fig1.add_subplot(gs[0, 0], projection='3d')

sigmas = 3
mean_value = np.mean(local_DoS)
std_value = np.std(local_DoS)
max_value, min_value = mean_value + sigmas * std_value, 0
color_map = plt.get_cmap("magma").reversed()
colors = color_map(np.linspace(0, 1, 20))
colors[0] = [1, 1, 1, 1]
color_map = LinearSegmentedColormap.from_list("custom_colormap", colors)
colormap = cm.ScalarMappable(norm=Normalize(vmin=min_value, vmax=max_value), cmap=color_map)
palette = seaborn.color_palette(palette='viridis_r', n_colors=200)
palette = [palette[0], palette[50], palette[100], palette[130], palette[-1]]
#
#
ax2.scatter(np.round(bulk_pos[:, 0], 2), np.round(bulk_pos[:, 1], 2), np.round(bulk_pos[:, 2], 2), facecolor='white', edgecolor='black', rasterized=True)
ax2.scatter(np.round(bulk_pos[:, 0], 2), np.round(bulk_pos[:, 1], 2), np.round(bulk_pos[:, 2], 2), c=local_DoS,
            cmap=color_map, vmin=min_value, vmax=max_value, rasterized=True)
ax2.set_box_aspect((3, 3, 10))
ax2.set_axis_off()
pos2 = ax2.get_position()
ax2.set_position([pos2.x0 - 0.02, pos2.y0 - 0.15, 0.25, 0.4])
plt.show()

