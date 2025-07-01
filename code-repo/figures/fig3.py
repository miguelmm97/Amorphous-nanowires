#%% Modules and setup

# Plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn


# Modules
from functions import *
from colorbar_marker import *


#%% Loading data
file_list = ['fig3-nonchiral.h5', 'fig3-chiral.h5']
data_dict = load_my_data(file_list, '../data')

# Loading data
flux   = data_dict[file_list[0]]['Simulation']['flux']
width  = data_dict[file_list[0]]['Simulation']['width']
N      = data_dict[file_list[0]]['Simulation']['Nx']
G      = data_dict[file_list[0]]['Simulation']['G_array']
L      = data_dict[file_list[0]]['Parameters']['Nz']
K      = data_dict[file_list[0]]['Simulation']['K_onsite']
G2     = data_dict[file_list[1]]['Simulation']['G_array']
K2     = data_dict[file_list[1]]['Simulation']['K_onsite']


#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['dodgerblue', 'limegreen', 'm', 'r', 'orange']
fontsize = 20
palette = seaborn.color_palette(palette='magma', n_colors=len(N))


# Figure 1
fig1 = plt.figure(figsize=(8, 3))
gs = GridSpec(1, 1, figure=fig1)
ax1 = fig1.add_subplot(gs[0, 0])

# Figure 1: Plots
for i in range(G.shape[0]):
    label = f'${N[i]}$'
    ax1.plot(flux, G[i, :], color=palette[i], linestyle='solid', label=label)
    ax1.plot(flux[::2], G2[i, ::2], color=palette[i], linestyle='None', label=None, marker='x')
ax1.plot(flux, np.ones((len(flux), )), color='k', linestyle='dashed', alpha=0.2)


ax1.legend(loc='center', ncol=2, frameon=False, fontsize=fontsize, handlelength=0.7, columnspacing=0.6, handletextpad=0.2,
           bbox_to_anchor=(0.5, 0.35))
ax1.text(1.12, 0.79, '$\\underline{N}$', fontsize=fontsize)
ax1.text(0.2, 0.75, '$\\underline{K}$', fontsize=fontsize)
ax1.plot(0.1, 0.46, marker='_', color='k', markersize=20)
ax1.plot(0.1, 0.63, marker='x', color='k', markersize=10)
ax1.text(0.2, 0.58, '$0.0$', fontsize=fontsize)
ax1.text(0.2, 0.42, '$0.3$', fontsize=fontsize)

ax1.set_xlabel("$\phi/\phi_0$", fontsize=fontsize, labelpad=-5)
ax1.set_ylabel("$G(e^2/h)$", fontsize=fontsize)
ax1.set_xlim(flux[0], 2.5)
ax1.set_ylim(0, 1.2)

ax1.tick_params(which='major', width=0.75, labelsize=10)
ax1.tick_params(which='major', length=6, labelsize=10)
ax1.tick_params(which='major', width=0.75, labelsize=fontsize)
ax1.tick_params(which='major', length=6, labelsize=fontsize)
fig1.tight_layout()

fig1.savefig('fig3.pdf', format='pdf')
plt.show()