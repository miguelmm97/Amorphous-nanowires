#%% modules setup

# Math and plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn
from matplotlib.patches import FancyArrowPatch

# modules
from modules.functions import *


#%% Loading data
file_list = ['Exp69.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-latex-figures')

# Parameters

Nx           = data_dict[file_list[0]]['Parameters']['Nx']
Ny           = data_dict[file_list[0]]['Parameters']['Ny']
Nz            = data_dict[file_list[0]]['Parameters']['Nz']
r            = data_dict[file_list[0]]['Parameters']['r']
t            = data_dict[file_list[0]]['Parameters']['t']
eps          = data_dict[file_list[0]]['Parameters']['eps']
lamb         = data_dict[file_list[0]]['Parameters']['lamb']
lamb_z       = data_dict[file_list[0]]['Parameters']['lamb_z']
mu_leads     = data_dict[file_list[0]]['Parameters']['mu_leads']

# Simulation data
G0          = data_dict[file_list[0]]['Simulation']['G_0']
Ghalf       = data_dict[file_list[0]]['Simulation']['G_half']
fermi       = data_dict[file_list[0]]['Simulation']['fermi']
width       = data_dict[file_list[0]]['Simulation']['width']


#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange']
palette0 = seaborn.color_palette(palette='viridis_r', n_colors=100)
palette = [palette0[0], palette0[33], palette0[66], palette0[-1]]
palette2 = [palette0[10], palette0[43], palette0[76], 'k']
marker_list=['o', 's', 'd', 'p', '*', 'h', '>', '<', 'X']
markersize = 5
fontsize=20

# Figure 1: Definition
fig1 = plt.figure(figsize=(8, 6))
gs = GridSpec(1, 1, figure=fig1, wspace=0.2, hspace=0.1)
ax1 = fig1.add_subplot(gs[0, 0])


# Upper panel: Plots
for i in range(G0.shape[-1]):
    ax1.plot(fermi, G0[:, i], color=palette[i], label=f'${width[i] :.2f}$')#, alpha=0.75)
    ax1.plot(fermi, Ghalf[:, i], color=palette[i], linestyle='dashed', alpha=0.7)
    # ax1.plot(fermi[::10], G0[:, i][::10], color=palette[i], marker='o', markersize=3, linestyle='None')#, alpha=0.75)
    # ax1.plot(fermi[::10], Ghalf[:, i][::10], color=palette[i], markersize=3, marker='o', linestyle='None')

ax1.text(0.12, 9.7, '$\\underline{w}$', fontsize=fontsize)
ax1.text(0.44, 9.7, '$\phi_{\mathrm{max}}$', fontsize=fontsize)
ax1.text(0.3, 8.5, '$\phi=0$', fontsize=fontsize)
# ax1.text(0.756, 6.5, '$\Delta E_F=0$', fontsize=fontsize)
ax1.legend(loc='upper left', frameon=False, fontsize=fontsize, bbox_to_anchor=(-0.02, 0.95), handlelength=1)
arrow1 = FancyArrowPatch((0.367, 8.41), (0.452, 7.97), arrowstyle='->', color='black', linewidth=1, mutation_scale=10)
arrow2 = FancyArrowPatch((0.482, 9.5), (0.528, 8.97), arrowstyle='->', color='black', linewidth=1, mutation_scale=10)
ax1.add_patch(arrow1)
ax1.add_patch(arrow2)

ax1.set_xlabel("$E_F$", fontsize=fontsize, labelpad=-1)
ax1.set_ylabel("$G[2e^2/h]$", fontsize=fontsize)
ax1.set_xlim(fermi[0], 1)
ax1.set_ylim(0, np.max(G0[:, 1]))
ax1.tick_params(which='major', width=0.75, labelsize=fontsize)
ax1.tick_params(which='major', length=6, labelsize=fontsize)


fig1.savefig('fig-G-vs-Ef.pdf', format='pdf', backend='pgf')
plt.show()
