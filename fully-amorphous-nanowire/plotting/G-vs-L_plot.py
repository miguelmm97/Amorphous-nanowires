#%% modules setup

# Math and plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn

# modules
from modules.functions import *


#%% Loading data
file_list = ['Exp43.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-cond-vs-L')

# Parameters
Ef           = data_dict[file_list[0]]['Parameters']['Ef']
Nx           = data_dict[file_list[0]]['Parameters']['Nx']
Ny           = data_dict[file_list[0]]['Parameters']['Ny']
r            = data_dict[file_list[0]]['Parameters']['r']
t            = data_dict[file_list[0]]['Parameters']['t']
eps          = data_dict[file_list[0]]['Parameters']['eps']
lamb         = data_dict[file_list[0]]['Parameters']['lamb']
lamb_z       = data_dict[file_list[0]]['Parameters']['lamb_z']
mu_leads     = data_dict[file_list[0]]['Parameters']['mu_leads']

# Simulation data
flux          = data_dict[file_list[0]]['Simulation']['flux']
width         = data_dict[file_list[0]]['Simulation']['width']
Nz            = data_dict[file_list[0]]['Simulation']['Nz']
G_array       = data_dict[file_list[0]]['Simulation']['G_array']

# Conductance decay with length for specific resnances
flux1, flux2 = 4.4, 4.6
idx1 = np.where(flux > flux1)[0][0]
idx2 = np.where(flux > flux2)[0][0]

G_resonance = [np.max(G_array[0, i, idx1: idx2]) for i in range(len(Nz))]

#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange']
palette1 = seaborn.color_palette(palette='magma', n_colors=G_array.shape[1])
palette2 = seaborn.color_palette(palette='magma', n_colors=5)
marker_list=['o', 's', 'd', 'p', '*', 'h', '>', '<', 'X']
markersize = 5
fontsize=20

# # Figure 1: Definition
# fig1 = plt.figure(figsize=(20, 15))
# gs = GridSpec(3, 2, figure=fig1, wspace=0.2, hspace=0.3)
# ax1 = fig1.add_subplot(gs[0, 0])
# ax2 = fig1.add_subplot(gs[0, 1])
# ax3 = fig1.add_subplot(gs[1, 0])
# ax4 = fig1.add_subplot(gs[1, 1])
# ax5 = fig1.add_subplot(gs[2, 0])
# ax6 = fig1.add_subplot(gs[2, 1])
# ax_vec = [ax1, ax2, ax3, ax4, ax5, ax6]
#
# # Figure 1: Plots
# for i in range(G_array.shape[0]):
#     ax = ax_vec[i]
#     ax.set_title(f'$w= {width[i]}$', fontsize=fontsize)
#     for j in range(G_array.shape[2]):
#         label = f'$\phi= {flux[j] :.2f}$'
#         ax.plot(Nz[::-1], G_array[i, :, j][::-1], color=palette1[j], linestyle='solid', marker=marker_list[j], label=label)
#
# # Figure 1: Format
# ax4.legend(ncol=2, frameon=False, fontsize=10)
# fig1.suptitle(f'$\mu_l= {mu_leads}$, $E_f= {Ef}$, $r= {r}$, $N_x= {Nx}$, $N_y = {Ny}$', y=0.93, fontsize=20)
# for ax in ax_vec:
#     ax.set_xlim(Nz[-1], Nz[0])
#     ax.set_ylim(0, np.max(G_array))
#     ax.tick_params(which='major', width=0.75, labelsize=10)
#     ax.tick_params(which='major', length=6, labelsize=10)
#     ax.set_xlabel("$L$", fontsize=fontsize)
#     ax.set_ylabel("$G[2e^2/h]$", fontsize=fontsize)



# Figure 1: Definition
fig1 = plt.figure(figsize=(15, 8))
gs = GridSpec(1, 1, figure=fig1, wspace=0.2, hspace=0.3)
ax1 = fig1.add_subplot(gs[0, 0])
ax_vec = [ax1]

# Figure 1: Plots
for i in range(G_array.shape[0]):
    ax = ax_vec[i]
    for j in range(G_array.shape[1]):
        label = f'$N_z= {Nz[j] :.2f}$'
        ax.plot(flux, G_array[i, j, :], color=palette1[j], linestyle='solid', label=label)

# Figure 1: Format
ax1.legend(ncol=2, frameon=False, fontsize=10)
fig1.suptitle(f'$\mu_l= {mu_leads}$, $E_f= {Ef}$, $r= {r}$, $N_x= {Nx}$, $N_y = {Ny}$, $w= {width[i]}$', y=0.93, fontsize=20)
for ax in ax_vec:
    ax.set_xlim(flux[0], flux[-1])
    ax.set_ylim(0, np.max(G_array))
    ax.tick_params(which='major', width=0.75, labelsize=10)
    ax.tick_params(which='major', length=6, labelsize=10)
    ax.set_xlabel("$\phi$", fontsize=fontsize)
    ax.set_ylabel("$G[2e^2/h]$", fontsize=fontsize)


fig2 = plt.figure()
gs = GridSpec(1, 1, figure=fig2, wspace=0.2, hspace=0.3)
ax1 = fig2.add_subplot(gs[0, 0])
ax1.plot(Nz, G_resonance, marker='o', linestyle='solid', color='dodgerblue')
ax1.set_xlabel('$N_z$')
ax1.set_ylabel('$G$')






fig1.savefig(f'../figures/{file_list[0]}-cond-vs-L.pdf', format='pdf', backend='pgf')
plt.show()
