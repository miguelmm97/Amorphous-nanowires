#%% modules setup

# Math and plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn

# modules
from modules.functions import *


#%% Loading data
file_list = ['Exp44.h5']
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

index_flux1 = [0, 42, 92, 125, 166]
index_flux2 = [0, 42, 113, 125, 166]
index_flux3 = [0, 42, 123, 125, 166]
index_flux4 = [0, 42, 92, 125, 224]
index_flux = [index_flux1, index_flux2, index_flux3, index_flux4]
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

# Figure 1: Definition
fig1 = plt.figure(figsize=(20, 6))
gs = GridSpec(2, 4, figure=fig1, wspace=0.2, hspace=0.6)
ax1 = fig1.add_subplot(gs[0, 0])
ax2 = fig1.add_subplot(gs[0, 1])
ax3 = fig1.add_subplot(gs[0, 2])
ax4 = fig1.add_subplot(gs[0, 3])
ax5 = fig1.add_subplot(gs[1, 0])
ax6 = fig1.add_subplot(gs[1, 1])
ax7 = fig1.add_subplot(gs[1, 2])
ax8 = fig1.add_subplot(gs[1, 3])

ax_vec = [ax1, ax2, ax3, ax4]
# Upper panel: Plots
for i in range(G_array.shape[0]):
    ax = ax_vec[i]
    # ax.set_title(f'$w= {width[i]}$', fontsize=fontsize)
    for j in range(5):
        index = index_flux[i]
        label = f'$\phi= {flux[index[j]] :.2f}$'
        ax.plot(Nz[::-1], G_array[i, :, index[j], 0][::-1], color=palette2[j], linestyle='solid', marker=marker_list[j],
                markersize=5, label=label)

# Upper panel: Format
ax2.legend(loc='upper center', ncol=9, frameon=False, fontsize=fontsize - 5, bbox_to_anchor=(1.05, 1.25))
for ax in ax_vec:
    ax.set(xticks=np.linspace(50, 200, 4))
    ax.set(yticks=[0, 0.5, 1], yticklabels=[])
    ax.set_xlim(Nz[-1], Nz[0])
    ax.set_ylim(0, 1.2)
    ax.tick_params(which='major', width=0.75, labelsize=fontsize)
    ax.tick_params(which='major', length=6, labelsize=fontsize)
    ax.set_xlabel("$L$", fontsize=fontsize)
ax1.set(yticks=[0, 0.5, 1], yticklabels=['0', '0.5', '1'])
ax1.set_ylabel("$G[2e^2/h]$", fontsize=fontsize)


# Lower panel: Plots
ax_vec = [ax5, ax6, ax7, ax8]
for i in range(G_array.shape[0]):
    ax = ax_vec[i]
    ax.plot(flux, 1 * np.ones(flux.shape), '--', color='Black', alpha=0.2)
    for j in range(G_array.shape[1] - 1, -1, -1):
        label = f'$L= {Nz[j]}$'
        ax.plot(flux, G_array[i, j, :, 0], color=palette1[j], linestyle='solid', label=label)


# Lower panel: Format
ax6.legend(loc='upper center', ncol=9, frameon=False, fontsize=fontsize - 5, bbox_to_anchor=(1.05, 1.25))
for ax in ax_vec:
    ax.set_xlim(flux[0], flux[-1])
    ax.set_ylim(0, 1.2)
    ax.set(yticks=[0, 0.5, 1], yticklabels=[])
    ax.tick_params(which='major', width=0.75, labelsize=fontsize)
    ax.tick_params(which='major', length=6, labelsize=fontsize)
    ax.set_xlabel("$\phi$", fontsize=fontsize)
ax5.set(yticks=[0, 0.5, 1], yticklabels=['0', '0.5', '1'])
ax5.set_ylabel("$G[2e^2/h]$", fontsize=fontsize)
ax5.text(0.3, 0.2, f'$w= {width[0]}$', fontsize=fontsize)
ax6.text(0.32, 0.2, f'$w= {width[1]}$', fontsize=fontsize)
ax7.text(0.35, 0.2, f'$w= {width[2]}$', fontsize=fontsize)
ax8.text(0.4, 0.2, f'$w= {width[3]}$', fontsize=fontsize)



# fig1.savefig('draft-fig4.pdf', format='pdf', backend='pgf')
plt.show()
