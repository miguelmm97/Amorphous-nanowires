#%% modules setup

# Math and plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn

# modules
from modules.functions import *


#%% Loading data
file_list = ['Exp22.h5']
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
disorder      = data_dict[file_list[0]]['Simulation']['disorder']
Nz            = data_dict[file_list[0]]['Simulation']['Nz']
G_array       = data_dict[file_list[0]]['Simulation']['G_array']

#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange']
palette = seaborn.color_palette(palette='magma', n_colors=G_array.shape[1])
markersize = 5
fontsize=20

# Figure 1: Definition
fig1 = plt.figure(figsize=(20, 10))
gs = GridSpec(2, 2, figure=fig1, wspace=0.13, hspace=0.3)
ax1_1 = fig1.add_subplot(gs[0, 0])
ax1_2 = fig1.add_subplot(gs[0, 1])
ax1_3 = fig1.add_subplot(gs[1, 0])
ax1_4 = fig1.add_subplot(gs[1, 1])
ax_vec = [ax1_1, ax1_2, ax1_3, ax1_4]


# Figure 1: Plots
disorder_sample = 0
for i in range(G_array.shape[0]):
    ax = ax_vec[i]
    ax.plot(flux, 1 * np.ones(flux.shape), '--', color='Black', alpha=0.2)
    ax.set_title(f'$w= {width[i]}$', fontsize=fontsize)
    for j in range(G_array.shape[1] - 1, -1, -1):
        label = f'$L= {Nz[j]}$'
        ax.plot(flux, G_array[i, j, :, disorder_sample], color=palette[j], linestyle='solid', label=label)


# Figure 1: Format
ax1_1.legend(ncol=2, frameon=False, fontsize=10)
fig1.suptitle(f'$\mu_l= {mu_leads}$, $E_f= {Ef}$, $r= {r}$, $N_x= {Nx}$, $N_y = {Ny}$', y=0.93, fontsize=20)
for ax in ax_vec:
    ax.set_xlim(flux[0], flux[-1])
    ax.set_ylim(0, np.max(G_array))
    ax.tick_params(which='major', width=0.75, labelsize=10)
    ax.tick_params(which='major', length=6, labelsize=10)
    ax.set_xlabel("$\phi$", fontsize=fontsize)
    ax.set_ylabel("$G[2e^2/h]$", fontsize=fontsize)
fig1.savefig(f'../figures/{file_list[0]}-peak-shape-width.pdf', format='pdf', backend='pgf')




# Figure 2: Definition
fig2 = plt.figure(figsize=(20, 10))
gs = GridSpec(2, 2, figure=fig2, wspace=0.13, hspace=0.3)
ax2_1 = fig2.add_subplot(gs[0, 0])
ax2_2 = fig2.add_subplot(gs[0, 1])
ax2_3 = fig2.add_subplot(gs[1, 0])
ax2_4 = fig2.add_subplot(gs[1, 1])
ax_vec = [ax2_1, ax2_2, ax2_3, ax2_4]


# Figure 2: Plots
width_sample = 0
for i in range(G_array.shape[-1]):
    ax = ax_vec[i]
    ax.plot(flux, 1 * np.ones(flux.shape), '--', color='Black', alpha=0.2)
    ax.set_title(f'$K= {disorder[i]}$', fontsize=fontsize)
    for j in range(G_array.shape[1] - 1, -1, -1):
        label = f'$L= {Nz[j]}$'
        ax.plot(flux, G_array[width_sample, j, :, i], color=palette[j], linestyle='solid', label=label)


# Figure 2: Format
ax2_1.legend(ncol=2, frameon=False, fontsize=10)
fig2.suptitle(f'$\mu_l= {mu_leads}$, $E_f= {Ef}$, $r= {r}$, $N_x= {Nx}$, $N_y = {Ny}$', y=0.93, fontsize=20)
for ax in ax_vec:
    ax.set_xlim(flux[0], flux[-1])
    ax.set_ylim(0, np.max(G_array))
    ax.tick_params(which='major', width=0.75, labelsize=10)
    ax.tick_params(which='major', length=6, labelsize=10)
    ax.set_xlabel("$\phi$", fontsize=fontsize)
    ax.set_ylabel("$G[2e^2/h]$", fontsize=fontsize)
fig2.savefig(f'../figures/{file_list[0]}-peak-shape-disorder.pdf', format='pdf', backend='pgf')


plt.show()
