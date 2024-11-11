#%% modules setup

# Math and plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn

# modules
from modules.functions import *


#%% Loading data
file_list = ['Exp1.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-cond-vs-width')

# Parameters
Nx           = data_dict[file_list[0]]['Parameters']['Nx']
Ny           = data_dict[file_list[0]]['Parameters']['Ny']
r            = data_dict[file_list[0]]['Parameters']['r']
t            = data_dict[file_list[0]]['Parameters']['t']
eps          = data_dict[file_list[0]]['Parameters']['eps']
lamb         = data_dict[file_list[0]]['Parameters']['lamb']
lamb_z       = data_dict[file_list[0]]['Parameters']['lamb_z']
mu_leads     = data_dict[file_list[0]]['Parameters']['mu_leads']

# Simulation data
Ef            = data_dict[file_list[0]]['Simulation']['Ef']
flux          = data_dict[file_list[0]]['Simulation']['flux']
width         = data_dict[file_list[0]]['Simulation']['width']
Nz            = data_dict[file_list[0]]['Simulation']['Nz']
G_array       = data_dict[file_list[0]]['Simulation']['G_array']

#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange']
palette1 = seaborn.color_palette(palette='magma', n_colors=G_array.shape[2])
palette2 = seaborn.color_palette(palette='magma', n_colors=5)
marker_list=['o', 's', 'd', 'p', '*', 'h', '>', '<', 'X']
markersize = 5
fontsize = 20

# Figure 1: Definition
fig1 = plt.figure(figsize=(20, 15))
gs = GridSpec(3, 2, figure=fig1, wspace=0.2, hspace=0.3)
Ncol = 2
Nrow = int(np.ceil(G_array.shape[2] / Ncol))

ax_vec = []
for i in range(Nrow):
    for j in range(Ncol):
        ax = fig1.add_subplot(gs[i, j])
        ax_vec.append(ax)


# Figure 1: Plots
for i in range(G_array.shape[2]):
    ax = ax_vec[i]
    ax.set_title(f'$E_f= {Ef[i]}$', fontsize=fontsize)
    for j in range(G_array.shape[1]):
        label = f'$\phi= {flux[j] :.2f}$'
        ax.plot(width, G_array[:, j, i, 0, 0], color=palette1[j], linestyle='solid', marker=marker_list[j], label=label)

# Figure 1: Format
ax_vec[0].legend(ncol=2, frameon=False, fontsize=10)
fig1.suptitle(f'$\mu_l= {mu_leads}$, $E_f= {Ef}$, $r= {r}$, $N_x= {Nx}$, $N_y = {Ny}$', y=0.93, fontsize=20)
for ax in ax_vec:
    ax.set_xlim(Nz[-1], Nz[0])
    ax.set_ylim(0, np.max(G_array))
    ax.tick_params(which='major', width=0.75, labelsize=10)
    ax.tick_params(which='major', length=6, labelsize=10)
    ax.set_xlabel("$L$", fontsize=fontsize)
    ax.set_ylabel("$G[2e^2/h]$", fontsize=fontsize)

fig1.savefig(f'../figures/{file_list[0]}-cond-vs-width.pdf', format='pdf', backend='pgf')
plt.show()
