#%% modules setup

# Math and plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn

# modules
from modules.functions import *


#%% Loading data
file_list = ['Exp20.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-cond-vs-flux-fully-amorphous')

# Parameters
Ef           = data_dict[file_list[0]]['Parameters']['Ef']
Nx           = data_dict[file_list[0]]['Parameters']['Nx']
Ny           = data_dict[file_list[0]]['Parameters']['Ny']
Nz           = data_dict[file_list[0]]['Parameters']['Nz']
r            = data_dict[file_list[0]]['Parameters']['r']
t            = data_dict[file_list[0]]['Parameters']['t']
eps          = data_dict[file_list[0]]['Parameters']['eps']
lamb         = data_dict[file_list[0]]['Parameters']['lamb']
lamb_z       = data_dict[file_list[0]]['Parameters']['lamb_z']
mu_leads     = data_dict[file_list[0]]['Parameters']['mu_leads']

# Simulation data
flux          = data_dict[file_list[0]]['Simulation']['flux']
G_array       = data_dict[file_list[0]]['Simulation']['G_array']
# G_array_typo  = data_dict[file_list[1]]['Simulation']['G_array']
width         = data_dict[file_list[0]]['Simulation']['width']

# Solve typo in the data for Exp7 and 8
try:
    aux = len(Ef)
except TypeError:
    Ef = [Ef]
    G_array = np.array([G_array])

# G_aux = np.ones((len(Ef), len(width), len(flux)))
# for i in range(len(width)):
#     if i < 2:
#         G_aux[:, i, :] = G_array[:, i, :]
#     elif i > 2:
#         G_aux[:, i, :] = G_array[:, i - 1, :]
#     else:
#         G_aux[:, i, :] = G_array_typo[:, 0, :]
# G_array = G_aux
# width = [0.001, 0.002, 0.02, 0.05, 0.1]
#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange']
palette = seaborn.color_palette(palette='Spectral', n_colors=G_array.shape[1])
markersize = 5
fontsize=20

# Figure 1: Definition
fig1 = plt.figure(figsize=(20, 10))
gs = GridSpec(3, 1, figure=fig1)
ax1 = fig1.add_subplot(gs[0, 0])
ax2 = fig1.add_subplot(gs[1, 0])
ax3 = fig1.add_subplot(gs[2, 0])
ax_vec = [ax1, ax2, ax3]

# Figure 1: Plots
for i in range(G_array.shape[1]):
    for j in range(len(Ef)):
        ax = ax_vec[j]
        label = None if (j % 2 != 0) else f'$w= {width[i]}$'
        ax.plot(flux, G_array[j, i, :], color=color_list[i], linestyle='solid', label=label)
        ax.plot(flux, 1 * np.ones(flux.shape), '--', color='Black', alpha=0.2)
        ax.text(0.5, 1.1, f'$E_f= {Ef[j]}$', fontsize=fontsize)


# Figure 1: Format
ax1.legend(ncol=5, frameon=False, fontsize=20)
fig1.suptitle(f'$\mu_l= {mu_leads}$, $r= {r}$, $N_x= {Nx}$, $N_y = {Ny}$, $N_z= {Nz}$', y=0.93, fontsize=20)
ylim = np.max(G_array)
for ax in ax_vec:
    ax.set_xlim(flux[0], flux[-1])
    ax.set_ylim(0, ylim)
    ax.tick_params(which='major', width=0.75, labelsize=10)
    ax.tick_params(which='major', length=6, labelsize=10)
    ax.set_xlabel("$\phi$", fontsize=fontsize)
    ax.set_ylabel("$G[2e^2/h]$", fontsize=fontsize)

fig1.savefig(f'../figures/{file_list[0]}.pdf', format='pdf', backend='pgf')
plt.show()
