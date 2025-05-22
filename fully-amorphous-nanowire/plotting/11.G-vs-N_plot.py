#%% modules setup

# Math and plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn

# modules
from modules.functions import *


#%% Loading data
file_list = ['Exp16.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-cond-vs-N')

# Parameters
Nz           = data_dict[file_list[0]]['Parameters']['Nz']
Ef           = data_dict[file_list[0]]['Parameters']['Ef']
r            = data_dict[file_list[0]]['Parameters']['r']
t            = data_dict[file_list[0]]['Parameters']['t']
eps          = data_dict[file_list[0]]['Parameters']['eps']
lamb         = data_dict[file_list[0]]['Parameters']['lamb']
lamb_z       = data_dict[file_list[0]]['Parameters']['lamb_z']
mu_leads     = data_dict[file_list[0]]['Parameters']['mu_leads']


# Simulation data
Nx            = data_dict[file_list[0]]['Simulation']['Nx']
Ny            = data_dict[file_list[0]]['Simulation']['Ny']
flux          = data_dict[file_list[0]]['Simulation']['flux']
width         = data_dict[file_list[0]]['Simulation']['width']
G_array       = data_dict[file_list[0]]['Simulation']['G_array']
K_onsite     = data_dict[file_list[0]]['Simulation']['K_onsite']



#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange']
palette1 = seaborn.color_palette(palette='magma', n_colors=G_array.shape[0])
palette2 = seaborn.color_palette(palette='magma', n_colors=5)
marker_list=['o', 's', 'd', 'p', '*', 'h', '>', '<', 'X']
markersize = 5
fontsize=20

# Figure 1: Definition
fig1 = plt.figure(figsize=(15, 8))
gs = GridSpec(1, 1, figure=fig1, wspace=0.2, hspace=0.3)
ax1 = fig1.add_subplot(gs[0, 0])

# Figure 1: Plots
for i in range(G_array.shape[0]):
    label = f'$N_z= {Nx[i]}$'
    ax1.plot(flux, G_array[i, :], color=palette1[i], linestyle='solid', label=label)
ax1.plot(flux, np.ones((len(flux), )), color='k', linestyle='dashed')

# Figure 1: Format
ax1.legend(ncol=2, frameon=False, fontsize=10)
fig1.suptitle(f'$\mu_l= {mu_leads}$, $E_f= {Ef}$, $r= {r}$, $L= {Nz}$, $w= {width}$, $K= {K_onsite:.2f}$', y=0.93, fontsize=20)
ax1.set_xlim(flux[0], flux[-1])
ax1.set_ylim(0, 1.2)
ax1.tick_params(which='major', width=0.75, labelsize=10)
ax1.tick_params(which='major', length=6, labelsize=10)
ax1.set_xlabel("$\phi/\phi_0$", fontsize=fontsize)
ax1.set_ylabel("$G[2e^2/h]$", fontsize=fontsize)



fig1.savefig(f'../figures/{file_list[0]}-cond-vs-N.pdf', format='pdf', backend='pgf')
plt.show()
