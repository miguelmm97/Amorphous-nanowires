#%% modules setup

# Math and plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# modules
from modules.functions import *


#%% Loading data
file_list = ['Exp3.h5']
# data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-cond-vs-Ef')
data_dict = load_my_data(file_list, '.')

# Parameters
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
flux_half     = data_dict[file_list[0]]['Simulation']['flux_max']
G_0           = data_dict[file_list[0]]['Simulation']['G_0']
G_half        = data_dict[file_list[0]]['Simulation']['G_half']
fermi         = data_dict[file_list[0]]['Simulation']['fermi']
width         = data_dict[file_list[0]]['Simulation']['width']

#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange']
marker_list=['o', 's', 'd', 'p', '*', 'h', '>', '<', 'X']
line_list = ['solid', 'dashed', 'dashdot', 'dotted']
markersize = 5
fontsize=20

# Figure 1: Definition
fig1 = plt.figure(figsize=(20, 10))
gs = GridSpec(1, 1, figure=fig1, wspace=0., hspace=0.)
ax1 = fig1.add_subplot(gs[0, 0])

# Figure 1: Plots
for i in range(len(G_0[0, :])):
    ax1.plot(fermi, G_0[:, i], color='#9A32CD', label=f'$\phi / \phi_0=0$', linestyle=line_list[i])
    ax1.plot(fermi, G_half[:, i], color='#3F6CFF', alpha=0.5, label=f'$\phi / \phi_0= {flux_half[i] :.2f}$', linestyle=line_list[i])
ax1.legend(ncol=1, frameon=False, fontsize=16)
fig1.suptitle(f'$\mu_l= {mu_leads}$, $r= {r}$, $N_x= {Nx}$, $N_y = {Ny}$, $N_z= {Nz}$', y=0.93, fontsize=20)

# Figure 1: Format
y_axis_ticks = [i for i in range(0, 14, 2)]
y_axis_labels = [str(i) for i in range(0, 14, 2)]
ax1.set_xlim(fermi[0], fermi[-1])
ax1.set_ylim(0, 14)
ax1.tick_params(which='major', width=0.75, labelsize=fontsize)
ax1.tick_params(which='major', length=6, labelsize=fontsize)
ax1.set_xlabel("$E_F / t$", fontsize=fontsize)
ax1.set_ylabel("$G(2e^2/h)$",fontsize=fontsize)
ax1.set(yticks=y_axis_ticks, yticklabels=y_axis_labels)

# fig1.savefig(f'../figures/{file_list[0]}-cond-vs-Ef.pdf', format='pdf', backend='pgf')
plt.show()
