#%% modules setup

# Math and plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# modules
from modules.functions import *


#%% Loading data
file_list = ['Exp4.h5']
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
width         = data_dict[file_list[0]]['Simulation']['width']

try:
    aux = len(Ef)
except TypeError:
    Ef = [Ef]
    G_array = np.array([G_array])
#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125', '#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']
style_list = ['solid', 'dashed']
markersize = 5

# Figure 1: Definition
fig1 = plt.figure(figsize=(10, 10))
gs = GridSpec(1, 1, figure=fig1)
ax1 = fig1.add_subplot(gs[0, 0])

# Figure 1: Plots
for i in range(G_array.shape[0]):
    for j in range(len(Ef)):
        ax1.plot(flux, G_array[j, i, :], color=color_list[i], linestyle=style_list[j], label=f'$w= {width[i]}$')
ax1.plot(flux, 1 * np.ones(flux.shape),  color='Black', alpha=0.5)

# Figure 1: Format
ax1.legend(ncol=2, frameon=False, fontsize=16)
fig1.suptitle(f'$\mu_l= {mu_leads}$, $r= {r}$, $N_x= {Nx}$, $N_y = {Ny}$, $N_z= {Nz}$', y=0.93, fontsize=20)
ylim = 1.3
for ax in [ax1]:
    ax.set_xlim(flux[0], flux[-1])
    ax.set_ylim(0, ylim)
    ax.tick_params(which='major', width=0.75, labelsize=10)
    ax.tick_params(which='major', length=6, labelsize=10)
    ax.set_xlabel("$\phi$", fontsize=10)
    ax.set_ylabel("$G[2e^2/h]$", fontsize=10)

fig1.savefig(f'../figures/{file_list[0]}.pdf', format='pdf', backend='pgf')
plt.show()
