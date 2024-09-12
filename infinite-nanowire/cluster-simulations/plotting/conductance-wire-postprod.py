#%% Modules setup

# Math and plotting
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Kwant
import kwant

import h5py

# Modules
from modules.functions import *
from modules.AmorphousLattice_2d import AmorphousLattice_2d
from modules.AmorphousWire_kwant import promote_to_kwant_nanowire

#%% Loading data
file_list = ['final-data-conductance-wire.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-cluster')

# Parameters
Nsamples     = data_dict[file_list[0]]['Parameters']['Nsamples']
Nx           = data_dict[file_list[0]]['Parameters']['Nx']
Ny           = data_dict[file_list[0]]['Parameters']['Ny']
n_layers     = data_dict[file_list[0]]['Parameters']['n_layers']
width        = data_dict[file_list[0]]['Parameters']['width']
r            = data_dict[file_list[0]]['Parameters']['r']
t            = data_dict[file_list[0]]['Parameters']['t']
eps          = data_dict[file_list[0]]['Parameters']['eps']
lamb         = data_dict[file_list[0]]['Parameters']['lamb']
lamb_z       = data_dict[file_list[0]]['Parameters']['lamb_z']
mu_leads     = data_dict[file_list[0]]['Parameters']['mu_leads']
flux0        = data_dict[file_list[0]]['Parameters']['flux0']
flux_half    = data_dict[file_list[0]]['Parameters']['flux_half']
params_dict  = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}

# Simulation data
fermi        = data_dict[file_list[0]]['Simulation']['fermi']
kz           = data_dict[file_list[0]]['Simulation']['kz']
avg_G0       = data_dict[file_list[0]]['Simulation']['avg_G0']
avg_G_half   = data_dict[file_list[0]]['Simulation']['avg_G_half']
std_G0       = data_dict[file_list[0]]['Simulation']['std_G0']
std_G_half   = data_dict[file_list[0]]['Simulation']['std_G_half']


#%% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
axcolour = ['#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125']

labelsize    = 10
major_width  = 0.75
major_length = 6
fontsize     = 20

error_0_up = avg_G0 + 0.5 * std_G0 / np.sqrt(Nsamples)
error_0_down = avg_G0 - 0.5 * std_G0 / np.sqrt(Nsamples)
error_half_up = avg_G_half + 0.5 * std_G_half / np.sqrt(Nsamples)
error_half_down = avg_G_half - 0.5 * std_G_half / np.sqrt(Nsamples)

# Conductance vs Fermi level
fig0 = plt.figure(figsize=(10, 6))
ax0 = fig0.gca()
ax0.plot(fermi, avg_G0, color='#9A32CD', label='$\phi / \phi_0=0$')
ax0.plot(fermi, avg_G_half, color='#3F6CFF', alpha=0.5, label=f'$\phi / \phi_0=0.56$ ')
ax0.fill_between(fermi, error_0_down, error_0_up, color='#9A32CD', alpha=0.3)
ax0.fill_between(fermi, error_half_down, error_half_up,  color='#3F6CFF', alpha=0.3)
ax0.legend(ncol=1, frameon=False, fontsize=fontsize)

y_axis_ticks = [i for i in range(0, 6, 2)]
y_axis_labels = [str(i) for i in range(0, 6, 2)]
ax0.set_xlim(fermi[0], 0.4)
ax0.set_ylim(0, 6)
ax0.tick_params(which='major', width=major_width, labelsize=labelsize)
ax0.tick_params(which='major', length=major_length, labelsize=labelsize)
ax0.set_xlabel("$E_F$ [$t$]", fontsize=fontsize)
ax0.set_ylabel("$G[2e^2/h]$",fontsize=fontsize)
ax0.set(yticks=y_axis_ticks, yticklabels=y_axis_labels)
fig0.suptitle(f'Average over ${Nsamples}$ samples with $\mu_l= {mu_leads}$, $N_x= {Nx}$, $N_y = {Ny}$, $N_z= {n_layers}$', y=0.93, fontsize=fontsize)
fig0.savefig(f'AB-osc.pdf', format='pdf', backend='pgf')
plt.show()




