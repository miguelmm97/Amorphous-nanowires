#%% Modules setup

# Math and plotting
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools

# Kwant
import kwant

# Managing data
import h5py
import os
import sys
from datetime import date

# Modules
from modules.functions import *
from modules.AmorphousLattice_2d import AmorphousLattice_2d
from modules.AmorphousWire_kwant import promote_to_kwant_nanowire

#%% Loading data
file_list = ['Exp6.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data_gap/data_oscillations_amorphous_cross_section')

# Parameters
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
x                 = data_dict[file_list[0]]['Simulation']['x_pos']
y                 = data_dict[file_list[0]]['Simulation']['y_pos']
fermi             = data_dict[file_list[0]]['Simulation']['fermi']
kz                = data_dict[file_list[0]]['Simulation']['kz']
G0                = data_dict[file_list[0]]['Simulation']['G0']
G_half            = data_dict[file_list[0]]['Simulation']['G_half']
bands_0           = data_dict[file_list[0]]['Simulation']['bands_0']
bands_half        = data_dict[file_list[0]]['Simulation']['bands_half']
bands_lead_0      = data_dict[file_list[0]]['Simulation']['bands_lead_0']
bands_lead_half   = data_dict[file_list[0]]['Simulation']['bands_lead_half']
bottom_0          = data_dict[file_list[0]]['Simulation']['bottom_0']
bottom_half       = data_dict[file_list[0]]['Simulation']['bottom_half']
bottom_lead_0     = data_dict[file_list[0]]['Simulation']['bottom_lead_0']
bottom_lead_half  = data_dict[file_list[0]]['Simulation']['bottom_lead_half']


# Prepare the lattice to plot
cross_section = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=width, r=r)
cross_section.build_lattice(from_x=x, from_y=y)
nanowire = promote_to_kwant_nanowire(cross_section, n_layers, params_dict, mu_leads=mu_leads).finalized()

#%% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
axcolour = ['#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125']

site_size  = 0.1
site_lw    = 0.01
site_color = 'm'
hop_color  = 'royalblue'
hop_lw     = 0.05
lead_color = 'r'


# Cross section snippet
fig0 = plt.figure(figsize=(10, 4))
gs = GridSpec(1, 2, figure=fig0)
ax0_1 = fig0.add_subplot(gs[0, 0])
ax0_2 = fig0.add_subplot(gs[0, 1], projection='3d')

cross_section.plot_lattice(ax0_1)
kwant.plot(nanowire, site_size=site_size, site_lw=site_lw, site_color=site_color, hop_lw=hop_lw, hop_color=hop_color,
           lead_site_size=site_size, lead_color=lead_color, lead_site_lw=site_lw, lead_hop_lw=hop_lw,
           ax=ax0_2)

ax0_2.set_axis_off()
ax0_2.margins(-0.49, -0.49, -0.49)




# Conductance vs Fermi level
fig1 = plt.figure(figsize=(20, 6))
gs = GridSpec(1, 3, figure=fig1)
ax1_1 = fig1.add_subplot(gs[0, 0])
ax1_2 = fig1.add_subplot(gs[0, 1])
ax1_3 = fig1.add_subplot(gs[0, 2])

ax1_1.plot(fermi, G0, color='#9A32CD', label=f'$\phi / \phi_0= {flux0}$')
for i in range(len(bottom_0)):
    ax1_1.plot(bottom_0[i] * np.ones((10, )), np.linspace(0, 100, 10), '--', color='#9A32CD', alpha=0.1)
    ax1_1.plot(bottom_lead_0[i] * np.ones((10,)), np.linspace(0, 100, 10), '--', color='#FF7D66', alpha=0.1)

ax1_2.plot(fermi, G_half, color='#3F6CFF', alpha=0.5, label=f'$\phi / \phi_0= {flux_half}$ ')
for i in range(len(bottom_half)):
    ax1_2.plot(bottom_half[i] * np.ones((10, )), np.linspace(0, 100, 10), '--', color='#9A32CD', alpha=0.1)
    ax1_2.plot(bottom_lead_half[i] * np.ones((10,)), np.linspace(0, 100, 10), '--', color='#FF7D66', alpha=0.1)

ax1_3.plot(fermi, G0, color='#9A32CD', label='$\phi / \phi_0=0$')
ax1_3.plot(fermi, G_half, color='#3F6CFF', alpha=0.5, label=f'$\phi / \phi_0=0.56$ ')
ax1_3.legend(ncol=1, frameon=False, fontsize=16)

y_axis_ticks = [i for i in range(0, 18, 2)]
y_axis_labels = [str(i) for i in range(0, 18, 2)]
for ax in [ax1_1, ax1_2, ax1_3]:
    ax.set_xlim(fermi[0], fermi[-1])
    ax.set_ylim(0, 18)
    ax.tick_params(which='major', width=0.75, labelsize=10)
    ax.tick_params(which='major', length=6, labelsize=10)
    ax.set_xlabel("$E_F$ [$t$]", fontsize=10)
    ax.set_ylabel("$G[2e^2/h]$",fontsize=10)
    ax.set(yticks=y_axis_ticks, yticklabels=y_axis_labels)

fig1.suptitle(f'$\mu_l= {mu_leads}$, $N_x= {Nx}$, $N_y = {Ny}$, $N_z= {n_layers}$', y=0.93, fontsize=20)
# fig2.savefig('AB-osc.pdf', format='pdf', backend='pgf')



# Bands for the module
fig3 = plt.figure(figsize=(10, 10))
gs = GridSpec(2, 2, figure=fig3)
ax3_1 = fig3.add_subplot(gs[0, 0])
ax3_2 = fig3.add_subplot(gs[0, 1])
ax3_3 = fig3.add_subplot(gs[1, 0])
ax3_4 = fig3.add_subplot(gs[1, 1])

for i in range(bands_0.shape[0]):
    ax3_1.plot(kz, bands_0[i, :], color='#3F6CFF', linewidth=0.5)
    ax3_2.plot(kz, bands_half[i, :], color='#3F6CFF', linewidth=0.5)

ax3_3.plot(kz, bands_lead_0, color='#3F6CFF', linewidth=0.5)
ax3_4.plot(kz, bands_lead_half, color='#3F6CFF', linewidth=0.5)

for ax in [ax3_1, ax3_2, ax3_3, ax3_4]:
    ax.plot(kz, fermi[0] * np.ones(kz.shape), '--', color='#00B5A1')
    ax.plot(kz, fermi[-1] * np.ones(kz.shape), '--', color='#00B5A1')

ax3_1.set_title(f'$\phi / \phi_0= {flux0}$ wire')
ax3_2.set_title(f'$\phi / \phi_0= {flux_half}$ wire')
ax3_3.set_title(f'$\phi / \phi_0= {flux0}$ leads')
ax3_4.set_title(f'$\phi / \phi_0= {flux_half}$ leads')

for ax in [ax3_1, ax3_2, ax3_3, ax3_4]:
    ax.set_xlabel('$k/a$')
    ax.set_ylabel('$E(k)/t$')
    ax.set_xlim(-pi, pi)
    ax.set_ylim(-10, 10)
    ax.tick_params(which='major', width=0.75, labelsize=10)
    ax.tick_params(which='major', length=6, labelsize=10)
    ax.set(xticks=[-pi, -pi/2, 0, pi/2, pi], xticklabels=['$-\pi$', '$-\pi/2$', '$0$', '$\pi/2$', '$\pi$'])

plt.show()




