#%% Modules setup

# Math and plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from numpy import pi

# modules
import kwant
from modules.functions import *
from modules.AmorphousWire_kwant import thermal_average, infinite_nanowire_kwant
#%% Loading data
file_list = ['Exp37.h5', 'Exp36.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-cond-vs-Ef')

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
params_dict  = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}

Nx_2           = data_dict[file_list[1]]['Parameters']['Nx']
Ny_2           = data_dict[file_list[1]]['Parameters']['Ny']
Nz_2           = data_dict[file_list[1]]['Parameters']['Nz']
mu_leads_2     = data_dict[file_list[1]]['Parameters']['mu_leads']



# File 1
G_0_1          = data_dict[file_list[0]]['Simulation']['G_0']
fermi_1        = data_dict[file_list[0]]['Simulation']['fermi']
width_1        = data_dict[file_list[0]]['Simulation']['width']

G_0_2          = data_dict[file_list[1]]['Simulation']['G_0']
fermi_2        = data_dict[file_list[1]]['Simulation']['fermi']
width_2        = data_dict[file_list[1]]['Simulation']['width']

kz = np.linspace(-pi, pi, 101)
wire_kwant = infinite_nanowire_kwant(Nx, Ny, params_dict, mu_leads=-1.).finalized()
bands = kwant.physics.Bands(wire_kwant, params=dict(flux=0))
bottom_leads = bands(0)
bands = [bands(k) for k in kz]


#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange']
marker_list=['o', 's', 'd', 'p', '*', 'h', '>', '<', 'X']
line_list = ['solid', 'dashed', 'dashdot', 'dotted']
markersize = 5
fontsize=20
site_size  = 0.1
site_lw    = 0.01
site_color = 'm'
hop_color  = 'royalblue'
hop_lw     = 0.05
lead_color = 'r'

fig1 = plt.figure(figsize=(15, 7))
gs = GridSpec(1, 2, figure=fig1, wspace=0.1, hspace=0.0)
ax1 = fig1.add_subplot(gs[0, 0])
ax2 = fig1.add_subplot(gs[0, 1])
ax_amor = [ax1, ax2]


for i in range(1):
    # Thermal average
    kBT = 0.05
    G01_th, Ef_th1 = thermal_average(G_0_1[:, i], fermi_1, kBT)
    G02_th, Ef_th2 = thermal_average(G_0_2[:, i], fermi_2, kBT)

    ax1.plot(fermi_1, G_0_1[:, i], color=color_list[i], label=f'$w= {width_1[i]}$', linestyle='solid')
    ax2.plot(fermi_2, G_0_2[:, i], color=color_list[i],linestyle='solid')
    ax1.plot(Ef_th1, G01_th, color='r', linestyle='dashed')
    ax2.plot(Ef_th2, G02_th, color='r', linestyle='dashed')


for energy in bottom_leads:
    ax1.plot(energy * np.ones((100, )), np.linspace(0, 10, 100), linestyle='dashed', color='dodgerblue', alpha=0.1)

for ax in ax_amor:
    y_axis_ticks = [i for i in range(0, 11, 2)]
    y_axis_labels = ['' for i in range(0, 11, 2)]
    ax.set_xlim(0, fermi_1[-1])
    ax.set_ylim(0, 10)
    ax.tick_params(which='major', width=0.75, labelsize=fontsize)
    ax.tick_params(which='major', length=6, labelsize=fontsize)
    ax.set_xlabel("$E_F / t$", fontsize=fontsize)
    ax.set(yticks=y_axis_ticks, yticklabels=y_axis_labels)
ax1.text(1, 8, f'$Nx={Nx}, N_z={Nz}, \mu= {mu_leads}$')
ax2.text(1, 8, f'$Nx={Nx_2}, N_z={Nz_2}, \mu= {mu_leads_2}$')

y_axis_ticks = [i for i in range(0, 11, 2)]
y_axis_labels = [str(i) for i in range(0, 11, 2)]
ax1.set_ylabel("$G(2e^2/h)$", fontsize=fontsize)
ax1.set(yticks=y_axis_ticks, yticklabels=y_axis_labels)
ax1.legend(ncol=1, frameon=False, fontsize=fontsize)
ax2.legend(ncol=1, frameon=False, fontsize=fontsize)


# Bands
fig2 = plt.figure(figsize=(10, 10))
gs = GridSpec(1, 1, figure=fig2)
ax1 = fig2.add_subplot(gs[0, 0])

ax1.plot(kz, bands, color='dodgerblue', linewidth=0.5)
ax1.plot(kz, 0. * np.ones(kz.shape), '--', color='Black', alpha=0.2)
ax1.plot(kz, 1 * np.ones(kz.shape), '--', color='Black', alpha=0.2)

ax1.set_xlabel('$k/a$')
ax1.set_ylabel('$E(k)/t$')
# ax1.set_xlim(-0.2, 0.2)
# ax1.set_ylim(-0.2, 0.2)
ax1.tick_params(which='major', width=0.75, labelsize=10)
ax1.tick_params(which='major', length=6, labelsize=10)








# fig1.savefig(f'draft-fig6.pdf', format='pdf', backend='pgf')
plt.show()





