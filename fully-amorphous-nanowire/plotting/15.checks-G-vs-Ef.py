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
file_list = ['Exp35.h5', 'Exp36.h5','Exp37.h5', 'Exp38.h5', 'Exp34.h5', 'Exp30.h5', 'Exp39.h5', 'Exp40.h5', 'Exp41.h5']
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
Nx_3           = data_dict[file_list[2]]['Parameters']['Nx']
Ny_3           = data_dict[file_list[2]]['Parameters']['Ny']
Nz_3           = data_dict[file_list[2]]['Parameters']['Nz']
mu_leads_3     = data_dict[file_list[2]]['Parameters']['mu_leads']
Nx_4           = data_dict[file_list[3]]['Parameters']['Nx']
Ny_4           = data_dict[file_list[3]]['Parameters']['Ny']
Nz_4           = data_dict[file_list[3]]['Parameters']['Nz']
mu_leads_4     = data_dict[file_list[3]]['Parameters']['mu_leads']
Nx_5           = data_dict[file_list[4]]['Parameters']['Nx']
Ny_5           = data_dict[file_list[4]]['Parameters']['Ny']
Nz_5           = data_dict[file_list[4]]['Parameters']['Nz']
mu_leads_5     = data_dict[file_list[4]]['Parameters']['mu_leads']
Nx_6           = data_dict[file_list[5]]['Parameters']['Nx']
Ny_6           = data_dict[file_list[5]]['Parameters']['Ny']
Nz_6           = data_dict[file_list[5]]['Parameters']['Nz']
mu_leads_6     = data_dict[file_list[5]]['Parameters']['mu_leads']
Nx_7           = data_dict[file_list[6]]['Parameters']['Nx']
Ny_7           = data_dict[file_list[6]]['Parameters']['Ny']
Nz_7           = data_dict[file_list[6]]['Parameters']['Nz']
mu_leads_7     = data_dict[file_list[6]]['Parameters']['mu_leads']
Nx_8           = data_dict[file_list[7]]['Parameters']['Nx']
Ny_8           = data_dict[file_list[7]]['Parameters']['Ny']
Nz_8           = data_dict[file_list[7]]['Parameters']['Nz']
mu_leads_8     = data_dict[file_list[7]]['Parameters']['mu_leads']
Nx_9           = data_dict[file_list[8]]['Parameters']['Nx']
Ny_9           = data_dict[file_list[8]]['Parameters']['Ny']
Nz_9           = data_dict[file_list[8]]['Parameters']['Nz']
mu_leads_9     = data_dict[file_list[8]]['Parameters']['mu_leads']


# File 1
G_0_1          = data_dict[file_list[0]]['Simulation']['G_0']
fermi_1        = data_dict[file_list[0]]['Simulation']['fermi']
width_1        = data_dict[file_list[0]]['Simulation']['width']
G_0_2          = data_dict[file_list[1]]['Simulation']['G_0']
fermi_2        = data_dict[file_list[1]]['Simulation']['fermi']
width_2        = data_dict[file_list[1]]['Simulation']['width']
G_0_3          = data_dict[file_list[2]]['Simulation']['G_0']
fermi_3        = data_dict[file_list[2]]['Simulation']['fermi']
width_3        = data_dict[file_list[2]]['Simulation']['width']
G_0_4          = data_dict[file_list[3]]['Simulation']['G_0']
fermi_4        = data_dict[file_list[3]]['Simulation']['fermi']
width_4        = data_dict[file_list[3]]['Simulation']['width']
G_0_5          = data_dict[file_list[4]]['Simulation']['G_0']
fermi_5        = data_dict[file_list[4]]['Simulation']['fermi']
width_5        = data_dict[file_list[4]]['Simulation']['width']
G_0_6          = data_dict[file_list[5]]['Simulation']['G_0']
fermi_6        = data_dict[file_list[5]]['Simulation']['fermi']
width_6        = data_dict[file_list[5]]['Simulation']['width']
G_0_7          = data_dict[file_list[6]]['Simulation']['G_0']
fermi_7        = data_dict[file_list[6]]['Simulation']['fermi']
width_7        = data_dict[file_list[6]]['Simulation']['width']
spectrum       = data_dict[file_list[6]]['Simulation']['eps']
G_0_8          = data_dict[file_list[7]]['Simulation']['G_0']
fermi_8        = data_dict[file_list[7]]['Simulation']['fermi']
width_8        = data_dict[file_list[7]]['Simulation']['width']
G_0_9          = data_dict[file_list[8]]['Simulation']['G_0']
fermi_9        = data_dict[file_list[8]]['Simulation']['fermi']
width_9        = data_dict[file_list[8]]['Simulation']['width']

# Bottom and top of the bands
kz = np.linspace(-pi, pi, 101)
wire_kwant = infinite_nanowire_kwant(5, 5, params_dict, mu_leads=-1.).finalized()
bands = kwant.physics.Bands(wire_kwant, params=dict(flux=0))
bottom_bands = bands(0)
top_bands = bands(pi)
bands = [bands(k) for k in kz]

# Number of open channels in the leads
Nchannels_lead = np.zeros(np.shape(fermi_9))
for i, E in enumerate(fermi_9):
    if E < bottom_bands[-1]:
        channel_openings = len(np.where((E > bottom_bands))[0])
    if E < top_bands[-1]:
        channel_closings = len(np.where((E > top_bands))[0])
    Nchannels_lead[i] = channel_openings - channel_closings

# Number of states per energy in the closed system
tol = 10 * (fermi_8[2] - fermi_8[0])
DoS_closed_nw = np.zeros(np.shape(fermi_8))
for i, E in enumerate(fermi_8):
    DoS_closed_nw[i] = len(np.where(np.abs(E - spectrum) < tol)[0])

#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange', 'forestgreen']
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
gs = GridSpec(2, 2, figure=fig1, wspace=0.1, hspace=0.2)
ax1 = fig1.add_subplot(gs[0, 0])
ax2 = fig1.add_subplot(gs[0, 1])
ax3 = fig1.add_subplot(gs[1, 1])
ax4 = fig1.add_subplot(gs[1, 0])
ax_amor = [ax1, ax2, ax3, ax4]


for i in range(1):
    # Thermal average
    kBT = 0.1
    G01_th, Ef_th1 = thermal_average(G_0_1[:, i], fermi_1, kBT)
    G02_th, Ef_th2 = thermal_average(G_0_2[:, i], fermi_2, kBT)
    G03_th, Ef_th3 = thermal_average(G_0_3[:, i], fermi_3, kBT)
    G04_th, Ef_th4 = thermal_average(G_0_4[:, i], fermi_4, kBT)
    G05_th, Ef_th5 = thermal_average(G_0_5[:, i], fermi_5, kBT)
    G06_th, Ef_th6 = thermal_average(G_0_6[:, i], fermi_6, kBT)
    G07_th, Ef_th7 = thermal_average(G_0_7[:, i], fermi_7, kBT)
    G08_th, Ef_th8 = thermal_average(G_0_8[:, i], fermi_8, kBT)
    G09_th, Ef_th9 = thermal_average(G_0_9[:, i], fermi_9, kBT)

    ax1.plot(fermi_1, G_0_1[:, i], color=color_list[1], label=f'$w= {width_1[i]}, N_x={Nx}, N_z={Nz}, \mu= {mu_leads}$', linestyle='solid', alpha=0.3)
    ax1.plot(fermi_6, G_0_6[:, i], color=color_list[2], label=f'$w= {width_6[i]}, N_x={Nx_6}, N_z={Nz_6}, \mu= {mu_leads_6}$',linestyle='solid', alpha=0.3)
    ax2.plot(fermi_2, G_0_2[:, i], color=color_list[1], label=f'$w= {width_2[i]}, N_x={Nx_2}, N_z={Nz_2}, \mu= {mu_leads_2},$' + 'smooth interface', linestyle='solid', alpha=0.3)
    ax2.plot(fermi_3, G_0_3[:, i], color=color_list[2], label=f'$w= {width_3[i]}, N_x={Nx_3}, N_z={Nz_3}, \mu= {mu_leads_3}$', linestyle='solid', alpha=0.3)
    ax2.plot(fermi_4, G_0_4[:, i], color=color_list[3], label=f'$w= {width_4[i]}, N_x={Nx_4}, N_z={Nz_4}, \mu= {mu_leads_4}$', linestyle='solid', alpha=0.3)
    ax2.plot(fermi_5, G_0_5[:, i], color=color_list[4], label=f'$w= {width_5[i]}, N_x={Nx_5}, N_z={Nz_5}, \mu= {mu_leads_5}$', linestyle='solid', alpha=0.3)
    ax3.plot(fermi_7, G_0_7[:, i], color=color_list[5], label=f'$w= {width_7[i]}, N_x={Nx_7}, N_z={Nz_7}, \mu= {mu_leads_7}$', linestyle='solid', alpha=0.3)
    ax2.plot(fermi_8, G_0_8[:, i], color=color_list[5], label=f'$w= {width_8[i]}, N_x={Nx_8}, N_z={Nz_8}, \mu= {mu_leads_8}$', linestyle='solid', alpha=0.3)
    ax4.plot(fermi_9, G_0_9[:, i], color=color_list[0], label=f'$w= {width_9[i]}, N_x={Nx_9}, N_z={Nz_9}, \mu= {mu_leads_9}$', linestyle='solid', alpha=0.3)
    ax1.plot(Ef_th1, G01_th, color=color_list[1], linestyle='dashed')
    ax2.plot(Ef_th2, G02_th, color=color_list[1], linestyle='dashed')
    ax2.plot(Ef_th3, G03_th, color=color_list[2], linestyle='dashed')
    ax2.plot(Ef_th4, G04_th, color=color_list[3], linestyle='dashed')
    ax2.plot(Ef_th5, G05_th, color=color_list[4], linestyle='dashed')
    ax1.plot(Ef_th6, G06_th, color=color_list[2], linestyle='dashed')
    ax3.plot(Ef_th7, G07_th, color=color_list[5], linestyle='dashed')
    ax2.plot(Ef_th8, G08_th, color=color_list[5], linestyle='dashed')
    ax4.plot(Ef_th9, G09_th, color=color_list[0], linestyle='dashed')
ax2.text(-0.9, 1, 'Lead gap', rotation='vertical')

ax4.plot(fermi_9, 0.2 * Nchannels_lead, linestyle='solid', color='blue', alpha=0.7)
ax3.plot(fermi_8, 0.2 * DoS_closed_nw, linestyle='solid', color='blue', alpha=0.7)
for energy in spectrum:
    if energy < 1.5:
        ax3.plot(energy * np.ones((100, )), np.linspace(0, 10, 100), linestyle='dashed', color='dodgerblue', alpha=0.1)

for ax in ax_amor:
    y_axis_ticks = [i for i in range(0, 11, 2)]
    y_axis_labels = ['' for i in range(0, 11, 2)]
    ax.set_xlim(-1, fermi_1[-1])
    ax.set_ylim(0, 5)
    ax.tick_params(which='major', width=0.75, labelsize=fontsize)
    ax.tick_params(which='major', length=6, labelsize=fontsize)
    ax.set_xlabel("$E_F / t$", fontsize=fontsize)
    ax.set(yticks=y_axis_ticks, yticklabels=y_axis_labels)
ax2.set_ylim(0, 5)
ax4.set_xlim(-1, 10)

y_axis_ticks = [i for i in range(0, 11, 2)]
y_axis_labels = [str(i) for i in range(0, 11, 2)]
ax1.set_ylabel("$G(2e^2/h)$", fontsize=fontsize)
ax4.set_ylabel("$G(2e^2/h)$", fontsize=fontsize)
ax1.set(yticks=y_axis_ticks, yticklabels=y_axis_labels)
ax1.legend(ncol=1, frameon=False, fontsize=10)
ax2.legend(ncol=1, frameon=False, fontsize=10)
ax3.legend(ncol=1, frameon=False, fontsize=10)
ax4.legend(ncol=1, frameon=False, fontsize=10)


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





