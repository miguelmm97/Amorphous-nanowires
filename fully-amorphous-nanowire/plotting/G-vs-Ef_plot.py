#%% modules setup

# Math and plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Kwant
import kwant

# modules
from modules.functions import *
from modules.AmorphousLattice_3d import AmorphousLattice_3d
from modules.FullyAmorphousWire_kwant import promote_to_kwant_nanowire3d

#%% Loading data
file_list = ['Exp10.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-cond-vs-Ef')
# data_dict = load_my_data(file_list, '.')

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

# Simulation data
flux_half     = data_dict[file_list[0]]['Simulation']['flux_max']
G_0           = data_dict[file_list[0]]['Simulation']['G_0']
G_half        = data_dict[file_list[0]]['Simulation']['G_half']
fermi         = data_dict[file_list[0]]['Simulation']['fermi']
width         = data_dict[file_list[0]]['Simulation']['width']
# x             = data_dict[file_list[0]]['Simulation']['x']
# y             = data_dict[file_list[0]]['Simulation']['y']
# z             = data_dict[file_list[0]]['Simulation']['z']

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

# Figure 1: Definition
fig1 = plt.figure(figsize=(10, 8))
gs = GridSpec(1, 1, figure=fig1, wspace=0., hspace=0.)
ax1 = fig1.add_subplot(gs[0, 0])

# Figure 1: Plots
for i in range(len(G_0[0, :])):
    ax1.plot(fermi, G_0[:, i], color=color_list[i], label=f'$w= {width[i]}$', linestyle='solid')
    ax1.plot(fermi, G_half[:, i], color=color_list[i], alpha=0.5, linestyle='dotted')
ax1.legend(ncol=1, frameon=False, fontsize=16)
fig1.suptitle(f'$\mu_l= {mu_leads}$, $r= {r}$, $N_x= {Nx}$, $N_y = {Ny}$, $N_z= {Nz}$', y=0.93, fontsize=20)
ax1.text(0.015, 1.2, '$\phi_{max}$', fontsize=fontsize)
ax1.text(0.015, 0.2, '$\phi = 0$', fontsize=fontsize)

# Figure 1: Format
y_axis_ticks = [i for i in range(0, 10, 2)]
y_axis_labels = [str(i) for i in range(0, 10, 2)]
ax1.set_xlim(fermi[0], fermi[-1])
ax1.set_ylim(0, 10)
ax1.tick_params(which='major', width=0.75, labelsize=fontsize)
ax1.tick_params(which='major', length=6, labelsize=fontsize)
ax1.set_xlabel("$E_F / t$", fontsize=fontsize)
ax1.set_ylabel("$G(2e^2/h)$",fontsize=fontsize)
ax1.set(yticks=y_axis_ticks, yticklabels=y_axis_labels)


# # Nanowire structure
# lattice = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=Nz, w=width[0], r=r)
# lattice.set_configuration(x, y, z)
# lattice.build_lattice()
# nanowire = promote_to_kwant_nanowire3d(lattice, params_dict, mu_leads=mu_leads).finalized()
#
# fig2 = plt.figure()
# gs = GridSpec(1, 1, figure=fig2)
# ax1 = fig1.add_subplot(gs[0, 0], projection='3d')
# kwant.plot(nanowire, site_size=site_size, site_lw=site_lw, site_color=site_color, hop_lw=hop_lw, hop_color=hop_color,
#            lead_site_size=site_size, lead_color=lead_color, lead_site_lw=site_lw, lead_hop_lw=hop_lw,
#            ax=ax1)
# ax1.set_axis_off()
# ax1.margins(-0.49, -0.49, -0.49)
#
#




fig1.savefig(f'../figures/{file_list[0]}-cond-vs-Ef.pdf', format='pdf', backend='pgf')
plt.show()
