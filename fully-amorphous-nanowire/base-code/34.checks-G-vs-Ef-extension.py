#%% Modules setup

# Math and plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from numpy import pi
import seaborn
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable



# modules
import kwant
from modules.functions import *
from modules.AmorphousWire_kwant import infinite_nanowire_kwant
from modules.AmorphousLattice_3d import AmorphousLattice_3d
from modules.FullyAmorphousWire_kwant import promote_to_kwant_nanowire3d
#%% Loading data
file_list = ['Exp76.h5', 'Exp77.h5', 'Exp78.h5', 'Exp79.h5', 'Exp80.h5', 'Exp81.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/local-simulations/data-cond-vs-Ef')

r              = data_dict[file_list[0]]['Parameters']['r']
t              = data_dict[file_list[0]]['Parameters']['t']
eps            = data_dict[file_list[0]]['Parameters']['eps']
lamb           = data_dict[file_list[0]]['Parameters']['lamb']
lamb_z         = data_dict[file_list[0]]['Parameters']['lamb_z']
params_dict  = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}


# Different N comparison.
# N=5
Nx0           = data_dict[file_list[2]]['Parameters']['Nx']
Ny0           = data_dict[file_list[2]]['Parameters']['Ny']
Nz0           = data_dict[file_list[2]]['Parameters']['Nz']
x0            = data_dict[file_list[2]]['Simulation']['x']
y0            = data_dict[file_list[2]]['Simulation']['y']
z0            = data_dict[file_list[2]]['Simulation']['z']
mu_leads0     = data_dict[file_list[2]]['Parameters']['mu_leads']
G0            = data_dict[file_list[2]]['Simulation']['G_0']
fermi0        = data_dict[file_list[2]]['Simulation']['fermi']
width0        = data_dict[file_list[2]]['Simulation']['width']
G0_cryst      = data_dict[file_list[3]]['Simulation']['G_0']

# N=10
Nx1           = data_dict[file_list[0]]['Parameters']['Nx']
Ny1           = data_dict[file_list[0]]['Parameters']['Ny']
Nz1           = data_dict[file_list[0]]['Parameters']['Nz']
x1            = data_dict[file_list[0]]['Simulation']['x']
y1            = data_dict[file_list[0]]['Simulation']['y']
z1            = data_dict[file_list[0]]['Simulation']['z']
mu_leads1     = data_dict[file_list[0]]['Parameters']['mu_leads']
G1            = data_dict[file_list[0]]['Simulation']['G_0']
fermi1        = data_dict[file_list[0]]['Simulation']['fermi']
width1        = data_dict[file_list[0]]['Simulation']['width']

# N=15
Nx2           = data_dict[file_list[1]]['Parameters']['Nx']
Ny2           = data_dict[file_list[1]]['Parameters']['Ny']
Nz2           = data_dict[file_list[1]]['Parameters']['Nz']
x2            = data_dict[file_list[1]]['Simulation']['x']
y2            = data_dict[file_list[1]]['Simulation']['y']
z2            = data_dict[file_list[1]]['Simulation']['z']
mu_leads2     = data_dict[file_list[1]]['Parameters']['mu_leads']
G2            = data_dict[file_list[1]]['Simulation']['G_0']
fermi2        = data_dict[file_list[1]]['Simulation']['fermi']
width2        = data_dict[file_list[1]]['Simulation']['width']

# N=20
Nx3           = data_dict[file_list[4]]['Parameters']['Nx']
Ny3           = data_dict[file_list[4]]['Parameters']['Ny']
Nz3           = data_dict[file_list[4]]['Parameters']['Nz']
x3            = data_dict[file_list[4]]['Simulation']['x']
y3            = data_dict[file_list[4]]['Simulation']['y']
z3            = data_dict[file_list[4]]['Simulation']['z']
mu_leads3     = data_dict[file_list[4]]['Parameters']['mu_leads']
G3            = data_dict[file_list[4]]['Simulation']['G_0']
fermi3        = data_dict[file_list[4]]['Simulation']['fermi']
width3        = data_dict[file_list[4]]['Simulation']['width']


# Different aspect ratio comparison
# N=10, L=50
Nx4           = data_dict[file_list[5]]['Parameters']['Nx']
Ny4           = data_dict[file_list[5]]['Parameters']['Ny']
Nz4           = data_dict[file_list[5]]['Parameters']['Nz']
x4            = data_dict[file_list[5]]['Simulation']['x']
y4            = data_dict[file_list[5]]['Simulation']['y']
z4            = data_dict[file_list[5]]['Simulation']['z']
mu_leads4     = data_dict[file_list[5]]['Parameters']['mu_leads']
G4            = data_dict[file_list[5]]['Simulation']['G_0']
fermi4        = data_dict[file_list[5]]['Simulation']['fermi']
width4        = data_dict[file_list[5]]['Simulation']['width']



#%% Bands
#
# kz = np.linspace(-pi, pi, 101)
# wire_kwant = infinite_nanowire_kwant(5, 5, params_dict, mu_leads=0.).finalized()
# bands0 = kwant.physics.Bands(wire_kwant, params=dict(flux=0))
# bands0 = [bands0(k) for k in kz]

# wire_kwant = infinite_nanowire_kwant(10, 10, params_dict, mu_leads=0.).finalized()
# bands1 = kwant.physics.Bands(wire_kwant, params=dict(flux=0))
# bands1 = [bands1(k) for k in kz]

# wire_kwant = infinite_nanowire_kwant(15, 15, params_dict, mu_leads=0.).finalized()
# bands2 = kwant.physics.Bands(wire_kwant, params=dict(flux=0))
# bands2 = [bands2(k) for k in kz]
#
# wire_kwant = infinite_nanowire_kwant(20, 20, params_dict, mu_leads=0.).finalized()
# bands3 = kwant.physics.Bands(wire_kwant, params=dict(flux=0))
# bands3 = [bands3(k) for k in kz]

#%% Density of states

# # Scattering states
# lattice = AmorphousLattice_3d(Nx=Nx1, Ny=Ny1, Nz=Nz1, w=width1, r=r)
# lattice.set_configuration(x1[0, :], y1[0, :], z1[0, :])
# lattice.build_lattice()
# nanowire = promote_to_kwant_nanowire3d(lattice, params_dict).finalized()
# site_pos_top = np.array([site.pos for site in nanowire.id_by_site])
# conducting_state = kwant.wave_function(nanowire, params=dict(flux=0., mu=-0.4, mu_leads=mu_leads1 - 0.4))
# downturn_state  =  kwant.wave_function(nanowire, params=dict(flux=0., mu=-1, mu_leads=mu_leads1 - 1))
# bulk_state      =  kwant.wave_function(nanowire, params=dict(flux=0., mu=-2, mu_leads=mu_leads1 - 2))
#
# # Total DoS through cuts
# N = np.linspace(2, Nx1/ 2, 10)
# conducting_DoS = np.zeros((len(N), ))
# downturn_DoS   = np.zeros((len(N), ))
# bulk_DoS       = np.zeros((len(N), ))
# for i, n in enumerate(N):
#     def bulk(site):
#         x, y = site.pos[0] - 0.5 * (Nx1 - 1), site.pos[1] - 0.5 * (Ny1 - 1)
#         cond1 = np.abs(x) < n
#         cond2 = np.abs(y) < n
#         return cond1 * cond2
#
#     total_density_operator = kwant.operator.Density(nanowire, where=bulk, sum=True)
#     conducting_DoS[i] = total_density_operator(conducting_state(0)[0])
#     downturn_DoS[i]   = total_density_operator(downturn_state(0)[0])
#     bulk_DoS[i]       = total_density_operator(bulk_state(0)[0])
#
# conducting_DoS = conducting_DoS / conducting_DoS[-1]
# downturn_DoS   = downturn_DoS / downturn_DoS[-1]
# bulk_DoS       = bulk_DoS / bulk_DoS[-1]
#
#
# # Local DoS
# def bulk(syst, R):
#     new_sites_x = tuple([site for site in syst.id_by_site if 0 < (site.pos[0] - 0.5 * (Nx1-1)) < R])
#     new_sites = tuple([site for site in new_sites_x if 0 < (site.pos[1] - 0.5 * (Ny1-1)) < R])
#     new_sites_pos = np.array([site.pos for site in new_sites])
#     return new_sites, new_sites_pos
# bulk_sites, bulk_pos = bulk(nanowire, Nx1/2 + 1)
# density_operator = kwant.operator.Density(nanowire, where=bulk_sites, sum=False)
#
# conducting_local_DoS = density_operator(conducting_state(0)[0])
# downturn_local_DoS   = density_operator(downturn_state(0)[0])
# bulk_local_DoS       = density_operator(bulk_state(0)[0])
# conducting_local_DoS = conducting_local_DoS / np.sum(conducting_local_DoS)
# downturn_local_DoS   = downturn_local_DoS / np.sum(downturn_local_DoS)
# bulk_local_DoS       = bulk_local_DoS / np.sum(bulk_local_DoS)
#


#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange', 'forestgreen', 'k']
palette = seaborn.color_palette(palette='magma_r', n_colors=5)
marker_list=['o', 's', 'd', 'p', '*', 'h', '>', '<', 'X']
line_list = ['solid', 'dashed', 'dashdot', 'dotted']
markersize = 5
fontsize = 20
site_size  = 0.1
site_lw    = 0.01
site_color = 'm'
hop_color  = 'royalblue'
hop_lw     = 0.05
lead_color = 'r'


# Figure 1
fig1 = plt.figure(figsize=(15, 8))
gs = GridSpec(1, 3, figure=fig1, wspace=0.1, hspace=0.2)
ax1 = fig1.add_subplot(gs[0, 0])
ax2 = fig1.add_subplot(gs[0, 1])
ax3 = fig1.add_subplot(gs[0, 2])
ax_amor = [ax1, ax2, ax3]

# N comparison
ax1.plot(fermi0, G0, color=palette[0], label=f'$w= {width0}, N_x={Nx0}, N_z={Nz0}, \mu= {mu_leads0}$', linestyle='solid', alpha=0.7)
ax1.plot(fermi1, G1, color=palette[1], label=f'$w= {width1}, N_x={Nx1}, N_z={Nz1}, \mu= {mu_leads1}$', linestyle='solid', alpha=0.7)
ax1.plot(fermi2, G2, color=palette[2], label=f'$w= {width2}, N_x={Nx2}, N_z={Nz2}, \mu= {mu_leads2}$', linestyle='solid', alpha=0.7)
ax1.plot(fermi2, G3, color=palette[3], label=f'$w= {width3}, N_x={Nx3}, N_z={Nz3}, \mu= {mu_leads3}$', linestyle='solid', alpha=0.7)
ax1.plot(fermi2, G0_cryst, color=palette[0], linestyle='solid', alpha=0.3)

# Aspect ratio comparison
ax2.plot(fermi4, G4, color=palette[0], label=f'$w= {width4}, N_x={Nx4}, N_z={Nz4}, \mu= {mu_leads4}$', linestyle='solid', alpha=0.7)
ax2.plot(fermi1, G1, color=palette[1], label=f'$w= {width1}, N_x={Nx1}, N_z={Nz1}, \mu= {mu_leads1}$', linestyle='solid', alpha=0.7)


ax1.legend(ncol=1, frameon=False, fontsize=10)
ax2.legend(ncol=1, frameon=False, fontsize=10)
for ax in ax_amor:
    y_axis_ticks = [i for i in range(0, 11, 2)]
    y_axis_labels = ['' for i in range(0, 11, 2)]
    ax.set_xlim(0, fermi0[-1])
    ax.set_ylim(0, 5)
    ax.tick_params(which='major', width=0.75, labelsize=fontsize)
    ax.tick_params(which='major', length=6, labelsize=fontsize)
    ax.set_xlabel("$E_F / t$", fontsize=fontsize)
    ax.set(yticks=y_axis_ticks, yticklabels=y_axis_labels)
ax2.set_ylim(0, 5)
y_axis_ticks = [i for i in range(0, 11, 2)]
y_axis_labels = [str(i) for i in range(0, 11, 2)]
ax1.set_ylabel("$G(2e^2/h)$", fontsize=fontsize)
ax1.set(yticks=y_axis_ticks, yticklabels=y_axis_labels)



# Bands
fig2 = plt.figure(figsize=(8, 8))
gs = GridSpec(1, 1, figure=fig2)
ax1 = fig2.add_subplot(gs[0, 0])
# ax2 = fig2.add_subplot(gs[0, 1])
# ax3 = fig2.add_subplot(gs[0, 2])
# ax4 = fig2.add_subplot(gs[0, 3])
#
# ax1.plot(kz, bands1, color='dodgerblue', linewidth=0.5)
# ax2.plot(kz, bands1, color='dodgerblue', linewidth=0.5)
# ax3.plot(kz, bands2, color='dodgerblue', linewidth=0.5)
# ax4.plot(kz, bands3, color='dodgerblue', linewidth=0.5)

ax1.set_xlabel('$k/a$')
ax1.set_ylabel('$E(k)/t$')
ax1.set_xlim(-1, 1)
ax1.set_ylim(0, 1)
ax1.tick_params(which='major', width=0.75, labelsize=10)
ax1.tick_params(which='major', length=6, labelsize=10)



# DoS
fig3 = plt.figure(figsize=(15, 6))
gs = GridSpec(1, 5, figure=fig3, wspace=0.5, hspace=0.3)
ax1 = fig3.add_subplot(gs[0, 0])
ax2 = fig3.add_subplot(gs[0, 1], projection='3d')
ax3 = fig3.add_subplot(gs[0, 2], projection='3d')
ax4 = fig3.add_subplot(gs[0, 3], projection='3d')

# sigmas = 3
# mean_value = np.mean(conducting_local_DoS)
# std_value = np.std(conducting_local_DoS)
# max_value, min_value = mean_value + sigmas * std_value, 0
# color_map = plt.get_cmap("magma").reversed()
# colors = color_map(np.linspace(0, 1, 20))
# colors[0] = [1, 1, 1, 1]
# color_map = LinearSegmentedColormap.from_list("custom_colormap", colors)
# colormap = cm.ScalarMappable(norm=Normalize(vmin=min_value, vmax=max_value), cmap=color_map)
# palette = seaborn.color_palette(palette='viridis_r', n_colors=200)
# palette = [palette[0], palette[50], palette[100], palette[130], palette[-1]]
#
#
# ax2.scatter(np.round(bulk_pos[:, 0], 2), np.round(bulk_pos[:, 1], 2), np.round(bulk_pos[:, 2], 2), facecolor='white', edgecolor='black', rasterized=True)
# ax2.scatter(np.round(bulk_pos[:, 0], 2), np.round(bulk_pos[:, 1], 2), np.round(bulk_pos[:, 2], 2), c=conducting_local_DoS,
#             cmap=color_map, vmin=min_value, vmax=max_value, rasterized=True)
# ax2.set_box_aspect((3, 3, 10))
# ax2.set_axis_off()
# pos2 = ax2.get_position()
# ax2.set_position([pos2.x0 - 0.02, pos2.y0 - 0.15, 0.25, 0.4])
#
# ax3.scatter(np.round(bulk_pos[:, 0], 2), np.round(bulk_pos[:, 1], 2), np.round(bulk_pos[:, 2], 2), facecolor='white', edgecolor='black', rasterized=True)
# ax3.scatter(np.round(bulk_pos[:, 0], 2), np.round(bulk_pos[:, 1], 2), np.round(bulk_pos[:, 2], 2), c=downturn_local_DoS,
#             cmap=color_map, vmin=min_value, vmax=max_value, rasterized=True)
# ax3.set_box_aspect((3, 3, 10))
# ax3.set_axis_off()
# pos3 = ax3.get_position()
# ax3.set_position([pos3.x0, pos3.y0 - 0.15, 0.25, 0.4])
#
# ax4.scatter(np.round(bulk_pos[:, 0], 2), np.round(bulk_pos[:, 1], 2), np.round(bulk_pos[:, 2], 2), facecolor='white', edgecolor='black', rasterized=True)
# ax4.scatter(np.round(bulk_pos[:, 0], 2), np.round(bulk_pos[:, 1], 2), np.round(bulk_pos[:, 2], 2), c=bulk_local_DoS,
#             cmap=color_map, vmin=min_value, vmax=max_value, rasterized=True)
# ax4.set_box_aspect((3, 3, 10))
# ax4.set_axis_off()
# pos4 = ax4.get_position()
# ax4.set_position([pos4.x0  + 0.02, pos4.y0 - 0.15, 0.25, 0.4])
#
# scatter_ax2 = fig3.add_axes([0.3, 0.38, 0.05, 0.05])
# scatter_ax2.text(0, 0, 'Conducting')
# scatter_ax2.set_xticks([])
# scatter_ax2.set_yticks([])
# scatter_ax2.set_axis_off()
# scatter_ax3 = fig3.add_axes([0.49, 0.38, 0.05, 0.05])
# scatter_ax3.text(0, 0, 'Downturn')
# scatter_ax3.set_xticks([])
# scatter_ax3.set_yticks([])
# scatter_ax3.set_axis_off()
# scatter_ax4 = fig3.add_axes([0.68, 0.38, 0.05, 0.05])
# scatter_ax4.text(0, 0, 'Bulk')
# scatter_ax4.set_xticks([])
# scatter_ax4.set_yticks([])
# scatter_ax4.set_axis_off()
#
#
# cbar_ax = fig3.add_subplot(gs[0, 4])
# divider = make_axes_locatable(cbar_ax)
# cax = divider.append_axes("right", size="10%", pad=-2)
# cbar = fig3.colorbar(colormap, cax=cax, orientation='vertical', ticks=[0., 0.0001, 0.0002, 0.0003])
# cbar_ax.set_axis_off()
# cbar.set_label(label='$\\vert \psi (r)\\vert ^2$', labelpad=10, fontsize=20)
# cbar.ax.tick_params(which='major', width=0.75, labelsize=fontsize)
# cbar.ax.set_yticklabels(['0', '1', '2', '3'])
# cbar.ax.set_title(r'$\quad \quad \times 10^{-4}$', x=0.8, fontsize=fontsize-10)
#
#
# ax1.plot(N, conducting_DoS, marker='o', linestyle='solid', color=color_list[2] , label=f'conducting')
# ax1.plot(N, downturn_DoS,   marker='^', linestyle='solid', color=color_list[1] , label=f'downturn')
# ax1.plot(N, bulk_DoS,       marker='*', linestyle='solid', color=color_list[-1], label=f'bulk')
# ax1.set_xlabel('$R$', fontsize=fontsize, labelpad=-1)
# ax1.set_ylabel('DoS', fontsize=fontsize, labelpad=-10)
# ax1.legend()
# label = ax1.yaxis.get_label()
# x, y = label.get_position()
# label.set_position((x, y + 0.2))
# # ax2.set_ylim(0, 1)
# ax1.tick_params(which='major', width=0.75, labelsize=fontsize)
# ax1.tick_params(which='major', length=6, labelsize=fontsize)
# ax1.tick_params(which='minor', width=0.75, labelsize=fontsize)
# ax1.tick_params(which='minor', length=3, labelsize=fontsize)
# ax1.set_yscale('log')



# fig1.savefig('GvsEf-checks.pdf', format='pdf')
# fig2.savefig('crystalline-bands.pdf', format='pdf')
fig3.savefig('DoS-check.pdf', format='pdf')

plt.show()
