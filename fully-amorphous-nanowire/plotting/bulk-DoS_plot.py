#%% Modules and setup

# Plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm

# Modules
from modules.functions import *


#%% Loading data
file_list = ['Exp5.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-bulk-dos')

# Parameters
Ef           = data_dict[file_list[0]]['Parameters']['Ef']
Nx           = data_dict[file_list[0]]['Parameters']['Nx']
Ny           = data_dict[file_list[0]]['Parameters']['Ny']
Nz           = data_dict[file_list[0]]['Parameters']['Nz']
r            = data_dict[file_list[0]]['Parameters']['r ']
t            = data_dict[file_list[0]]['Parameters']['t ']
eps          = data_dict[file_list[0]]['Parameters']['eps']
lamb         = data_dict[file_list[0]]['Parameters']['lamb']
lamb_z       = data_dict[file_list[0]]['Parameters']['lamb_z']
mu_leads     = data_dict[file_list[0]]['Parameters']['mu_leads']
flux1        = data_dict[file_list[0]]['Parameters']['flux1']
flux2        = data_dict[file_list[0]]['Parameters']['flux2']
idx1         = data_dict[file_list[0]]['Parameters']['idx1']
idx2         = data_dict[file_list[0]]['Parameters']['idx2']


# Simulation data
flux                   = data_dict[file_list[0]]['Simulation']['flux']
width                  = data_dict[file_list[0]]['Simulation']['width']
G_array                = data_dict[file_list[0]]['Simulation']['G_array']
N                      = data_dict[file_list[0]]['Simulation']['N']
N_local                = data_dict[file_list[0]]['Simulation']['N_local']
bulk_tot_density1      = data_dict[file_list[0]]['Simulation']['bulk_tot_density1']
bulk_tot_density2      = data_dict[file_list[0]]['Simulation']['bulk_tot_density2']
bulk_density1          = data_dict[file_list[0]]['Simulation']['DoS1']
bulk_density2          = data_dict[file_list[0]]['Simulation']['DoS2']
cut_pos                = data_dict[file_list[0]]['Simulation']['cuts']

# Normalisation for the plots
sigmas = 3
mean_value = np.mean(np.array([bulk_density1[f'{len(N_local) - 1}'], bulk_density2[f'{len(N_local) - 1}']]))
std_value = np.std(np.array([bulk_density1[f'{len(N_local) - 1}'], bulk_density2[f'{len(N_local) - 1}']]))
max_value, min_value = mean_value + sigmas * std_value, 0

#%% Figure 1
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['dodgerblue', 'limegreen', 'm', 'r', 'orange']
fontsize = 15


# Figure 1
fig1 = plt.figure()
gs = GridSpec(2, 1, figure=fig1, wspace=0.1, hspace=0.35)
ax1 = fig1.add_subplot(gs[0, 0])
ax2 = fig1.add_subplot(gs[1, 0])


ax2.plot(N, bulk_tot_density1, marker='o', linestyle='solid', color='limegreen', label='(a)')
ax2.plot(N, bulk_tot_density2, marker='o', linestyle='solid', color='dodgerblue', label='(b)')
ax2.set_xlabel('$n_x$', fontsize=fontsize)
ax2.set_ylabel('Norm. DoS', fontsize=fontsize)
ax2.legend(frameon=False, fontsize=15)


ax_vec = [ax1]
ax1.set_title(f'$w= {width}$, $E_f = {Ef}$', fontsize=fontsize)
for i in range(G_array.shape[1]):
    for j in range(len(Ef)):
        ax = ax_vec[j]
        label = None if (j % 2 != 0) else f'$w= {width[i]}$'
        ax.plot(flux, G_array[j, i, :], color=color_list[i], linestyle='solid', label=label)
        ax.plot(flux, 1 * np.ones(flux.shape), '--', color='Black', alpha=0.2)
        ax.text(flux1 + 0.05, G_array[j, i, idx1] + 0.2, f'$(a)$', fontsize=fontsize)
        ax.text(flux2 + 0.05, G_array[j, i, idx2] - 0.2, f'$(b)$', fontsize=fontsize)
        ax.plot(flux2, G_array[j, i, idx2], 'ob')
        ax.plot(flux1, G_array[j, i, idx1], 'ob')

# Figure 1: Format
# ax1.legend(ncol=5, frameon=False, fontsize=20)
ylim = np.max(G_array) + 0.2
for ax in ax_vec:
    ax.set_xlim(flux[0], flux[-1])
    ax.set_ylim(0, ylim)
    ax.tick_params(which='major', width=0.75, labelsize=10)
    ax.tick_params(which='major', length=6, labelsize=10)
    ax.set_xlabel("$\phi$", fontsize=fontsize)
    ax.set_ylabel("$G[2e^2/h]$", fontsize=fontsize)

#%% Figure 2

# Defining a colormap
color_map = plt.get_cmap("magma").reversed()
colors = color_map(np.linspace(0, 1, 20))
colors[0] = [1, 1, 1, 1]
color_map = LinearSegmentedColormap.from_list("custom_colormap", colors)
colormap = cm.ScalarMappable(norm=Normalize(vmin=min_value, vmax=max_value), cmap=color_map)


# Figure 1
fig2 = plt.figure(figsize=(12, 10))
gs = GridSpec(2, 4, figure=fig2, wspace=0.1, hspace=0.02)
ax1 = fig2.add_subplot(gs[0, 0], projection='3d')
ax2 = fig2.add_subplot(gs[0, 1], projection='3d')
ax3 = fig2.add_subplot(gs[0, 2], projection='3d')
ax4 = fig2.add_subplot(gs[1, 0], projection='3d')
ax5 = fig2.add_subplot(gs[1, 1], projection='3d')
ax6 = fig2.add_subplot(gs[1, 2], projection='3d')



for i, ax in enumerate(fig2.axes):
    if i<3:
        ax.set_title(f'$n_x, n_y < {N_local[i] :.2f}$')
        ax.scatter(cut_pos[f'{i}'][:, 0], cut_pos[f'{i}'][:, 1], cut_pos[f'{i}'][:, 2], facecolor='white', edgecolor='black')
        ax.scatter(cut_pos[f'{i}'][:, 0], cut_pos[f'{i}'][:, 1], cut_pos[f'{i}'][:, 2], c=bulk_density1[f'{i}'], cmap=color_map, vmin=min_value,
                vmax=max_value)
    else:
        ax.set_title(f'$n_x, n_y < {N_local[i - 3] :.2f}$')
        ax.scatter(cut_pos[f'{i - 3}'][:, 0], cut_pos[f'{i - 3}'][:, 1], cut_pos[f'{i - 3}'][:, 2], facecolor='white', edgecolor='black')
        ax.scatter(cut_pos[f'{i - 3}'][:, 0], cut_pos[f'{i - 3}'][:, 1], cut_pos[f'{i - 3}'][:, 2], c=bulk_density2[f'{i - 3}'], cmap=color_map,
                   vmin=min_value, vmax=max_value)

    ax.set_box_aspect((1, 1, 5))
    ax.set_axis_off()


cbar_ax = fig2.add_subplot(gs[0, -1])
divider = make_axes_locatable(cbar_ax)
cax = divider.append_axes("left", size="10%", pad=0)
cbar = fig2.colorbar(colormap, cax=cax, orientation='vertical')
cbar_ax.set_axis_off()
cbar.set_label(label='$\\vert \psi (r)\\vert ^2$', labelpad=10, fontsize=20)


# fig1.savefig(f'../figures/{file_list[0]}-total-DoS-bulk.pdf', format='pdf')
# fig2.savefig(f'../figures/{file_list[0]}-bulk-cuts.pdf', format='pdf')

plt.show()