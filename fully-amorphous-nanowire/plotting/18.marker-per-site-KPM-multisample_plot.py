#%% Modules and setup

# Plotting
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm

# Modules
from modules.functions import *
from modules.colorbar_marker import *


#%% Loading data
file_list = ['Exp26.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/cluster-simulations/data-cluster-marker-per-site/data-cluster-marker-per-site-statistics')

# Parameters
Nx           = data_dict[file_list[0]]['Parameters']['Nx']
Ny           = data_dict[file_list[0]]['Parameters']['Ny']
Nz           = data_dict[file_list[0]]['Parameters']['Nz']
r            = data_dict[file_list[0]]['Parameters']['r ']
t            = data_dict[file_list[0]]['Parameters']['t ']
eps          = data_dict[file_list[0]]['Parameters']['eps']
lamb         = data_dict[file_list[0]]['Parameters']['lamb']
lamb_z       = data_dict[file_list[0]]['Parameters']['lamb_z']


# Simulation data
z0              = data_dict[file_list[0]]['Simulation']['z0']
z1              = data_dict[file_list[0]]['Simulation']['z1']
x               = data_dict[file_list[0]]['Simulation']['x']
y               = data_dict[file_list[0]]['Simulation']['y']
z               = data_dict[file_list[0]]['Simulation']['z']
marker          = data_dict[file_list[0]]['Simulation']['marker']
width           = data_dict[file_list[0]]['Simulation']['width']
avg_marker      = data_dict[file_list[0]]['Simulation']['avg_marker']
std_marker      = data_dict[file_list[0]]['Simulation']['std_marker']
avg_radius      = data_dict[file_list[0]]['Simulation']['avg_radius']
prob_dist       = data_dict[file_list[0]]['prob_dist']
bins_marker     = data_dict[file_list[0]]['bins_marker']
error_bar_up    = np.array(avg_marker) + 0.5 * np.array(std_marker)
error_bar_down  = np.array(avg_marker) - 0.5 * np.array(std_marker)


#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange']
marker_list=['o', 's', 'd', 'p', '*', 'h', '>', '<', 'X']
markersize = 5
fontsize=20
palette = seaborn.color_palette(palette='magma_r', n_colors=len(avg_radius))


# Figure 1
fig1 = plt.figure(figsize=(10, 8))
gs = GridSpec(2, 6, figure=fig1, wspace=0.5, hspace=0.5)
ax1 = fig1.add_subplot(gs[:, :5])


# # Defining a colormap
vmin, vmax = min(bins_marker['0']),  1 * max(bins_marker[str(len(avg_radius) - 1)])
cbar_ticks = np.arange(np.ceil(vmin), np.floor(vmax) + 1, 1)
hex_list = ['#ff416d', '#ff7192', '#ffa0b6', '#ffd0db', '#ffffff', '#cfdaff', '#9fb6ff', '#6f91ff', '#3f6cff']
cmap = get_continuous_cmap(hex_list)
divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
colormap = cm.ScalarMappable(norm=divnorm, cmap=cmap)
gradient = np.linspace(vmin, vmax, 256).reshape(-1, 1)
#
# Defining the colorbar
cbar_ax = fig1.add_subplot(gs[:, 5])
divider = make_axes_locatable(cbar_ax)
cax = divider.append_axes("left", size="1%", pad=-20)
cbar_ax.set_axis_off()
cax.imshow(gradient, aspect='auto', cmap=cmap, norm=divnorm, origin='lower')
tick_locs = (cbar_ticks - vmin) / (vmax - vmin) * gradient.shape[0]
cax.set_yticks(tick_locs)
cax.set_yticklabels([f"{t:.0f}" for t in cbar_ticks])
cax.set_xticks([])
cax.set_ylabel('$\\nu(r)$', labelpad=10, fontsize=20)
cax.yaxis.set_label_position('right')
cax.tick_params(which='major', width=0.75, labelsize=fontsize)

# Figure 1
fig1.suptitle(f'$N= {Nx}$, $L= {Nz}$, $w={width :.2f}$', y=0.93, fontsize=20)
ax1.scatter(x, y, c=marker, facecolor='white', edgecolor='black')
ax1.scatter(x, y, c=marker, norm=divnorm, cmap=get_continuous_cmap(hex_list),  linewidths=2.5)
ax1.set_xlabel('$x$', fontsize=fontsize)
ax1.set_ylabel('$y$', fontsize=fontsize)
ax1.set_xlim(-np.max(x) - 0.2, np.max(x) + 0.2)
ax1.set_ylim(-np.max(y) - 0.2, np.max(y) + 0.2)

theta = np.linspace(0, 2 * np.pi, 200)
for i, rad in enumerate(avg_radius):
    x_rad, y_rad = rad * np.cos(theta), rad * np.sin(theta)
    ax1.plot(x_rad, y_rad, marker='none', linestyle='dashed', color=palette[i], alpha=0.5)



# Figure 2
fig2 = plt.figure(figsize=(8, 8))
gs = GridSpec(2, 1, figure=fig2, wspace=0.2, hspace=0.3)
ax1 = fig2.add_subplot(gs[0, 0])
ax2 = fig2.add_subplot(gs[1, 0])

ax1.plot(avg_radius, avg_marker, marker='o', color='#3F6CFF', label=f'$w= {width}$')
ax1.fill_between(avg_radius, error_bar_down, error_bar_up, color='#3F6CFF', alpha=0.3)
ax1.set_xlabel('Cross section radius', fontsize=fontsize)
ax1.set_ylabel('$\langle\\nu \\rangle$', fontsize=fontsize)
ax1.set_ylim(-1.5, 1)
ax1.set_xlim(min(avg_radius), max(avg_radius))
ax1.tick_params(which='major', width=0.75, labelsize=fontsize)
ax1.tick_params(which='major', length=6, labelsize=fontsize)
ax1.legend()
# ax1.set_xscale('log')
fig2.suptitle(f'$N= {Nx}$, $L= {Nz}$, $w={width :.2f}$', y=0.93, fontsize=20)

# for i, (probs, markers) in enumerate(zip(prob_dist, bins_marker)):
keys = np.array([int(item) for item in prob_dist.keys()])
keys = np.sort(keys)
for i, key in enumerate(keys):
    probs = [prob_dist[str(key)][i] for i in range(len(prob_dist[str(key)])) if np.abs(prob_dist[str(key)][i]) > 1e-6]
    bins = [bins_marker[str(key)][i] for i in range(len(bins_marker[str(key)])) if np.abs(prob_dist[str(key)][i]) > 1e-6]
    # ax2.plot(bins, probs, marker='o', color=palette[int(key)], label=f'$r= {avg_radius[int(key)] :.2f}$')
    ax2.plot(bins, probs, marker='o', color=palette[i], label=f'$r= {avg_radius[i] :.2f}$')
ax2.plot(np.ones((10, )) * (-1), np.linspace(0, 10, 10), color='grey', linestyle='dashed')
ax2.plot(np.zeros((10, )), np.linspace(0, 10, 10), color='grey', linestyle='dashed')
ax2.set_ylim(0, 0.5)
ax2.set_xlim(vmin, vmax)
ax2.set_xlabel('$\\nu$', fontsize=fontsize)
ax2.set_ylabel('$P(\\nu)$', fontsize=fontsize)
ax2.tick_params(which='major', width=0.75, labelsize=fontsize)
ax2.tick_params(which='major', length=6, labelsize=fontsize)
ax2.legend(ncol=3)



# Figure 3
fig3 = plt.figure(figsize=(8, 8))
gs = GridSpec(len(avg_radius), 1, figure=fig3, wspace=0., hspace=0.3)

for i, key in enumerate(keys):

    ax = fig3.add_subplot(gs[i, 0])
    probs = [prob_dist[str(key)][i] for i in range(len(prob_dist[str(key)])) if np.abs(prob_dist[str(key)][i]) > 1e-6]
    bins = [bins_marker[str(key)][i] for i in range(len(bins_marker[str(key)])) if np.abs(prob_dist[str(key)][i]) > 1e-6]
    ax.plot(bins, probs, marker='o', markersize=3.5, color=palette[i], label=f'$r= {avg_radius[i] :.2f}$')
    ax.plot(np.ones((10, )) * (-1), np.linspace(0, 10, 10), color='grey', linestyle='dashed')
    ax.plot(np.zeros((10, )), np.linspace(0, 10, 10), color='grey', linestyle='dashed')
    ax.set_ylim(0, 0.5)
    ax.set_xlim(-2, 8)
    # ax.set_ylabel('$P(\\nu)$', fontsize=fontsize)
    ax.tick_params(which='major', width=0.75, labelsize=fontsize)
    ax.tick_params(which='major', length=6, labelsize=fontsize)
    ax.set_yticks(ticks=[0, 0.5], labels=[])

    if key==len(avg_radius) - 1:
        xticks = np.arange(-2, 8)
        xlabels = [str(i) for i in xticks]
        ax.set_xticks(ticks=xticks, labels=xlabels)
        ax.set_xlabel('$\\nu$', fontsize=fontsize)
        ax.set_yticks(ticks=[0, 0.5], labels=['0', '0.5'])
    else:
        ax.set_xticks(ticks=[])
    ax.legend()

fig3.text(0.04, 0.5, '$P(\\nu)$', va='center', rotation='vertical', fontsize=fontsize)

fig1.savefig(f'../figures/wire-cross-section-branchcut.pdf', format='pdf')
# fig2.savefig(f'../figures/wire-stats-16-w4.pdf', format='pdf')
# fig3.savefig(f'../figures/wire-probs-16-w4.pdf', format='pdf')
plt.show()

