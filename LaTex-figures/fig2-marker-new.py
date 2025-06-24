#%% Modules and setup

# Plotting
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.gridspec as gridspec
import seaborn
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm

# Modules
from modules.functions import *
from modules.colorbar_marker import *


#%% Loading data
file_list = ['Exp39.h5', 'Exp40.h5', 'Exp41.h5', 'Exp42.h5'] # ['Exp7.h5', 'Exp8.h5', 'Exp9.h5', 'Exp10.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/cluster-simulations/'
                                    'data-cluster-marker-per-site/data-cluster-marker-per-site-statistics')

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
z0              = [data_dict[file]['Simulation']['z0'] for file in file_list]
z1              = [data_dict[file]['Simulation']['z1'] for file in file_list]
x               = [data_dict[file]['Simulation']['x'] for file in file_list]
y               = [data_dict[file]['Simulation']['y'] for file in file_list]
z               = [data_dict[file]['Simulation']['z'] for file in file_list]
marker          = [data_dict[file]['Simulation']['marker'] for file in file_list]
width           = [data_dict[file]['Simulation']['width'] for file in file_list]
avg_marker      = [data_dict[file]['Simulation']['avg_marker'] for file in file_list]
std_marker      = [data_dict[file]['Simulation']['std_marker'] for file in file_list]
r_edges         = [data_dict[file]['Simulation']['r_bin_edges'] for file in file_list]
prob_dist       = [data_dict[file]['prob_dist'] for file in file_list]
bins_marker     = [data_dict[file]['bins_marker'] for file in file_list]


#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange']
marker_list=['o', 's', 'd', 'p', '*', 'h', '>', '<', 'X']
markersize = 5
fontsize=20



# Figure 1
fig1 = plt.figure(figsize=(12, 8))
outer_gs = GridSpec(3, 4, figure=fig1, wspace=0.15, hspace=0.4)

# Defining a colormap
vmin, vmax = -1.5,  5
cbar_ticks = np.arange(np.ceil(vmin), np.floor(vmax) + 1, 1)
hex_list = ['#ff416d', '#ff7192', '#ffa0b6', '#ffd0db', '#ffffff', '#cfdaff', '#9fb6ff', '#6f91ff', '#3f6cff']
cmap = get_continuous_cmap(hex_list)
divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
colormap = cm.ScalarMappable(norm=divnorm, cmap=cmap)
gradient = np.linspace(vmin, vmax, 256).reshape(-1, 1)


# Figure 1
theta = np.linspace(0, 2 * np.pi, 200)
for i in range(len(file_list)):

    # Scatter plots: Marker per site
    # plots
    ax_scatt = fig1.add_subplot(outer_gs[0, i])
    ax_probs = fig1.add_subplot(outer_gs[1:, i])
    ax_scatt.scatter(x[i], y[i], c=marker[i], norm=divnorm, cmap=cmap, s=1)
    palette = seaborn.color_palette(palette='magma_r', n_colors=len(r_edges[i]))
    cmap_probs = LinearSegmentedColormap.from_list('stab_colormap', palette, N=len(r_edges[i]))
    colormap_cbar_probs = cm.ScalarMappable(norm=Normalize(vmin=min(r_edges[i]), vmax=max(r_edges[i])), cmap=cmap_probs)

    # Style
    ax_scatt.set_xlim(-np.max(x[i]) - 0.2, np.max(x[i]) + 0.2)
    ax_scatt.set_ylim(-np.max(y[i]) - 0.2, np.max(y[i]) + 0.2)
    ax_scatt.set_xticks(ticks=[-Nx/2, 0, Nx/2])
    if i==0:
        ax_scatt.set_yticks(ticks=[-Ny/2, 0, Ny/2])
        ax_scatt.set_ylabel('$y$', fontsize=fontsize)
        ax_scatt.set_xlabel('$x$', fontsize=fontsize, labelpad=-3)
    else:
        ax_scatt.set_yticks(ticks=[-Ny / 2, 0, Ny / 2], labels=[])
        ax_scatt.set_xlabel('$x$', fontsize=fontsize, labelpad=-3)
    ax_scatt.minorticks_on()
    xminor_ticks = [-Nx/4, Nx/4]
    ax_scatt.xaxis.set_minor_locator(plt.FixedLocator(xminor_ticks))
    ax_scatt.yaxis.set_minor_locator(plt.FixedLocator(xminor_ticks))
    ax_scatt.tick_params(which='major', width=0.75, labelsize=fontsize)
    ax_scatt.tick_params(which='major', length=6, labelsize=fontsize)
    ax_scatt.tick_params(which='minor', width=0.75, labelsize=fontsize)
    ax_scatt.tick_params(which='minor', length=3, labelsize=fontsize)


    # Probability distribution plots
    ax_probs.set_axis_off()
    inner_gs = gridspec.GridSpecFromSubplotSpec(len(r_edges[i]), 1, subplot_spec=outer_gs[1:, i], wspace=0.1, hspace=0.0)
    keys = np.sort(np.array([int(item) for item in prob_dist[i].keys()]))
    keys = [str(key) for key in keys]
    for j, key in enumerate(keys):
        # plots
        probs = [prob_dist[i][key][k]   for k in range(len(prob_dist[i][key]))   if np.abs(prob_dist[i][key][k]) > 1e-6]
        bins =  [bins_marker[i][key][k] for k in range(len(bins_marker[i][key])) if np.abs(prob_dist[i][key][k]) > 1e-6]
        ax = fig1.add_subplot(inner_gs[j, 0])
        ax.plot(bins, probs, marker='None', color=palette[j], label=f'$r= {r_edges[i][j] :.2f}$')
        ax.fill_between(bins, probs, color=palette[j], alpha=0.5)
        ax.plot(np.ones((10,)) * (-1), np.linspace(0, 10, 10), color='grey', linestyle='dashed', alpha=0.3)
        ax.plot(np.zeros((10,)), np.linspace(0, 10, 10), color='grey', linestyle='dashed', alpha=0.3)

        # Style
        ax.set_ylim(0, 0.3)
        ax.set_xlim(-2, 7)
        if j == len(keys) - 1:
            ax.set_xticks(ticks=[0, 5])
            ax.set_xlabel('$\\nu$', fontsize=fontsize, labelpad=-10)
        else:
            ax.set_xticks(ticks=[], labels=[])
        if i==0 and j == len(keys) - 1:
            ax.set_yticks(ticks=[0, 0.3])
        else:
            ax.set_yticks(ticks=[], labels=[])
        xminor_ticks = [-2, -1, 1, 2, 3, 4, 6, 7]
        ax.xaxis.set_minor_locator(plt.FixedLocator(xminor_ticks))
        ax.tick_params(which='major', width=0.75, labelsize=fontsize)
        ax.tick_params(which='major', length=6, labelsize=fontsize)
        ax.tick_params(which='minor', width=0.75, labelsize=fontsize)
        ax.tick_params(which='minor', length=3, labelsize=fontsize)

    # Colorbars
    if i==len(file_list) - 1:
        cax_scatt = inset_axes(ax_scatt, width='5%', height='100%', loc='center right', bbox_to_anchor=(0.1, 0.0, 1, 1),
                         bbox_transform=ax_scatt.transAxes, borderpad=0)
        cax_scatt.imshow(gradient, aspect='auto', cmap=cmap, norm=divnorm, origin='lower')
        tick_locs = (cbar_ticks - vmin) / (vmax - vmin) * gradient.shape[0]
        cax_scatt.set_yticks(tick_locs)
        cax_scatt.set_yticklabels([f"{t:.0f}" for t in cbar_ticks])
        cax_scatt.set_xticks([])
        cax_scatt.set_ylabel('$\\nu(r)$', labelpad=10, fontsize=20)
        cax_scatt.yaxis.set_label_position('right')
        cax_scatt.yaxis.set_ticks_position('right')
        cax_scatt.tick_params(which='major', width=0.75, labelsize=fontsize)

        cax_probs = inset_axes(ax_probs, width='5%', height='95%', loc='center right', bbox_to_anchor=(0.1, 0.028, 1, 1),
                         bbox_transform=ax_probs.transAxes, borderpad=0)
        cbar = fig1.colorbar(colormap_cbar_probs, cax=cax_probs, orientation='vertical', boundaries=r_edges[i])
        cax_probs.set_xticks([])
        cax_probs.set_yticks(ticks=[*r_edges[i][:]], labels=[f'${rad :.1f}$' for rad in r_edges[i]])
        cax_probs.set_ylabel('$r$', labelpad=-1, fontsize=20)
        cax_probs.yaxis.set_label_position('right')
        cax_probs.yaxis.set_ticks_position('right')
        cax_probs.tick_params(which='major', width=0.75, labelsize=fontsize)
        cax_probs.invert_yaxis()



fig1.text(0.075, 0.35, '$P(\\nu)$', va='center', rotation='vertical', fontsize=fontsize)
fig1.text(0.13, 0.91, '$(a)$', va='center', fontsize=fontsize-3)
fig1.text(0.33, 0.91, '$(b)$', va='center', fontsize=fontsize-3)
fig1.text(0.53, 0.91, '$(c)$', va='center', fontsize=fontsize-3)
fig1.text(0.73, 0.91, '$(d)$', va='center', fontsize=fontsize-3)
fig1.text(0.27, 0.58, '$(e)$', va='center', fontsize=fontsize-3)
fig1.text(0.47, 0.58, '$(f)$', va='center', fontsize=fontsize-3)
fig1.text(0.67, 0.58, '$(g)$', va='center', fontsize=fontsize-3)
fig1.text(0.87, 0.58, '$(h)$', va='center', fontsize=fontsize-3)
fig1.text(0.18, 0.91, f'$w={width[0]}$', va='center', fontsize=fontsize-3)
fig1.text(0.38, 0.91, f'$w={width[1]}$', va='center', fontsize=fontsize-3)
fig1.text(0.58, 0.91, f'$w={width[2]}$', va='center', fontsize=fontsize-3)
fig1.text(0.78, 0.91, f'$w={width[3]}$', va='center', fontsize=fontsize-3)


fig1.savefig('fig2-marker-new.pdf', format='pdf')
plt.show()

