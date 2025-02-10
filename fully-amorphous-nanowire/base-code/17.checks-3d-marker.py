#%% Modules and setup

# Plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.ticker as ticker
from matplotlib import cm
import seaborn

# Modules
from modules.functions import *
from modules.colorbar_marker import *


#%% Loading data
file_list = ['data-cluster.h5', 'data-cluster2.h5', 'data-cluster-random1.h5', 'data-cluster-bulk-full-range.h5',
             'data-cluster-bulk.h5', 'data-cluster-try.h5', 'data-cluster-Nz-dep.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-marker-vs-cross-section')

# Simulation with parent lattices on a wire shaped region
avg_marker1        = data_dict[file_list[0]]['Plot1']['avg_marker']
avg_marker2        = data_dict[file_list[1]]['Plot1']['avg_marker']
std_marker1        = data_dict[file_list[0]]['Plot1']['std_marker']
std_marker2        = data_dict[file_list[1]]['Plot1']['std_marker']
width1             = data_dict[file_list[0]]['Plot1']['width']
width2             = data_dict[file_list[1]]['Plot1']['width']
Nx1                = data_dict[file_list[0]]['Plot1']['Nx']
avg_marker1        = np.concatenate((avg_marker1, avg_marker2))
std_marker1        = np.concatenate((std_marker1, std_marker2))
width1             = np.concatenate((width1, width2))
error_bar_up1      = avg_marker1 + 0.5 * std_marker1
error_bar_down1    = avg_marker1 - 0.5 * std_marker1

# Simulation with random lattices in a wire shaped region
avg_marker2        = data_dict[file_list[2]]['Plot1']['avg_marker']
std_marker2        = data_dict[file_list[2]]['Plot1']['std_marker']
width2             = data_dict[file_list[2]]['Plot1']['width']
error_bar_up2      = avg_marker2 + 0.5 * std_marker2
error_bar_down2    = avg_marker2 - 0.5 * std_marker2
Nx2                = data_dict[file_list[2]]['Plot1']['Nx']

# Simulation with random lattices on a cubic region
avg_marker3        = data_dict[file_list[3]]['Plot1']['avg_marker']
std_marker3        = data_dict[file_list[3]]['Plot1']['std_marker']
width3             = data_dict[file_list[3]]['Plot1']['width']
error_bar_up3      = avg_marker3 + 0.5 * std_marker3
error_bar_down3    = avg_marker3 - 0.5 * std_marker3
Nx3                = data_dict[file_list[3]]['Plot1']['Nx']

# Simulation with random lattices in a wire shaped region varying Nz
avg_marker4        = data_dict[file_list[5]]['Plot1']['avg_marker']
std_marker4        = data_dict[file_list[5]]['Plot1']['std_marker']
width4             = data_dict[file_list[5]]['Plot1']['width']
error_bar_up4      = avg_marker4 + 0.5 * std_marker4
error_bar_down4    = avg_marker4 - 0.5 * std_marker4
Nz4                = np.arange(10, 22)
Nx4                = data_dict[file_list[5]]['Plot1']['Nx']



#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['dodgerblue', 'limegreen', 'm', 'r', 'orange']
fontsize = 15
palette = seaborn.color_palette(palette='magma', n_colors=len(Nx1))
palette2 = seaborn.color_palette(palette='viridis', n_colors=len(width3))
palette3 = seaborn.color_palette(palette='magma', n_colors=len(Nz4))


# Figure 1
fig1 = plt.figure(figsize=(10, 7))
gs = GridSpec(2, 2, figure=fig1, wspace=0.5, hspace=0.5)
ax1 = fig1.add_subplot(gs[0, 0])
ax2 = fig1.add_subplot(gs[0, 1])
ax3 = fig1.add_subplot(gs[1, 0])
ax4 = fig1.add_subplot(gs[1, 1])

# Figure1: Plots
ax1.set_title('Parent lattices, wire shaped')
for i in range(len(Nx1)):
    ax1.plot(width1, avg_marker1[:, i], marker='o', linestyle='solid', color=palette[i], label=f'${Nx1[i]}$')
    ax1.fill_between(width1, error_bar_down1[:, i], error_bar_up1[:, i], color=palette[i], alpha=0.3)
ax2.set_title('Random lattices, wire shaped')
for i in range(len(Nx2)):
    ax2.plot(width2, avg_marker2[:, i], marker='o', linestyle='solid', color=palette[i], label=f'${Nx2[i]}$')
    ax2.fill_between(width2, error_bar_down2[:, i], error_bar_up2[:, i], color=palette[i], alpha=0.3)
ax3.set_title('Random lattices, cubic shaped')
for i in range(len(Nx3)):
    ax3.plot(width3, avg_marker3[:, i], marker='o', linestyle='solid', color=palette[i], label=f'${Nx3[i]}$')
    ax3.fill_between(width3, error_bar_down3[:, i], error_bar_up3[:, i], color=palette[i], alpha=0.3)
ax4.set_title('Random lattices, wire shaped')
for i in range(len(Nz4)):
    ax4.plot(width4, avg_marker4[:, i], marker='o', linestyle='solid', color=palette3[i], label=f'${Nz4[i]}$')
    ax4.fill_between(width4, error_bar_down4[:, i], error_bar_up4[:, i], color=palette3[i], alpha=0.3)


# Figure 1: Legends and details of each plot
ax1.legend(ncol=2, frameon=False, fontsize=13, handlelength=1, columnspacing=0.5, bbox_to_anchor=(0.6, 0.75))
ax1.text(0.58, -0.23, '$\\underline{N_{x, y}}$', fontsize=fontsize - 2)
ax1.text(0.01, -0.15, '$N_z = 15$', fontsize=fontsize - 2)
ax1.text(0.35, -0.9, f'$\\vert x, y, z \\vert< {0.4}$' + '$N_{x, y}$', fontsize=fontsize - 2)

ax2.legend(ncol=2, frameon=False, fontsize=13, handlelength=1, columnspacing=0.5, bbox_to_anchor=(0.6, 0.75))
ax2.text(0.58, -0.23, '$\\underline{N_{x, y}}$', fontsize=fontsize - 2)
ax2.text(0.01, -0.15, '$N_z = 15$', fontsize=fontsize - 2)
ax2.text(0.35, -0.9, f'$\\vert x, y, z \\vert< {0.4}$' + '$N_{x, y}$', fontsize=fontsize - 2)

ax3.legend(ncol=2, frameon=False, fontsize=13, handlelength=1, columnspacing=0.5, bbox_to_anchor=(0.6, 0.75))
ax3.text(0.58, -0.23, '$\\underline{N_{x, y, z}}$', fontsize=fontsize - 2)
ax3.text(0.35, -0.9, f'$\\vert x, y, z \\vert< {0.4}$' + '$N_{x, y, z}$', fontsize=fontsize - 2)

ax4.legend(ncol=2, frameon=False, fontsize=13, handlelength=1, columnspacing=0.5, bbox_to_anchor=(0.6, 0.75))
ax4.text(0.58, -0.23, '$\\underline{N_{z}}$', fontsize=fontsize - 2)
ax4.text(0.35, -0.9, f'$\\vert x, y, z \\vert< {0.4}$' + '$N_{z}$', fontsize=fontsize - 2)
ax4.text(0.01, -0.15, f'$N_x, N_y = {Nx4}$', fontsize=fontsize - 2)


# Figure 1: Overall format of the plots
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xlabel('$w$', fontsize=fontsize)
    ax.set_ylabel('$\overline{\\nu}$', fontsize=fontsize)
    ax.set_ylim([-1, 0])
    ax.set_xlim([0, 0.8])
    majorsy = [-1, -0.5, 0]
    minorsy = [-0.75, -0.25]
    ax.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
    ax.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
    ax.tick_params(which='major', width=0.75, labelsize=fontsize)
    ax.tick_params(which='major', length=6, labelsize=fontsize)
    ax.tick_params(which='minor', width=0.75, labelsize=fontsize)
    ax.tick_params(which='minor', length=3, labelsize=fontsize)
plt.show()

