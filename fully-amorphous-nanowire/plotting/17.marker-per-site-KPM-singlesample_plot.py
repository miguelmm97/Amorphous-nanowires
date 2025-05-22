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
file_list = ['exp-24.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-cluster-marker-per-site')

# Parameters
Nx, Ny, Nz = 12, 12, 80
# Nx           = data_dict[file_list[0]]['Parameters']['Nx']
# Ny           = data_dict[file_list[0]]['Parameters']['Ny']
# Nz           = data_dict[file_list[0]]['Parameters']['Nz']
# r            = data_dict[file_list[0]]['Parameters']['r ']
# t            = data_dict[file_list[0]]['Parameters']['t ']
# eps          = data_dict[file_list[0]]['Parameters']['eps']
# lamb         = data_dict[file_list[0]]['Parameters']['lamb']
# lamb_z       = data_dict[file_list[0]]['Parameters']['lamb_z']


# Simulation data
z01           = data_dict[file_list[0]]['Simulation']['z0']
z11           = data_dict[file_list[0]]['Simulation']['z1']
x1           = data_dict[file_list[0]]['Simulation']['x']
y1           = data_dict[file_list[0]]['Simulation']['y']
z1           = data_dict[file_list[0]]['Simulation']['z']
marker1      = data_dict[file_list[0]]['Simulation']['local_marker']
width1       = data_dict[file_list[0]]['Simulation']['width']
# x1, y1 = x1 - 0.5 * (Nx - 1),  y1 - 0.5 * (Ny - 1)
# marker1 = np.real(marker1)



# Local marker distribution as a function of r
num_bins = 15
radius = np.sqrt((x1 ** 2) + (y1 ** 2))
r_min, r_max = radius.min(), radius.max()
bin_edges = np.linspace(r_min, r_max, num_bins + 1)
bin_indices = np.digitize(radius, bin_edges) - 1
binned_samples = [[] for _ in range(num_bins)]
avg_radius = 0.5 * (bin_edges[:-1] + bin_edges[1:])
for idx, bin_idx in enumerate(bin_indices):
    if 0 <= bin_idx < num_bins:
            binned_samples[bin_idx].append(marker1[idx])
binned_samples = [np.array(bin) for bin in binned_samples]

# Statistics of the distribution: Average and standard deviation
avg_marker = np.array([np.mean(binned_samples[i]) for i in range(len(binned_samples))])
std_marker = np.array([np.std(binned_samples[i]) for i in range(len(binned_samples))])
avg_radius = [avg_radius[i] for i in range(len(avg_radius)) if not math.isnan(avg_marker[i]) and avg_marker[i] < 20]
std_marker = [std_marker[i] for i in range(len(std_marker)) if not math.isnan(avg_marker[i]) and avg_marker[i] < 20]
avg_marker = [x for x in avg_marker if not math.isnan(x) and x < 20]
error_bar_up    = np.array(avg_marker) + 0.5 * np.array(std_marker)
error_bar_down  = np.array(avg_marker) - 0.5 * np.array(std_marker)

# Probability distribution P(\nu)|_r
prob_dist = []
bin_marker_list = []
for i, lst in enumerate(binned_samples):
    counts, bin_edges = np.histogram(lst, bins='auto')
    bin_marker_list.append(0.5 * (bin_edges[:-1] + bin_edges[1:]))
    prob_dist.append(counts / len(lst))




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


# Defining a colormap
vmin, vmax = 0.75 * np.min(marker1), 0.75 * np.ceil(np.max(marker1))
cbar_ticks = np.arange(np.ceil(vmin), np.floor(vmax) + 1, 1)
hex_list = ['#ff416d', '#ff7192', '#ffa0b6', '#ffd0db', '#ffffff', '#cfdaff', '#9fb6ff', '#6f91ff', '#3f6cff']
cmap = get_continuous_cmap(hex_list)
divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
colormap = cm.ScalarMappable(norm=divnorm, cmap=cmap)
gradient = np.linspace(vmin, vmax, 256).reshape(-1, 1)

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
cax.set_ylabel('$\\nu$', labelpad=10, fontsize=20)
cax.yaxis.set_label_position('right')
cax.tick_params(which='major', width=0.75, labelsize=fontsize)

# Figure 1
fig1.suptitle(f'$N= {Nx}$, $L= {Nz}$, $z_0= {z01}$, $z_1= {z11}$, $w={width1 :.2f}$', y=0.93, fontsize=20)
ax1.scatter(x1, y1, c=marker1, facecolor='white', edgecolor='black')
ax1.scatter(x1, y1, c=marker1, norm=divnorm, cmap=get_continuous_cmap(hex_list),  linewidths=2.5)
ax1.set_xlabel('$x$', fontsize=fontsize)
ax1.set_ylabel('$y$', fontsize=fontsize)
ax1.set_xlim(-np.max(x1) - 0.2, np.max(x1) + 0.2)
ax1.set_ylim(-np.max(y1) - 0.2, np.max(y1) + 0.2)

theta = np.linspace(0, 2 * np.pi, 200)
for i, rad in enumerate(avg_radius):
    x, y = rad * np.cos(theta), rad * np.sin(theta)
    ax1.plot(x, y, marker='none', linestyle='dashed', color=palette[i], alpha=0.5)



# Figure 2
fig2 = plt.figure(figsize=(8, 8))
gs = GridSpec(2, 1, figure=fig2, wspace=0.2, hspace=0.3)
ax1 = fig2.add_subplot(gs[0, 0])
ax2 = fig2.add_subplot(gs[1, 0])

ax1.plot(avg_radius, avg_marker, marker='o', color='#3F6CFF', label=f'$w= {width1}$')
ax1.fill_between(avg_radius, error_bar_down, error_bar_up, color='#3F6CFF', alpha=0.3)
ax1.set_xlabel('Cross section radius', fontsize=fontsize)
ax1.set_ylabel('$\\nu(r)$', fontsize=fontsize)
ax1.set_ylim(vmin, 1)
ax1.tick_params(which='major', width=0.75, labelsize=fontsize)
ax1.tick_params(which='major', length=6, labelsize=fontsize)
ax1.legend()
# ax1.set_xscale('log')
fig2.suptitle(f'$N= {Nx}$, $L= {Nz}$, $z_0= {z01}$, $z_1= {z11}$, $w={width1 :.2f}$', y=0.93, fontsize=20)

for i, (probs, markers) in enumerate(zip(prob_dist, bin_marker_list)):
    ax2.plot(markers, probs, marker='o', color=palette[i], label=f'$r= {avg_radius[i] :.2f}$')
ax2.plot(np.ones((10, )) * (-1), np.linspace(0, 10, 10), color='grey', linestyle='dashed')
ax2.plot(np.zeros((10, )), np.linspace(0, 10, 10), color='grey', linestyle='dashed')
ax2.set_ylim(0, 0.5)
ax2.set_xlabel('$\\nu$', fontsize=fontsize)
ax2.set_ylabel('$P(\\nu)$', fontsize=fontsize)
ax2.tick_params(which='major', width=0.75, labelsize=fontsize)
ax2.tick_params(which='major', length=6, labelsize=fontsize)
ax2.legend(ncol=3)



# Figure 3
fig3 = plt.figure(figsize=(8, 8))
gs = GridSpec(len(avg_radius), 1, figure=fig3, wspace=0., hspace=0.3)

for i, (probs, markers) in enumerate(zip(prob_dist, bin_marker_list)):
    ax = fig3.add_subplot(gs[i, 0])
    ax.plot(markers, probs, marker='o', color=palette[i], label=f'$r= {avg_radius[i] :.2f}$')
    ax.plot(np.ones((10, )) * (-1), np.linspace(0, 10, 10), color='grey', linestyle='dashed')
    ax.plot(np.zeros((10, )), np.linspace(0, 10, 10), color='grey', linestyle='dashed')
    ax.set_ylim(0, 0.5)
    ax.set_xlim(np.min(marker1), np.max(marker1))
    # ax.set_ylabel('$P(\\nu)$', fontsize=fontsize)
    ax.tick_params(which='major', width=0.75, labelsize=fontsize)
    ax.tick_params(which='major', length=6, labelsize=fontsize)
    ax.set_yticks(ticks=[0, 0.5], labels=[])
    ax.legend()

# ax.set_yticks(ticks=[0, 0.5], labels=['0', '0.5'])
ax.set_xticks(ticks=np.arange(-1, np.ceil(np.max(marker1)) + 1))
ax.set_xlabel('$\\nu$', fontsize=fontsize)
plt.show()
