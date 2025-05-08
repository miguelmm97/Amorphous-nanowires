#%% Modules and setup

# Plotting
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm

# Modules
from modules.functions import *
from modules.colorbar_marker import *


#%% Loading data
file_list = ['Exp4.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-marker-per-site')

# Parameters
Nx           = data_dict[file_list[0]]['Parameters']['Nx']
Ny           = data_dict[file_list[0]]['Parameters']['Ny']
Nz           = data_dict[file_list[0]]['Parameters']['Nz']
r            = data_dict[file_list[0]]['Parameters']['r ']
t            = data_dict[file_list[0]]['Parameters']['t ']
eps          = data_dict[file_list[0]]['Parameters']['eps']
lamb         = data_dict[file_list[0]]['Parameters']['lamb']
lamb_z       = data_dict[file_list[0]]['Parameters']['lamb_z']
flux         = data_dict[file_list[0]]['Parameters']['flux']

# Simulation data
pos  = data_dict[file_list[0]]['Simulation']['position']
local_marker = data_dict[file_list[0]]['Simulation']['local_marker']
width        = data_dict[file_list[0]]['Simulation']['width']
radius = np.sqrt(((pos[:, 0] - 0.5 * (Nx - 1)) ** 2) + ((pos[:, 1] - 0.5 * (Ny -1)) ** 2))
z_min, z_max = 0.2 * (Nz-1), 0.8 * (Nz-1)

# Statistics
num_bins = 10
r_min, r_max = radius.min(), radius.max()
bin_edges = np.linspace(r_min, r_max, num_bins + 1)
bin_indices = np.digitize(radius, bin_edges) - 1
binned_samples = [[] for _ in range(num_bins)]
for idx, bin_idx in enumerate(bin_indices):
    if 0 <= bin_idx < num_bins:
        if z_min <= pos[idx, 2] < z_max:
            binned_samples[bin_idx].append(local_marker[idx])
binned_samples = [np.array(bin) for bin in binned_samples]
avg_marker = np.array([np.mean(binned_samples[i]) for i in range(len(binned_samples))])
avg_radius = 0.5 * (bin_edges[:-1] + bin_edges[1:])
avg_radius = [avg_radius[i] for i in range(len(avg_radius)) if not math.isnan(avg_marker[i])]
avg_marker = [x for x in avg_marker if not math.isnan(x)]



# Scatter plot of the cross-section
x = pos[:, 0] - 0.5 * (Nx - 1)
y = pos[:, 1] - 0.5 * (Ny - 1)
cond1 = pos[:, 2] < z_max
cond2 = z_min <= pos[:, 2]
x_scatter, y_scatter, marker_scatter =  x[cond1 * cond2], y[cond1 * cond2], local_marker[cond1 * cond2]
#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange']
marker_list=['o', 's', 'd', 'p', '*', 'h', '>', '<', 'X']
markersize = 5
fontsize=20

# Figure 1: Definition
fig1 = plt.figure(figsize=(8, 8))
gs = GridSpec(1, 1, figure=fig1, wspace=0.2, hspace=0.3)
ax1 = fig1.add_subplot(gs[0, 0])
ax_vec = [ax1]

ax1.plot(avg_radius, avg_marker, marker='o', color='dodgerblue')
ax1.set_xlabel('Cross section radius', fontsize=fontsize)
ax1.set_ylabel('$\\nu(r)$', fontsize=fontsize)
ax1.set_ylim(-1, max(avg_marker))
ax1.tick_params(which='major', width=0.75, labelsize=fontsize)
ax1.tick_params(which='major', length=6, labelsize=fontsize)





fig2 = plt.figure(figsize=(8, 6))
gs = GridSpec(4, 4, figure=fig2, wspace=0.1, hspace=0.3)
ax1 = fig2.add_subplot(gs[:, :3])

# Defining a colormap
divnorm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=-0.5, vmax=1)
hex_list = ['#ff416d', '#ff7192', '#ffa0b6', '#ffd0db', '#ffffff', '#cfdaff', '#9fb6ff', '#6f91ff', '#3f6cff']
colormap = cm.ScalarMappable(norm=Normalize(vmin=-1, vmax=1), cmap=get_continuous_cmap(hex_list))

# Figure 1
# Plot
ax1.scatter(x_scatter, y_scatter, c=marker_scatter, facecolor='white', edgecolor='black')
ax1.scatter(x_scatter, y_scatter, c=marker_scatter, norm=divnorm, cmap=get_continuous_cmap(hex_list),  linewidths=2.5)
ax1.set_xlabel('$x$', fontsize=fontsize)
ax1.set_ylabel('$y$', fontsize=fontsize)


cbar_ax = fig2.add_subplot(gs[:, 3])
divider = make_axes_locatable(cbar_ax)
cax = divider.append_axes("left", size="1%", pad=-20)
cbar = fig1.colorbar(colormap, cax=cax, orientation='vertical')
cbar_ax.set_axis_off()
cbar.set_label(label='$\\nu$', labelpad=0, fontsize=20)
cbar.ax.tick_params(which='major', width=0.75, labelsize=fontsize)
cbar.ax.ticklabel_format(style='sci')
plt.show()