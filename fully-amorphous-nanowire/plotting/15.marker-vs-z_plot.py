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
file_list = ['Exp8.h5']
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
pos          = data_dict[file_list[0]]['Simulation']['position']
local_marker = data_dict[file_list[0]]['Simulation']['local_marker']
width        = data_dict[file_list[0]]['Simulation']['width']

pos[:, 0] = pos[:, 0] - 0.5 * (Nx - 1)
pos[:, 1] = pos[:, 1] - 0.5 * (Ny - 1)

#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange']
marker_list=['o', 's', 'd', 'p', '*', 'h', '>', '<', 'X']
markersize = 5
fontsize=20
palette = seaborn.color_palette(palette='magma_r', n_colors=4)

# Defining a colormap
divnorm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=-0.5, vmax=1)
hex_list = ['#ff416d', '#ff7192', '#ffa0b6', '#ffd0db', '#ffffff', '#cfdaff', '#9fb6ff', '#6f91ff', '#3f6cff']
colormap = cm.ScalarMappable(norm=Normalize(vmin=-1, vmax=1), cmap=get_continuous_cmap(hex_list))


# Scatter plot for different cross-sections
fig1 = plt.figure(figsize=(25, 8))
gs = GridSpec(2, 6, figure=fig1, wspace=0.4, hspace=0.3)

z_bins = np.linspace(-1, Nz, 10)
for i in range(len(z_bins) - 1):
    cond_1 = pos[:, 2] < z_bins[i+1]
    cond_2 = z_bins[i] <= pos[:, 2]
    cond = cond_1 * cond_2

    ax = fig1.add_subplot(gs[i // 5, i % 5])
    ax.scatter(pos[:, 0][cond], pos[:, 1][cond], c=local_marker[cond], facecolor='white', edgecolor='black')
    ax.scatter(pos[:, 0][cond], pos[:, 1][cond], c=local_marker[cond], norm=divnorm, cmap=get_continuous_cmap(hex_list), linewidths=2.5)
    ax.set_xlabel('$x$', fontsize=fontsize)
    ax.set_ylabel('$y$', fontsize=fontsize)
    ax.set_title(f'${z_bins[i] :.2f} < z < {z_bins[i+1] :.2f}$')


cbar_ax = fig1.add_subplot(gs[:, 5])
divider = make_axes_locatable(cbar_ax)
cax = divider.append_axes("left", size="1%", pad=-20)
cbar = fig1.colorbar(colormap, cax=cax, orientation='vertical')
cbar_ax.set_axis_off()
cbar.set_label(label='$\\nu$', labelpad=0, fontsize=20)
cbar.ax.tick_params(which='major', width=0.75, labelsize=fontsize)
cbar.ax.ticklabel_format(style='sci')


plt.show()

