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
file_list = ['Exp10.h5', 'Exp9.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-marker-per-site')

# File 1
N1           = data_dict[file_list[0]]['Parameters']['Nx']
Nz1           = data_dict[file_list[0]]['Parameters']['Nz']
local_marker1  = data_dict[file_list[0]]['Simulation']['local_marker']
bin_edges1     = data_dict[file_list[0]]['Simulation']['bin_edges']
width1         = data_dict[file_list[0]]['Simulation']['width']
z_min1         = data_dict[file_list[0]]['Simulation']['z_min']
avg_radius1    = 0.5 * (bin_edges1[:-1] + bin_edges1[1:])

# File 2
N2             = data_dict[file_list[1]]['Parameters']['Nx']
Nz2            = data_dict[file_list[1]]['Parameters']['Nz']
local_marker2  = data_dict[file_list[1]]['Simulation']['local_marker']
bin_edges2     = data_dict[file_list[1]]['Simulation']['bin_edges']
width2         = data_dict[file_list[1]]['Simulation']['width']
z_min2         = data_dict[file_list[1]]['Simulation']['z_min']
avg_radius2    = 0.5 * (bin_edges2[:-1] + bin_edges2[1:])

#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange']
marker_list=['o', 's', 'd', 'p', '*', 'h', '>', '<', 'X']
markersize = 5
fontsize=20
palette = seaborn.color_palette(palette='magma_r', n_colors=len(width1))

# Figure 1: Definition
fig1 = plt.figure(figsize=(8, 8))
gs = GridSpec(1, 1, figure=fig1, wspace=0.2, hspace=0.3)
ax1 = fig1.add_subplot(gs[0, 0])
ax_vec = [ax1]

for i in range(len(width1)):
    ax1.plot(avg_radius1, local_marker1[i, :], marker='o', color=palette[i], label=f'$w= {width1[i]}$, $N= {N1}$', linestyle='dashed')
    ax1.plot(avg_radius2, local_marker2[i, :], marker='o', color=palette[i], label=f'$w= {width2[i]}$, $N= {N2}$')
ax1.set_xlabel('$r_{xy}$', fontsize=fontsize)
ax1.set_ylabel('$\langle\\nu(r)\\rangle_{r_{xy}}$', fontsize=fontsize)
ax1.set_ylim(-1, 2)
ax1.tick_params(which='major', width=0.75, labelsize=fontsize)
ax1.tick_params(which='major', length=6, labelsize=fontsize)
ax1.legend()
# ax1.set_xscale('log')
fig1.suptitle(f'$L= {Nz1}$, $z_0= {z_min1}$', y=0.93, fontsize=20)


plt.show()