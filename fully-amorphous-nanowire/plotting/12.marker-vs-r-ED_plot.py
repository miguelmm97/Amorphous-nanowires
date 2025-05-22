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
# file_list = ['Exp12.h5', 'Exp14.h5', 'Exp13.h5', 'Exp15.h5']  more restricted in z
file_list = ['Exp16.h5', 'Exp18.h5', 'Exp17.h5', 'Exp19.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-marker-per-site-stats')

# Parameters
Nx           = data_dict[file_list[0]]['Parameters']['Nx']
Ny           = data_dict[file_list[0]]['Parameters']['Ny']
Nz           = data_dict[file_list[0]]['Parameters']['Nz']
r            = data_dict[file_list[0]]['Parameters']['r ']
t            = data_dict[file_list[0]]['Parameters']['t ']
eps          = data_dict[file_list[0]]['Parameters']['eps']
lamb         = data_dict[file_list[0]]['Parameters']['lamb']
lamb_z       = data_dict[file_list[0]]['Parameters']['lamb_z']
z_min        = data_dict[file_list[0]]['Simulation']['z_min']
z_max        = data_dict[file_list[0]]['Simulation']['z_max']

# Simulation data
avg_radius1  = data_dict[file_list[0]]['Simulation']['avg_radius']
avg_marker1  = data_dict[file_list[0]]['Simulation']['avg_marker']
x1           = data_dict[file_list[0]]['Simulation']['x']
y1           = data_dict[file_list[0]]['Simulation']['y']
marker1      = data_dict[file_list[0]]['Simulation']['marker']
width1       = data_dict[file_list[0]]['Simulation']['width']

avg_radius2  = data_dict[file_list[1]]['Simulation']['avg_radius']
avg_marker2  = data_dict[file_list[1]]['Simulation']['avg_marker']
x2           = data_dict[file_list[1]]['Simulation']['x']
y2           = data_dict[file_list[1]]['Simulation']['y']
marker2      = data_dict[file_list[1]]['Simulation']['marker']
width2       = data_dict[file_list[1]]['Simulation']['width']

avg_radius3  = data_dict[file_list[2]]['Simulation']['avg_radius']
avg_marker3  = data_dict[file_list[2]]['Simulation']['avg_marker']
x3           = data_dict[file_list[2]]['Simulation']['x']
y3           = data_dict[file_list[2]]['Simulation']['y']
marker3      = data_dict[file_list[2]]['Simulation']['marker']
width3       = data_dict[file_list[2]]['Simulation']['width']

avg_radius4  = data_dict[file_list[3]]['Simulation']['avg_radius']
avg_marker4  = data_dict[file_list[3]]['Simulation']['avg_marker']
x4           = data_dict[file_list[3]]['Simulation']['x']
y4           = data_dict[file_list[3]]['Simulation']['y']
marker4      = data_dict[file_list[3]]['Simulation']['marker']
width4       = data_dict[file_list[3]]['Simulation']['width']



#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange']
marker_list=['o', 's', 'd', 'p', '*', 'h', '>', '<', 'X']
markersize = 5
fontsize=20
palette = seaborn.color_palette(palette='magma_r', n_colors=4)

# Figure 1: Definition
fig1 = plt.figure(figsize=(8, 8))
gs = GridSpec(1, 1, figure=fig1, wspace=0.2, hspace=0.3)
ax1 = fig1.add_subplot(gs[0, 0])
ax_vec = [ax1]


ax1.plot(avg_radius1, avg_marker1, marker='o', color=palette[0], label=f'$w= {width1}$')
ax1.plot(avg_radius2, avg_marker2, marker='o', color=palette[1], label=f'$w= {width2}$')
ax1.plot(avg_radius3, avg_marker3, marker='o', color=palette[2], label=f'$w= {width3}$')
ax1.plot(avg_radius4, avg_marker4, marker='o', color=palette[3], label=f'$w= {width4}$')
ax1.set_xlabel('Cross section radius', fontsize=fontsize)
ax1.set_ylabel('$\\nu(r)$', fontsize=fontsize)
ax1.set_ylim(-1, max(avg_marker1))
ax1.tick_params(which='major', width=0.75, labelsize=fontsize)
ax1.tick_params(which='major', length=6, labelsize=fontsize)
ax1.legend()
ax1.set_xscale('log')
fig1.suptitle(f' $r= {r}$, $L= {Nz}$,  $N= {Nx}$, $z0= {z_min}$, $z1= {z_max}$', y=0.93, fontsize=20)




#
fig2 = plt.figure(figsize=(10, 8))
gs = GridSpec(4, 5, figure=fig2, wspace=0.5, hspace=0.5)
ax1 = fig2.add_subplot(gs[:2, :2])
ax2 = fig2.add_subplot(gs[:2, 2:4])
ax3 = fig2.add_subplot(gs[2:, :2])
ax4 = fig2.add_subplot(gs[2:, 2:4])

# Defining a colormap
divnorm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=-0.5, vmax=1)
hex_list = ['#ff416d', '#ff7192', '#ffa0b6', '#ffd0db', '#ffffff', '#cfdaff', '#9fb6ff', '#6f91ff', '#3f6cff']
colormap = cm.ScalarMappable(norm=Normalize(vmin=-1, vmax=1), cmap=get_continuous_cmap(hex_list))

# Figure 1
# Plot
ax1.scatter(x1, y1, c=marker1, facecolor='white', edgecolor='black')
ax1.scatter(x1, y1, c=marker1, norm=divnorm, cmap=get_continuous_cmap(hex_list),  linewidths=2.5)
ax1.set_xlabel('$x$', fontsize=fontsize)
ax1.set_ylabel('$y$', fontsize=fontsize)

ax2.scatter(x2, y2, c=marker2, facecolor='white', edgecolor='black')
ax2.scatter(x2, y2, c=marker2, norm=divnorm, cmap=get_continuous_cmap(hex_list),  linewidths=2.5)
ax2.set_xlabel('$x$', fontsize=fontsize)
ax2.set_ylabel('$y$', fontsize=fontsize)

ax3.scatter(x3, y3, c=marker3, facecolor='white', edgecolor='black')
ax3.scatter(x3, y3, c=marker3, norm=divnorm, cmap=get_continuous_cmap(hex_list),  linewidths=2.5)
ax3.set_xlabel('$x$', fontsize=fontsize)
ax3.set_ylabel('$y$', fontsize=fontsize)

ax4.scatter(x4, y4, c=marker4, facecolor='white', edgecolor='black')
ax4.scatter(x4, y4, c=marker4, norm=divnorm, cmap=get_continuous_cmap(hex_list),  linewidths=2.5)
ax4.set_xlabel('$x$', fontsize=fontsize)
ax4.set_ylabel('$y$', fontsize=fontsize)


cbar_ax = fig2.add_subplot(gs[:, 4])
divider = make_axes_locatable(cbar_ax)
cax = divider.append_axes("left", size="1%", pad=-20)
cbar = fig1.colorbar(colormap, cax=cax, orientation='vertical')
cbar_ax.set_axis_off()
cbar.set_label(label='$\\nu$', labelpad=0, fontsize=20)
cbar.ax.tick_params(which='major', width=0.75, labelsize=fontsize)
cbar.ax.ticklabel_format(style='sci')
plt.show()
