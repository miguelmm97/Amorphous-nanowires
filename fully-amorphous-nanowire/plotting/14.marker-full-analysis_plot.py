#%% Modules and setup

# Plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
import seaborn

# Modules
from modules.functions import *
from modules.colorbar_marker import *


import sys
from datetime import date


#%% Loading data
file_list = ['Exp7.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-marker-full-analysis')

# Plot 1
avg_marker         = data_dict[file_list[0]]['Plot1']['avg_marker']
width              = data_dict[file_list[0]]['Plot1']['width']
Nx                 = data_dict[file_list[0]]['Plot1']['Nx']

# Plot 2
Nx_plot            = data_dict[file_list[0]]['Plot2']['Nx_plot']
w                  = data_dict[file_list[0]]['Plot2']['w']
pos1               = data_dict[file_list[0]]['Plot2']['pos1']
pos2               = data_dict[file_list[0]]['Plot2']['pos2']
pos3               = data_dict[file_list[0]]['Plot2']['pos3']
marker1            = data_dict[file_list[0]]['Plot2']['marker1']
marker2            = data_dict[file_list[0]]['Plot2']['marker2']
marker3            = data_dict[file_list[0]]['Plot2']['marker3']
cutoff             = data_dict[file_list[0]]['Plot2']['cutoff']
avg_marker2        = data_dict[file_list[0]]['Plot2']['avg_marker2']

# Plot 3
cutoff_sequence    = data_dict[file_list[0]]['Plot3']['cutoff_sequence']
marker_transition  = data_dict[file_list[0]]['Plot3']['marker_transition']
width3             = data_dict[file_list[0]]['Plot3']['width']


#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['dodgerblue', 'limegreen', 'm', 'r', 'orange']
fontsize = 20
palette = seaborn.color_palette(palette='magma', n_colors=len(Nx))
palette2 = seaborn.color_palette(palette='magma', n_colors=len(width))


# Figure 1
fig1 = plt.figure()
gs = GridSpec(1, 1, figure=fig1, wspace=0.0, hspace=0.02)
ax1 = fig1.add_subplot(gs[0, 0])

for i in range(len(Nx)):
    ax1.plot(width, avg_marker[:, i], marker='o', linestyle='solid', color=palette[i], label=f'$n_x= {Nx[i]}$')

ax1.legend(frameon=False, fontsize=15)
ax1.set_xlabel('$w$', fontsize=fontsize)
ax1.set_ylabel('$\overline{\\nu}$', fontsize=fontsize)
ax1.set_ylim([-1, 0.1])
ax1.set_xlim([0, width[-1]])
ax1.tick_params(which='major', width=0.75, labelsize=fontsize)
ax1.tick_params(which='major', length=6, labelsize=fontsize)
ax1.tick_params(which='minor', width=0.75, labelsize=fontsize)
ax1.tick_params(which='minor', length=3, labelsize=fontsize)



# Figure 2
# Defining a colormap
divnorm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=-0.5, vmax=0)
hex_list = ['#ff416d', '#ff7192', '#ffa0b6', '#ffd0db', '#ffffff', '#cfdaff', '#9fb6ff', '#6f91ff', '#3f6cff']
colormap = cm.ScalarMappable(norm=Normalize(vmin=-1, vmax=0), cmap=get_continuous_cmap(hex_list))

# Figure 1
fig2 = plt.figure(figsize=(8, 5))
gs = GridSpec(1, 3, figure=fig2, wspace=0.0, hspace=0.02)
ax1 = fig2.add_subplot(gs[0, 0], projection='3d')
ax2 = fig2.add_subplot(gs[0, 1], projection='3d')
ax3 = fig2.add_subplot(gs[0, 2], projection='3d')
fig2.suptitle(f'$N_x= {Nx_plot}, w= {w :.2f}$', fontsize=fontsize)

ax1.scatter(pos1[0], pos1[1], pos1[2], c=marker1, facecolor='white', edgecolor='black')
ax2.scatter(pos2[0], pos2[1], pos2[2], c=marker2, facecolor='white', edgecolor='black')
ax3.scatter(pos3[0], pos3[1], pos3[2], c=marker3, facecolor='white', edgecolor='black')
scatters1 = ax1.scatter(pos1[0], pos1[1], pos1[2], c=marker1, norm=divnorm, cmap=get_continuous_cmap(hex_list),  linewidths=2.5)
scatters2 = ax2.scatter(pos2[0], pos2[1], pos2[2], c=marker2, norm=divnorm, cmap=get_continuous_cmap(hex_list),  linewidths=2.5)
scatters3 = ax3.scatter(pos3[0], pos3[1], pos3[2], c=marker3, norm=divnorm, cmap=get_continuous_cmap(hex_list),  linewidths=2.5)
limx, limy, limz = ax3.get_xlim(), ax3.get_ylim(), ax3.get_zlim()
ax1.set_xlim(limx)
ax1.set_ylim(limy)
ax1.set_zlim(limz)
ax2.set_xlim(limx)
ax2.set_ylim(limy)
ax2.set_zlim(limz)

for i, ax in enumerate(fig2.axes):
    ax.set_title(f'$n_x, n_y < {cutoff[i] :.2f}$ \\newline \quad $\\nu= {avg_marker2[i] :.2f}$ ', fontsize=fontsize)
    ax.set_box_aspect((3, 3, 10))
    ax.set_axis_off()

cbar_ax = fig2.add_subplot(gs[0, :])
divider = make_axes_locatable(cbar_ax)
cax = divider.append_axes("bottom", size="10%", pad=0)
cbar = fig2.colorbar(colormap, cax=cax, orientation='horizontal')
cbar_ax.set_axis_off()
cbar.set_label(label='$\\nu$', labelpad=10, fontsize=20)
cbar.ax.tick_params(which='major', width=0.75, labelsize=fontsize)
cbar.ax.ticklabel_format(style='sci')




# Figure 3
fig3 = plt.figure()
gs = GridSpec(1, 1, figure=fig3, wspace=0.0, hspace=0.02)
ax1 = fig3.add_subplot(gs[0, 0])

for i in range(len(width)):
    ax1.plot(cutoff_sequence, marker_transition[i, :], marker='o', linestyle='solid', color=palette2[i], label=f'$w= {width[i] :.2f}$')
ax1.set_xlabel('$S_{x, y}$ ', fontsize=fontsize)
ax1.set_ylabel('$\overline{\\nu}$', fontsize=fontsize)
ax1.set_ylim([-1, 0])
ax1.set_xlim([cutoff_sequence[0], 1])
ax1.legend(ncol=3, frameon=False, fontsize=10)
plt.show()

