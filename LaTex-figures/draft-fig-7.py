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
file_list = ['draft-fig7.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-latex-figures')

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
fig1 = plt.figure(figsize=(10, 7))
gs = GridSpec(2, 6, figure=fig1, wspace=0.4, hspace=0.3)
ax1 = fig1.add_subplot(gs[0, :3])
ax2 = fig1.add_subplot(gs[0, 3:])
ax3 = fig1.add_subplot(gs[1, :2],  projection='3d')
ax4 = fig1.add_subplot(gs[1, 2:4], projection='3d')
ax5 = fig1.add_subplot(gs[1, 4:],  projection='3d')

for i in range(len(Nx)):
    ax1.plot(width, avg_marker[:, i], marker='o', linestyle='solid', color=palette[i], label=f'${Nx[i]}$')

lgnd = ax1.legend(ncol=2, frameon=False, fontsize=13, handlelength=1, bbox_to_anchor=(0.35, 0.35), columnspacing=0.5)
ax1.text(0.05, -0.1, '$\\underline{N_{x, y}}$', fontsize=fontsize-2)
ax1.text(0.35, -0.8, '$N_z = 15$', fontsize=fontsize-2)
ax1.text(0.23, -0.95, f'$\\vert x, y, z \\vert< {0.4}$' + '$N_{x, y, z}$', fontsize=fontsize-2)
ax1.set_xlabel('$w$', fontsize=fontsize)
ax1.set_ylabel('$\overline{\\nu}$', fontsize=fontsize)
ax1.set_ylim([-1, 0])
ax1.set_xlim([0, width[-1]])
majorsy = [-1, -0.5, 0]
minorsy = [-0.75, -0.25]
ax1.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
ax1.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
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
# fig2.suptitle(f'$N_x= {Nx_plot}, w= {w :.2f}$', fontsize=fontsize)

ax3.scatter(pos1[0], pos1[1], pos1[2], c=marker1, facecolor='white', edgecolor='black')
ax4.scatter(pos2[0], pos2[1], pos2[2], c=marker2, facecolor='white', edgecolor='black')
ax5.scatter(pos3[0], pos3[1], pos3[2], c=marker3, facecolor='white', edgecolor='black')
scatters1 = ax3.scatter(pos1[0], pos1[1], pos1[2], c=marker1, norm=divnorm, cmap=get_continuous_cmap(hex_list),  linewidths=2.5)
scatters2 = ax4.scatter(pos2[0], pos2[1], pos2[2], c=marker2, norm=divnorm, cmap=get_continuous_cmap(hex_list),  linewidths=2.5)
scatters3 = ax5.scatter(pos3[0], pos3[1], pos3[2], c=marker3, norm=divnorm, cmap=get_continuous_cmap(hex_list),  linewidths=2.5)
limx, limy, limz = ax5.get_xlim(), ax5.get_ylim(), ax5.get_zlim()
ax3.set_xlim(limx)
ax3.set_ylim(limy)
ax3.set_zlim(limz)
ax4.set_xlim(limx)
ax4.set_ylim(limy)
ax4.set_zlim(limz)

for i, ax in enumerate([ax3, ax4, ax5]):
    # ax.set_title(f'$n_x, n_y < {cutoff[i] :.2f}$ \\newline \quad $\\nu= {avg_marker2[i] :.2f}$ ', fontsize=fontsize)
    ax.set_box_aspect((3, 3, 10))
    ax.set_axis_off()

cbar_ax = fig1.add_subplot(gs[1, :])
divider = make_axes_locatable(cbar_ax)
cax = divider.append_axes("bottom", size="10%", pad=0)
cbar = fig1.colorbar(colormap, cax=cax, orientation='horizontal')
cbar_ax.set_axis_off()
cbar.set_label(label='$\\nu$', labelpad=10, fontsize=20)
cbar.ax.tick_params(which='major', width=0.75, labelsize=fontsize)
cbar.ax.ticklabel_format(style='sci')


scatter_ax3 = fig1.add_axes([0.08, 0.3, 0.05, 0.05])
scatter_ax3.set_xticks([])
scatter_ax3.set_yticks([])
scatter_ax3.set_axis_off()
scatter_ax3.text(-0.5, 0, '$\\vert x, y \\vert <' + f'{cutoff[0] :.1f}' + 'N_{x, y}$', fontsize=fontsize - 2)
scatter_ax3.text(0, -1, '$\\vert z \\vert < N_z$ ', fontsize=fontsize - 2)
scatter_ax3.text(0, -2, f'$\\nu= {avg_marker2[0] :.2f}$ ', fontsize=fontsize - 2)

scatter_ax4 = fig1.add_axes([0.35, 0.3, 0.05, 0.05])
scatter_ax4.set_xticks([])
scatter_ax4.set_yticks([])
scatter_ax4.set_axis_off()
scatter_ax4.text(0, 0, '$S_{x,y}' + f'= {cutoff[1] :.1f} (\%)$', fontsize=fontsize - 2)
scatter_ax4.text(0, -1, f'$\\nu= {avg_marker2[1] :.2f}$ ', fontsize=fontsize - 2)

scatter_ax5 = fig1.add_axes([0.6, 0.3, 0.05, 0.05])
scatter_ax5.set_xticks([])
scatter_ax5.set_yticks([])
scatter_ax5.set_axis_off()
scatter_ax5.text(0, 0, '$S_{x,y}' + f'= {cutoff[2] :.1f} (\%)$', fontsize=fontsize - 2)
scatter_ax5.text(0, -1, f'$\\nu= {avg_marker2[2] :.2f}$ ', fontsize=fontsize - 2)

scatter_ax5 = fig1.add_axes([0.85, 0.3, 0.05, 0.05])
scatter_ax5.set_xticks([])
scatter_ax5.set_yticks([])
scatter_ax5.set_axis_off()
scatter_ax5.text(0, 0, '$n_{x, y}' + f'= {12}$', fontsize=fontsize - 2)
scatter_ax5.text(0, -1, f'$w={w :.2f}$ ', fontsize=fontsize - 2)







# Figure 3
for i in range(len(width)):
    ax2.plot(cutoff_sequence, marker_transition[i, :], marker='o', linestyle='solid', color=palette2[i], label=f'${width[i] :.2f}$')
ax2.set_xlabel('$\\vert x, y, z \\vert$', fontsize=fontsize)
# ax2.set_ylabel('$\overline{\\nu}$', fontsize=fontsize)
ax2.set_ylim([-1, 0])
ax2.set_xlim([cutoff_sequence[0], 1])
# ax2.legend(ncol=2, frameon=False, fontsize=13, handlelength=1)
ax2.text(0.82, -0.97, f'$w= {width[0] :.2f}$', fontsize=15)
ax2.text(0.55, -0.085, f'$w= {width[-1] :.2f}$', fontsize=15)
ax2.text(0.25, -0.15, '$N_{x, y} = 12$', fontsize=15)
# ax2.text(0.25, -0.2, '$N_{z} = 15$', fontsize=15)
majorsy = [-1, -0.5, 0]
ylabels = ['', '', '']
ax2.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
ax2.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
ax2.tick_params(which='major', width=0.75, labelsize=fontsize)
ax2.tick_params(which='major', length=6, labelsize=fontsize)
ax2.tick_params(which='minor', width=0.75, labelsize=fontsize)
ax2.tick_params(which='minor', length=3, labelsize=fontsize)
ax2.set(yticklabels=ylabels)
plt.show()

