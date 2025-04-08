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
file_list = ['Exp10.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-latex-figures')

# Data
width              = data_dict[file_list[0]]['Parameters']['width']
pos1               = data_dict[file_list[0]]['Simulation']['Xcuts']
pos2               = data_dict[file_list[0]]['Simulation']['Ycuts']
pos3               = data_dict[file_list[0]]['Simulation']['Zcuts']
marker             = data_dict[file_list[0]]['Simulation']['marker_cuts']
cutoff             = data_dict[file_list[0]]['Simulation']['cutoff_sequence']
nsites             = data_dict[file_list[0]]['Simulation']['Nsites_in_cut']


idx_width1 = 4
idx_cutoff1, idx_cutoff2, idx_cutoff3 = 2, 6, -1
n1, n2, n3 = int(nsites[idx_width1, idx_cutoff1]), int(nsites[idx_width1, idx_cutoff2]), int(nsites[idx_width1, idx_cutoff3])
xcut1, ycut1, zcut1 = pos1[idx_width1, :n1, idx_cutoff1], pos2[idx_width1, :n1, idx_cutoff1], pos3[idx_width1, :n1, idx_cutoff1]
xcut2, ycut2, zcut2 = pos1[idx_width1, :n2, idx_cutoff2], pos2[idx_width1, :n2, idx_cutoff2], pos3[idx_width1, :n2, idx_cutoff2]
xcut3, ycut3, zcut3 = pos1[idx_width1, :n3, idx_cutoff3], pos2[idx_width1, :n3, idx_cutoff3], pos3[idx_width1, :n3, idx_cutoff3]
marker1, marker2, marker3 = marker[idx_width1, :n1, idx_cutoff1], marker[idx_width1, :n2, idx_cutoff2], marker[idx_width1, :n3, idx_cutoff3]
avg_marker1, avg_marker2, avg_marker3 = np.mean(marker1), np.mean(marker2), np.mean(marker3)


idx_width2, idx_width3, idx_width4 = 3, 7, 10
idx_cutoff = 5
n1, n2, n3 = int(nsites[idx_width2, idx_cutoff]), int(nsites[idx_width2, idx_cutoff]), int(nsites[idx_width2, idx_cutoff])
xcut1_2, ycut1_2, zcut1_2 = pos1[idx_width2, :n1, idx_cutoff], pos2[idx_width2, :n1, idx_cutoff], pos3[idx_width2, :n1, idx_cutoff2]
xcut2_2, ycut2_2, zcut2_2 = pos1[idx_width3, :n2, idx_cutoff], pos2[idx_width3, :n2, idx_cutoff], pos3[idx_width3, :n2, idx_cutoff3]
xcut3_2, ycut3_2, zcut3_2 = pos1[idx_width4, :n3, idx_cutoff], pos2[idx_width4, :n3, idx_cutoff], pos3[idx_width4, :n3, idx_cutoff]
marker1_2, marker2_2, marker3_2 = marker[idx_width2, :n1, idx_cutoff], marker[idx_width3, :n2, idx_cutoff], marker[idx_width4, :n3, idx_cutoff]
avg_marker1_2, avg_marker2_2, avg_marker3_2 = np.mean(marker1_2), np.mean(marker2_2), np.mean(marker3_2)


#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['dodgerblue', 'limegreen', 'm', 'r', 'orange']
fontsize = 20

plt.rcParams.update({
    'axes.facecolor': 'black',
    'figure.facecolor': 'black',
    'savefig.facecolor': 'black',
    'axes.edgecolor': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'text.color': 'white',
    'legend.edgecolor': 'white',
    'legend.facecolor': 'black',
})


# Figure 1
fig1 = plt.figure(figsize=(8, 6))
gs = GridSpec(2, 3, figure=fig1, wspace=0.1, hspace=0.3)
ax1 = fig1.add_subplot(gs[0, 0],  projection='3d')
ax2 = fig1.add_subplot(gs[0, 1], projection='3d')
ax3 = fig1.add_subplot(gs[0, 2],  projection='3d')
ax4 = fig1.add_subplot(gs[1, 0],  projection='3d')
ax5 = fig1.add_subplot(gs[1, 1], projection='3d')
ax6 = fig1.add_subplot(gs[1, 2],  projection='3d')

# Figure 2
# Defining a colormap
divnorm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=-0.5, vmax=1)
hex_list = ['#ff416d', '#ff7192', '#ffa0b6', '#ffd0db', '#ffffff', '#cfdaff', '#9fb6ff', '#6f91ff', '#3f6cff']
colormap = cm.ScalarMappable(norm=Normalize(vmin=-1, vmax=1), cmap=get_continuous_cmap(hex_list))

# Figure 1
# Plot
ax1.scatter(xcut1, ycut1, zcut1, c=marker1, facecolor='white', edgecolor='black')
ax2.scatter(xcut2, ycut2, zcut2, c=marker2, facecolor='white', edgecolor='black')
ax3.scatter(xcut3, ycut3, zcut3, c=marker3, facecolor='white', edgecolor='black')
scatters1 = ax1.scatter(xcut1, ycut1, zcut1, c=marker1, norm=divnorm, cmap=get_continuous_cmap(hex_list),  linewidths=2.5)
scatters2 = ax2.scatter(xcut2, ycut2, zcut2, c=marker2, norm=divnorm, cmap=get_continuous_cmap(hex_list),  linewidths=2.5)
scatters3 = ax3.scatter(xcut3, ycut3, zcut3, c=marker3, norm=divnorm, cmap=get_continuous_cmap(hex_list),  linewidths=2.5)

ax4.scatter(xcut1_2, ycut1_2, zcut1_2, c=marker1_2, facecolor='white', edgecolor='black')
ax5.scatter(xcut2_2, ycut2_2, zcut2_2, c=marker2_2, facecolor='white', edgecolor='black')
ax6.scatter(xcut3_2, ycut3_2, zcut3_2, c=marker3_2, facecolor='white', edgecolor='black')
scatters4 = ax4.scatter(xcut1_2, ycut1_2, zcut1_2, c=marker1_2, norm=divnorm, cmap=get_continuous_cmap(hex_list),  linewidths=2.5)
scatters5 = ax5.scatter(xcut2_2, ycut2_2, zcut2_2, c=marker2_2, norm=divnorm, cmap=get_continuous_cmap(hex_list),  linewidths=2.5)
scatters6 = ax6.scatter(xcut3_2, ycut3_2, zcut3_2, c=marker3_2, norm=divnorm, cmap=get_continuous_cmap(hex_list),  linewidths=2.5)




# Limits
limx, limy, limz = ax3.get_xlim(), ax3.get_ylim(), ax3.get_zlim()
ax1.set_xlim(limx)
ax1.set_ylim(limy)
ax1.set_zlim(limz)
ax2.set_xlim(limx)
ax2.set_ylim(limy)
ax2.set_zlim(limz)
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6]):
    ax.set_box_aspect((3, 3, 5))
    ax.set_axis_off()

# Titles
ax1.set_title(f'$ R = {cutoff[idx_cutoff1] * 6:.2f}$ \n$\\nu= {avg_marker1 :.2f}$ ', fontsize=fontsize, pad=-5)
ax2.set_title(f'$ R = {cutoff[idx_cutoff2] * 6:.2f}$ \n$\\nu= {avg_marker2 :.2f}$', fontsize=fontsize, pad=-5)
ax3.set_title(f'$ R = {cutoff[idx_cutoff3] * 6:.2f}$ \n$\\nu= {avg_marker3 :.2f}$', fontsize=fontsize, pad=-5)
ax4.set_title(f'$ w = {width[idx_width2] :.2f}$ \n $\\nu= {avg_marker1_2 :.2f}$', fontsize=fontsize, pad=-25)
ax5.set_title(f'$ w = {width[idx_width3] :.2f}$ \n $\\nu= {avg_marker2_2 :.2f}$', fontsize=fontsize, pad=-25)
ax6.set_title(f'$ w = {width[idx_width4] :.2f}$ \n $\\nu= {avg_marker3_2 :.2f}$', fontsize=fontsize, pad=-25)


cbar_ax = fig1.add_subplot(gs[1, :])
divider = make_axes_locatable(cbar_ax)
cax = divider.append_axes("bottom", size="5%", pad=0)
cbar = fig1.colorbar(colormap, cax=cax, orientation='horizontal')
cbar_ax.set_axis_off()
cbar.set_label(label='$\\nu$', labelpad=0, fontsize=20)
cbar.ax.tick_params(which='major', width=0.75, labelsize=fontsize)
cbar.ax.ticklabel_format(style='sci')

# Text
scatter_ax1 = fig1.add_axes([0.02, 0.7, 0.05, 0.05])
scatter_ax2 = fig1.add_axes([0.02, 0.3, 0.05, 0.05])
scatter_ax1.set_xticks([])
scatter_ax1.set_yticks([])
scatter_ax1.set_axis_off()
scatter_ax2.set_xticks([])
scatter_ax2.set_yticks([])
scatter_ax2.set_axis_off()
scatter_ax1.text(0, 0, f'$w= {width[idx_width1] :.2f}$', fontsize=fontsize)
scatter_ax2.text(0, 0, f' $R= {cutoff[idx_cutoff] * 6 :.2f}$', fontsize=fontsize)
# scatter_ax1.text(0, -2, f'$\\nu= {avg_marker2 :.2f}$ ', fontsize=fontsize - 2)

scatter_ax1 = fig1.add_axes([0.12, 0.91, 0.05, 0.05])
scatter_ax1.set_xticks([])
scatter_ax1.set_yticks([])
scatter_ax1.set_axis_off()
scatter_ax1.text(0, 0, f'$(a)$', fontsize=fontsize)

scatter_ax1 = fig1.add_axes([0.39, 0.91, 0.05, 0.05])
scatter_ax1.set_xticks([])
scatter_ax1.set_yticks([])
scatter_ax1.set_axis_off()
scatter_ax1.text(0, 0, f'$(b)$', fontsize=fontsize)

scatter_ax1 = fig1.add_axes([0.66, 0.91, 0.05, 0.05])
scatter_ax1.set_xticks([])
scatter_ax1.set_yticks([])
scatter_ax1.set_axis_off()
scatter_ax1.text(0, 0, f'$(c)$', fontsize=fontsize)
#
scatter_ax1 = fig1.add_axes([0.12, 0.47, 0.05, 0.05])
scatter_ax1.set_xticks([])
scatter_ax1.set_yticks([])
scatter_ax1.set_axis_off()
scatter_ax1.text(0, 0, f'$(d)$', fontsize=fontsize)
#
scatter_ax1 = fig1.add_axes([0.39, 0.47, 0.05, 0.05])
scatter_ax1.set_xticks([])
scatter_ax1.set_yticks([])
scatter_ax1.set_axis_off()
scatter_ax1.text(0, 0, f'$(e)$', fontsize=fontsize)
#
scatter_ax1 = fig1.add_axes([0.66, 0.47, 0.05, 0.05])
scatter_ax1.set_xticks([])
scatter_ax1.set_yticks([])
scatter_ax1.set_axis_off()
scatter_ax1.text(0, 0, f'$(f)$', fontsize=fontsize)


fig1.savefig('fig2-marker.pdf', format='pdf')
plt.show()

