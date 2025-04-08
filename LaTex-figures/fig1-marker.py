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
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import PatchCollection
from drawarrow import fig_arrow

# Modules
from modules.functions import *
from modules.colorbar_marker import *


#%% Loading data
file_list = ['Exp9.h5', 'data-cluster-bulk-full-range.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-latex-figures')

# Plot 1
avg_marker         = data_dict[file_list[1]]['Plot1']['avg_marker']
std_marker         = data_dict[file_list[1]]['Plot1']['std_marker']
width              = data_dict[file_list[1]]['Plot1']['width']
Nx                 = data_dict[file_list[1]]['Plot1']['Nx']
error_bar_up       = avg_marker + 0.5 * std_marker
error_bar_down     = avg_marker - 0.5 * std_marker

# Plot 2
cutoff_sequence    = data_dict[file_list[0]]['Simulation']['cutoff_sequence']
marker_transition  = data_dict[file_list[0]]['Simulation']['marker_transition']
width3             = data_dict[file_list[0]]['Parameters']['width']


#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['dodgerblue', 'limegreen', 'm', 'r', 'orange']
fontsize = 20
# palette = seaborn.color_palette(palette='magma_r', n_colors=len(Nx))
# palette2 = seaborn.color_palette(palette='viridis_r', n_colors=len(width3))

magma = cm.get_cmap('magma_r')
palette = [magma(i) for i in np.linspace(0., 0.7, len(Nx))]
palette2 = seaborn.color_palette(palette='viridis', n_colors=len(width3))

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
gs = GridSpec(1, 1, figure=fig1, wspace=0.1)
ax1 = fig1.add_subplot(gs[0, 0])
ax2 = ax1.inset_axes([0.56, 0.1, 0.4, 0.4], )

# Plot
for i in range(len(Nx)):
    ax1.plot(width, avg_marker[:, i], marker='o', linestyle='solid', color=palette[i], label=f'${Nx[i]}$')
    ax1.fill_between(width, error_bar_down[:, i], error_bar_up[:, i], color=palette[i], alpha=0.3)

# Legend and text
lgnd = ax1.legend(loc='upper left', ncol=1, frameon=False, fontsize=fontsize, handlelength=1, columnspacing=0.5, labelspacing=0.2, bbox_to_anchor=(0, 0.97))
ax1.text(0.09, 0.04, '$\\underline{N}$', fontsize=fontsize)
ax1.text(0.2, -0.0, '$N_s = 100$', fontsize=fontsize)
ax1.text(0.5, -0.25, f'$\\vert x, y, z \\vert< {0.4}N$', fontsize=fontsize)

# Labels and limits
ax1.set_xlabel('$w$', fontsize=fontsize)
ax1.set_ylabel('$\overline{\\nu}$', fontsize=fontsize, labelpad=-5)
ax1.set_ylim([-1, 0.1])
ax1.set_xlim([0, 0.8])

# Tick params
majorsy = [-1, - 0.75, -0.5, -0.25, 0]
ax1.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
ax1.tick_params(which='major', width=0.75, labelsize=fontsize)
ax1.tick_params(which='major', length=6, labelsize=fontsize)
ax1.tick_params(which='minor', width=0.75, labelsize=fontsize)
ax1.tick_params(which='minor', length=3, labelsize=fontsize)


# cutoff_sequence = 0.5 * Nx[0] * cutoff_sequence

# Plot
for i in range(len(width3)):
    ax2.plot(cutoff_sequence, marker_transition[i, :], marker='o', linestyle='solid', color=palette2[i], label=f'${width3[i] :.2f}$', alpha=0.7)

# Arrow and text
start_point = (0.9, -0.9)
end_point = (0.55, 0.15)
arrow = FancyArrowPatch(start_point, end_point, connectionstyle="arc3,rad=0.3", color=(0.5, 0.5, 0.5), lw=1,  mutation_scale=20, arrowstyle='simple', alpha=0.3)
ax2.add_patch(arrow)
ax2.text(0.78, -0.99, f'$w= {width3[0] :.0f}$', fontsize=20)
ax2.text(0.22, 0.07, f'$w= {width3[-1] :.1f}$', fontsize=20)
ax1.text(0.55, -0.43, '$N = 12$', fontsize=20)

# Labels and limits
ax2.set_xlabel('$R$', fontsize=fontsize, labelpad=-15)
# ax2.xaxis.label.set_position((0.6, 0.1))
ax2.set_ylabel('$\overline{\\nu}$', fontsize=fontsize)
ax2.set_ylim([-1, 0.22])
ax2.set_xlim([cutoff_sequence[0], 1])

# Tick params
minorsy = [-0.75, -0.25]
majorsy = [-1, -0.5, 0]
majorsx = [0.5, 1]
minorsx = [0.25, 0.75]
xlabels = [f'{0.5 * Nx[0] * 0.5 :.1f}', f'{0.5 * Nx[0] :.1f}']
ax2.set(xticks=majorsx, xticklabels=xlabels)
ax2.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
ax2.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
ax2.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
ax2.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))
ax2.tick_params(which='major', width=0.75, labelsize=fontsize)
ax2.tick_params(which='major', length=6, labelsize=fontsize)
ax2.tick_params(which='minor', width=0.75, labelsize=fontsize)
ax2.tick_params(which='minor', length=3, labelsize=fontsize)


fig1.savefig('fig1-marker.pdf', format='pdf')
plt.show()

