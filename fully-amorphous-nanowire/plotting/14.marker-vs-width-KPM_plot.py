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
import matplotlib.ticker as ticker
import seaborn

# Modules
from modules.functions import *
from modules.colorbar_marker import *

#%% Loading data
file_list = ['data-cluster-marker-KPM-L=150.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-cluster')

# Simulation data
N            = data_dict[file_list[0]]['Simulation']['N']
num_vecs     = data_dict[file_list[0]]['Simulation']['num_vecs']
num_moments  = data_dict[file_list[0]]['Simulation']['num_moments']
width        = data_dict[file_list[0]]['Simulation']['width']
avg_marker   = data_dict[file_list[0]]['Simulation']['avg_marker']
std_marker   = data_dict[file_list[0]]['Simulation']['std_marker']
med_marker   = data_dict[file_list[0]]['Simulation']['med_marker']
mode_marker   = data_dict[file_list[0]]['Simulation']['mode_marker']
marker       = data_dict[file_list[0]]['Simulation']['marker']
Nsamples     = data_dict[file_list[0]]['Simulation']['Nsamples']
prob_dist_10    = data_dict[file_list[0]]['prob-dist-0']
prob_dist_12    = data_dict[file_list[0]]['prob-dist-1']
prob_dist_14    = data_dict[file_list[0]]['prob-dist-2']
prob_dist_16    = data_dict[file_list[0]]['prob-dist-3']
prob_dist_18    = data_dict[file_list[0]]['prob-dist-4']


# Parameters
Nz           = data_dict[file_list[0]]['Parameters']['Nz']
cutoff       = data_dict[file_list[0]]['Parameters']['cutoff']
t            = data_dict[file_list[0]]['Parameters']['t']
eps          = data_dict[file_list[0]]['Parameters']['eps']
lamb         = data_dict[file_list[0]]['Parameters']['lamb']
lamb_z       = data_dict[file_list[0]]['Parameters']['lamb_z']

error_bar_up    = avg_marker + 0.5 * std_marker
error_bar_down  = avg_marker - 0.5 * std_marker

#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['dodgerblue', 'limegreen', 'm', 'r', 'orange']
fontsize = 20
palette = seaborn.color_palette(palette='magma_r', n_colors=len(N))
# palette2 = seaborn.color_palette(palette='viridis_r', n_colors=len(width3))


# Figure 1
fig1 = plt.figure(figsize=(8, 6))
gs = GridSpec(1, 1, figure=fig1, wspace=0.1)
ax1 = fig1.add_subplot(gs[0, 0])

# Plot
for i in range(0, len(N)-1):
    ax1.plot(width, avg_marker[i, :], marker='o', linestyle='solid', color=palette[i], label=f'${N[::-1][i]}$')
    ax1.fill_between(width, error_bar_down[i, :], error_bar_up[i, :], color=palette[i], alpha=0.3)

# Legend and text
lgnd = ax1.legend(loc='upper left', ncol=1, frameon=False, fontsize=fontsize, handlelength=1, columnspacing=0.5, labelspacing=0.2, bbox_to_anchor=(0, 0.97))
ax1.text(0.09, 0.04, '$\\underline{N}$', fontsize=fontsize)
ax1.text(0.2, -0.0, f'$N_s = {Nsamples}$', fontsize=fontsize)
ax1.text(0.5, -0.6, f'$R< {0.4 * 0.5}N_i$', fontsize=fontsize)

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



fig2 = plt.figure(figsize=(8, 8))
gs = GridSpec(3, 1, figure=fig2, wspace=0.3, hspace=0.5)
ax1 = fig2.add_subplot(gs[0, 0])
ax2 = fig2.add_subplot(gs[1, 0])
ax3 = fig2.add_subplot(gs[2, 0])
ax1_inset = ax1.inset_axes([0.75, 0.35, 0.2, 0.6], )
ax2_inset = ax2.inset_axes([0.75, 0.35, 0.2, 0.6], )
ax3_inset = ax3.inset_axes([0.75, 0.35, 0.2, 0.6], )

ax1.plot(prob_dist_18['bins_width_2'], prob_dist_18['prob_dist_width_2'],   marker='o', linestyle='solid',   color=palette[-2], label=f'$N=18$')
ax2.plot(prob_dist_18['bins_width_9' ], prob_dist_18['prob_dist_width_9' ], marker='o', linestyle='solid', color=palette[-2], label=f'$N=18$')
ax3.plot(prob_dist_18['bins_width_16'], prob_dist_18['prob_dist_width_16'], marker='o', linestyle='solid', color=palette[-2], label=f'$N=18$')

ax1.plot(prob_dist_16['bins_width_2'],  prob_dist_16['prob_dist_width_2'], marker='o', linestyle='solid',  color=palette[-3],  label=f'$N=16$')
ax2.plot(prob_dist_16['bins_width_9' ], prob_dist_16['prob_dist_width_9' ], marker='o', linestyle='solid', color=palette[-3],  label=f'$N=16$')
ax3.plot(prob_dist_16['bins_width_16'], prob_dist_16['prob_dist_width_16'], marker='o', linestyle='solid', color=palette[-3],  label=f'$N=16$')

ax1.plot(prob_dist_14['bins_width_2'],  prob_dist_14['prob_dist_width_2'], marker='o', linestyle='solid',  color=palette[-4],  label=f'$N=14$')
ax2.plot(prob_dist_14['bins_width_9' ], prob_dist_14['prob_dist_width_9' ], marker='o', linestyle='solid', color=palette[-4],  label=f'$N=14$')
ax3.plot(prob_dist_14['bins_width_16'], prob_dist_14['prob_dist_width_16'], marker='o', linestyle='solid', color=palette[-4],  label=f'$N=14$')

ax1.plot(prob_dist_12['bins_width_2'],  prob_dist_12['prob_dist_width_2'], marker='o', linestyle='solid',  color=palette[-5],  label=f'$N=12$')
ax2.plot(prob_dist_12['bins_width_9' ], prob_dist_12['prob_dist_width_9' ], marker='o', linestyle='solid', color=palette[-5],  label=f'$N=12$')
ax3.plot(prob_dist_12['bins_width_16'], prob_dist_12['prob_dist_width_16'], marker='o', linestyle='solid', color=palette[-5],  label=f'$N=12$')

ax1.plot(prob_dist_10['bins_width_2'],  prob_dist_10['prob_dist_width_2'], marker='o', linestyle='solid',  color=palette[-6],  label=f'$N=10$')
ax2.plot(prob_dist_10['bins_width_9' ], prob_dist_10['prob_dist_width_9' ], marker='o', linestyle='solid', color=palette[-6],  label=f'$N=10$')
ax3.plot(prob_dist_10['bins_width_16'], prob_dist_10['prob_dist_width_16'], marker='o', linestyle='solid', color=palette[-6],  label=f'$N=10$')


print(N[1:][::-1])
ax1_inset.plot(N[1:][::-1], mode_marker[:, 2][:-1], marker='o', linestyle='solid', color='royalblue')
ax2_inset.plot(N[1:][::-1], mode_marker[:, 9][:-1], marker='o', linestyle='solid', color='royalblue')
ax3_inset.plot(N[1:][::-1], mode_marker[:, 16][:-1], marker='o', linestyle='solid', color='royalblue')
ax1_inset.set_ylim([-1, 0])
ax2_inset.set_ylim([-1, 0])
ax3_inset.set_ylim([-1, 0])
ax1_inset.set_xlabel('$N$',          fontsize=fontsize - 5)
ax1_inset.set_ylabel('mode$(\\nu)$', fontsize=fontsize - 5)
ax2_inset.set_xlabel('$N$',          fontsize=fontsize - 5)
ax2_inset.set_ylabel('mode$(\\nu)$', fontsize=fontsize - 5)
ax3_inset.set_xlabel('$N$',          fontsize=fontsize - 5)
ax3_inset.set_ylabel('mode$(\\nu)$', fontsize=fontsize - 5)

ax1_inset.tick_params(which='major', width=0.75, labelsize=fontsize - 5)
ax1_inset.tick_params(which='major', length=6,   labelsize=fontsize - 5)
ax1_inset.tick_params(which='minor', width=0.75, labelsize=fontsize - 5)
ax1_inset.tick_params(which='minor', length=3,   labelsize=fontsize - 5)
ax2_inset.tick_params(which='major', width=0.75, labelsize=fontsize - 5)
ax2_inset.tick_params(which='major', length=6,   labelsize=fontsize - 5)
ax2_inset.tick_params(which='minor', width=0.75, labelsize=fontsize - 5)
ax2_inset.tick_params(which='minor', length=3,   labelsize=fontsize - 5)
ax3_inset.tick_params(which='major', width=0.75, labelsize=fontsize - 5)
ax3_inset.tick_params(which='major', length=6,   labelsize=fontsize - 5)
ax3_inset.tick_params(which='minor', width=0.75, labelsize=fontsize - 5)
ax3_inset.tick_params(which='minor', length=3,   labelsize=fontsize - 5)


ax1.legend(loc='upper left')
ax2.legend(loc='upper left')
ax3.legend(loc='upper left')
ax1.set_title(f'$w= {width[2] :.2f}$', fontsize=fontsize)
ax2.set_title(f'$w= {width[9] :.2f}$', fontsize=fontsize)
ax3.set_title(f'$w= {width[16] :.2f}$', fontsize=fontsize)

ax1.set_xlim([-1.5, 1.5])
ax1.set_ylim([0, 0.3])
ax2.set_xlim([-1.5, 1.5])
ax2.set_ylim([0, 0.3])
ax3.set_xlim([-1.5, 1.5])
ax3.set_ylim([0, 0.3])
ax3.set_xlabel('$\\nu$', fontsize=fontsize)
ax1.set_ylabel('$P(\\nu)$', fontsize=fontsize)
ax2.set_ylabel('$P(\\nu)$', fontsize=fontsize)
ax3.set_ylabel('$P(\\nu)$', fontsize=fontsize)
ax1.tick_params(which='major', width=0.75, labelsize=fontsize)
ax1.tick_params(which='major', length=6, labelsize=fontsize)
ax1.tick_params(which='minor', width=0.75, labelsize=fontsize)
ax1.tick_params(which='minor', length=3, labelsize=fontsize)
ax2.tick_params(which='major', width=0.75, labelsize=fontsize)
ax2.tick_params(which='major', length=6, labelsize=fontsize)
ax2.tick_params(which='minor', width=0.75, labelsize=fontsize)
ax2.tick_params(which='minor', length=3, labelsize=fontsize)
ax3.tick_params(which='major', width=0.75, labelsize=fontsize)
ax3.tick_params(which='major', length=6, labelsize=fontsize)
ax3.tick_params(which='minor', width=0.75, labelsize=fontsize)
ax3.tick_params(which='minor', length=3, labelsize=fontsize)

fig1.savefig(f'../figures/marker-vs-width.pdf', format='pdf')
fig2.savefig(f'../figures/probs.pdf', format='pdf')


plt.show()