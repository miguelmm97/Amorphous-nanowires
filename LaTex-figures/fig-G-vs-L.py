#%% Modules and setup

# Plotting
import numpy as np
from scipy.optimize import curve_fit
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
file_list = ['draft-fig8.h5', 'draft-fig-8-perfect-mode.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-latex-figures')

# Plot 1
G       = data_dict[file_list[0]]['Simulation']['G_array']
Nz      = data_dict[file_list[0]]['Simulation']['Nz']
width   = data_dict[file_list[0]]['Simulation']['width']
flux    = data_dict[file_list[0]]['Simulation']['flux']
Nx      = data_dict[file_list[0]]['Parameters']['Nx']

# Plot 2
flux2   = data_dict[file_list[1]]['Simulation']['flux']
width2  = data_dict[file_list[1]]['Simulation']['width']
Nz2      = data_dict[file_list[1]]['Simulation']['Nz']
G2      = data_dict[file_list[1]]['Simulation']['G_array']


# Exponential fit
def funcfit(L, C, xi): return C * np.exp(- L / xi)
x = Nz[::-1] - 50
#%% Isolation of different resonances
bool1 = flux > 0.17
bool2 = flux < 0.4
bool = bool1 * bool2
peaks1 = np.max(G[0, :, bool, 0], axis=0)
idx1 = [np.where(G[0, i, :, 0] == peaks1[i])[0][0] for i in range(len(Nz))]

fit1, covariance1 = curve_fit(funcfit, x, peaks1[::-1], p0=[1.5, 50])
exp1 = funcfit(x, fit1[0], fit1[1])

bool1 = flux < 2
bool2 = flux > 1.4
bool = bool1 * bool2
peaks2 = np.max(G[0, :, bool, 0], axis=0)
idx2 = [np.where(G[0, i, :, 0] == peaks2[i])[0][0] for i in range(len(Nz))][:-1]
peaks2 = peaks2[:-1]
fit2, covariance2 = curve_fit(funcfit, x[1:], peaks2[::-1], p0=[1.5, 50])
exp2 = funcfit(x, fit2[0], fit2[1])

bool1 = flux < 3.52
bool2 = flux > 3.2
bool = bool1 * bool2
peaks3 = np.max(G[0, :, bool, 0], axis=0)
idx3 = [np.where(G[0, i, :, 0] == peaks3[i])[0][0] for i in range(len(Nz))]
fit3, covariance3 = curve_fit(funcfit, x, peaks3[::-1], p0=[1.5, 50])
exp3 = funcfit(x, fit3[0], fit3[1])

#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['dodgerblue', 'limegreen', 'm', 'r', 'orange']
fontsize = 20
palette = seaborn.color_palette(palette='magma', n_colors=len(Nz))
palette1 = seaborn.color_palette(palette='magma', n_colors=len(Nz2))
palette2 = seaborn.color_palette(palette='magma', n_colors=len(width))


# Figure 1
fig1 = plt.figure(figsize=(8, 6))
gs = GridSpec(2, 4, figure=fig1, wspace=0.25, hspace=0.3)
ax1 = fig1.add_subplot(gs[0, 0:2])
ax2 = fig1.add_subplot(gs[1, :])
# ax3 = ax2.inset_axes([0.1, 0.65, 0.32, 0.32], )
ax3 = fig1.add_subplot(gs[0, 2:])


# Plot 1
ax1.plot(flux2, 1 * np.ones(flux2.shape), '--', color='Black', alpha=0.2)
for j in range(G2.shape[1] - 1, -1, -1):
    label = f'$L= {Nz2[j]}$'
    ax1.plot(flux2, G2[2, j, :, 0], color=palette1[j], linestyle='solid', label=label)
ax1.text(0.1, 1.05, f'$w={width2[2] :.2f}$', fontsize=fontsize)
# ax1.text(0.1, 1.15, f'$N_x=N_y={Nx}$', fontsize=fontsize-3)

ax1.set_xlim(flux2[0], flux2[-1])
ax1.set_ylim(0, 1.25)
ax1.set(yticks=[0, 0.5, 1])
ax1.tick_params(which='major', width=0.75, labelsize=fontsize)
ax1.tick_params(which='major', length=6, labelsize=fontsize)
ax1.set_xlabel("$\phi$", fontsize=fontsize, labelpad=-10)
ax1.set_ylabel("$G[2e^2/h]$", fontsize=fontsize)
ax1.tick_params(which='major', width=0.75, labelsize=fontsize)
ax1.tick_params(which='major', length=6, labelsize=fontsize )


# Plot 2
ax2.plot(flux, 1 * np.ones(flux.shape), '--', color='Black', alpha=0.2)
for i in range(G.shape[1]):
    label = f'${Nz[i]}$'
    ax2.plot(flux, G[0, i, :, 0], color=palette[i], linestyle='solid', label=label)
ax2.plot(flux[idx1], peaks1, marker='o', color=color_list[0], linestyle='None', alpha=0.5)
ax2.plot(flux[idx2], peaks2, marker='s', color=color_list[1], linestyle='None', alpha=0.5)
ax2.plot(flux[idx3], peaks3, marker='^', color=color_list[2], linestyle='None', alpha=0.5, markersize=7)

ax2.legend(ncol=4, frameon=False, fontsize=fontsize, columnspacing=0.3, handlelength=1, labelspacing=0.2, bbox_to_anchor=(0.45, 0.43))
ax2.text(0.37, 1.7, '$\\underline{L}$', fontsize=fontsize)
ax2.text(4, 1.8, f'$w={width[0] :.2f}$', fontsize=fontsize)

ax2.set_xlabel("$\phi$", fontsize=fontsize)
ax2.set_ylabel("$G[2e^2/h]$", fontsize=fontsize)
ax2.set_xlim(flux[0], flux[-1])
ax2.set_ylim(0, np.max(G) + 0.2)
ax2.set(yticks=[0, 0.5, 1, 1.5, 2])
ax2.tick_params(which='major', width=0.75, labelsize=fontsize)
ax2.tick_params(which='major', length=6, labelsize=fontsize)


# PLot 3
ax3.plot(Nz[::-1], 1 * np.ones(Nz[::-1].shape), '--', color='Black', alpha=0.2)
ax3.plot(Nz[::-1], peaks1[::-1], marker='o', color=color_list[0], linestyle='None',  alpha=0.5)
ax3.plot(Nz[::-1], exp1, color=color_list[0], linestyle='dashed')
ax3.plot(Nz[::-1][1:], peaks2[::-1], marker='s', color=color_list[1], linestyle='None',  alpha=0.5)
ax3.plot(Nz[::-1][1:], exp2[1:], color=color_list[1], linestyle='dashed')
ax3.plot(Nz[::-1], peaks3[::-1], marker='*', color=color_list[2], linestyle='None',  alpha=0.5)
ax3.plot(Nz[::-1], exp3, color=color_list[2], linestyle='dashed')

ax3.set_xlabel("$N_z$", fontsize=fontsize, labelpad=-10)
ax3.set_xlim(60, 180)
ax3.set_ylim(0, 1.25)
ax3.tick_params(which='major', width=0.75, labelsize=fontsize)
ax3.tick_params(which='major', length=6, labelsize=fontsize)
ax3.set(yticks=[0, 0.5, 1], yticklabels=[])
ax3.set(xticks=[60, 100, 140, 180])

fig1.savefig('fig-G-vs-L.pdf', format='pdf')
plt.show()