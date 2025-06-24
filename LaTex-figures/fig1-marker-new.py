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
import matplotlib.ticker as ticker


# Modules
from modules.functions import *
from modules.colorbar_marker import *


#%% Logging setup
loger_main = logging.getLogger('main')
loger_main.setLevel(logging.INFO)

stream_handler = colorlog.StreamHandler()
formatter = ColoredFormatter(
    '%(black)s%(asctime) -5s| %(blue)s%(name) -10s %(black)s| %(cyan)s %(funcName) '
    '-40s %(black)s|''%(log_color)s%(levelname) -10s | %(message)s',
    datefmt=None,
    reset=True,
    log_colors={
        'TRACE': 'black',
        'DEBUG': 'purple',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)

stream_handler.setFormatter(formatter)
loger_main.addHandler(stream_handler)

#%% Loading data marker
file_list = ['data-cluster-marker-KPM-L=150.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/cluster-simulations/data-cluster-marker-vs-width')

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

# Parameters
L            = data_dict[file_list[0]]['Parameters']['Nz']
cutoff       = data_dict[file_list[0]]['Parameters']['cutoff']
t            = data_dict[file_list[0]]['Parameters']['t']
eps          = data_dict[file_list[0]]['Parameters']['eps']
lamb         = data_dict[file_list[0]]['Parameters']['lamb']
lamb_z       = data_dict[file_list[0]]['Parameters']['lamb_z']

error_bar_up    = avg_marker + 0.5 * std_marker
error_bar_down  = avg_marker - 0.5 * std_marker


#%% Loading data OPDM
# file_list = ['Exp10.h5', 'Exp11.h5', 'Exp12.h5', 'Exp13.h5', 'Exp14.h5', 'Exp15.h5']
# data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/local-simulations/data-OPDM')

# # Load data
# Nx           = data_dict[file_list[0]]['Parameters']['Nx']
# Ny           = data_dict[file_list[0]]['Parameters']['Nx']
# Nz           = data_dict[file_list[0]]['Parameters']['Nz']
# OPDM1        = data_dict[file_list[0]]['Simulation']['OPDM_r']
# OPDM2        = data_dict[file_list[1]]['Simulation']['OPDM_r']
# OPDM3        = data_dict[file_list[2]]['Simulation']['OPDM_r']
# OPDM4        = data_dict[file_list[3]]['Simulation']['OPDM_r']
# OPDM5        = data_dict[file_list[4]]['Simulation']['OPDM_r']
# OPDM6        = data_dict[file_list[5]]['Simulation']['OPDM_r']
# radius1      = np.real(data_dict[file_list[0]]['Simulation']['r'])
# radius2      = np.real(data_dict[file_list[1]]['Simulation']['r'])
# radius3      = np.real(data_dict[file_list[2]]['Simulation']['r'])
# radius4      = np.real(data_dict[file_list[3]]['Simulation']['r'])
# radius5      = np.real(data_dict[file_list[4]]['Simulation']['r'])
# radius6      = np.real(data_dict[file_list[5]]['Simulation']['r'])
# width1       = data_dict[file_list[0]]['Simulation']['width']
# width2       = data_dict[file_list[1]]['Simulation']['width']
# width3       = data_dict[file_list[2]]['Simulation']['width']
# width4       = data_dict[file_list[3]]['Simulation']['width']
# width5       = data_dict[file_list[4]]['Simulation']['width']
# width6       = data_dict[file_list[5]]['Simulation']['width']


# # Histograms of the data
# r_bins1, binned_opdm1 = bin_my_samples(radius1, OPDM1, num_bins=15)
# r_bins2, binned_opdm2 = bin_my_samples(radius2, OPDM2, num_bins=15)
# r_bins3, binned_opdm3 = bin_my_samples(radius3, OPDM3, num_bins=15)
# r_bins4, binned_opdm4 = bin_my_samples(radius4, OPDM4, num_bins=15)
# r_bins5, binned_opdm5 = bin_my_samples(radius5, OPDM5, num_bins=15)
# r_bins6, binned_opdm6 = bin_my_samples(radius6, OPDM6, num_bins=15)
# OPDM_sum1 = np.real(np.array([np.mean(binned_opdm1[i]) for i in range(len(binned_opdm1))if len(binned_opdm1[i]) != 0]))
# OPDM_sum2 = np.real(np.array([np.mean(binned_opdm2[i]) for i in range(len(binned_opdm2))if len(binned_opdm2[i]) != 0]))
# OPDM_sum3 = np.real(np.array([np.mean(binned_opdm3[i]) for i in range(len(binned_opdm3))if len(binned_opdm3[i]) != 0]))
# OPDM_sum4 = np.real(np.array([np.mean(binned_opdm4[i]) for i in range(len(binned_opdm4))if len(binned_opdm4[i]) != 0]))
# OPDM_sum5 = np.real(np.array([np.mean(binned_opdm5[i]) for i in range(len(binned_opdm5))if len(binned_opdm5[i]) != 0]))
# OPDM_sum6 = np.real(np.array([np.mean(binned_opdm6[i]) for i in range(len(binned_opdm6))if len(binned_opdm6[i]) != 0]))
# rad1 = np.array([r_bins1[i] for i in range(len(binned_opdm1)) if len(binned_opdm1[i]) != 0])
# rad2 = np.array([r_bins2[i] for i in range(len(binned_opdm2)) if len(binned_opdm2[i]) != 0])
# rad3 = np.array([r_bins3[i] for i in range(len(binned_opdm3)) if len(binned_opdm3[i]) != 0])
# rad4 = np.array([r_bins4[i] for i in range(len(binned_opdm4)) if len(binned_opdm4[i]) != 0])
# rad5 = np.array([r_bins5[i] for i in range(len(binned_opdm5)) if len(binned_opdm5[i]) != 0])
# rad6 = np.array([r_bins6[i] for i in range(len(binned_opdm6)) if len(binned_opdm6[i]) != 0])

#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['dodgerblue', 'limegreen', 'm', 'r', 'orange']
fontsize = 20
palette = seaborn.color_palette(palette='magma_r', n_colors=len(N))
palette2 = seaborn.color_palette(palette='viridis_r', n_colors=200)
palette2 = [palette2[0], palette2[40], palette2[80], palette2[120], palette2[160], palette2[-1]]


# Figure 1
fig1 = plt.figure(figsize=(8, 6))
gs = GridSpec(1, 1, figure=fig1, wspace=0.1)
ax1 = fig1.add_subplot(gs[0, 0])
# ax1_inset =  ax1.inset_axes([0.67, 0.13, 0.3, 0.3], )

# Plot
ax1.plot(np.linspace(0, np.max(width), 10), np.zeros((10, )), '--', color='Black', alpha=0.2)
for i in range(0, len(N) - 1):
    ax1.plot(width, avg_marker[i, :], marker='o', linestyle='solid', color=palette[i], label=f'${N[len(N)-i -1]}$')
    ax1.fill_between(width, error_bar_down[i, :], error_bar_up[i, :], color=palette[i], alpha=0.3)

# Legend and text
lgnd = ax1.legend(loc='upper left', ncol=1, frameon=False, fontsize=fontsize, handlelength=1, columnspacing=0.5, labelspacing=0.2, bbox_to_anchor=(0.05, 0.77))
ax1.text(0.135, -0.17, '$\\underline{N}$', fontsize=fontsize)
ax1.text(0.5, -0.55, f'$N_s = {100}$', fontsize=fontsize)
ax1.text(0.5, -0.65, f'$\\vert x_i \\vert < 0.2 N_i$', fontsize=fontsize)
ax1.text(0.5, -0.75, f'$L = 150$', fontsize=fontsize)

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



# ax1_inset.plot(rad1, OPDM_sum1, marker='o', color=palette2[0], label=f'$0$', markersize=4)
# ax1_inset.plot(rad2, OPDM_sum2, marker='o', color=palette2[1], label=f'${width2}$', markersize=4)
# ax1_inset.plot(rad3, OPDM_sum3, marker='o', color=palette2[2], label=f'${width3}$', markersize=4)
# ax1_inset.plot(rad4, OPDM_sum4, marker='o', color=palette2[3], label=f'${width4}$', markersize=4)
# ax1_inset.plot(rad5, OPDM_sum5, marker='o', color=palette2[4], label=f'${width5}$', markersize=4)
# ax1_inset.plot(rad6, OPDM_sum6, marker='o', color=palette2[5], label=f'${width6}$', markersize=4)
# ax1_inset.set_xlabel('$\\vert x - y \\vert$', fontsize=fontsize-5)
# ax1_inset.set_ylabel('$\sum_{r, \\alpha} \\vert \\rho(x, y) \\vert^{2}$', fontsize=fontsize-5)
# ax1_inset.set_ylim(1e-8, 1)
# ax1_inset.set_xlim(0, 10)
# ax1_inset.tick_params(which='major', width=0.75, labelsize=fontsize-5)
# ax1_inset.tick_params(which='major', length=6, labelsize=fontsize-5)
# ax1_inset.legend(loc='upper center', ncol=2, frameon=False, fontsize=fontsize-5, handlelength=1, columnspacing=0.5, labelspacing=0.2, bbox_to_anchor=(0.6, 1.45))
# ax1_inset.set_yscale('log')
# label = ax1_inset.xaxis.get_label()
# x, y = label.get_position()
# label.set_position((x, y + 0.5))
# ax1.text(0.54, -0.48, '$\\underline{w}$', fontsize=fontsize)

fig1.savefig('fig1-marker-new.pdf', format='pdf')
plt.show()
