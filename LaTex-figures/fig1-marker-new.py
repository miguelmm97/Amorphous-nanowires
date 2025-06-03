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
file_list = ['Exp7.h5', 'Exp4.h5', 'Exp5.h5', 'Exp6.h5'] #, 'exp-23.h5', 'exp-24.h5', 'exp-25.h5', 'exp-26.h5', 'exp-27.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/local-simulations/data-OPDM')

# Parameters
Nx           = data_dict[file_list[0]]['Parameters']['Nx']
Ny           = data_dict[file_list[0]]['Parameters']['Nx']
Nz           = data_dict[file_list[0]]['Parameters']['Nz']
x = data_dict[file_list[0]]['Simulation']['x']
y = data_dict[file_list[0]]['Simulation']['y']
z = data_dict[file_list[0]]['Simulation']['z']

OPDM1 = data_dict[file_list[0]]['Simulation']['OPDM_r']
radius1 = np.real(data_dict[file_list[0]]['Simulation']['r_3d'])
width1        = data_dict[file_list[0]]['Simulation']['width']

OPDM2 = data_dict[file_list[1]]['Simulation']['OPDM_r']
radius2 = np.real(data_dict[file_list[1]]['Simulation']['r_3d'])
width2        = data_dict[file_list[1]]['Simulation']['width']

OPDM3 = data_dict[file_list[2]]['Simulation']['OPDM_r']
radius3 = np.real(data_dict[file_list[2]]['Simulation']['r_3d'])
width3        = data_dict[file_list[2]]['Simulation']['width']

OPDM4 = data_dict[file_list[3]]['Simulation']['OPDM_r']
radius4 = np.real(data_dict[file_list[3]]['Simulation']['r_3d'])
width4        = data_dict[file_list[3]]['Simulation']['width']

num_bins = 15
r_min, r_max = radius1.min(), radius1.max()
bin_edges = np.linspace(r_min, r_max, num_bins + 1)
bin_indices = np.digitize(radius1, bin_edges) - 1
binned_samples1 = [[] for _ in range(num_bins)]
avg_radius1 = 0.5 * (bin_edges[:-1] + bin_edges[1:])
for idx, bin_idx in enumerate(bin_indices):
    if 0 <= bin_idx < num_bins:
        binned_samples1[bin_idx].append(OPDM1[idx])
binned_samples1 = [np.array(bin) for bin in binned_samples1]


num_bins = 15
r_min, r_max = radius2.min(), radius2.max()
bin_edges = np.linspace(r_min, r_max, num_bins + 1)
bin_indices = np.digitize(radius2, bin_edges) - 1
binned_samples2 = [[] for _ in range(num_bins)]
avg_radius2 = 0.5 * (bin_edges[:-1] + bin_edges[1:])
for idx, bin_idx in enumerate(bin_indices):
    if 0 <= bin_idx < num_bins:
        binned_samples2[bin_idx].append(OPDM2[idx])
binned_samples2 = [np.array(bin) for bin in binned_samples2]


num_bins = 15
r_min, r_max = radius3.min(), radius3.max()
bin_edges = np.linspace(r_min, r_max, num_bins + 1)
bin_indices = np.digitize(radius3, bin_edges) - 1
binned_samples3 = [[] for _ in range(num_bins)]
avg_radius3 = 0.5 * (bin_edges[:-1] + bin_edges[1:])
for idx, bin_idx in enumerate(bin_indices):
    if 0 <= bin_idx < num_bins:
        binned_samples3[bin_idx].append(OPDM3[idx])
binned_samples3 = [np.array(bin) for bin in binned_samples3]

num_bins = 15
r_min, r_max = radius4.min(), radius4.max()
bin_edges = np.linspace(r_min, r_max, num_bins + 1)
bin_indices = np.digitize(radius4, bin_edges) - 1
binned_samples4 = [[] for _ in range(num_bins)]
avg_radius4 = 0.5 * (bin_edges[:-1] + bin_edges[1:])
for idx, bin_idx in enumerate(bin_indices):
    if 0 <= bin_idx < num_bins:
        binned_samples4[bin_idx].append(OPDM4[idx])
binned_samples4 = [np.array(bin) for bin in binned_samples4]


# Statistics of the distribution: Average and standard deviation
OPDM_sum1 = np.array([np.sum(binned_samples1[i]) for i in range(len(binned_samples1))if len(binned_samples1[i]) != 0])
OPDM_sum2 = np.array([np.sum(binned_samples2[i]) for i in range(len(binned_samples2))if len(binned_samples2[i]) != 0])
OPDM_sum3 = np.array([np.sum(binned_samples3[i]) for i in range(len(binned_samples3))if len(binned_samples3[i]) != 0])
OPDM_sum4 = np.array([np.sum(binned_samples4[i]) for i in range(len(binned_samples4))if len(binned_samples4[i]) != 0])
rad1 = np.array([avg_radius1[i] for i in range(len(binned_samples1))if len(binned_samples1[i]) != 0])
rad2 = np.array([avg_radius2[i] for i in range(len(binned_samples2))if len(binned_samples2[i]) != 0])
rad3 = np.array([avg_radius3[i] for i in range(len(binned_samples3))if len(binned_samples3[i]) != 0])
rad4 = np.array([avg_radius4[i] for i in range(len(binned_samples4))if len(binned_samples4[i]) != 0])


#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['dodgerblue', 'limegreen', 'm', 'r', 'orange']
fontsize = 20
palette = seaborn.color_palette(palette='magma_r', n_colors=len(N))
palette2 = seaborn.color_palette(palette='viridis_r', n_colors=200)
palette2 = [palette2[0], palette2[50], palette2[100], palette2[150]]


# Figure 1
fig1 = plt.figure(figsize=(8, 6))
gs = GridSpec(1, 1, figure=fig1, wspace=0.1)
ax1 = fig1.add_subplot(gs[0, 0])
ax1_inset =  ax1.inset_axes([0.67, 0.13, 0.3, 0.3], )

# Plot
ax1.plot(np.linspace(0, np.max(width), 10), np.zeros((10, )), '--', color='Black', alpha=0.2)
for i in range(0, len(N) - 1):
    ax1.plot(width, avg_marker[i, :], marker='o', linestyle='solid', color=palette[i], label=f'${N[len(N)-i -1]}$')
    ax1.fill_between(width, error_bar_down[i, :], error_bar_up[i, :], color=palette[i], alpha=0.3)

# Legend and text
lgnd = ax1.legend(loc='upper left', ncol=2, frameon=False, fontsize=fontsize, handlelength=1, columnspacing=0.5, labelspacing=0.2, bbox_to_anchor=(0.05, 0.85))
ax1.text(0.135, -0.08, '$\\underline{N}$', fontsize=fontsize)
ax1.text(0.05, -0.45, f'$N_s = {200}$', fontsize=fontsize)
ax1.text(0.05, -0.55, f'$\\vert x_i \\vert < 0.2 N_i$', fontsize=fontsize)
ax1.text(0.05, -0.65, f'$L = 150$', fontsize=fontsize)

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



ax1_inset.plot(rad2, OPDM_sum2, marker='o', color=palette2[0], label=f'${width2}$')
ax1_inset.plot(rad4, OPDM_sum4, marker='o', color=palette2[1], label=f'${width4}$')
ax1_inset.plot(rad3, OPDM_sum3, marker='o', color=palette2[2], label=f'${width3}$')
ax1_inset.plot(rad1, OPDM_sum1, marker='o', color=palette2[3], label=f'${width1}$')
ax1_inset.set_xlabel('$\\vert x - y \\vert$', fontsize=fontsize-5)
ax1_inset.set_ylabel('$\sum_{r, \\alpha} \\vert \\rho(x, y) \\vert^{2}$', fontsize=fontsize-5)
# a_insetx1.set_ylim(-1.5, 1)
ax1_inset.set_xlim(min(avg_radius1), max(avg_radius1))
ax1_inset.tick_params(which='major', width=0.75, labelsize=fontsize-5)
ax1_inset.tick_params(which='major', length=6, labelsize=fontsize-5)
ax1_inset.legend(loc='upper center', ncol=2, frameon=False, fontsize=fontsize-5, handlelength=1, columnspacing=0.5, labelspacing=0.2, bbox_to_anchor=(0.6, 1.45))
ax1_inset.set_yscale('log')
label = ax1_inset.xaxis.get_label()
x, y = label.get_position()
label.set_position((x, y + 0.5))
ax1.text(0.54, -0.48, '$\\underline{w}$', fontsize=fontsize)

fig1.savefig('fig1-marker-new.pdf', format='pdf')
plt.show()
