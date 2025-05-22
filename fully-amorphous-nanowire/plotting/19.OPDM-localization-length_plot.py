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

#%% Loading data
file_list = ['Exp7.h5', 'Exp4.h5', 'Exp5.h5', 'Exp6.h5'] #, 'exp-23.h5', 'exp-24.h5', 'exp-25.h5', 'exp-26.h5', 'exp-27.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-OPDM')

# Parameters
Nx           = data_dict[file_list[0]]['Parameters']['Nx']
Ny           = data_dict[file_list[0]]['Parameters']['Nx']
Nz           = data_dict[file_list[0]]['Parameters']['Nz']
r            = data_dict[file_list[0]]['Parameters']['r ']
t            = data_dict[file_list[0]]['Parameters']['t ']
eps          = data_dict[file_list[0]]['Parameters']['eps']
lamb         = data_dict[file_list[0]]['Parameters']['lamb']
lamb_z       = data_dict[file_list[0]]['Parameters']['lamb_z']



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







font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange']
marker_list=['o', 's', 'd', 'p', '*', 'h', '>', '<', 'X']
markersize = 5
fontsize=20
palette = seaborn.color_palette(palette='viridis_r', n_colors=200)
palette = [palette[0], palette[50], palette[100], palette[150]]


# Figure 2
fig1 = plt.figure(figsize=(8, 8))
gs = GridSpec(1, 1, figure=fig1, wspace=0.2, hspace=0.3)
ax1 = fig1.add_subplot(gs[0, 0])


ax1.plot(rad2, OPDM_sum2, marker='o', color=palette[0], label=f'$w= {width2}$')
ax1.plot(rad3, OPDM_sum3, marker='o', color=palette[2], label=f'$w= {width3}$')
ax1.plot(rad4, OPDM_sum4, marker='o', color=palette[1], label=f'$w= {width4}$')
ax1.plot(rad1, OPDM_sum1, marker='o', color=palette[3], label=f'$w= {width1}$')
ax1.set_xlabel('$\\vert x - y \\vert$', fontsize=fontsize)
ax1.set_ylabel('$\sum_{r, \\alpha} \\vert \\rho(x, y) \\vert^{2}$', fontsize=fontsize)
# ax1.set_ylim(-1.5, 1)
ax1.set_xlim(min(avg_radius1), max(avg_radius1))
ax1.tick_params(which='major', width=0.75, labelsize=fontsize)
ax1.tick_params(which='major', length=6, labelsize=fontsize)
ax1.legend()
ax1.set_yscale('log')

fig1.savefig(f'../figures/opdm-localisation.pdf', format='pdf')
plt.show()