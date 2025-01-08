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


#%% Loading data
file_list = ['Exp2.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-marker-vs-cross-section')

# Parameters
Nx           = data_dict[file_list[0]]['Parameters']['Nx']
Ny           = data_dict[file_list[0]]['Parameters']['Ny']
Nz           = data_dict[file_list[0]]['Parameters']['Nz']
r            = data_dict[file_list[0]]['Parameters']['r ']
t            = data_dict[file_list[0]]['Parameters']['t ']
eps          = data_dict[file_list[0]]['Parameters']['eps']
lamb         = data_dict[file_list[0]]['Parameters']['lamb']
lamb_z       = data_dict[file_list[0]]['Parameters']['lamb_z']
flux         = data_dict[file_list[0]]['Parameters']['flux']
# width        = data_dict[file_list[0]]['Parameters']['width']
width = 0.1

# Simulation data
X            = data_dict[file_list[0]]['Simulation']['x']
Y            = data_dict[file_list[0]]['Simulation']['y']
Z            = data_dict[file_list[0]]['Simulation']['z']
local_marker = data_dict[file_list[0]]['Simulation']['local_marker']


# Calculation of the marker for the different wires
pos = {}
marker = {}
avg_marker = np.zeros((len(width), len(Nx)))
def bulk(x, y, z, local_marker, cutoff, nx, ny, include_last_layer=True):
    x_pos, y_pos = x - 0.5 * nx, y - 0.5 * ny
    cond1 = np.abs(x_pos) < cutoff
    cond2 = np.abs(y_pos) < cutoff
    cond = cond1 * cond2
    if not include_last_layer:
        cond3 = 5 < z
        cond4 = z < Nz - 6
        cond = cond1 * cond2 * cond3 * cond4
    return x[cond], y[cond], z[cond], local_marker[cond]

for i in range(len(width)):
    pos[i] = {}
    marker[i] = {}
    for j, n in enumerate(Nx):
        cutoff_bulk = 0.6 * 0.5 * n
        Nsites = int(n * n * Nz)
        x, y, z, marker_xy = bulk(X[i, j, :Nsites], Y[i, j, :Nsites], Z[i, j, :Nsites], local_marker[i, j, :Nsites], cutoff_bulk,
                                  n, n, include_last_layer=False)
        pos[i][j] = np.array([[x], [y], [z]])
        marker[i][j] = marker_xy
        avg_marker[i, j] = np.mean(marker[i][j])


#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['dodgerblue', 'limegreen', 'm', 'r', 'orange']
fontsize = 15
palette = seaborn.color_palette(palette='magma', n_colors=len(Nx))


# Figure 1
fig1 = plt.figure()
gs = GridSpec(1, 1, figure=fig1, wspace=0.0, hspace=0.02)
ax1 = fig1.add_subplot(gs[0, 0])

for i in range(len(Nx)):
    ax1.plot(width, avg_marker[:, i], marker='o', linestyle='solid', color=palette[i])

ax1.set_xlabel('$w$', fontsize=fontsize)
ax1.set_ylabel('$\overline{\\nu}$', fontsize=fontsize)
ax1.set_ylim([-1, 0.1])
plt.show()



