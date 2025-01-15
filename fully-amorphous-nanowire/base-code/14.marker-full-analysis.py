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
loger_main.info('Loading data...')
file_list = ['Exp7.h5']
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
width        = data_dict[file_list[0]]['Parameters']['width']


# Simulation data
X            = data_dict[file_list[0]]['Simulation']['x']
Y            = data_dict[file_list[0]]['Simulation']['y']
Z            = data_dict[file_list[0]]['Simulation']['z']
local_marker = data_dict[file_list[0]]['Simulation']['local_marker']
loger_main.info('Data loaded successfully')

#%% Main
# Local marker for the different wires
pos = {}
marker = {}
avg_marker = np.zeros((len(width), len(Nx)))
def bulk(x, y, z, local_marker, cutoff_xy, cutoff_z, nx, ny, full_z=True, full_xy=True):

    # Coordinates
    x_pos, y_pos = x - 0.5 * nx, y - 0.5 * ny
    cond1 = np.abs(x_pos) < cutoff_xy
    cond2 = np.abs(y_pos) < cutoff_xy
    cond3 = (0.5 * Nz - cutoff_z) < z
    cond4 = (0.5 * Nz + cutoff_z) > z

    # Cutoff conditions
    if full_xy and full_z:
        return x, y, z, local_marker
    elif full_z:
        cond = cond1 * cond2
    else:
        cond = cond1 * cond2 * cond3 * cond4
    return x[cond], y[cond], z[cond], local_marker[cond]

loger_main.info('Calculating average marker...')
for i in range(len(width)):
    pos[i] = {}
    marker[i] = {}
    for j, n in enumerate(Nx):
        cutoff_bulk = 0.4 * 0.5 * n
        cutoff_z = cutoff_bulk
        Nsites = int(n * n * Nz)
        x, y, z, marker_xy = bulk(X[i, j, :Nsites], Y[i, j, :Nsites], Z[i, j, :Nsites], local_marker[i, j, :Nsites],
                                  cutoff_bulk, cutoff_z, n, n, full_z=False, full_xy=False)
        pos[i][j] = np.array([[x], [y], [z]])
        marker[i][j] = marker_xy
        avg_marker[i, j] = np.mean(marker[i][j])


# Local marker for different cuts of a particular wire
loger_main.info('Calculating example cuts')
idx_w = 4
idx_n = -1
Nsites = int(Nx[idx_n] * Nx[idx_n]* Nz)
c1 = 0.4
cutoff_xy1 = c1 * 0.5 * Nx[idx_n]
cutoff_z1 = cutoff_xy1
x_plot1, y_plot1, z_plot1, marker_plot1 = bulk(X[idx_w, idx_n, :Nsites], Y[idx_w, idx_n, :Nsites], Z[idx_w, idx_n, :Nsites],
                                           local_marker[idx_w, idx_n, :Nsites], cutoff_xy1, cutoff_z1, Nx[idx_n], Nx[idx_n],
                                           full_z=False, full_xy=False)
c2 = 0.6
cutoff_xy2 = c2 * 0.5 * Nx[idx_n]
cutoff_z2 = 0.8 * 0.5 * Nx[idx_n]
x_plot2, y_plot2, z_plot2, marker_plot2 = bulk(X[idx_w, idx_n, :Nsites], Y[idx_w, idx_n, :Nsites], Z[idx_w, idx_n, :Nsites],
                                           local_marker[idx_w, idx_n, :Nsites], cutoff_xy2, cutoff_z2, Nx[idx_n], Nx[idx_n],
                                           full_z=False, full_xy=False)

cutoff_xy3 = 0.5 * Nx[idx_n]
cutoff_z3 = 0.5 * Nz
x_plot3, y_plot3, z_plot3, marker_plot3 = bulk(X[idx_w, idx_n, :Nsites], Y[idx_w, idx_n, :Nsites], Z[idx_w, idx_n, :Nsites],
                                           local_marker[idx_w, idx_n, :Nsites], cutoff_xy3, cutoff_z3, Nx[idx_n], Nx[idx_n],
                                           full_z=True, full_xy=True)
cutoff = [c1, c2, 1]
avg_marker_plot = [np.mean(marker_plot1), np.mean(marker_plot2), np.mean(marker_plot3)]



# Marker bulk-surface transition
loger_main.info('Calculating bulk-surface transition...')
idx_n2 = -1
cutoff_sequence = np.linspace(0.2, 1, 10)
marker_transition = np.zeros((len(width), len(cutoff_sequence)))
for i in range(len(width)):
    for j, c in enumerate(cutoff_sequence):
        cutoff_xy = c * 0.5 * Nx[idx_n2]
        cutoff_z = c * 0.5 * Nz
        full_system = False if j<(len(cutoff_sequence) - 1) else True
        marker_transition[i, j] = np.mean(bulk(X[i, idx_n2, :Nsites], Y[i, idx_n2, :Nsites], Z[i, idx_n2, :Nsites],
                                               local_marker[i, idx_n2, :Nsites], cutoff_xy, cutoff_z, Nx[idx_n2], Nx[idx_n2],
                                               full_z=full_system, full_xy=full_system)[3])




#%% Saving data
data_dir = '/home/mfmm/Projects/amorphous-nanowires/data/data-marker-full-analysis'
file_list = os.listdir(data_dir)
expID = get_fileID(file_list, common_name='Exp')
filename = '{}{}{}'.format('Exp', expID, '.h5')
filepath = os.path.join(data_dir, filename)

loger_main.info('Saving data...')
with h5py.File(filepath, 'w') as f:

    plot1 = f.create_group('Plot1')
    store_my_data(plot1, 'avg_marker', avg_marker)
    store_my_data(plot1, 'width',      width)
    store_my_data(plot1, 'Nx',         Nx)

    plot2 = f.create_group('Plot2')
    store_my_data(plot2, 'Nx_plot',     Nx[idx_n])
    store_my_data(plot2, 'w',           width[idx_w])
    store_my_data(plot2, 'pos1',        [x_plot1, y_plot1, z_plot1])
    store_my_data(plot2, 'pos2',        [x_plot2, y_plot2, z_plot2])
    store_my_data(plot2, 'pos3',        [x_plot3, y_plot3, z_plot3])
    store_my_data(plot2, 'marker1',     marker_plot1)
    store_my_data(plot2, 'marker2',     marker_plot2)
    store_my_data(plot2, 'marker3',     marker_plot3)
    store_my_data(plot2, 'cutoff',      cutoff)
    store_my_data(plot2, 'avg_marker2', avg_marker_plot)

    plot3 = f.create_group('Plot3')
    store_my_data(plot3, 'cutoff_sequence', cutoff_sequence)
    store_my_data(plot3, 'marker_transition', marker_transition)
    store_my_data(plot3, 'width',      width)

    # Attributes
    attr_my_data(plot1, "Date",       str(date.today()))
    attr_my_data(plot1, "Code_path",  sys.argv[0])

loger_main.info('Data saved successfully')

