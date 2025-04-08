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
file_list = ['Exp8.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-marker-vs-cross-section')

# Parameters
Nx           = data_dict[file_list[0]]['Parameters']['Nx'][0]
Ny           = data_dict[file_list[0]]['Parameters']['Ny'][0]
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


def lattice_cut(x, y, z, local_marker, cutoff, nx, ny, nz, full_z=False, full_xy=False, center=np.array([0, 0, 0])):
    # Coordinates and conditions to pertain to the cut
    x_pos, y_pos, z_pos = x - 0.5 * nx, y - 0.5 * ny, z - 0.5 * nz
    cond_x_pos, cond_x_neg = x_pos < center[0] + cutoff[0], center[0] - cutoff[0] < x_pos
    cond_y_pos, cond_y_neg = y_pos < center[1] + cutoff[1], center[1] - cutoff[1] < y_pos
    cond_z_pos, cond_z_neg = z_pos < center[2] + cutoff[2], center[2] - cutoff[2] < z_pos
    loger_main.info(f'{-cutoff[0] + center[0] + 0.5 * nx}  < x < {cutoff[0] + center[0] + 0.5 * nx}')
    loger_main.info(f'{-cutoff[1] + center[1] + 0.5 * nx}  < y < {cutoff[1] + center[1] + 0.5 * nx}')
    loger_main.info(f'{-cutoff[2] + center[2] + 0.5 * nx}  < z < {cutoff[2] + center[2] + 0.5 * nx}')

    # Cutoff conditions
    if full_xy and full_z:
        return x, y, z, local_marker
    elif full_z:
        cond = cond_x_pos * cond_x_neg * cond_y_pos * cond_y_neg
    else:
        cond = cond_x_pos * cond_x_neg * cond_y_pos * cond_y_neg * cond_z_pos * cond_z_neg

    return x[cond], y[cond], z[cond], local_marker[cond]

#%% Main

# Preallocation
Nsites = Nx * Ny * Nz
cutoff_sequence = np.linspace(0.2, 1, 10)
marker_transition   = np.zeros((len(width), len(cutoff_sequence)))
Xcuts               = np.zeros((len(width), Nsites, len(cutoff_sequence)))
Ycuts               = np.zeros((len(width), Nsites, len(cutoff_sequence)))
Zcuts               = np.zeros((len(width), Nsites, len(cutoff_sequence)))
marker_cuts         = np.zeros((len(width), Nsites, len(cutoff_sequence)))
Nsites_in_cut       = np.zeros((len(width), len(cutoff_sequence)))

# Local marker for different cuts of the lattice
loger_main.info('Calculating bulk-surface transition...')
for i in range(len(width)):
    for j, cutoff in enumerate(cutoff_sequence):
        cutoff_vec = [cutoff * 0.5 * Nx, cutoff * 0.5 * Ny, cutoff * 0.5 * Nz]
        full_system = True if j == (len(cutoff_sequence) - 1) else False
        x, y, z, marker = lattice_cut(X[i, 0, :Nsites], Y[i, 0, :Nsites], Z[i, 0, :Nsites], local_marker[i, 0, :Nsites],
                                 cutoff_vec, Nx, Ny, Nz, full_z=full_system, full_xy=full_system)
        Xcuts[i, :len(x), j], Ycuts[i, :len(x), j], Zcuts[i, :len(x), j], marker_cuts[i, :len(x), j] = x, y, z, marker
        marker_transition[i, j] = np.mean(marker_cuts[i, :len(x), j])
        Nsites_in_cut[i, j] = len(x)

#%% Saving data
data_dir = '/home/mfmm/Projects/amorphous-nanowires/data/data-marker-full-analysis'
file_list = os.listdir(data_dir)
expID = get_fileID(file_list, common_name='Exp')
filename = '{}{}{}'.format('Exp', expID, '.h5')
filepath = os.path.join(data_dir, filename)


with h5py.File(filepath, 'w') as f:

    # Simulation folder
    simulation = f.create_group('Simulation')
    store_my_data(simulation, 'cutoff_sequence',  cutoff_sequence)
    store_my_data(simulation, 'marker_transition', marker_transition)
    store_my_data(simulation, 'marker_cuts',        marker_cuts)
    store_my_data(simulation, 'Xcuts',              Xcuts)
    store_my_data(simulation, 'Ycuts',              Ycuts)
    store_my_data(simulation, 'Zcuts',              Zcuts)
    store_my_data(simulation, 'Nsites_in_cut',     Nsites_in_cut)


    # Parameters folder
    parameters = f.create_group('Parameters')
    store_my_data(parameters, 'flux',  flux)
    store_my_data(parameters, 'width', width)
    store_my_data(parameters, 'Nx',      Nx)
    store_my_data(parameters, 'Ny',      Ny)
    store_my_data(parameters, 'Nz',      Nz)
    store_my_data(parameters, 'r ',      r)
    store_my_data(parameters, 't ',      t)
    store_my_data(parameters, 'eps',     eps)
    store_my_data(parameters, 'lamb',    lamb)
    store_my_data(parameters, 'lamb_z',  lamb_z)

    # Attributes
    attr_my_data(parameters, "Date",       str(date.today()))
    attr_my_data(parameters, "Code_path",  sys.argv[0])

loger_main.info('Data saved correctly')







