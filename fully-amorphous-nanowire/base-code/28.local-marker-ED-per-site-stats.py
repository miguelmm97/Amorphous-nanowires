#%% Modules and setup

# Plotting
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
file_list = ['Exp8.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-marker-per-site')

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

# Simulation data
pos  = data_dict[file_list[0]]['Simulation']['position']
local_marker = data_dict[file_list[0]]['Simulation']['local_marker']
width        = data_dict[file_list[0]]['Simulation']['width']
radius = np.sqrt(((pos[:, 0] - 0.5 * (Nx - 1)) ** 2) + ((pos[:, 1] - 0.5 * (Ny -1)) ** 2))
z_min, z_max = 0.2 * (Nz-1), 0.8 * (Nz-1)

# Statistics
num_bins = 10
r_min, r_max = radius.min(), radius.max()
bin_edges = np.linspace(r_min, r_max, num_bins + 1)
bin_indices = np.digitize(radius, bin_edges) - 1
binned_samples = [[] for _ in range(num_bins)]
for idx, bin_idx in enumerate(bin_indices):
    if 0 <= bin_idx < num_bins:
        if z_min <= pos[idx, 2] < z_max:
            binned_samples[bin_idx].append(local_marker[idx])
binned_samples = [np.array(bin) for bin in binned_samples]
avg_marker = np.array([np.mean(binned_samples[i]) for i in range(len(binned_samples))])
avg_radius = 0.5 * (bin_edges[:-1] + bin_edges[1:])
avg_radius = [avg_radius[i] for i in range(len(avg_radius)) if not math.isnan(avg_marker[i])]
avg_marker = [x for x in avg_marker if not math.isnan(x)]




# Scatter plot of the cross-section
x = pos[:, 0] - 0.5 * (Nx - 1)
y = pos[:, 1] - 0.5 * (Ny - 1)
cond1 = pos[:, 2] < z_max
cond2 = z_min <= pos[:, 2]
x_scatter, y_scatter, marker_scatter =  x[cond1 * cond2], y[cond1 * cond2], local_marker[cond1 * cond2]

#%% Saving data
data_dir = '/home/mfmm/Projects/amorphous-nanowires/data/data-marker-per-site-stats'
file_list = os.listdir(data_dir)
expID = get_fileID(file_list, common_name='Exp')
filename = '{}{}{}'.format('Exp', expID, '.h5')
filepath = os.path.join(data_dir, filename)


with h5py.File(filepath, 'w') as f:

    # Simulation folder
    simulation = f.create_group('Simulation')
    store_my_data(simulation, 'avg_marker', avg_marker)
    store_my_data(simulation, 'avg_radius', avg_radius)
    store_my_data(simulation, 'x',           x_scatter)
    store_my_data(simulation, 'y',           y_scatter)
    store_my_data(simulation, 'marker',      marker_scatter)
    store_my_data(simulation, 'width',       width)
    store_my_data(simulation, 'pos',         pos)
    store_my_data(simulation, 'local_marker', local_marker)
    store_my_data(simulation, 'z_min',        z_min)
    store_my_data(simulation, 'z_max',        z_max)




    # Parameters folder
    parameters = f.create_group('Parameters')
    store_my_data(parameters, 'Nx',      Nx)
    store_my_data(parameters, 'Ny',      Ny)
    store_my_data(parameters, 'Nz',      Nz)
    store_my_data(parameters, 'r ',      r)
    store_my_data(parameters, 't ',      t)
    store_my_data(parameters, 'eps',     eps)
    store_my_data(parameters, 'lamb',    lamb)
    store_my_data(parameters, 'lamb_z',  lamb_z)



loger_main.info('Data saved correctly')