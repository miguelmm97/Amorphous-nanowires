#%% Modules and setup

# Plotting
import math
import numpy as np

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
file_list = ['exp-0.h5', 'exp-2.h5',  'exp-3.h5',  'exp-4.h5']
# file_list = ['exp-14.h5', 'exp-15.h5', 'exp-16.h5', 'exp-17.h5', 'exp-18.h5', 'exp-19.h5', 'exp-20.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/cluster-simulations/data-cluster-marker-per-site/data-12')
# Parameters
Nx           = data_dict[file_list[0]]['Parameters']['Nx']
Ny           = data_dict[file_list[0]]['Parameters']['Nx']
Nz           = data_dict[file_list[0]]['Parameters']['Nz']
r            = data_dict[file_list[0]]['Parameters']['r ']
t            = data_dict[file_list[0]]['Parameters']['t ']
eps          = data_dict[file_list[0]]['Parameters']['eps']
lamb         = data_dict[file_list[0]]['Parameters']['lamb']
lamb_z       = data_dict[file_list[0]]['Parameters']['lamb_z']
width        = data_dict[file_list[0]]['Simulation']['width']

# Loading data
z0, z1 = [], []
x, y, z, marker = np.array([]), np.array([]), np.array([]), np.array([])
for i in range(len(file_list)):
    z0.append(data_dict[file_list[i]]['Simulation']['z0'])
    z1.append(data_dict[file_list[i]]['Simulation']['z1'])
    x = np.concatenate((x, data_dict[file_list[i]]['Simulation']['x']))
    y = np.concatenate((y, data_dict[file_list[i]]['Simulation']['y']))
    z = np.concatenate((z, data_dict[file_list[i]]['Simulation']['z']))
    marker = np.concatenate((marker, data_dict[file_list[i]]['Simulation']['local_marker']))

x, y = x - 0.5 * (Nx - 1),  y - 0.5 * (Ny - 1)
radius = np.sqrt((x ** 2) + (y ** 2))
marker = np.real(marker)



#%% Data processing

# Probability distribution P(\nu)|_r
num_bins = 15
r_min, r_max = radius.min(), radius.max()
bin_edges = np.linspace(r_min, r_max, num_bins + 1)
bin_indices = np.digitize(radius, bin_edges) - 1
binned_samples = [[] for _ in range(num_bins)]
for idx, bin_idx in enumerate(bin_indices):
    if 0 <= bin_idx < num_bins:
        if not math.isnan(marker[idx]): # and np.abs(marker[idx]) < 30:
            binned_samples[bin_idx].append(marker[idx])
binned_samples = [np.array(bin) for bin in binned_samples]

avg_radius0 = 0.5 * (bin_edges[:-1] + bin_edges[1:])
prob_dist, avg_radius = {}, []
bin_marker_list = {}
for i, lst in enumerate(binned_samples):
    if len(lst) != 0:
        counts, bin_edges = np.histogram(lst, bins='auto')
        bin_marker_list[i] = (0.5 * (bin_edges[:-1] + bin_edges[1:]))
        prob_dist[i] = counts / len(lst)
        avg_radius.append(avg_radius0[i])

# Statistics of the distribution: Average and standard deviation
avg_marker = np.array([np.mean(binned_samples[i]) for i in range(len(binned_samples))if len(binned_samples[i]) != 0])
std_marker = np.array([np.std(binned_samples[i]) for i in range(len(binned_samples)) if len(binned_samples[i]) != 0])


#%% Saving data
data_dir = '/home/mfmm/Projects/amorphous-nanowires/data/cluster-simulations/data-cluster-marker-per-site/data-cluster-marker-per-site-statistics'
file_list = os.listdir(data_dir)
expID = get_fileID(file_list, common_name='Exp')
filename = '{}{}{}'.format('Exp', expID, '.h5')
filepath = os.path.join(data_dir, filename)


with h5py.File(filepath, 'w') as f:

    # Simulation folder
    distribution =  f.create_group('prob_dist')
    bins_marker  = f.create_group('bins_marker')
    simulation   = f.create_group('Simulation')
    store_my_dict(distribution,                         prob_dist)
    store_my_dict(bins_marker,                    bin_marker_list)
    store_my_data(simulation, 'avg_marker',            avg_marker)
    store_my_data(simulation, 'std_marker',            std_marker)
    store_my_data(simulation, 'avg_radius',            avg_radius)
    store_my_data(simulation, 'marker',                    marker)
    store_my_data(simulation, 'x',                              x)
    store_my_data(simulation, 'y',                              y)
    store_my_data(simulation, 'z',                              z)
    store_my_data(simulation, 'z0',                             z0)
    store_my_data(simulation, 'z1',                             z1)
    store_my_data(simulation, 'width',                       width)

    # Parameters folder
    parameters = f.create_group('Parameters')
    store_my_data(parameters, 'Nx',                            Nx)
    store_my_data(parameters, 'Ny',                            Ny)
    store_my_data(parameters, 'Nz',                            Nz)
    store_my_data(parameters, 'r ',                            r)
    store_my_data(parameters, 't ',                            t)
    store_my_data(parameters, 'eps',                           eps)
    store_my_data(parameters, 'lamb',                          lamb)
    store_my_data(parameters, 'lamb_z',                        lamb_z)


loger_main.info('Data saved correctly')

