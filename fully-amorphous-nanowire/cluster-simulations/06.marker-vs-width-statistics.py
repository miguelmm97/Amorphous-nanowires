#%% Modules and setup

# Math
import numpy as np

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


#%% Preallocation
with open('gen-params-marker-vs-width.txt', 'r') as f:
    Nsamples = len(f.readlines())

file_path = os.path.join('data', 'exp-0.h5')
with h5py.File(file_path, 'r') as f:
    Nx = f['Parameters']['Nx'][()]
    Ny = f['Parameters']['Ny'][()]
    Nz = f['Parameters']['Nz'][()]
    width = f['Parameters']['width'][()]
Nsites = int(Nz * np.max(Nx) * np.max(Ny))
avg_marker_samples = np.zeros((Nsamples, len(width), len(Nx)))


#%% Main
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

for i, file in enumerate(os.listdir('data')):
    if file.endswith('h5'):
        file_path = os.path.join('data', file)
        with h5py.File(file_path, 'r') as f:

            # Loading data
            loger_main.info('Loading data...')
            X = f['Simulation']['x'][()]
            Y = f['Simulation']['y'][()]
            Z = f['Simulation']['z'][()]
            local_marker = f['Simulation']['local_marker'][()]

            # Calculating average marker for each realisation
            loger_main.info('Calculating average marker per realisation...')
            for j in range(len(width)):
                for k, n in enumerate(Nx):
                    cutoff_bulk = 0.4 * 0.5 * n
                    cutoff_z = cutoff_bulk
                    Nsites = int(n * n * Nz)
                    x, y, z, marker_xy = bulk(X[j, k, :Nsites], Y[j, k, :Nsites], Z[j, k, :Nsites],
                                              local_marker[j, k, :Nsites],
                                              cutoff_bulk, cutoff_z, n, n, full_z=False, full_xy=False)
                    avg_marker_samples[i, j, k] = np.mean(marker_xy)


# Calculating statistics
loger_main.info('Calculating statistics...')
avg_marker = np.mean(avg_marker_samples, axis=0)
std_marker = np.std(avg_marker_samples, axis=0)

#%% Saving data
data_dir = '.'
filename = 'data-cluster.h5'
filepath = os.path.join(data_dir, filename)

loger_main.info('Saving data...')
with h5py.File(filepath, 'w') as f:

    plot1 = f.create_group('Plot1')
    store_my_data(plot1, 'avg_marker', avg_marker)
    store_my_data(plot1, 'std_marker', avg_marker)
    store_my_data(plot1, 'width',      width)
    store_my_data(plot1, 'Nx',         Nx)

    # Attributes
    attr_my_data(plot1, "Date",       str(date.today()))
    attr_my_data(plot1, "Code_path",  sys.argv[0])

loger_main.info('Data saved successfully')

