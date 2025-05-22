#%% modules set up

# Math and plotting
import numpy as np
from numpy.linalg import eigh
import scipy.sparse
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Kwant
import kwant

# modules
from modules.functions import *
from modules.AmorphousLattice_3d import AmorphousLattice_3d
from modules.FullyAmorphousWire_kwant import promote_to_kwant_nanowire3d, local_marker_KPM_rshell

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

#%% Variables

Nx, Ny, Nz = 15, 15, 100
r          = 1.3
width      = 0.1
t          = 1
eps        = 4 * t
lamb       = 1 * t
lamb_z     = 1.8 * t
params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}
flux_value = 0

# Radii distribution
num_bins_r, num_bins_z = 10, 10
diagonal =  np.sqrt(2 * (0.5 * (Nx - 1)) ** 2)
r_min, r_max = 0.8, diagonal + 0.1 * diagonal
z_min, z_max = -0.2, (Nz - 1) + 0.1 * (Nz - 1)
bin_edges_r = np.linspace(r_min, r_max, num_bins_r + 1)
bin_edges_z = np.linspace(z_min, z_max, num_bins_z + 1)
local_marker = np.zeros(num_bins_r, num_bins_z)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

#%% Main

loger_main.info(f'Generating lattice')
lattice = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=Nz, w=width, r=r)
lattice.build_lattice(restrict_connectivity=False)
nanowire = promote_to_kwant_nanowire3d(lattice, params_dict, attach_leads=False).finalized()
S = scipy.sparse.kron(np.eye(Nx * Ny * Nz), np.kron(sigma_z, sigma_z), format='csr')


# Local marker through KPM + Stochastic trace algorithm
for i, (z0, z1) in enumerate(zip(bin_edges_z[:-1], bin_edges_z[1:])):
    for j, (r0, r1) in enumerate(zip(bin_edges_r[:-1], bin_edges_r[1:])):

        loger_main.info('Calculating bulk marker through KPM algorithm')
        local_marker[i, j] = local_marker_KPM_rshell(nanowire, S, r0, r1, z0, Nx, Ny, Nz, num_moments=5000, num_vecs=5, z_max=z1)
        loger_main.info(f' z: {i}/{len(bin_edges_z) - 1}, radius: {j}/{len(bin_edges_r) - 1}, marker KPM: {local_marker[i, j] :.5f}')



#%% Saving data
data_dir = '/home/mfmm/Projects/amorphous-nanowires/data/data-marker-per-site'
file_list = os.listdir(data_dir)
expID = get_fileID(file_list, common_name='Exp')
filename = '{}{}{}'.format('Exp', expID, '.h5')
filepath = os.path.join(data_dir, filename)


with h5py.File(filepath, 'w') as f:

    # Simulation folder
    simulation = f.create_group('Simulation')
    store_my_data(simulation, 'local_marker',  local_marker)
    store_my_data(simulation, 'bin_edges_r',   bin_edges_r)
    store_my_data(simulation, 'bin_edges_z',   bin_edges_z)
    store_my_data(simulation, 'width',         width)


    # Parameters folder
    parameters = f.create_group('Parameters')
    store_my_data(parameters, 'flux',          flux_value)
    store_my_data(parameters, 'Nx',            Nx)
    store_my_data(parameters, 'Ny',            Ny)
    store_my_data(parameters, 'Nz',            Nz)
    store_my_data(parameters, 'r ',            r)
    store_my_data(parameters, 't ',            t)
    store_my_data(parameters, 'eps',           eps)
    store_my_data(parameters, 'lamb',          lamb)
    store_my_data(parameters, 'lamb_z',        lamb_z)

    # Attributes
    attr_my_data(parameters, "Date",       str(date.today()))
    attr_my_data(parameters, "Code_path",  sys.argv[0])

loger_main.info('Data saved correctly')



