#%% modules set up

# Math and plotting
import numpy as np
from numpy.linalg import eigh
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Kwant
import kwant

# modules
from modules.functions import *
from modules.AmorphousLattice_3d import AmorphousLattice_3d
from modules.FullyAmorphousWire_kwant import promote_to_kwant_nanowire3d, spectrum, local_marker

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
# file_list = ['Exp23.h5']
# data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-cond-vs-flux-fully-amorphous')
#
# # Parameters
# Nx           = data_dict[file_list[0]]['Parameters']['Nx']
# Ny           = data_dict[file_list[0]]['Parameters']['Ny']
# Nz           = data_dict[file_list[0]]['Parameters']['Nz']
# r            = data_dict[file_list[0]]['Parameters']['r']
# t            = data_dict[file_list[0]]['Parameters']['t']
# eps          = data_dict[file_list[0]]['Parameters']['eps']
# lamb         = data_dict[file_list[0]]['Parameters']['lamb']
# lamb_z       = data_dict[file_list[0]]['Parameters']['lamb_z']
# mu_leads     = data_dict[file_list[0]]['Parameters']['mu_leads']
# params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}
#
# # Simulation data
# x             = data_dict[file_list[0]]['Simulation']['x']
# y             = data_dict[file_list[0]]['Simulation']['y']
# z             = data_dict[file_list[0]]['Simulation']['z']
# width         = data_dict[file_list[0]]['Simulation']['width']
# flux          = data_dict[file_list[0]]['Simulation']['flux']

# Variables
# idx =33
# flux_value = flux[idx]

# Variables (calculation from scratch)
Nx, Ny, Nz = 10, 10, 25
r          = 1.3
width      = 0.4
t          = 1
eps        = 4 * t
lamb       = 1 * t
lamb_z     = 1.8 * t
params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}
flux_value = 0

#%% Main

# Fully amorphous wire
loger_main.info('Generating fully amorphous lattice...')
lattice = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=Nz, w=width, r=r)
# lattice.set_configuration(x, y, z)
lattice.build_lattice(restrict_connectivity=False)
lattice.generate_disorder(K_hopp=0., K_onsite=0.)
nanowire = promote_to_kwant_nanowire3d(lattice, params_dict, mu_leads=0, attach_leads=False).finalized()
loger_main.info('Nanowire promoted to Kwant successfully.')

# Spectrum of the closed system
# H = nanowire.hamiltonian_submatrix(params=dict(flux=flux_value), sparse=True)
# eps, psi = sla.eigsh(H.tocsc(), k=50, sigma=0)
loger_main.info('Calculating spectrum:')
H = nanowire.hamiltonian_submatrix(params=dict(flux=flux_value))
eps, _, rho = spectrum(H)


# Local marker
loger_main.info('Calculating local marker...')
site_pos = np.array([site.pos for site in nanowire.id_by_site])
x, y, z = site_pos[:, 0], site_pos[:, 1], site_pos[:, 2]
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
chiral_sym = np.kron(np.eye(len(x)), np.kron(sigma_z, sigma_z))
local_marker = local_marker(x, y, z, rho, chiral_sym)


#%% Saving data
data_dir = '/home/mfmm/Projects/amorphous-nanowires/data/data-local-marker'
file_list = os.listdir(data_dir)
expID = get_fileID(file_list, common_name='Exp')
filename = '{}{}{}'.format('Exp', expID, '.h5')
filepath = os.path.join(data_dir, filename)


with h5py.File(filepath, 'w') as f:

    # Simulation folder
    simulation = f.create_group('Simulation')
    store_my_data(simulation, 'local_marker', local_marker)
    store_my_data(simulation, 'x', lattice.x)
    store_my_data(simulation, 'y', lattice.y)
    store_my_data(simulation, 'z', lattice.z)

    # Parameters folder
    parameters = f.create_group('Parameters')
    store_my_data(parameters, 'width',   width)
    store_my_data(parameters, 'flux',  flux_value)
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





