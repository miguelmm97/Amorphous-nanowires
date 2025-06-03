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
from modules.FullyAmorphousWire_kwant import promote_to_kwant_nanowire3d, OPDM_per_site_cross_section_KPM

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

z0 = 39.5
z1 = 40.5
Nx, Ny, Nz = 8, 8, 150
r          = 1.3
width      = 0.05
t          = 1
eps        = 4 * t
lamb       = 1 * t
lamb_z     = 1.8 * t
params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}
flux_value = 0
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

#%% Main

# Fully amorphous wire
loger_main.info(f'Generating lattice')
lattice = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=Nz, w=width, r=r)
lattice.build_lattice(restrict_connectivity=False)
nanowire = promote_to_kwant_nanowire3d(lattice, params_dict, attach_leads=False).finalized()

# Local marker through KPM + Stochastic trace algorithm
loger_main.info('Calculating OPDM')
OPDM_r, r_3d, indices, x, y, z = OPDM_per_site_cross_section_KPM(nanowire, Nx, Ny, Nz, z0, z1, Ef=0., num_moments=1000, bounds=None)

#%% Saving data
data_dir = '/home/mfmm/Projects/amorphous-nanowires/data/local-simulations/data-OPDM'
file_list = os.listdir(data_dir)
expID = get_fileID(file_list, common_name='Exp')
filename = '{}{}{}'.format('Exp', expID, '.h5')
filepath = os.path.join(data_dir, filename)


with h5py.File(filepath, 'w') as f:

    # Simulation folder
    simulation = f.create_group('Simulation')
    store_my_data(simulation, 'OPDM_r',  np.array(OPDM_r))
    store_my_data(simulation, 'r_3d',    np.array(r_3d))
    store_my_data(simulation, 'width',     width)
    store_my_data(simulation, 'x',             x)
    store_my_data(simulation, 'y',             y)
    store_my_data(simulation, 'z',             z)
    store_my_data(simulation, 'z0',           z0)
    store_my_data(simulation, 'z1',           z1)
    store_my_data(simulation, 'indices', np.array(indices))


    # Parameters folder
    parameters = f.create_group('Parameters')
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



