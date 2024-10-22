#%% modules set up

# Math and plotting
import numpy as np

# Kwant
import kwant

# modules
from modules.functions import *
from modules.AmorphousLattice_3d import AmorphousLattice_3d
from modules.FullyAmorphousWire_kwant import promote_to_kwant_nanowire3d

# Saving attributes
from datetime import date
import sys


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
"""
We check that the fully amorphous wire reduces to the translation invariant one.
"""

Nx, Ny, Nz       = 10, 10, 100                # Number of sites in the cross-section
r                = 1.3                        # Nearest-neighbour cutoff distance
t                = 1                          # Hopping
eps              = 4 * t                      # Onsite orbital hopping (in units of t)
lamb             = 1 * t                      # Spin-orbit coupling in the cross-section (in units of t)
lamb_z           = 1.8 * t                    # Spin-orbit coupling along z direction
mu_leads         = - 0 * t                    # Chemical potential at the leads
flux             = np.linspace(0, 1.5, 100)   # Magnetic flux
width            = [0.00001]                  # Amorphous width 0.0001, 0.02, 0.05,
Ef               = [0.01]                        # Fermi energy
params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}

# Preallocation
kwant_nw_dict = {}
lattice_dict = {}
G_array = np.zeros((len(Ef), len(width), len(flux)), dtype=np.float64)
#%% Main

# Generate nanowires
for i, w in enumerate(width):
    loger_main.info('Generating amorphous lattice...')
    lattice = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=Nz, w=w, r=r)
    lattice.build_lattice()
    lattice.generate_disorder()
    lattice_dict[i] = lattice
    kwant_nw_dict[i] = promote_to_kwant_nanowire3d(lattice, params_dict, mu_leads=mu_leads).finalized()
    loger_main.info('Nanowire promoted to Kwant successfully.')

# Calculate conductance
for key in kwant_nw_dict.keys():
    for i, phi in enumerate(flux):
        for k, E in enumerate(Ef):
            S = kwant.smatrix(kwant_nw_dict[key], Ef[k], params=dict(flux=phi))
            G_array[k, key, i] = S.transmission(1, 0)
            loger_main.info(f'Width: {key} / {len(width) - 1}, flux: {i} / {len(flux)}, Ef: {Ef[k]}'
                            f'|| G: {G_array[k, key, i] :.2f}')



#%% Saving data

data_dir = '/home/mfmm/Projects/amorphous-nanowires/data/data-cond-vs-flux-fully-amorphous'
file_list = os.listdir(data_dir)
expID = get_fileID(file_list, common_name='Exp')
filename = '{}{}{}'.format('Exp', expID, '.h5')
filepath = os.path.join(data_dir, filename)

with h5py.File(filepath, 'w') as f:

    # Simulation folder
    simulation = f.create_group('Simulation')
    store_my_data(simulation, 'flux',      flux)
    store_my_data(simulation, 'width',     width)
    store_my_data(simulation, 'G_array',   G_array)

    # Parameters folder
    parameters = f.create_group('Parameters')
    store_my_data(parameters, 'Ef',          Ef)
    store_my_data(parameters, 'Nx',          Nx)
    store_my_data(parameters, 'Ny',          Ny)
    store_my_data(parameters, 'Nz',          Nz)
    store_my_data(parameters, 'r',           r)
    store_my_data(parameters, 't',           t)
    store_my_data(parameters, 'eps',         eps)
    store_my_data(parameters, 'lamb',        lamb)
    store_my_data(parameters, 'lamb_z',      lamb_z)
    store_my_data(parameters, 'mu_leads',    mu_leads)


    # Attributes
    attr_my_data(parameters, "Date",       str(date.today()))
    attr_my_data(parameters, "Code_path",  sys.argv[0])

loger_main.info('Data saved correctly')

