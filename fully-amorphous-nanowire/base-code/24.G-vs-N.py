#%% modules set up

# Math and plotting
import numpy as np
import scipy.sparse

# Kwant
import kwant

# modules
from modules.functions import *
from modules.AmorphousLattice_3d import AmorphousLattice_3d, take_cut_from_parent_wire
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
Nz               = 50                        # Length
Nx               = np.linspace(8, 15, 8, dtype=np.int32)[::-1]   # Sites in the cross-section
Ny               = Nx                         # Sites in the cross-section
r                = 1.3                        # Nearest-neighbour cutoff distance
t                = 1                          # Hopping
eps              = 4 * t                      # Onsite orbital hopping (in units of t)
lamb             = 1 * t                      # Spin-orbit coupling in the cross-section (in units of t)
lamb_z           = 1.8 * t                    # Spin-orbit coupling along z direction
mu_leads         = - 1 * t                    # Chemical potential at the leads
flux             = np.linspace(0, 1, 100)     # Magnetic flux
width            = 0.15                        # Amorphous width 0.0001, 0.02, 0.05,
Ef               = 0.02                       # Fermi energy
K_onsite         = 0.1                        # Onsite disorder
params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}

# Preallocation
kwant_nw_dict = {}
lattice_dict = {}
G_array = np.zeros((len(Nx), len(flux)), dtype=np.float64)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
#%% Main

# Generate parent nanowire
full_lattice = AmorphousLattice_3d(Nx=np.max(Nx), Ny=np.max(Ny), Nz=Nz, w=width, r=r)
full_lattice.build_lattice()
X = full_lattice.x
Y = full_lattice.y
Z = full_lattice.z
full_lattice.generate_onsite_disorder(K_onsite=K_onsite)

for i, n in enumerate(Nx):

    # Taking a cut of the parent cross-section
    lattice = take_cut_from_parent_wire(full_lattice, Nx_new=n, Ny_new=n, keep_disorder=True)
    nanowire = promote_to_kwant_nanowire3d(lattice, params_dict).finalized()

    # Checks for chiral symmetry
    # closed_wire = promote_to_kwant_nanowire3d(lattice, params_dict, attach_leads=False).finalized()
    # H = closed_wire.hamiltonian_submatrix(params=dict(flux=0., mu=-Ef))
    # S = np.kron(np.eye(Nx[0] * Ny[0] * Nz), np.kron(sigma_z, sigma_z))
    # loger_main.info(f'Chiral symmetry: {np.allclose(S @ H @ S , -H)}')

    # Calculating conductance
    for j, phi in enumerate(flux):
        S = kwant.smatrix(nanowire, 0., params=dict(flux=phi, mu=-Ef, mu_leads=mu_leads - Ef))
        G_array[i, j] = S.transmission(1, 0)
        loger_main.info(f'N: {n}, {i} / {len(Nx) - 1}, flux: {phi :.2f}, {j} / {len(flux) - 1} || G: {G_array[i, j] :.2f}')



#%% Saving data

data_dir = '/home/mfmm/Projects/amorphous-nanowires/data/data-cond-vs-N'
file_list = os.listdir(data_dir)
expID = get_fileID(file_list, common_name='Exp')
filename = '{}{}{}'.format('Exp', expID, '.h5')
filepath = os.path.join(data_dir, filename)

with h5py.File(filepath, 'w') as f:

    # Simulation folder
    simulation = f.create_group('Simulation')
    store_my_data(simulation, 'flux',        flux)
    store_my_data(simulation, 'width',       width)
    store_my_data(simulation, 'G_array',     G_array)
    store_my_data(simulation, 'x',           X)
    store_my_data(simulation, 'y',           Y)
    store_my_data(simulation, 'z',           Z)
    store_my_data(simulation, 'Nx',          Nx)
    store_my_data(simulation, 'Ny',          Ny)
    store_my_data(simulation, 'K_onsite',    K_onsite)


    # Parameters folder
    parameters = f.create_group('Parameters')
    store_my_data(parameters, 'Ef',          Ef)
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

