#%% modules set up

# Math
import numpy as np


# Kwant
import kwant

# modules
from modules.functions import *
from modules.AmorphousLattice_3d import AmorphousLattice_3d, take_cut_from_parent_wire
from modules.FullyAmorphousWire_kwant import promote_to_kwant_nanowire3d, crystal_nanowire_kwant
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
Nx, Ny           = 10, 10                                     # Number of sites in the cross-section
r                = 1.3                                        # Nearest-neighbour cutoff distance
t                = 1                                          # Hopping
eps              = 4 * t                                      # Onsite orbital hopping (in units of t)
lamb             = 1 * t                                      # Spin-orbit coupling in the cross-section (in units of t)
lamb_z           = 1.8 * t                                    # Spin-orbit coupling along z direction
mu_leads         = - 1 * t                                    # Chemical potential at the leads
Ef               = np.linspace(0, 0.7, 6)                     # Fermi energy
width            = np.linspace(0.01, 2, 20)                   # Amorphous width
K_vec            = [0.]                                       # Disorder strength
Nz               = [200]                                      # Length of the wire
flux             = np.linspace(0, 1, 5)                       # Flux
params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}

# Preallocation
G_array = np.zeros((len(width), len(flux), len(Ef), len(Nz), len(K_vec)), dtype=np.float64)
disorder_dict = {}

#%% Main

# Generate nanowires
for i, w in enumerate(width):

    # Generating lattice structure of the wire
    full_lattice = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=np.max(Nz), w=w, r=r)
    full_lattice.build_lattice()

    for j, K in enumerate(K_vec):

        # Generating disorder realisation of the wire
        full_lattice.generate_disorder(K_onsite=0., K_hopp=K)
        disorder_dict[j] = full_lattice.disorder

        for k, L in enumerate(Nz):

            # Selecting different cuts of the wire for each disorder realisation
            lattice = take_cut_from_parent_wire(full_lattice, Nz_new=L, keep_disorder=True)
            nanowire = promote_to_kwant_nanowire3d(lattice, params_dict, mu_leads=mu_leads).finalized()

            # Calculating conductance
            for l, eps in enumerate(Ef):
                for m, phi in enumerate(flux):

                    S = kwant.smatrix(nanowire, eps, params=dict(flux=phi))
                    G_array[i, m, l, j, k] = S.transmission(1, 0)
                    loger_main.info(f'Width: {i} / {len(width) - 1}, Disorder: {j} / {len(K_vec) - 1}, L: {k} /'
                                    f'{len(Nz) - 1},  Ef: {l} / {len(Ef) - 1}, flux: {m} / {len(flux) - 1} '
                                    f'|| G: {G_array[i, m, l, j, k] :.2f}')


#%% Saving data
data_dir = '/home/mfmm/Projects/amorphous-nanowires/data/data-cond-vs-width'
file_list = os.listdir(data_dir)
expID = get_fileID(file_list, common_name='Exp')
filename = '{}{}{}'.format('Exp', expID, '.h5')
filepath = os.path.join(data_dir, filename)


with h5py.File(filepath, 'w') as f:

    # Simulation folder
    simulation = f.create_group('Simulation')
    disorder = simulation.create_group('Disorder')
    store_my_data(simulation, 'Nz',       Nz)
    store_my_data(simulation, 'flux',     flux)
    store_my_data(simulation, 'Ef',       Ef)
    store_my_data(simulation, 'width',    width)
    store_my_data(simulation, 'K',        K_vec)
    store_my_data(simulation, 'G_array',  G_array)
    store_my_dict(simulation['Disorder'], disorder_dict)

    # Parameters folder
    parameters = f.create_group('Parameters')
    store_my_data(parameters, 'Nx',            Nx)
    store_my_data(parameters, 'Ny',            Ny)
    store_my_data(parameters, 'r',             r)
    store_my_data(parameters, 't',             t)
    store_my_data(parameters, 'eps',           eps)
    store_my_data(parameters, 'lamb',          lamb)
    store_my_data(parameters, 'lamb_z',        lamb_z)
    store_my_data(parameters, 'mu_leads',      mu_leads)

    # Attributes
    attr_my_data(parameters, "Date",       str(date.today()))
    attr_my_data(parameters, "Code_path",  sys.argv[0])

loger_main.info('Data saved correctly')

