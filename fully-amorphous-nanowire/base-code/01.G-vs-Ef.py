# %% modules set up

# Math and plotting
import numpy as np
import sys
from datetime import date

# Kwant
import kwant

# modules
from modules.functions import *
from modules.AmorphousLattice_3d import AmorphousLattice_3d
from modules.FullyAmorphousWire_kwant import promote_to_kwant_nanowire3d, select_perfect_transmission_flux, spectrum

# %% Logging setup
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

# %% Variables
Nx, Ny, Nz = 15, 15, 100 # Number of sites in the cross-section
width = [0.1]            # Spread of the Gaussian distribution for the lattice sites
r = 1.3                  # Nearest-neighbour cutoff distance
t = 1                    # Hopping
eps = 4 * t              # Onsite orbital hopping (in units of t)
lamb = 1 * t             # Spin-orbit coupling in the cross-section (in units of t)
lamb_z = 1.8 * t         # Spin-orbit coupling along z direction
mu_leads = -0 * t        # Chemical potential at the leads
fermi = np.linspace(0, 2, 500)  # Fermi energy
K_hopp = 0.
K_onsite = 0.
params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}

# Preallocation
G_0 = np.zeros((len(fermi), len(width)))
G_half = np.zeros((len(fermi), len(width)))
flux_max = np.zeros((len(width),))
X = np.zeros((len(width), Nx * Ny * Nz))
Y = np.zeros((len(width), Nx * Ny * Nz))
Z = np.zeros((len(width), Nx * Ny * Nz))
# G_0    = np.zeros((len(fermi), len(K_onsite)))
# G_half = np.zeros((len(fermi), len(K_onsite)))
# flux_max = np.zeros((len(K_onsite), ))
# %% Main

# Fully amorphous wire

for j, w in enumerate(width):
# for j, k in enumerate(K_onsite):

    loger_main.info('Generating fully amorphous lattice...')
    lattice = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=Nz, w=w, r=r)
    lattice.build_lattice(restrict_connectivity=False)
    # lattice.generate_disorder(K_hopp=K_hopp, K_onsite=K_onsite)
    nanowire = promote_to_kwant_nanowire3d(lattice, params_dict).finalized()
    X[j, :] = lattice.x
    Y[j, :] = lattice.y
    Z[j, :] = lattice.z
    loger_main.info('Nanowire promoted to Kwant successfully.')

    # Scanning for flux that gives perfect transmission at the Dirac point
    # flux_max[j], Gmax = select_perfect_transmission_flux(nanowire, flux0=0., flux_end=1, Ef=0.01, mu_leads=mu_leads)
    # loger_main.info(f'Flux for perfect transmission: {flux_max[j]}, Conductance at the Dirac point: {Gmax}')
    # flux_max = 0.7

    # Conductance calculation for different flux values
    for i, Ef in enumerate(fermi):
        S0 = kwant.smatrix(nanowire, 0., params=dict(flux=0., mu=-Ef, mu_leads=mu_leads - Ef))
        G_0[i, j] = S0.transmission(1, 0)
        # G_half[i, j] = 0.
        # S1 = kwant.smatrix(nanowire, 0., params=dict(flux=flux_max[j], mu=-Ef, mu_leads=mu_leads - Ef))
        # G_half[i, j] = S1.transmission(1, 0)
        loger_main.info(
            f'Ef: {i} / {len(fermi) - 1}, width: {j} / {len(width) - 1} || G0: {G_0[i, j] :.2f} || Ghalf: {G_half[i, j] :.2f}')
        # loger_main.info(
        # f'Ef: {i} / {len(fermi) - 1}, K: {j} / {len(K_onsite) - 1} || G0: {G_0[i, j] :.2f} || Ghalf: {G_half[i, j] :.2f}')

# nanowire_closed = promote_to_kwant_nanowire3d(lattice, params_dict, mu_leads=0, attach_leads=False).finalized()
# H = nanowire_closed.hamiltonian_submatrix(params=dict(flux=0))
# eps, _, _ = spectrum(H)

# %% Saving data
data_dir = '/home/mfmm/Projects/amorphous-nanowires/data/local-simulations/data-cond-vs-Ef'
file_list = os.listdir(data_dir)
expID = get_fileID(file_list, common_name='Exp')
filename = '{}{}{}'.format('Exp', expID, '.h5')
filepath = os.path.join(data_dir, filename)

with h5py.File(filepath, 'w') as f:
    # Simulation folder
    simulation = f.create_group('Simulation')
    store_my_data(simulation, 'width', width)
    store_my_data(simulation, 'K_onsite', K_onsite)
    store_my_data(simulation, 'fermi', fermi)
    store_my_data(simulation, 'flux_max', flux_max)
    store_my_data(simulation, 'G_0', G_0)
    store_my_data(simulation, 'G_half', G_half)
    store_my_data(simulation, 'x', X)
    store_my_data(simulation, 'y', Y)
    store_my_data(simulation, 'z', Z)
    store_my_data(simulation, 'eps', eps)
    # store_my_data(simulation, 'disorder',      lattice.disorder)

    # Parameters folder
    parameters = f.create_group('Parameters')
    store_my_data(parameters, 'Nx', Nx)
    store_my_data(parameters, 'Ny', Ny)
    store_my_data(parameters, 'Nz', Nz)
    store_my_data(parameters, 'r', r)
    store_my_data(parameters, 't', t)
    store_my_data(parameters, 'eps', eps)
    store_my_data(parameters, 'lamb', lamb)
    store_my_data(parameters, 'lamb_z', lamb_z)
    store_my_data(parameters, 'mu_leads', mu_leads)

    # Attributes
    attr_my_data(parameters, "Date", str(date.today()))
    attr_my_data(parameters, "Code_path", sys.argv[0])

loger_main.info('Data saved correctly')
