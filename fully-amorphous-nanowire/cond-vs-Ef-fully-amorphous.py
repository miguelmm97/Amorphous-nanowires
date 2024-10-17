#%% modules set up

# Math and plotting
import sys
from datetime import date

# Kwant
import kwant

# modules
from modules.functions import *
from modules.AmorphousLattice_3d import AmorphousLattice_3d
from modules.FullyAmorphousWire_kwant import promote_to_kwant_nanowire3d, select_perfect_transmission_flux

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
We calculate the conductance of the nanowire vs Fermi energy.
"""

Nx, Ny, Nz       = 10, 10, 100                # Number of sites in the cross-section
width            = 0.1                        # Spread of the Gaussian distribution for the lattice sites
r                = 1.3                        # Nearest-neighbour cutoff distance
t                = 1                          # Hopping
eps              = 4 * t                      # Onsite orbital hopping (in units of t)
lamb             = 1 * t                      # Spin-orbit coupling in the cross-section (in units of t)
lamb_z           = 1.8 * t                    # Spin-orbit coupling along z direction
mu_leads         = 0 * t                      # Chemical potential at the leads
fermi            = np.linspace(0, 1, 200)     # Fermi energy
K_hopp           = 0.
K_onsite         = 0.
params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}

# Preallocation
G_0    = np.zeros(fermi.shape)
G_half = np.zeros(fermi.shape)

#%% Main

# Fully amorphous wire
loger_main.info('Generating fully amorphous lattice...')
lattice = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=Nz, w=width, r=r)
lattice.build_lattice()
lattice.generate_disorder(K_hopp=K_hopp, K_onsite=K_onsite)
nanowire = promote_to_kwant_nanowire3d(lattice, params_dict, mu_leads=mu_leads).finalized()
loger_main.info('Nanowire promoted to Kwant successfully.')

# Scanning for flux that gives perfect transmission at the Dirac point
flux_max, Gmax = select_perfect_transmission_flux(nanowire)
loger_main.info(f'Flux for perfect transmission: {flux_max}, Conductance at the Dirac point: {Gmax}')

# Conductance calculation for different flux values
for i, Ef in enumerate(fermi):
    loger_main.info(f'Calculating conductance for Ef: {i} / {fermi.shape[0] - 1}...')
    S0 = kwant.smatrix(nanowire, Ef, params=dict(flux=0.))
    G_0[i] = S0.transmission(1, 0)
    S1 = kwant.smatrix(nanowire, Ef, params=dict(flux=flux_max))
    G_half[i] = S1.transmission(1, 0)

#%% Saving data
data_dir = '/home/mfmm/Projects/amorphous-nanowires/data/data-cond-vs-Ef'
file_list = os.listdir(data_dir)
expID = get_fileID(file_list, common_name='Exp')
filename = '{}{}{}'.format('Exp', expID, '.h5')
filepath = os.path.join(data_dir, filename)


with h5py.File(filepath, 'w') as f:

    # Simulation folder
    simulation = f.create_group('Simulation')
    store_my_data(simulation, 'fermi',         fermi)
    store_my_data(simulation, 'flux_max',      flux_max)
    store_my_data(simulation, 'G_0',           G_0)
    store_my_data(simulation, 'G_half',        G_half)
    store_my_data(simulation, 'disorder',      lattice.disorder)

    # Parameters folder
    parameters = f.create_group('Parameters')
    store_my_data(parameters, 'Nx',            Nx)
    store_my_data(parameters, 'Ny',            Ny)
    store_my_data(parameters, 'Nz',            Nz)
    store_my_data(parameters, 'width',         width)
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

