#%% modules set up

# Math and plotting
import numpy as np

# Kwant
import kwant

# modules
from modules.functions import *
from modules.AmorphousLattice_3d import AmorphousLattice_3d, take_cut_from_parent_wire
from modules.FullyAmorphousWire_kwant import promote_to_kwant_nanowire3d, spectrum, local_marker_1d

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

Nz         = 50
Nx         = 4
Ny         = 4
r          = 1.3
width      = np.linspace(1e-5, 0.5, 15)
t          = 1
eps        = 4 * t
lamb       = 1 * t
lamb_z     = 1.8 * t
params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}
flux = [1.3]  # np.linspace(0, 5, 25)

# Preallocation
Nsites = Nx * Ny * Nz
marker = np.zeros((len(width), len(flux), Nz))
G = np.zeros((len(width), len(flux)))
X = np.zeros((len(width), len(flux), Nsites))
Y = np.zeros((len(width), len(flux), Nsites))
Z = np.zeros((len(width), len(flux), Nsites))
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

#%% Main

loger_main.info(f'Starting calculation...')
for i, w in enumerate(width):

    # Generating lattice structure of the wire
    lattice = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=Nz, w=w, r=r)
    lattice.build_lattice()
    lattice.generate_disorder(K_onsite=0., K_hopp=0)
    nanowire_leads = promote_to_kwant_nanowire3d(lattice, params_dict, attach_leads=True).finalized()
    nanowire_closed = promote_to_kwant_nanowire3d(lattice, params_dict, attach_leads=False).finalized()


    for j, phi in enumerate(flux):

        # Conductance
        S0 = kwant.smatrix(nanowire_leads, 0., params=dict(flux=phi, mu=0., mu_leads=0.))
        G[i, j] = S0.transmission(1, 0)

        # Spectrum of the closed system
        H = nanowire_closed.hamiltonian_submatrix(params=dict(flux=phi, mu=0., mu_leads=0.))
        eps, _, rho = spectrum(H)

        # Local marker
        site_pos = np.array([site.pos for site in nanowire_closed.id_by_site])
        x, y, z = site_pos[:, 0], site_pos[:, 1], site_pos[:, 2]
        X[i, j, :len(x)], Y[i, j, :len(x)], Z[i, j, :len(x)] = x, y, z
        chiral_sym = np.kron(np.eye(len(x)), np.kron(sigma_z, sigma_z))
        marker[i, j, :len(x)] = local_marker_1d(z, rho, chiral_sym, Nx, Ny, Nz)

        # Outputs
        loger_main.info(f'width: {i}/{len(width) - 1}, flux: {j}/{len(flux) - 1}, '
                        f'marker: {np.mean(marker[i, j, 10: Nz - 10]) :.2f}, '
                        f'Conductance: {G[i, j] :.2f}')
        loger_main.debug(f'Checking symmetry... SHS=-H: {np.allclose(chiral_sym @ H @ chiral_sym, -H)}, '
                         f'PS + SP = S: {np.allclose(rho @ chiral_sym + chiral_sym @ rho, chiral_sym)} ')



#%% Saving data
# data_dir = '/home/mfmm/Projects/amorphous-nanowires/data/data-marker-vs-cross-section'
# file_list = os.listdir(data_dir)
# expID = get_fileID(file_list, common_name='Exp')
# filename = '{}{}{}'.format('Exp', expID, '.h5')
# filepath = os.path.join(data_dir, filename)
#
#
# with h5py.File(filepath, 'w') as f:
#
#     # Simulation folder
#     simulation = f.create_group('Simulation')
#     store_my_data(simulation, 'local_marker', marker)
#     store_my_data(simulation, 'x', X)
#     store_my_data(simulation, 'y', Y)
#     store_my_data(simulation, 'z', Z)
#
#     # Parameters folder
#     parameters = f.create_group('Parameters')
#     store_my_data(parameters, 'flux',   flux)
#     store_my_data(parameters, 'width', width)
#     store_my_data(parameters, 'Nx',      Nx)
#     store_my_data(parameters, 'Ny',      Ny)
#     store_my_data(parameters, 'Nz',      Nz)
#     store_my_data(parameters, 'r ',      r)
#     store_my_data(parameters, 't ',      t)
#     store_my_data(parameters, 'eps',     eps)
#     store_my_data(parameters, 'lamb',    lamb)
#     store_my_data(parameters, 'lamb_z',  lamb_z)
#
#     # Attributes
#     attr_my_data(parameters, "Date",       str(date.today()))
#     attr_my_data(parameters, "Code_path",  sys.argv[0])
#
# loger_main.info('Data saved correctly')
#
