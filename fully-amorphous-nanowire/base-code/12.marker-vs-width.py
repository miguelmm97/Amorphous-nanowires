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

#%% Variables

Nx, Ny, Nz = 10, 10, 25
r          = 1.3
width      = np.linspace(0.000001, 0.4, 10)
t          = 1
eps        = 4 * t
lamb       = 1 * t
lamb_z     = 1.8 * t
params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}
flux_value = 0
cutoff = 2.5

bulk_marker = np.zeros((len(width), ))

#%% Main

def bulk(x, y, z, local_marker, cutoff, include_last_layer=True):
    x_pos, y_pos = x - 0.5 * Nx, y - 0.5 * Ny
    cond1 = np.abs(x_pos) < cutoff
    cond2 = np.abs(y_pos) < cutoff
    cond = cond1 * cond2
    if not include_last_layer:
        cond3 = 0.5 < z
        cond4 = z < Nz - 1.5
        cond = cond1 * cond2 * cond3 * cond4
    return x[cond], y[cond], z[cond], local_marker[cond]

# Fully amorphous wire
for i, w in enumerate(width):

    lattice = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=Nz, w=w, r=r)
    lattice.build_lattice(restrict_connectivity=False)
    lattice.generate_disorder(K_hopp=0., K_onsite=0.)
    nanowire = promote_to_kwant_nanowire3d(lattice, params_dict, mu_leads=0, attach_leads=False).finalized()

    # Spectrum of the closed system
    H = nanowire.hamiltonian_submatrix(params=dict(flux=flux_value))
    eps, _, rho = spectrum(H)

    # Local marker
    site_pos = np.array([site.pos for site in nanowire.id_by_site])
    x, y, z = site_pos[:, 0], site_pos[:, 1], site_pos[:, 2]
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    chiral_sym = np.kron(np.eye(len(x)), np.kron(sigma_z, sigma_z))
    marker = local_marker(x, y, z, rho, chiral_sym)
    x_cut, y_cut, z_cut, marker_cut = bulk(x, y, z, marker, cutoff, include_last_layer=False)
    bulk_marker[i]= np.mean(marker_cut)
    loger_main.info(f'width: {i}/{len(width) - 1}, marker: {bulk_marker[i]}')



#%% Saving data
data_dir = '/home/mfmm/Projects/amorphous-nanowires/data/data-marker-vs-width'
file_list = os.listdir(data_dir)
expID = get_fileID(file_list, common_name='Exp')
filename = '{}{}{}'.format('Exp', expID, '.h5')
filepath = os.path.join(data_dir, filename)


with h5py.File(filepath, 'w') as f:

    # Simulation folder
    simulation = f.create_group('Simulation')
    store_my_data(simulation, 'local_marker', bulk_marker)
    store_my_data(simulation, 'width',   width)


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
    store_my_data(parameters, 'cutoff', cutoff)

    # Attributes
    attr_my_data(parameters, "Date",       str(date.today()))
    attr_my_data(parameters, "Code_path",  sys.argv[0])

loger_main.info('Data saved correctly')




