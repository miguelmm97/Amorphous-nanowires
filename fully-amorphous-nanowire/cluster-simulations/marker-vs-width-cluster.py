#%% Modules and setup

# Math and plotting
import numpy as np

# Kwant
import kwant

# modules
from modules.functions import *
from modules.AmorphousLattice_3d import AmorphousLattice_3d, take_cut_from_parent_wire
from modules.FullyAmorphousWire_kwant import promote_to_kwant_nanowire3d, spectrum, local_marker

# Utilities
import argparse
import sys
from datetime import date


# Arguments to submit to the cluster
parser = argparse.ArgumentParser(description='Local marker vs width')
parser.add_argument('-l', '--line', type=int, help='Select line number', default=None)
parser.add_argument('-f', '--file', type=str, help='Select file name', default='params.txt')
parser.add_argument('-M', '--outdir', type=str, help='Select the base name of the output file', default='outdir')
parser.add_argument('-o', '--outbase', type=str, help='Select the base name of the output file', default='exp')
args = parser.parse_args()



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

Nz         = 15
Nx         = np.arange(6, 13)
Ny         = Nx
r          = 1.3
width      = np.linspace(1e-5, 0.5, 15)
t          = 1
eps        = 4 * t
lamb       = 1 * t
lamb_z     = 1.8 * t
params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}
flux_value = 0

# Preallocation
Nsites = int(Nz * np.max(Nx) * np.max(Ny))
marker = np.zeros((len(width), len(Nx), Nsites))
X = np.zeros((len(width), len(Nx), Nsites))
Y = np.zeros((len(width), len(Nx), Nsites))
Z = np.zeros((len(width), len(Nx), Nsites))
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

#%% Main

for i, w in enumerate(width):

    # Generating lattice structure of the wire
    full_lattice = AmorphousLattice_3d(Nx=np.max(Nx), Ny=np.max(Ny), Nz=Nz, w=w, r=r)
    full_lattice.build_lattice()
    full_lattice.generate_disorder(K_onsite=0., K_hopp=0)

    for j, n in enumerate(Nx):

        # Selecting different cuts of the wire for each disorder realisation
        lattice = take_cut_from_parent_wire(full_lattice, Nx_new=n, Ny_new=n, Nz_new=Nz, keep_disorder=True)
        nanowire = promote_to_kwant_nanowire3d(lattice, params_dict, mu_leads=0, attach_leads=False).finalized()

        # Spectrum of the closed system
        H = nanowire.hamiltonian_submatrix(params=dict(flux=flux_value))
        eps, _, rho = spectrum(H)

        # Local marker
        site_pos = np.array([site.pos for site in nanowire.id_by_site])
        x, y, z = site_pos[:, 0], site_pos[:, 1], site_pos[:, 2]
        X[i, j, :len(x)], Y[i, j, :len(x)], Z[i, j, :len(x)] = x, y, z
        chiral_sym = np.kron(np.eye(len(x)), np.kron(sigma_z, sigma_z))
        marker[i, j, :len(x)] = local_marker(x, y, z, rho, chiral_sym)
        loger_main.info(f'width: {i}/{len(width) - 1}, Nx: {j}/{len(Nx) - 1}')



#%% Saving data
outfile = '{}-{}.h5'.format(args.outbase, args.line)
filepath = os.path.join(args.outdir, outfile)


with h5py.File(filepath, 'w') as f:

    # Simulation folder
    simulation = f.create_group('Simulation')
    store_my_data(simulation, 'local_marker', marker)
    store_my_data(simulation, 'x', X)
    store_my_data(simulation, 'y', Y)
    store_my_data(simulation, 'z', Z)

    # Parameters folder
    parameters = f.create_group('Parameters')
    store_my_data(parameters, 'flux',  flux_value)
    store_my_data(parameters, 'width', width)
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

