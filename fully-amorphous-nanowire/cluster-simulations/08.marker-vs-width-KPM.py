#%% modules set up

# Math and plotting
import numpy as np
import scipy.sparse

# Kwant
import kwant

# modules
from modules.functions import *
from modules.AmorphousLattice_3d import AmorphousLattice_3d
from modules.FullyAmorphousWire_kwant import promote_to_kwant_nanowire3d, local_marker_KPM
import config
import sys
from datetime import date

# Cluster managing
import argparse
import h5py
import os

# Arguments to submit to the cluster
parser = argparse.ArgumentParser(description='')
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

# Static variables for the simulation group
loger_main.info('Loading variables from the .toml file')
variables = config.variables_marker_vs_width_KPM

cutoff      = 0.4 * 0.5
width       = variables['width']
Nz          = variables['Nz']
r           = variables['r']
t           = variables['t']
Ef          = variables['Ef']
num_moments = variables['num_moments']
num_vecs    = variables['num_vecs']
eps         = 4 * t
lamb        = 1 * t
lamb_z      = 1.8 * t
params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


# Input data from SLURM
loger_main.info('Loading variables from the parameter file')
if args.line is not None:
    print("Line number:", args.line)
    with open(args.file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i == args.line:
                params = line.split()
                N = params[0]
                sample = params[2]
else:
    raise IOError("No line number was given")
if N or sample is None:
    raise ValueError("Not loading input parameters")

#%% Main

bulk_marker = np.zeros((len(width), ))

for i, w in enumerate(width):

    loger_main.info(f'Generating lattice for w: {w}')
    lattice = AmorphousLattice_3d(Nx=N, Ny=N, Nz=Nz, w=w, r=r)
    lattice.build_lattice(restrict_connectivity=False)
    nanowire = promote_to_kwant_nanowire3d(lattice, params_dict, attach_leads=False).finalized()
    S = scipy.sparse.kron(np.eye(N * N * Nz), np.kron(sigma_z, sigma_z), format='csr')

    # Local marker through KPM + Stochastic trace algorithm
    loger_main.info('Calculating bulk marker through KPM algorithm')
    bulk_marker[i] = local_marker_KPM(nanowire, S, N, N, Nz, Ef=0., num_moments=num_moments, num_vecs=num_vecs, bounds=None)
    loger_main.info(f'width: {i}/{len(width) - 1}, marker KPM: {bulk_marker[i] :.5f}')


#%% Saving data
outfile = '{}-{}.h5'.format(args.outbase, args.line)
filepath = os.path.join(args.outdir, outfile)

with h5py.File(filepath, 'w') as f:

    # Simulation folder
    simulation = f.create_group('Simulation')
    store_my_data(simulation, 'bulk_marker', bulk_marker)
    store_my_data(simulation, 'width',   width)
    store_my_data(simulation, 'num_moments', num_moments)
    store_my_data(simulation, 'num_vecs', num_vecs)


    # Parameters folder
    parameters = f.create_group('Parameters')
    store_my_data(parameters, 'N',       N)
    store_my_data(parameters, 'Nz',      Nz)
    store_my_data(parameters, 'r ',      r)
    store_my_data(parameters, 't ',      t)
    store_my_data(parameters, 'eps',     eps)
    store_my_data(parameters, 'lamb',    lamb)
    store_my_data(parameters, 'lamb_z',  lamb_z)
    store_my_data(parameters, 'cutoff',  cutoff)

    # Attributes
    attr_my_data(parameters, "Date",       str(date.today()))
    attr_my_data(parameters, "Code_path",  sys.argv[0])

loger_main.info('Data saved correctly')




