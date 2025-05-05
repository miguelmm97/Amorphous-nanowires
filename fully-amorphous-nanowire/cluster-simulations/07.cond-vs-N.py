#%% modules set up

# Math and plotting
import numpy as np

# Kwant
import kwant

# modules
from modules.functions import *
from modules.AmorphousLattice_3d import AmorphousLattice_3d
from modules.FullyAmorphousWire_kwant import promote_to_kwant_nanowire3d
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
variables = config.variables_cond_vs_N

width       = variables['width']
Nz          = variables['Nz']
r           = variables['r']
t           = variables['t']
mu_leads    = variables['mu_leads']
flux_max    = variables['flux_max']
flux_min    = variables['flux_min']
flux_L      = variables['flux_L']
Ef          = variables['Ef']
K_onsite    = variables['K_onsite']
eps         = 4 * t
lamb        = 1 * t
lamb_z      = 1.8 * t
mu_leads    = mu_leads * t
flux        = np.linspace(flux_min, flux_max, flux_L)
params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}


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
loger_main.info('Generating amorphous lattice...')
lattice = AmorphousLattice_3d(Nx=N, Ny=N, Nz=Nz, w=width, r=r)
lattice.build_lattice()
lattice.generate_disorder(K_onsite=K_onsite, K_hopp=0.)
nanowire = promote_to_kwant_nanowire3d(lattice, params_dict).finalized()

# Calculate conductance
Gmax = 0.
for j, phi in enumerate(flux):
    S = kwant.smatrix(nanowire, 0., params=dict(flux=phi, mu=-Ef, mu_leads=mu_leads - Ef))
    G = S.transmission(1, 0)
    loger_main.info(f'N: {N}, flux: {phi :.2f}, {j} / {len(flux) - 1} || G: {G :.2f}')
    if G > Gmax:
        Gmax = G

#%% Saving data

outfile = '{}-{}.h5'.format(args.outbase, args.line)
filepath = os.path.join(args.outdir, outfile)

loger_main.info('Saving data...')
with h5py.File(filepath, 'w') as f:

    # Simulation folder
    simulation = f.create_group('Simulation')
    store_my_data(simulation, 'N',             N)
    store_my_data(simulation, 'Ef',            Ef)
    store_my_data(simulation, 'flux',          flux)
    store_my_data(simulation, 'width',         width)
    store_my_data(simulation, 'Gmax',          Gmax)
    store_my_data(simulation, 'sample',        sample)
    store_my_dict(simulation['Disorder'],      lattice.disorder)

    # Parameters folder
    parameters = f.create_group('Parameters')
    store_my_data(simulation, 'Nz',            Nz)
    store_my_data(parameters, 'K_onsite',      K_onsite)
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

