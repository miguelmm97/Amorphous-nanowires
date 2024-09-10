#%% Modules set up

# Math
import numpy as np
from numpy import pi

# Kwant
import kwant

# Modules
from modules.functions import *
from modules.AmorphousLattice_2d import AmorphousLattice_2d
from modules.AmorphousWire_kwant import promote_to_kwant_nanowire, infinite_nanowire_kwant
from modules.InfiniteNanowire import InfiniteNanowire_FuBerg

# Cluster managing
import argparse
import h5py
import os


# Arguments to submit to the cluster
parser = argparse.ArgumentParser(description='Argument parser for the XYZ model simulation')
parser.add_argument('-l', '--line', type=int, help='Select line number', default=None)
parser.add_argument('-f', '--file', type=str, help='Select file name', default='params.txt')
parser.add_argument('-M', '--outdir', type=str, help='Select the base name of the output file', default='outdir')
parser.add_argument('-o', '--outbase', type=str, help='Select the base name of the output file', default='outXYZ')
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
"""
We calculate the conductance and bands for an amorphous cross-section wire.
"""

Nx, Ny    = 5, 5                     # Number of sites in the cross-section
n_layers  = 120                        # Number of cross-section layers
width     = 0.1                        # Spread of the Gaussian distribution for the lattice sites
r         = 1.3                        # Nearest-neighbour cutoff distance
t         = 1                          # Hopping
eps       = 4 * t                      # Onsite orbital hopping (in units of t)
lamb      = 1 * t                      # Spin-orbit coupling in the cross-section (in units of t)
lamb_z    = 1.8 * t                    # Spin-orbit coupling along z direction
mu_leads  = -1 * t                     # Chemical potential at the leads
flux0     = 0.                         # O flux
flux_half = 0.56                       # Close to half flux
fermi     = np.linspace(0, 1, 2)     # Fermi level for calculating the conductance
kz        = np.linspace(-pi, pi, 101)  # Transversal momentum to the wire
params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}


# Preallocation
G_0 = np.zeros(fermi.shape)
G_half = np.zeros(fermi.shape)
bottom_0 = np.zeros((int(Nx * Ny * 4), ))
bottom_half = np.zeros((int(Nx * Ny * 4), ))
bands_0 = np.zeros((int(Nx * Ny * 4), len(kz)))
bands_half = np.zeros((int(Nx * Ny * 4), len(kz)))

#%% Main

# Initiaise amorphous nanowire for transport
loger_main.info('Generating amorphous cross section...')
cross_section = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=width, r=1.3)
cross_section.build_lattice()
nanowire = promote_to_kwant_nanowire(cross_section, n_layers, params_dict, mu_leads=mu_leads).finalized()
loger_main.info('Nanowire promoted to Kwant successfully.')


# Conductance calculation for different flux values
for i, Ef in enumerate(fermi):
    loger_main.info(f'Calculating conductance for Ef: {i} / {fermi.shape[0] - 1}...')
    S0 = kwant.smatrix(nanowire, Ef, params=dict(flux=flux0))
    G_0[i] = S0.transmission(1, 0)
    S1 = kwant.smatrix(nanowire, Ef, params=dict(flux=flux_half))
    G_half[i] = S1.transmission(1, 0)


# Calculating bands in the scattering region
loger_main.info(f'Calculating bands for the nanowires...')
nanowire_0 = InfiniteNanowire_FuBerg(lattice=cross_section, t=t, eps=eps, lamb=lamb, lamb_z=lamb_z, flux=flux0)
nanowire_0.get_bands()
nanowire_half = InfiniteNanowire_FuBerg(lattice=cross_section, t=t, eps=eps, lamb=lamb, lamb_z=lamb_z, flux=flux_half)
nanowire_half.get_bands()

# Bottom and storage
index0 = int(np.floor(len(nanowire_0.energy_bands[0]) / 2))
for i in nanowire_0.energy_bands.keys():
    bands_0[i, :] = nanowire_0.energy_bands[i]
    bands_half[i, :] = nanowire_half.energy_bands[i]
    bottom_0[i] = nanowire_0.energy_bands[i][index0]
    bottom_half[i] = nanowire_half.energy_bands[i][index0]


# Calculating bands in the leads
loger_main.info(f'Calculating bands for the leads...')
lead = infinite_nanowire_kwant(Nx, Ny, params_dict, mu_leads=mu_leads).finalized()
bands = kwant.physics.Bands(lead, params=dict(flux=flux0))
bands_lead0 = [bands(k) for k in kz]
bottom_lead_0 = bands(0)

bands = kwant.physics.Bands(lead, params=dict(flux=flux_half))
bands_lead_half = [bands(k) for k in kz]
bottom_lead_half = bands(0)




#%% Saving data
outfile = '{}-{}.h5'.format(args.outbase, args.line)
filepath = os.path.join(args.outdir, outfile)

with h5py.File(filepath, 'w') as f:

    # Simulation folder
    simulation = f.create_group('Simulation')
    store_my_data(simulation, 'fermi',               fermi)
    store_my_data(simulation, 'kz',                  kz)
    store_my_data(simulation, 'G0',                  G_0)
    store_my_data(simulation, 'G_half',              G_half)
    store_my_data(simulation, 'bands_0',             bands_0)
    store_my_data(simulation, 'bands_half',          bands_half)
    store_my_data(simulation, 'bands_lead_0',        bands_lead0)
    store_my_data(simulation, 'bands_lead_half',     bands_lead_half)
    store_my_data(simulation, 'bottom_0',            bottom_0)
    store_my_data(simulation, 'bottom_half',         bottom_half)
    store_my_data(simulation, 'bottom_lead_0',       bottom_lead_0)
    store_my_data(simulation, 'bottom_lead_half',    bottom_lead_half)
    store_my_data(simulation, 'x_pos',               cross_section.x)
    store_my_data(simulation, 'y_pos',               cross_section.y)


    # Parameters folder
    parameters = f.create_group('Parameters')
    store_my_data(parameters, 'Nx',         Nx)
    store_my_data(parameters, 'Ny',         Ny)
    store_my_data(parameters, 'n_layers',   n_layers)
    store_my_data(parameters, 'width',      width)
    store_my_data(parameters, 'r',          r)
    store_my_data(parameters, 't',          t)
    store_my_data(parameters, 'eps',        eps)
    store_my_data(parameters, 'lamb',       lamb)
    store_my_data(parameters, 'lamb_z',     lamb_z)
    store_my_data(parameters, 'mu_leads',   mu_leads)
    store_my_data(parameters, 'flux0',      flux0)
    store_my_data(parameters, 'flux_half',  flux_half)


    # Attributes
    attr_my_data(parameters, "Date",       str(date.today()))
    attr_my_data(parameters, "Code_path",  sys.argv[0])

loger_main.info('Data saved correctly')




