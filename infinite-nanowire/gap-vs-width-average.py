#%% Modules setup

# Math and plotting
from numpy import pi
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

# Tracking time
import time

# Managing logging
import logging
import colorlog
from colorlog import ColoredFormatter

# Managing data
import h5py
import os
import sys
from datetime import date

# Modules
from InfiniteNanowire import InfiniteNanowire_FuBerg
from functions import get_fileID, store_my_data, attr_my_data

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

w_0, w_end, Nw = 0.00000001, 0.15, 100    # Flux array parameters
Nsamples   = 1000                 # Number of amorphous lattice samples
Nx, Ny     = 10, 10               # Number of sites in the cross-section
flux       = 0.55                 # Spread of the Gaussian distribution for the lattice sites
r          = 1.3                  # Nearest-neighbour cutoff distance
t          = 1                    # Hopping
eps        = 4 * t                # Onsite orbital hopping (in units of t)
lamb       = 1 * t                # Spin-orbit coupling in the cross-section (in units of t)
lamb_z     = 1.8 * t              # Spin-orbit coupling along z direction

#%% Main
width = np.linspace(w_0, w_end, Nw, dtype=np.float64)
gap = np.zeros((Nw, Nsamples))
n_rejected = np.zeros((Nw, ))

for i, w in enumerate(width):
    for sample in range(Nsamples):
        loger_main.info(f'width: {i}/{Nw - 1}, Sample: {sample}/{Nsamples - 1}')
        wire = InfiniteNanowire_FuBerg(Nx=Nx, Ny=Ny, w=w, r=r, flux=flux, t=t, eps=eps, lamb=lamb, lamb_z=lamb_z)
        try:
            wire.build_lattice()
            wire.get_boundary()
            wire.get_bands(k_0=0, k_end=0, Nk=1)
            gap[i, sample] = wire.get_gap()
        except ValueError as error:
            loger_main.error(f'Something went wrong: {error}')

#%% Saving data

file_list = os.listdir('../../data-width-gap')
expID = get_fileID(file_list, common_name='width-gap')
filename = '{}{}{}'.format('width-gap', expID, '.h5')
filepath = os.path.join('../../data-width-gap', filename)

with h5py.File(filepath, 'w') as f:

    # Simulation folder
    simulation = f.create_group('Simulation')
    store_my_data(simulation, 'gap', gap)
    store_my_data(simulation, 'n_rejected', n_rejected)
    store_my_data(simulation, 'width', flux)


    # Parameters folder
    parameters = f.create_group('Parameters')
    store_my_data(parameters, 'Nsamples',     Nsamples)
    store_my_data(parameters, 'w_0',          w_0)
    store_my_data(parameters, 'w_end',        w_end)
    store_my_data(parameters, 'Nw',           Nw)
    store_my_data(parameters, 'Nx',           Nx)
    store_my_data(parameters, 'Ny',           Ny)
    store_my_data(parameters, 'flux',         flux)
    store_my_data(parameters, 'r',            r)
    store_my_data(parameters, 't',            t)
    store_my_data(parameters, 'eps',          eps)
    store_my_data(parameters, 'lamb',         lamb)
    store_my_data(parameters, 'lamb_z',       lamb_z)

    # Attributes
    attr_my_data(parameters, "Date",       str(date.today()))
    attr_my_data(parameters, "Code_path",  sys.argv[0])