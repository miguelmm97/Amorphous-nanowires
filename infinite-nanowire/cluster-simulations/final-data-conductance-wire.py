import os
import h5py
import numpy as np
from modules.functions import *

"""
Generate final data file from the raw data.
"""
print('Opening files...')
#%% Loading raw data
data_dir = 'data'
Nsamples = len(os.listdir(data_dir))

# Parameters (same for every file in the simulation) and preallocation
sample_file_path = os.path.join(data_dir, 'exp-0.h5')
with h5py.File(sample_file_path, 'r') as f:

    # Simulation parameters
    Nx         = f['Parameters/Nx'][()]
    Ny         = f['Parameters/Ny'][()]
    n_layers   = f['Parameters/n_layers'][()]
    width      = f['Parameters/width'][()]
    r          = f['Parameters/r'][()]
    t          = f['Parameters/t'][()]
    eps        = f['Parameters/eps'][()]
    lamb       = f['Parameters/lamb'][()]
    lamb_z     = f['Parameters/lamb_z'][()]
    mu_leads   = f['Parameters/mu_leads'][()]
    flux0      = f['Parameters/flux0'][()]
    flux_half  = f['Parameters/flux_half'][()]

    # Preallocation
    kz         = f['Simulation/kz'][()]
    fermi      = f['Simulation/fermi'][()]
    G0         = np.zeros((len(fermi), Nsamples))
    G_half     = np.zeros((len(fermi), Nsamples))

print('loaded...')

# Data from the simulation
n = 0
for file in os.listdir(data_dir):
    file_path = os.path.join(data_dir, file)
    print('file:', file)

    with h5py.File(file_path, 'r') as f:
        G0[:, n]      = f['Simulation/G0'][()]
        G_half[:, n]  = f['Simulation/G_half'][()]
    n += 1


#%% Data analysis
avg_G0      = np.mean(G0, axis=1)
avg_G_half  = np.mean(G_half, axis=1)
std_G0      = np.std(G0, axis=1)
std_G_half  = np.std(G_half, axis=1)

#%% Saving data
outfile = 'final-data-conductance-wire.h5'
filepath = outfile

with h5py.File(filepath, 'w') as f:

    # Simulation folder
    simulation = f.create_group('Simulation')
    store_my_data(simulation, 'fermi',        fermi)
    store_my_data(simulation, 'kz',           kz)
    store_my_data(simulation, 'avg_G0',       avg_G0)
    store_my_data(simulation, 'avg_G_half',   avg_G_half)
    store_my_data(simulation, 'std_G0',       std_G0)
    store_my_data(simulation, 'std_G_half',   std_G_half)


    # Parameters folder
    parameters = f.create_group('Parameters')
    store_my_data(parameters, 'Nsamples',   Nsamples)
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
print('Done...')