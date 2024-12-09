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
from modules.FullyAmorphousWire_kwant import promote_to_kwant_nanowire3d, select_perfect_transmission_flux

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

#%% Loading data
file_list = ['Exp23.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-cond-vs-flux-fully-amorphous')

# Parameters
Nx           = data_dict[file_list[0]]['Parameters']['Nx']
Ny           = data_dict[file_list[0]]['Parameters']['Ny']
Nz           = data_dict[file_list[0]]['Parameters']['Nz']
r            = data_dict[file_list[0]]['Parameters']['r']
t            = data_dict[file_list[0]]['Parameters']['t']
eps          = data_dict[file_list[0]]['Parameters']['eps']
lamb         = data_dict[file_list[0]]['Parameters']['lamb']
lamb_z       = data_dict[file_list[0]]['Parameters']['lamb_z']
mu_leads     = data_dict[file_list[0]]['Parameters']['mu_leads']
params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}

# Simulation data
x             = data_dict[file_list[0]]['Simulation']['x']
y             = data_dict[file_list[0]]['Simulation']['y']
z             = data_dict[file_list[0]]['Simulation']['z']
width         = data_dict[file_list[0]]['Simulation']['width']
flux          = data_dict[file_list[0]]['Simulation']['flux']


# Variables
idx =33
flux_value = flux[idx]


#%% Main

# Fully amorphous wire
loger_main.info('Generating fully amorphous lattice...')
lattice = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=Nz, w=width, r=r)
lattice.set_configuration(x, y, z)
lattice.build_lattice(restrict_connectivity=False)
lattice.generate_disorder(K_hopp=0., K_onsite=0.)
nanowire = promote_to_kwant_nanowire3d(lattice, params_dict, mu_leads=0, attach_leads=False).finalized()
loger_main.info('Nanowire promoted to Kwant successfully.')

# Spectrum of the closed system
H = nanowire.hamiltonian_submatrix(params=dict(flux=flux_value), sparse=True)
eps, psi = sla.eigsh(H.tocsc(), k=50, sigma=0)
# H = nanowire.hamiltonian_submatrix(params=dict(flux=flux_value))
# eps, psi = np.linalg.eigh(H)

idx = eps.argsort()
eps = eps[idx]
psi = psi[:, idx]


#%% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange']



fig1 = plt.figure()
gs = GridSpec(1, 1, figure=fig1, wspace=0, hspace=0)
ax1 = fig1.add_subplot(gs[0, 0])

ax1.plot(np.arange(len(eps)), eps, 'o', color='dodgerblue')
ax1.set_xlabel('Eigenstate')
ax1.set_ylabel('$\epsilon$')
plt.show()