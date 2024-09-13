#%% Modules set up

# Math
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Kwant
import kwant

# Modules
from functions import *
from AmorphousLattice_3d import AmorphousLattice_3d

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

Nx, Ny, Nz = 5, 5, 10                    # Number of sites in the cross-section
width      = 0.1                        # Spread of the Gaussian distribution for the lattice sites
r          = 1.3                        # Nearest-neighbour cutoff distance
t          = 1                          # Hopping
eps        = 4 * t                      # Onsite orbital hopping (in units of t)
lamb       = 1 * t                      # Spin-orbit coupling in the cross-section (in units of t)
lamb_z     = 1.8 * t                    # Spin-orbit coupling along z direction


#%% Main

# Initiaise amorphous nanowire for transport
loger_main.info('Generating amorphous lattice...')
nanowire = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=Nz, w=width, r=1.3)
nanowire.build_lattice()


fig0 = plt.figure(figsize=(10, 4))
gs = GridSpec(1, 2, figure=fig0)
ax0_0 = fig0.add_subplot(gs[0, 0], projection='3d')
ax0_1 = fig0.add_subplot(gs[0, 1])

nanowire.plot_lattice(ax0_0)
nanowire.plot_lattice_projection(ax0_1)
plt.show()