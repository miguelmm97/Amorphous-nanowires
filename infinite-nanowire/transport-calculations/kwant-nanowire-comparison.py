#%% Modules set up

# Math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Kwant
import kwant
import tinyarray as ta

# Modules
from modules.functions import *
from modules.AmorphousLattice_2d import AmorphousLattice_2d
from modules.AmorphousWire_kwant import promote_to_transport_nanowire

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
We compare the structure and bands of the kwant nanowire and the one generated with our own class.
"""

Nx, Ny    = 10, 10       # Number of sites in the cross-section
n_layers  = 100         # Number of cross-section layers
width     = 0.1        # Spread of the Gaussian distribution for the lattice sites
r         = 1.3        # Nearest-neighbour cutoff distance
flux      = 2.3        # Flux threaded through the cross-section (in units of flux quantum)
t         = 1          # Hopping
eps       = 4 * t      # Onsite orbital hopping (in units of t)
lamb      = 1 * t      # Spin-orbit coupling in the cross-section (in units of t)
lamb_z    = 1.8 * t    # Spin-orbit coupling along z direction

params_dict = {
    't': t,
    'eps': eps,
    'lamb': lamb,
    'lamb_z': lamb_z,
    'flux': flux
}

#%% Kwant wire
loger_main.info('Generating amorphous cross section:')
cross_section = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=width, r=r)
cross_section.build_lattice()
nanowire_kwant = promote_to_transport_nanowire(cross_section, n_layers, params_dict)
loger_main.info('Nanowire promoted to Kwant succesfully.')

site_size  = 0.1
site_lw    = 0.01
site_color = 'm'
hop_color  = 'royalblue'
hop_lw     = 0.05
lead_color = 'r'

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
kwant.plot(nanowire_kwant, site_size=site_size, site_lw=site_lw, site_color=site_color, hop_lw=hop_lw, hop_color=hop_color,
           lead_site_size=site_size, lead_color=lead_color, lead_site_lw=site_lw, lead_hop_lw=hop_lw,
           ax=ax)
ax.set_axis_off()
plt.show()
