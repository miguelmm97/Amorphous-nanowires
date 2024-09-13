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
from AmorphousWire_kwant import promote_to_kwant_nanowire
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
params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}
mu_leads = 0 * t


#%% Main

# Initiaise amorphous nanowire for transport
loger_main.info('Generating amorphous lattice...')
lattice = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=Nz, w=width, r=1.3)
lattice.build_lattice()
nanowire = promote_to_kwant_nanowire(lattice, params_dict, mu_leads=mu_leads).finalized()
loger_main.info('Nanowire promoted to Kwant successfully.')



site_size  = 0.1
site_lw    = 0.01
site_color = 'm'
hop_color  = 'royalblue'
hop_lw     = 0.05
lead_color = 'r'


fig0 = plt.figure(figsize=(10, 4))
gs = GridSpec(1, 2, figure=fig0)
ax0_0 = fig0.add_subplot(gs[0, 0], projection='3d')
ax0_1 = fig0.add_subplot(gs[0, 1])

lattice.plot_lattice(ax0_0)
lattice.plot_lattice_projection(ax0_1)


fig1 = plt.figure(figsize=(6, 6))
gs = GridSpec(1, 1, figure=fig1)
ax1 = fig1.add_subplot(gs[0, 0], projection='3d')
kwant.plot(nanowire, site_size=site_size, site_lw=site_lw, site_color=site_color, hop_lw=hop_lw, hop_color=hop_color,
           lead_site_size=site_size, lead_color=lead_color, lead_site_lw=site_lw, lead_hop_lw=hop_lw,
           ax=ax1)
ax1.set_axis_off()
ax1.margins(-0.49, -0.49, -0.49)

plt.show()