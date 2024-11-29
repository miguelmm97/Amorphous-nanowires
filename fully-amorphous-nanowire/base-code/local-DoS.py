#%% modules set up

# Math and plotting
import numpy as np
import sys
from datetime import date

# Kwant
import kwant

# modules
from modules.functions import *
from modules.AmorphousLattice_3d import AmorphousLattice_3d
from modules.FullyAmorphousWire_kwant import promote_to_kwant_nanowire3d, select_perfect_transmission_flux

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm

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
Nx, Ny, Nz       = 10, 10, 25                 # Number of sites in the cross-section
width            = 0.0001                     # Spread of the Gaussian distribution for the lattice sites
r                = 1.3                        # Nearest-neighbour cutoff distance
t                = 1                          # Hopping
eps              = 4 * t                      # Onsite orbital hopping (in units of t)
lamb             = 1 * t                      # Spin-orbit coupling in the cross-section (in units of t)
lamb_z           = 1.8 * t                    # Spin-orbit coupling along z direction
mu_leads         = - 1 * t                    # Chemical potential at the leads
fermi            = 0.1                        # Fermi Energy
K_hopp           = 0.                         # Hopping disorder
K_onsite         = 0.                         # Onsite disorder
params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}


#%% Main

loger_main.info('Generating fully amorphous lattice...')
lattice = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=Nz, w=width, r=r)
lattice.build_lattice(restrict_connectivity=False)
lattice.generate_disorder(K_hopp=K_hopp, K_onsite=K_onsite)
nanowire = promote_to_kwant_nanowire3d(lattice, params_dict, mu_leads=mu_leads).finalized()
loger_main.info('Nanowire promoted to Kwant successfully.')

# Calculating the scattering wavefunctions at certain energies
loger_main.info('Calculating scattering wave functions...')
top_state = kwant.wave_function(nanowire, energy=0., params=dict(flux=0.5))
triv_state = kwant.wave_function(nanowire, energy=0.1, params=dict(flux=0.))
loger_main.info('Scattering wave functions calculated successfully')

# Calculating local density of states
loger_main.info('Calculating local Dos...')
local_density = kwant.operator.Density(nanowire)
top_density = local_density(top_state(0)[0])
triv_density = local_density(triv_state(0)[0])
max_value = np.max(np.array([top_density, triv_density]))
r = np.array([site.pos for site in nanowire.id_by_site])



font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Defining a colormap without negatives
color_map = plt.get_cmap("gist_heat_r")
colors = color_map(np.linspace(0, 1, 20))
colors[0] = [1, 1, 1, 1]
color_map = LinearSegmentedColormap.from_list("custom_colormap", colors)
colormap = cm.ScalarMappable(norm=Normalize(vmin=0, vmax=max_value), cmap=color_map)


fig1 = plt.figure()
gs = GridSpec(1, 3, figure=fig1, wspace=0.2, hspace=0.0)
ax1 = fig1.add_subplot(gs[0, 0], projection='3d')
ax2 = fig1.add_subplot(gs[0, 1], projection='3d')

ax1.scatter(r[:, 0], r[:, 1], r[:, 2], c=top_density, cmap=color_map, vmin=0, vmax=max_value)
ax1.get_proj = lambda: np.dot(Axes3D.get_proj(ax1), np.diag([0.5, 0.5, 1, 1]))
ax1.set_axis_off()

ax2.scatter(r[:, 0], r[:, 1], r[:, 2], c=triv_density, cmap=color_map, vmin=0, vmax=max_value)
ax2.get_proj = lambda: np.dot(Axes3D.get_proj(ax2), np.diag([0.5, 0.5, 1, 1]))
ax2.set_axis_off()


cbar_ax = fig1.add_subplot(gs[0, 2])
divider = make_axes_locatable(cbar_ax)
cax = divider.append_axes("left", size="10%", pad=0)
cbar = fig1.colorbar(colormap, cax=cax, orientation='vertical')
cbar_ax.set_axis_off()
cbar.set_label(label='$\\vert \psi \\vert ^2$', fontsize=20)


# plt.colorbar()


# kwant.plotter.density(nanowire, top_density)
# kwant.plotter.density(nanowire, triv_density)
plt.show()