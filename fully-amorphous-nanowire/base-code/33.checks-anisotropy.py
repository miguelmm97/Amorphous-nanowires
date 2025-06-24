#%% modules set up

# Math and plotting
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm

# modules
from modules.functions import *
from modules.AmorphousLattice_3d import AmorphousLattice_3d
from modules.FullyAmorphousWire_kwant import promote_to_kwant_nanowire3d, spectrum, local_marker, local_marker_per_site_cross_section_KPM,  \
    local_marker_per_site_cross_section_KPM_try
from modules.colorbar_marker import *

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


#%% Variables

Nx, Ny, Nz = 8, 8, 20
r          = 1.3
width      = 0.3
t          = 1
eps        = 4 * t
lamb       = 1 * t
lamb_z     = 1.8 * t
params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}
flux_value = 0
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

#%% Main

# Fully amorphous wire
loger_main.info(f'Generating lattice')
lattice = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=Nz, w=width, r=r)
lattice.build_lattice(restrict_connectivity=False)
nanowire = promote_to_kwant_nanowire3d(lattice, params_dict, attach_leads=False).finalized()


# Local marker
loger_main.info('Calculating local marker through ED')
H = nanowire.hamiltonian_submatrix(params=dict(flux=flux_value, mu=0.))
eps, _, rho = spectrum(H)
site_pos = np.array([site.pos for site in nanowire.id_by_site])
x, y, z = site_pos[:, 0], site_pos[:, 1], site_pos[:, 2]
chiral_sym = np.kron(np.eye(len(x)), np.kron(sigma_z, sigma_z))
marker_ED = local_marker(x - 0.5 * (Nx - 1), y - 0.5 * (Ny - 1), z - 0.5 * (Nz - 1), rho, chiral_sym)
#
# # Cross section range we are interested in
z0, z1 = 4.5, 15.5
cond1 = z < z1
cond2 = z0 < z
cond = cond1 * cond2
marker_ED = marker_ED[cond]
# # x, y = x[cond], y[cond]

# Local marker through KPM + Stochastic trace algorithm
loger_main.info('Calculating local marker through KPM algorithm')
S = scipy.sparse.kron(np.eye(Nx * Ny * Nz), np.kron(sigma_z, sigma_z), format='csr')
marker_KPM_try, _, _, _ = local_marker_per_site_cross_section_KPM_try(nanowire, S, Nx, Ny, Nz, z0, z1, Ef=0, num_moments=500)
marker_KPM, x, y, z = local_marker_per_site_cross_section_KPM(nanowire, S, Nx, Ny, Nz, z0, z1, Ef=0, num_moments=500)
x, y = x - 0.5 * (Nx - 1),  y - 0.5 * (Ny - 1)

#%% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange']
marker_list=['o', 's', 'd', 'p', '*', 'h', '>', '<', 'X']
markersize = 5
fontsize=20

# Figure 1
fig1 = plt.figure(figsize=(10, 8))
gs = GridSpec(1, 4, figure=fig1, wspace=0.5, hspace=0.5)
ax1 = fig1.add_subplot(gs[0, 0])
ax2 = fig1.add_subplot(gs[0, 1])
ax3 = fig1.add_subplot(gs[0, 2])


# Defining a colormap
vmin, vmax = -1.5,  1.5
cbar_ticks = np.arange(np.ceil(vmin), np.floor(vmax) + 1, 1)
hex_list = ['#ff416d', '#ff7192', '#ffa0b6', '#ffd0db', '#ffffff', '#cfdaff', '#9fb6ff', '#6f91ff', '#3f6cff']
cmap = get_continuous_cmap(hex_list)
divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
colormap = cm.ScalarMappable(norm=divnorm, cmap=cmap)
gradient = np.linspace(vmin, vmax, 256).reshape(-1, 1)

# Defining the colorbar
cbar_ax = fig1.add_subplot(gs[0, 3])
divider = make_axes_locatable(cbar_ax)
cax = divider.append_axes("left", size="1%", pad=-20)
cbar_ax.set_axis_off()
cax.imshow(gradient, aspect='auto', cmap=cmap, norm=divnorm, origin='lower')
tick_locs = (cbar_ticks - vmin) / (vmax - vmin) * gradient.shape[0]
cax.set_yticks(tick_locs)
cax.set_yticklabels([f"{t:.0f}" for t in cbar_ticks])
cax.set_xticks([])
cax.set_ylabel('$\\nu(r)$', labelpad=10, fontsize=20)
cax.yaxis.set_label_position('right')
cax.tick_params(which='major', width=0.75, labelsize=fontsize)

# Figure 1
fig1.suptitle(f'$N= {Nx}$, $L= {Nz}$, $w={width :.2f}$', y=0.93, fontsize=20)
ax1.scatter(x, y, c=marker_ED, facecolor='white', edgecolor='black')
ax1.scatter(x, y, c=marker_ED, norm=divnorm, cmap=get_continuous_cmap(hex_list),  linewidths=2.5)
ax1.set_xlabel('$x$', fontsize=fontsize)
ax1.set_ylabel('$y$', fontsize=fontsize)
ax1.set_xlim(-np.max(x) - 0.2, np.max(x) + 0.2)
ax1.set_ylim(-np.max(y) - 0.2, np.max(y) + 0.2)

ax2.scatter(x, y, c=marker_KPM, facecolor='white', edgecolor='black')
ax2.scatter(x, y, c=marker_KPM, norm=divnorm, cmap=get_continuous_cmap(hex_list),  linewidths=2.5)
ax2.set_xlabel('$x$', fontsize=fontsize)
ax2.set_ylabel('$y$', fontsize=fontsize)
ax2.set_xlim(-np.max(x) - 0.2, np.max(x) + 0.2)
ax2.set_ylim(-np.max(y) - 0.2, np.max(y) + 0.2)

ax3.scatter(x, y, c=marker_KPM_try, facecolor='white', edgecolor='black')
ax3.scatter(x, y, c=marker_KPM_try, norm=divnorm, cmap=get_continuous_cmap(hex_list),  linewidths=2.5)
ax3.set_xlabel('$x$', fontsize=fontsize)
ax3.set_ylabel('$y$', fontsize=fontsize)
ax3.set_xlim(-np.max(x) - 0.2, np.max(x) + 0.2)
ax3.set_ylim(-np.max(y) - 0.2, np.max(y) + 0.2)

plt.show()