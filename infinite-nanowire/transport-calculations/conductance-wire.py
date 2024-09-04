#%% Modules set up

# Math
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

# Kwant
import kwant

# Modules
from modules.functions import *
from modules.AmorphousLattice_2d import AmorphousLattice_2d
from modules.AmorphousWire_kwant import promote_to_kwant_nanowire
from modules.InfiniteNanowire import InfiniteNanowire_FuBerg

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

Nx, Ny    = 10, 10                # Number of sites in the cross-section
n_layers  = 20                  # Number of cross-section layers
width     = 0.1                 # Spread of the Gaussian distribution for the lattice sites
r         = 1.3                 # Nearest-neighbour cutoff distance
t         = 1                   # Hopping
eps       = 4 * t               # Onsite orbital hopping (in units of t)
lamb      = 1 * t               # Spin-orbit coupling in the cross-section (in units of t)
lamb_z    = 1.8 * t             # Spin-orbit coupling along z direction
fermi = np.linspace(0, 2, 50)   # Fermi level for calculating the conductance
params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}

# Preallocation
G_0 = np.zeros(fermi.shape)
G_half = np.zeros(fermi.shape)

#%% Main

# Initiaise amorphous nanowire for transport
loger_main.info('Generating amorphous cross section:')
cross_section = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=width, r=1.3)
cross_section.build_lattice()
nanowire = promote_to_kwant_nanowire(cross_section, n_layers, params_dict).finalized()
loger_main.info('Nanowire promoted to Kwant successfully.')

# Conductance calculation for different flux values
for i, Ef in enumerate(fermi):
    loger_main.info(f'Calculating conductance for Ef: {i} / {fermi.shape[0] - 1}...')
    S0 = kwant.smatrix(nanowire, Ef, params=dict(flux=0.))
    G_0[i] = S0.transmission(1, 0)
    S1 = kwant.smatrix(nanowire, Ef, params=dict(flux=0.56))
    G_half[i] = S1.transmission(1, 0)

# Calculating bottom energy of every band
loger_main.info(f'Calculating bottom bands...')
nanowire_infinite_0 = InfiniteNanowire_FuBerg(lattice=cross_section, t=t, eps=eps, lamb=lamb, lamb_z=lamb_z, flux=0.)
bottom_bands_0 = nanowire_infinite_0.get_bands(k_0=0, k_end=0, Nk=1, extract=True)[0]
nanowire_infinite_0.get_bands()

nanowire_infinite_half = InfiniteNanowire_FuBerg(lattice=cross_section, t=t, eps=eps, lamb=lamb, lamb_z=lamb_z, flux=0.56)
bottom_bands_half = nanowire_infinite_half.get_bands(k_0=0, k_end=0, Nk=1, extract=True)[0]
nanowire_infinite_half.get_bands()

#%% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
axcolour = ['#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125']

site_size  = 0.1
site_lw    = 0.01
site_color = 'm'
hop_color  = 'royalblue'
hop_lw     = 0.05
lead_color = 'r'

# Cross section snippet
fig0 = plt.figure()
ax0 = fig0.gca()
cross_section.plot_lattice(ax0)

# Nnaowire structure
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
kwant.plot(nanowire, site_size=site_size, site_lw=site_lw, site_color=site_color, hop_lw=hop_lw, hop_color=hop_color,
           lead_site_size=site_size, lead_color=lead_color, lead_site_lw=site_lw, lead_hop_lw=hop_lw,
           ax=ax1)
ax1.set_axis_off()


# Conductance vs Fermi level
fig2 = plt.figure()
ax2 = fig2.gca()
ax2.plot(fermi, G_0, color='#9A32CD', label='$\phi / \phi_0=0$')
ax2.plot(fermi, G_half, color='#3F6CFF', alpha=0.5, label=f'$\phi / \phi_0=0.56$ ')
for i in bottom_bands_0.keys():
    ax2.plot(bottom_bands_0[i] * np.ones((10, )), np.linspace(0, 100, 10), '--', color='#9A32CD', alpha=0.1)
for i in bottom_bands_half.keys():
    ax2.plot(bottom_bands_half[i] * np.ones((10,)), np.linspace(0, 100, 10), '--', color='#3F6CFF', alpha=0.1)

y_axis_ticks = [i for i in range(0, 100, 2)]
y_axis_labels = [str(i) for i in range(0, 100, 2)]
ax2.set_xlim(fermi[0], fermi[-1])
ax2.set_ylim(0, np.max(G_0))
ax2.tick_params(which='major', width=0.75, labelsize=10)
ax2.tick_params(which='major', length=6, labelsize=10)
ax2.set_xlabel("$E_F$ [$t$]", fontsize=10)
ax2.set_ylabel("$G[2e^2/h]$",fontsize=10)
ax2.legend(ncol=1, frameon=False, fontsize=16)
ax2.set(yticks=y_axis_ticks, yticklabels=y_axis_labels)


fig4 = plt.figure(figsize=(6, 6))
gs = GridSpec(1, 2, figure=fig4)
ax4_1 = fig4.add_subplot(gs[0, 0])
ax4_2 = fig4.add_subplot(gs[0, 1])

# Bands for the module
for i in nanowire_infinite_0.energy_bands.keys():
    ax4_1.plot(nanowire_infinite_0.kz, nanowire_infinite_0.energy_bands[i], color='#3F6CFF', linewidth=0.5)
    ax4_2.plot(nanowire_infinite_half.kz, nanowire_infinite_half.energy_bands[i], color='#3F6CFF', linewidth=0.5)

ax4_1.set_xlabel('$k/a$')
ax4_1.set_ylabel('$E(k)/t$')
ax4_1.set_xlim(-pi, pi)
ax4_1.tick_params(which='major', width=0.75, labelsize=10)
ax4_1.tick_params(which='major', length=6, labelsize=10)
ax4_1.set(xticks=[-pi, -pi/2, 0, pi/2, pi], xticklabels=['$-\pi$', '$-\pi/2$', '$0$', '$\pi/2$', '$\pi$'])
ax4_1.set_title(f'$\phi / \phi_0=0$')

ax4_2.set_xlabel('$k/a$')
ax4_2.set_ylabel('$E(k)/t$')
ax4_2.set_xlim(-pi, pi)
ax4_2.tick_params(which='major', width=0.75, labelsize=10)
ax4_2.tick_params(which='major', length=6, labelsize=10)
ax4_2.set(xticks=[-pi, -pi/2, 0, pi/2, pi], xticklabels=['$-\pi$', '$-\pi/2$', '$0$', '$\pi/2$', '$\pi$'])
ax4_2.set_title(f'$\phi / \phi_0=0.5$')
plt.show()