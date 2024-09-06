#%% Modules set up

# Math
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
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

Nx, Ny    = 10, 10              # Number of sites in the cross-section
n_layers  = 20                  # Number of cross-section layers
width     = 0.1                 # Spread of the Gaussian distribution for the lattice sites
r         = 1.3                 # Nearest-neighbour cutoff distance
t         = 1                   # Hopping
eps       = 4 * t               # Onsite orbital hopping (in units of t)
lamb      = 1 * t               # Spin-orbit coupling in the cross-section (in units of t)
lamb_z    = 1.8 * t             # Spin-orbit coupling along z direction
mu_leads  = 0                   # Chemical potential at the leads
fermi = np.linspace(0, 2, 500)  # Fermi level for calculating the conductance
params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}

# Preallocation
G_0 = np.zeros(fermi.shape)
G_half = np.zeros(fermi.shape)

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
    S0 = kwant.smatrix(nanowire, Ef, params=dict(flux=0.))
    G_0[i] = S0.transmission(1, 0)
    S1 = kwant.smatrix(nanowire, Ef, params=dict(flux=0.56))
    G_half[i] = S1.transmission(1, 0)


# Calculating bands in the scattering region
loger_main.info(f'Calculating bands for the nanowires...')
nanowire_0 = InfiniteNanowire_FuBerg(lattice=cross_section, t=t, eps=eps, lamb=lamb, lamb_z=lamb_z, flux=0.)
nanowire_0.get_bands()
nanowire_half = InfiniteNanowire_FuBerg(lattice=cross_section, t=t, eps=eps, lamb=lamb, lamb_z=lamb_z, flux=0.56)
nanowire_half.get_bands()


# Calculating bands in the leads
loger_main.info('Generating lead cross section...')
lead_cross_section = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=0.00000000001, r=1.3)
lead_cross_section.build_lattice()

loger_main.info(f'Calculating bands for the leads...')
lead0 = InfiniteNanowire_FuBerg(lattice=lead_cross_section, t=t, eps=eps, lamb=lamb, lamb_z=lamb_z, flux=0.)
lead0.get_bands()
lead_half = InfiniteNanowire_FuBerg(lattice=lead_cross_section, t=t, eps=eps, lamb=lamb, lamb_z=lamb_z, flux=0.56)
lead_half.get_bands()

index0 = int(np.floor(len(lead0.energy_bands[0]) / 2))
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

# Nanowire structure
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
kwant.plot(nanowire, site_size=site_size, site_lw=site_lw, site_color=site_color, hop_lw=hop_lw, hop_color=hop_color,
           lead_site_size=site_size, lead_color=lead_color, lead_site_lw=site_lw, lead_hop_lw=hop_lw,
           ax=ax1)
ax1.set_axis_off()



# Conductance vs Fermi level
fig2 = plt.figure(figsize=(6, 10))
gs = GridSpec(1, 3, figure=fig2)
ax2_1 = fig2.add_subplot(gs[0, 0])
ax2_2 = fig2.add_subplot(gs[0, 1])
ax2_3 = fig2.add_subplot(gs[0, 2])

ax2_1.plot(fermi, G_0, color='#9A32CD', label='$\phi / \phi_0=0$')
for i in nanowire_0.energy_bands.keys():
    ax2_1.plot(nanowire_0.energy_bands[i][index0] * np.ones((10, )), np.linspace(0, 100, 10), '--', color='#9A32CD', alpha=0.1)
    ax2_1.plot(lead0.energy_bands[i][index0] * np.ones((10,)), np.linspace(0, 100, 10), '--', color='#FF7D66', alpha=0.1)

ax2_2.plot(fermi, G_half, color='#3F6CFF', alpha=0.5, label=f'$\phi / \phi_0=0.56$ ')
for i in nanowire_half.energy_bands.keys():
    ax2_2.plot(nanowire_half.energy_bands[i][index0] * np.ones((10, )), np.linspace(0, 100, 10), '--', color='#9A32CD', alpha=0.1)
    ax2_2.plot(lead_half.energy_bands[i][index0] * np.ones((10,)), np.linspace(0, 100, 10), '--', color='#FF7D66', alpha=0.1)

ax2_3.plot(fermi, G_0, color='#9A32CD', label='$\phi / \phi_0=0$')
ax2_3.plot(fermi, G_half, color='#3F6CFF', alpha=0.5, label=f'$\phi / \phi_0=0.56$ ')
ax2_3.legend(ncol=1, frameon=False, fontsize=16)

y_axis_ticks = [i for i in range(0, 30, 2)]
y_axis_labels = [str(i) for i in range(0, 30, 2)]
for ax in [ax2_1, ax2_2, ax2_3]:
    ax.set_xlim(fermi[0], fermi[-1])
    ax.set_ylim(0, 30)
    ax.tick_params(which='major', width=0.75, labelsize=10)
    ax.tick_params(which='major', length=6, labelsize=10)
    ax.set_xlabel("$E_F$ [$t$]", fontsize=10)
    ax.set_ylabel("$G[2e^2/h]$",fontsize=10)
    ax.set(yticks=y_axis_ticks, yticklabels=y_axis_labels)

# fig2.savefig('AB-osc.pdf', format='pdf', backend='pgf')



# Bands for the module
fig4 = plt.figure(figsize=(10, 10))
gs = GridSpec(2, 2, figure=fig4)
ax4_1 = fig4.add_subplot(gs[0, 0])
ax4_2 = fig4.add_subplot(gs[0, 1])
ax4_3 = fig4.add_subplot(gs[1, 0])
ax4_4 = fig4.add_subplot(gs[1, 1])

for i in nanowire_0.energy_bands.keys():
    ax4_1.plot(nanowire_0.kz, nanowire_0.energy_bands[i], color='#3F6CFF', linewidth=0.5)
    ax4_2.plot(nanowire_half.kz, nanowire_half.energy_bands[i], color='#3F6CFF', linewidth=0.5)
    ax4_3.plot(lead0.kz, lead0.energy_bands[i], color='#3F6CFF', linewidth=0.5)
    ax4_4.plot(lead_half.kz, lead_half.energy_bands[i], color='#3F6CFF', linewidth=0.5)

ax4_1.set_title(f'$\phi / \phi_0=0$ wire')
ax4_2.set_title(f'$\phi / \phi_0=0.56$ wire')
ax4_3.set_title(f'$\phi / \phi_0=0$ leads')
ax4_4.set_title(f'$\phi / \phi_0=0.56$ leads')

for ax in [ax4_1, ax4_2, ax4_3, ax4_4]:
    ax.set_xlabel('$k/a$')
    ax.set_ylabel('$E(k)/t$')
    ax.set_xlim(-pi, pi)
    ax.tick_params(which='major', width=0.75, labelsize=10)
    ax.tick_params(which='major', length=6, labelsize=10)
    ax.set(xticks=[-pi, -pi/2, 0, pi/2, pi], xticklabels=['$-\pi$', '$-\pi/2$', '$0$', '$\pi/2$', '$\pi$'])




plt.show()
