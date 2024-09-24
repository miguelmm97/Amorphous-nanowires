#%% modules set up

# Math and plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from numpy import pi

# Kwant
import kwant

# modules
from modules.functions import *
from modules.AmorphousLattice_3d import AmorphousLattice_3d
from modules.FullyAmorphousWire_kwant import promote_to_kwant_nanowire3d
from modules.AmorphousLattice_2d import AmorphousLattice_2d
from modules.AmorphousWire_kwant import promote_to_kwant_nanowire

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
We check that the fully amorphous wire reduces to the translation invariant one.
"""

Nx, Ny, Nz = 5, 5, 20                    # Number of sites in the cross-section
width      = 0.1                        # Spread of the Gaussian distribution for the lattice sites
r          = 1.3                        # Nearest-neighbour cutoff distance
t          = 1                          # Hopping
eps        = 4 * t                      # Onsite orbital hopping (in units of t)
lamb       = 1 * t                      # Spin-orbit coupling in the cross-section (in units of t)
lamb_z     = 1.8 * t                    # Spin-orbit coupling along z direction
mu_leads   = 0 * t                      # Chemical potential at the leads
fermi = np.linspace(0, 1, 2)
params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}


#%% Check structure

# Fully amorphous wire
loger_main.info('Generating fully amorphous lattice...')
lattice = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=Nz, w=width, r=1.3)
lattice.build_lattice()
nanowire1 = promote_to_kwant_nanowire3d(lattice, params_dict, mu_leads=mu_leads).finalized()
loger_main.info('Nanowire promoted to Kwant successfully.')


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

fig0 = plt.figure(figsize=(10, 4))
gs = GridSpec(1, 2, figure=fig0)
ax0_0 = fig0.add_subplot(gs[0, 0], projection='3d')
ax0_1 = fig0.add_subplot(gs[0, 1])

lattice.plot_lattice(ax0_0)
lattice.plot_lattice_projection(ax0_1)

fig1 = plt.figure(figsize=(6, 6))
gs = GridSpec(1, 1, figure=fig1)
ax1 = fig1.add_subplot(gs[0, 0], projection='3d')
kwant.plot(nanowire1, site_size=site_size, site_lw=site_lw, site_color=site_color, hop_lw=hop_lw, hop_color=hop_color,
           lead_site_size=site_size, lead_color=lead_color, lead_site_lw=site_lw, lead_hop_lw=hop_lw, ax=ax1)
ax1.set_axis_off()
ax1.margins(-0.49, -0.49, -0.49)


#%% Check limiting case of conductance

# Crystalline wire using our Amorphous module
loger_main.info('Generating amorphous cross section:')
cross_section = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=0.000000001, r=1.3)
cross_section.build_lattice()
nanowire2 = promote_to_kwant_nanowire(cross_section, Nz, params_dict).finalized()
loger_main.info('Nanowire promoted to Kwant successfully.')

# Fully amorphous wire
loger_main.info('Generating fully amorphous lattice...')
lattice = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=Nz, w=0.000000001, r=1.3)
lattice.build_lattice()
nanowire3 = promote_to_kwant_nanowire3d(lattice, params_dict, mu_leads=mu_leads).finalized()
loger_main.info('Nanowire promoted to Kwant successfully.')

G_cryst_0 = np.zeros(fermi.shape)
G_cryst_half = np.zeros(fermi.shape)
G_amorphous_0 = np.zeros(fermi.shape)
G_amorphous_half = np.zeros(fermi.shape)

for i, Ef in enumerate(fermi):
    loger_main.info(f'Calculating conductance for Ef: {i} / {fermi.shape[0] - 1}...')

    # Module nanowire
    S1 = kwant.smatrix(nanowire2, Ef, params=dict(flux=0.))
    G_cryst_0[i] = S1.transmission(1, 0)

    S2 = kwant.smatrix(nanowire2, Ef, params=dict(flux=0.56))
    G_cryst_half[i] = S2.transmission(1, 0)

    # Kwant nanowire
    S3 = kwant.smatrix(nanowire3, Ef, params=dict(flux=0.))
    G_amorphous_0[i] = S3.transmission(1, 0)

    S4 = kwant.smatrix(nanowire3, Ef, params=dict(flux=0.56))
    G_amorphous_half[i] = S4.transmission(1, 0)



fig2 = plt.figure(figsize=(4, 6))
gs = GridSpec(1, 2, figure=fig2)
ax2_1 = fig2.add_subplot(gs[0, 0])
ax2_2 = fig2.add_subplot(gs[0, 1])

# 0 flux
ax2_1.plot(fermi, G_cryst_0, 'o', color='#9A32CD', markersize=5, label='T. I model')
ax2_1.plot(fermi, G_amorphous_0, color='#3F6CFF', alpha=0.5, label=f'amorphous model ')
ax2_1.set_title('$\phi / \phi_0=0$')

# half flux
ax2_2.plot(fermi, G_cryst_half, 'o', color='#9A32CD', markersize=5, label='T.I model')
ax2_2.plot(fermi, G_amorphous_half, color='#3F6CFF', alpha=0.5, label=f'amorphous model ')
ax2_2.set_title('$\phi / \phi_0=0.56$')


y_axis_ticks = [i for i in range(0, 10, 2)]
y_axis_labels = [str(i) for i in range(0, 10, 2)]
for ax in [ax2_1, ax2_2]:
    ax.set_xlim(fermi[0], fermi[-1])
    ax.set_ylim(0, 10)
    ax.tick_params(which='major', width=0.75, labelsize=10)
    ax.tick_params(which='major', length=6, labelsize=10)
    ax.set_xlabel("$E_F$ [$t$]", fontsize=10)
    ax.set_ylabel("$G[2e^2/h]$",fontsize=10)
    ax.set(yticks=y_axis_ticks, yticklabels=y_axis_labels)
    ax.legend(ncol=1, frameon=False, fontsize=16)
plt.show()