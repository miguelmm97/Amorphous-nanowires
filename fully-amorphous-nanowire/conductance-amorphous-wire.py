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
from modules.FullyAmorphousWire_kwant import promote_to_kwant_nanowire3d, select_perfect_transmission_flux

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
We calculate the conductance of the nanowire vs Fermi energy.
"""

Nx, Ny, Nz       = 5, 5, 20                   # Number of sites in the cross-section
width            = 0.01                        # Spread of the Gaussian distribution for the lattice sites
r                = 1.3                        # Nearest-neighbour cutoff distance
t                = 1                          # Hopping
eps              = 4 * t                      # Onsite orbital hopping (in units of t)
lamb             = 1 * t                      # Spin-orbit coupling in the cross-section (in units of t)
lamb_z           = 1.8 * t                    # Spin-orbit coupling along z direction
mu_leads         = 0 * t                      # Chemical potential at the leads
fermi            = np.linspace(0, 2, 2)       # Fermi energy
params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}

# Preallocation
G_0    = np.zeros(fermi.shape)
G_half = np.zeros(fermi.shape)

#%% Main

# Fully amorphous wire
loger_main.info('Generating fully amorphous lattice...')
lattice = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=Nz, w=width, r=r)
lattice.build_lattice()
nanowire = promote_to_kwant_nanowire3d(lattice, params_dict, mu_leads=mu_leads).finalized()
loger_main.info('Nanowire promoted to Kwant successfully.')

# Scanning for flux that gives perfect transmission at the Dirac point
flux_max, Gmax = select_perfect_transmission_flux(nanowire)
loger_main.info(f'Flux for perfect transmission: {flux_max}, COnductance at the Dirac point: {Gmax}')

# Conductance calculation for different flux values
for i, Ef in enumerate(fermi):
    loger_main.info(f'Calculating conductance for Ef: {i} / {fermi.shape[0] - 1}...')
    S0 = kwant.smatrix(nanowire, Ef, params=dict(flux=0.))
    G_0[i] = S0.transmission(1, 0)
    S1 = kwant.smatrix(nanowire, Ef, params=dict(flux=flux_max))
    G_half[i] = S1.transmission(1, 0)


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
fig0 = plt.figure(figsize=(10, 4))
gs = GridSpec(1, 2, figure=fig0)
ax0_1 = fig0.add_subplot(gs[0, 0], projection='3d')
ax0_2 = fig0.add_subplot(gs[0, 1])
fig0_1 = plt.figure(figsize=(2, 2))
gs = GridSpec(1, 1, figure=fig0_1)
ax0_3 = fig0_1.add_subplot(gs[0, 0], projection='3d')


lattice.plot_lattice(ax0_1)
lattice.plot_lattice_projection(ax0_2)
kwant.plot(nanowire, site_size=site_size, site_lw=site_lw, site_color=site_color, hop_lw=hop_lw, hop_color=hop_color,
           lead_site_size=site_size, lead_color=lead_color, lead_site_lw=site_lw, lead_hop_lw=hop_lw, ax=ax0_3)
ax0_3.set_axis_off()
ax0_3.margins(-0.49, -0.49, -0.49)



# Conductance vs Fermi level
fig1 = plt.figure(figsize=(10, 10))
gs = GridSpec(1, 1, figure=fig1)
ax1_1 = fig1.add_subplot(gs[0, 0])

ax1_1.plot(fermi, G_0, color='#9A32CD', label=f'$\phi / \phi_0= {0.}$')
ax1_1.plot(fermi, G_half, color='#3F6CFF', alpha=0.5, label=f'$\phi / \phi_0= {flux_max :.2f}$ ')
ax1_1.legend(ncol=1, frameon=False, fontsize=16)

ylim = 10
y_axis_ticks = [i for i in range(0, ylim, 2)]
y_axis_labels = [str(i) for i in range(0, ylim, 2)]
for ax in [ax1_1]:
    ax.set_xlim(fermi[0], fermi[-1])
    ax.set_ylim(0, ylim)
    ax.tick_params(which='major', width=0.75, labelsize=10)
    ax.tick_params(which='major', length=6, labelsize=10)
    ax.set_xlabel("$E_F$ [$t$]", fontsize=10)
    ax.set_ylabel("$G[2e^2/h]$",fontsize=10)
    ax.set(yticks=y_axis_ticks, yticklabels=y_axis_labels)

fig1.suptitle(f'$\mu_l= {mu_leads}$, $w= {width}$, $r= {r}$, $N_x= {Nx}$, $N_y = {Ny}$, $N_z= {Nz}$', y=0.93, fontsize=20)
fig1.savefig('fully-amorphous-AB-osc.pdf', format='pdf', backend='pgf')
plt.show()
