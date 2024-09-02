#%% Modules set up

# Math
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi

# Kwant
import kwant

# Modules
from modules.functions import *
from modules.AmorphousLattice_2d import AmorphousLattice_2d
from modules.AmorphousWire_kwant import promote_to_kwant_nanowire, crystal_nanowire_kwant

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

Nx, Ny    = 5, 5                    # Number of sites in the cross-section
n_layers  = 10                      # Number of cross-section layers
width     = 0.1                     # Spread of the Gaussian distribution for the lattice sites
r         = 1.3                     # Nearest-neighbour cutoff distance
flux      = 0.0                     # Flux threaded through the cross-section (in units of flux quantum)
t         = 1                       # Hopping
eps       = 4 * t                   # Onsite orbital hopping (in units of t)
lamb      = 1 * t                   # Spin-orbit coupling in the cross-section (in units of t)
lamb_z    = 1.8 * t                 # Spin-orbit coupling along z direction
k = np.linspace(-pi, pi, 10001)     # Momentum along the regular direction

params_dict = {
    't': t,
    'eps': eps,
    'lamb': lamb,
    'lamb_z': lamb_z,
    'flux': flux
}

#%% Comparison of two crystalline wires

# Crystalline wire through the amorphous construction
loger_main.info('Generating amorphous cross section:')
cross_section = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=0.00001, r=1.3)
cross_section.build_lattice()
nanowire = promote_to_kwant_nanowire(cross_section, n_layers, params_dict)
nanowire = nanowire.finalized()
loger_main.info('Nanowire promoted to Kwant successfully.')

# Crystalline wire directly through kwant
nanowire_kwant = crystal_nanowire_kwant(Nx, Ny, n_layers + 2, params_dict)
nanowire_kwant = nanowire_kwant.finalized()


# Conductance calculation
fermi = np.linspace(0, 2, 50)
G = np.zeros(fermi.shape)
G_kwant = np.zeros(fermi.shape)
for i, Ef in enumerate(fermi):
    loger_main.info(f'Calculating conductance for Ef: {i} / {fermi.shape[0] - 1}...')
    S = kwant.smatrix(nanowire, Ef)
    G[i] = S.transmission(1, 0)
    S_kwant = kwant.smatrix(nanowire_kwant, Ef)
    G_kwant[i] = S_kwant.transmission(1, 0)



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


fig0 = plt.figure()
ax0 = fig0.gca()
cross_section.plot_lattice(ax0)


fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
kwant.plot(nanowire, site_size=site_size, site_lw=site_lw, site_color=site_color, hop_lw=hop_lw, hop_color=hop_color,
           lead_site_size=site_size, lead_color=lead_color, lead_site_lw=site_lw, lead_hop_lw=hop_lw,
           ax=ax1)
ax1.set_axis_off()


fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
kwant.plot(nanowire_kwant, site_size=site_size, site_lw=site_lw, site_color=site_color, hop_lw=hop_lw, hop_color=hop_color,
           lead_site_size=site_size, lead_color=lead_color, lead_site_lw=site_lw, lead_hop_lw=hop_lw,
           ax=ax2)
ax2.set_axis_off()

fig3 = plt.figure()
ax3 = fig3.gca()
ax3.plot(fermi, G, color='#3F6CFF')
ax3.plot(fermi, G_kwant, 'o', color='#00B5A1', alpha=0.5, markersize=5)
ax3.set_xlim(fermi[0], fermi[-1])
ax3.set_ylim(0, np.max(G))
ax3.tick_params(which='major', width=0.75, labelsize=10)
ax3.tick_params(which='major', length=6, labelsize=10)
ax3.set_xlabel("$E_F$ [$t$]", fontsize=10)
ax3.set_ylabel("$G[2e^2/h]$",fontsize=10)
plt.show()