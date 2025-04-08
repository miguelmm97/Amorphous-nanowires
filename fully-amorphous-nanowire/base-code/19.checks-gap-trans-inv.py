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
We check that the fully amorphous wire reduces to the translation invariant one.
"""

Nx, Ny, Nz       = 10, 10, 200                    # Number of sites in the cross-section
r                = 1.3                            # Nearest-neighbour cutoff distance
t                = 1                              # Hopping
eps              = 4 * t                          # Onsite orbital hopping (in units of t)
lamb             = 1 * t                          # Spin-orbit coupling in the cross-section (in units of t)
lamb_z           = 1.8 * t                        # Spin-orbit coupling along z direction
mu_leads         = -1 * t                         # Chemical potential at the leads
flux             = 0.                             # Magnetic flux
width            = [0.2]                          # Amorphous width
Ef               = [0]                            # Fermi energy
params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}

kwant_nw_dict = {}
lattice_dict = {}
# gap_array = np.zeros((len(width), len(flux)), dtype=np.float64)
#%% Main

# Generate nanowires
for i, w in enumerate(width):
    loger_main.info('Generating translation invariance amorphous lattice...')
    lattice = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=w, r=r)
    lattice.build_lattice()
    lattice_dict[i] = lattice
    kwant_nw_dict[i] = promote_to_kwant_nanowire(lattice, Nz, params_dict).finalized()
    loger_main.info('Nanowire promoted to Kwant successfully.')


for key in lattice_dict.keys():
    S = kwant.smatrix(kwant_nw_dict[key], 0., params=dict(flux=flux, mu=0., mu_leads=-1))
    G = S.transmission(1, 0)
    # Gap
    nanowire = InfiniteNanowire_FuBerg(lattice=lattice_dict[key], t=t, eps=eps, lamb=lamb, lamb_z=lamb_z, flux=flux)
    nanowire.get_bands() # (k_0=0, k_end=0, Nk=1)
        # gap_array[key, i] = nanowire.get_gap()

#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


fig1 = plt.figure()
gs = GridSpec(1, 1, figure=fig1, wspace=0.3, hspace=0.4)
ax1 = fig1.add_subplot(gs[0, 0])

for key in nanowire.energy_bands.keys():
    ax1.plot(np.linspace(-pi, pi, 101), nanowire.energy_bands[key], color='dodgerblue', linewidth=0.5)
    # ax1.plot(kz, 0.04 * np.ones(kz.shape), '--', color='Black', alpha=0.2)
ax1.text(-0.1, 0, f'$G= {G :.2f}$')
ax1.set_xlabel('$k/a$')
ax1.set_ylabel('$E(k)/t$')
ax1.set_xlim(-0.2, 0.2)
ax1.set_ylim(-0.2, 0.2)
ax1.tick_params(which='major', width=0.75, labelsize=10)
ax1.tick_params(which='major', length=6, labelsize=10)
# 1ax.set(xticks=[-pi, -pi/2, 0, pi/2, pi], xticklabels=['$-\pi$', '$-\pi/2$', '$0$', '$\pi/2$', '$\pi$'])
# ax1.set_title(f'$\phi / \phi_0={phi :.2f}$')
# fig1.savefig(f'figures/bands-vs-flux.pdf', format='pdf', backend='pgf')
plt.show()
