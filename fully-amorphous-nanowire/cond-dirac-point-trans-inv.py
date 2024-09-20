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

Nx, Ny, Nz       = 5, 5, 50                   # Number of sites in the cross-section
r                = 1.3                        # Nearest-neighbour cutoff distance
t                = 1                          # Hopping
eps              = 4 * t                      # Onsite orbital hopping (in units of t)
lamb             = 1 * t                      # Spin-orbit coupling in the cross-section (in units of t)
lamb_z           = 1.8 * t                    # Spin-orbit coupling along z direction
mu_leads         = 1 * t                      # Chemical potential at the leads
flux             = np.linspace(0, 3, 500)     # Magnetic flux
width            = [0.0001, 0.02, 0.05, 0.1]  # Amorphous width
params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}

kwant_nw_dict = {}
lattice_dict = {}
G_array = np.zeros((len(width), len(flux)), dtype=np.float64)
gap_array = np.zeros((len(width), len(flux)), dtype=np.float64)
#%% Main

# Generate nanowires
for i, w in enumerate(width):
    loger_main.info('Generating translation invariance amorphous lattice...')
    lattice = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=w, r=r)
    lattice.build_lattice()
    lattice_dict[i] = lattice
    kwant_nw_dict[i] = promote_to_kwant_nanowire(lattice, Nz, params_dict, mu_leads=mu_leads).finalized()
    loger_main.info('Nanowire promoted to Kwant successfully.')


dirac_point = 0.0
for key in kwant_nw_dict.keys():
    for j, phi in enumerate(flux):

        # Conductance
        S = kwant.smatrix(kwant_nw_dict[key], dirac_point, params=dict(flux=phi))
        G_array[key, j] = S.transmission(1, 0)

        # Gap
        nanowire = InfiniteNanowire_FuBerg(lattice=lattice_dict[key], t=t, eps=eps, lamb=lamb, lamb_z=lamb_z, flux=phi)
        nanowire.get_bands(k_0=0, k_end=0, Nk=1)
        gap_array[key, j] = nanowire.get_gap()

        loger_main.info(f'Width: {key} / {len(width) - 1}, flux: {j} / {len(flux)} || G: {G_array[key, j] :.2e}, '
                        f'gap: {gap_array[key, j] :.4e}')


#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125', '#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']
markersize = 5

# Figure 1: Definition
fig1 = plt.figure(figsize=(10, 10))
gs = GridSpec(1, 1, figure=fig1)
ax1 = fig1.add_subplot(gs[0, 0])

# Figure 1: Plots
for key in kwant_nw_dict.keys():
    ax1.plot(flux, G_array[key, :], color=color_list[key], label=f'$w= {width[key]}$')
    ax1.plot(flux, gap_array[key, :], color=color_list[key], linestyle='dashed', alpha=0.2)
ax1.plot(flux, 1 * np.ones(flux.shape),  color='Black', alpha=0.5)

# Figure 1: Format
ax1.legend(ncol=2, frameon=False, fontsize=16)
fig1.suptitle(f'$\mu_l= {mu_leads}$, $r= {r}$, $N_x= {Nx}$, $N_y = {Ny}$, $N_z= {Nz}$', y=0.93, fontsize=20)
ylim = 1.3
for ax in [ax1]:
    ax.set_xlim(flux[0], flux[-1])
    ax.set_ylim(0, ylim)
    ax.tick_params(which='major', width=0.75, labelsize=10)
    ax.tick_params(which='major', length=6, labelsize=10)
    ax.set_xlabel("$\phi$", fontsize=10)
    ax.set_ylabel("$G[2e^2/h]$", fontsize=10)

fig1.savefig('conductance-dirac-point_try.pdf', format='pdf', backend='pgf')
plt.show()



