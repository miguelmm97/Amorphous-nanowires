#%% Modules setup

# Math and plotting
import numpy as np
import matplotlib.pyplot as plt

# Modules
from modules.functions import *
from modules.AmorphousLattice_2d import AmorphousLattice_2d
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
We take a look at different realisations of amorphous wires and their Aharanov Bohm oscillations.
"""
Nx, Ny    = 10, 10     # Number of sites in the cross-section
width     = 0.1        # Spread of the Gaussian distribution for the lattice sites
r         = 1.3        # Nearest-neighbour cutoff distance
t         = 1          # Hopping
eps       = 4 * t      # Onsite orbital hopping (in units of t)
lamb      = 1 * t      # Spin-orbit coupling in the cross-section (in units of t)
lamb_z    = 1.8 * t    # Spin-orbit coupling along z direction

# Preallocation
flux  = np.linspace(0., 6, 10, dtype=np.float64)
gap1 = np.zeros((len(flux), ))
gap2 = np.zeros((len(flux), ))
gap3 = np.zeros((len(flux), ))
gap4 = np.zeros((len(flux), ))

#%% Main

# Different wire configurations
cross_section1 = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=width, r=r)
cross_section1.build_lattice()

cross_section2 = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=width, r=r)
cross_section2.build_lattice()

cross_section3 = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=width, r=r)
cross_section3.build_lattice()

cross_section4 = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=width, r=r)
cross_section4.build_lattice()


for i, phi in enumerate(flux):
    loger_main.info(f'flux: {i}/{len(flux)-1}')

    # Configuration 1
    wire1 = InfiniteNanowire_FuBerg(lattice=cross_section1, t=t, eps=eps, lamb=lamb, lamb_z=lamb_z, flux=phi)
    wire1.get_bands(k_0=0, k_end=0, Nk=1)
    gap1[i] = wire1.get_gap()

    # Configuration 2
    wire2 = InfiniteNanowire_FuBerg(lattice=cross_section2, t=t, eps=eps, lamb=lamb, lamb_z=lamb_z, flux=phi)
    wire2.get_bands(k_0=0, k_end=0, Nk=1)
    gap2[i] = wire2.get_gap()

    # Configuration 3
    wire3 = InfiniteNanowire_FuBerg(lattice=cross_section3, t=t, eps=eps, lamb=lamb, lamb_z=lamb_z, flux=phi)
    wire3.get_bands(k_0=0, k_end=0, Nk=1)
    gap3[i] = wire3.get_gap()

    # Configuration 4
    wire4 = InfiniteNanowire_FuBerg(lattice=cross_section4, t=t, eps=eps, lamb=lamb, lamb_z=lamb_z, flux=phi)
    wire4.get_bands(k_0=0, k_end=0, Nk=1)
    gap4[i] = wire4.get_gap()
#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125', '#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']

fig1 = plt.figure(figsize=(6, 6))
ax1 = fig1.gca()
cross_section1.plot_lattice(ax1)
ax1.set_title(f'$w=$ {width}, $r=$ {r}')

fig2 = plt.figure(figsize=(6, 6))
ax2 = fig2.gca()
cross_section2.plot_lattice(ax2)
ax2.set_title(f'$w=$ {width}, $r=$ {r}')

fig3 = plt.figure(figsize=(10, 6))
ax3 = fig3.gca()
# ax2.plot(flux, gap, '.', color=color_list[3], markersize=7)
ax3.plot(flux, gap1, color=color_list[8], linewidth=1)
ax3.plot(flux, gap2, color=color_list[7], linewidth=1)
ax3.plot(flux, gap3, color=color_list[6], linewidth=1)
ax3.plot(flux, gap4, color=color_list[5], linewidth=1)
ax3.set_xlabel('$\phi/\phi_0$')
ax3.set_ylabel('$E_g$')
ax3.set_xlim(flux[0], flux[-1])
ax3.set_ylim(0, np.max([gap1, gap2, gap3, gap4]) + 0.1 * np.max([gap1, gap2, gap3, gap4]))
ax3.tick_params(which='major', width=0.75, labelsize=10)
ax3.tick_params(which='major', length=6, labelsize=10)
fig2.suptitle(f'$N_x=N_y=$ {Nx}, $w=$ {width}, $r=$ {r}, $\epsilon=$ {eps}, $\lambda=$ {lamb}, $\lambda_z=$ {lamb_z}')
fig3.savefig("amorphous-AB-oscillations-7.pdf")
plt.show()