#%% Modules setup

# Math and plotting
from numpy import pi
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

# Tracking time
import time

# Managing logging
import logging
import colorlog
from colorlog import ColoredFormatter

# Modules
from InfiniteNanowire import InfiniteNanowire_FuBerg
import functions

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

Nx, Ny    = 6, 6       # Number of sites in the cross-section
width     = 0.01       # Spread of the Gaussian distribution for the lattice sites
r         = 1.3        # Nearest-neighbour cutoff distance
t         = 1          # Hopping
eps       = 4 * t      # Onsite orbital hopping (in units of t)
lamb      = 0.5 * t    # Spin-orbit coupling in the cross-section (in units of t)
lamb_z    = 0.5 * t    # Spin-orbit coupling along z direction

#%% Main
flux  = np.linspace(0., 5., 100, dtype=np.float64)
sample_wire = InfiniteNanowire_FuBerg(Nx=Nx, Ny=Ny, w=width, r=r, flux=0., t=t, eps=eps, lamb=lamb, lamb_z=lamb_z)
sample_wire.build_lattice()
sample_wire.get_boundary()
gap = np.zeros((len(flux), ))

for i, phi in enumerate(flux):
    loger_main.info(f'flux: {i}/{len(flux)-1}')
    wire = InfiniteNanowire_FuBerg(Nx=Nx, Ny=Ny, w=width, r=r, flux=phi, t=t, eps=eps, lamb=lamb, lamb_z=lamb_z)
    wire.build_lattice(from_x=sample_wire.x, from_y=sample_wire.y)
    wire.get_boundary()
    wire.get_bands()
    gap[i] = wire.get_gap()

#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125', '#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']

fig1 = plt.figure(figsize=(6, 6))
ax1 = fig1.gca()
sample_wire.plot_lattice(ax1)
ax1.set_title(f'$w=$ {width}, $r=$ {r}')

fig2 = plt.figure(figsize=(6, 6))
ax2 = fig2.gca()
ax2.plot(flux, gap, '.', color=color_list[3], markersize=7)
ax2.plot(flux, gap, color=color_list[8], linewidth=0.5)
ax2.set_xlabel('$\phi/\phi_0$')
ax2.set_ylabel('$E_g$')
ax2.set_xlim(flux[0], flux[-1])
ax2.tick_params(which='major', width=0.75, labelsize=10)
ax2.tick_params(which='major', length=6, labelsize=10)
fig2.suptitle(f'$N_x=N_y=$ {Nx}, $w=$ {width}, $r=$ {r}, $\epsilon=$ {eps}, $\lambda=$ {lamb}, $\lambda_z=$ {lamb_z}')
plt.show()