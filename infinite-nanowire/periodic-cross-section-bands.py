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

Nx, Ny    = 8, 8       # Number of sites in the cross-section
width     = 0.01       # Spread of the Gaussian distribution for the lattice sites
r         = 1.3        # Nearest-neighbour cutoff distance
flux      = 1          # Flux threaded through the cross-section (in units of flux quantum)
t         = 1          # Hopping
eps       = 4 * t      # Onsite orbital hopping (in units of t)
lamb      = 0.5 * t    # Spin-orbit coupling in the cross-section (in units of t)
lamb_z    = 0.5 * t    # Spin-orbit coupling along z direction

#%% Main
wire = InfiniteNanowire_FuBerg(Nx=Nx, Ny=Ny, w=width, r=r, flux=flux, t=t, eps=eps, lamb=lamb, lamb_z=lamb_z)
wire.build_lattice()
wire.get_boundary()
bands, eigenstates = wire.get_bands()



#%% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125', '#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']

fig1 = plt.figure(figsize=(6, 6))
ax1 = fig1.gca()
wire.plot_lattice(ax1)
ax1.set_title(f'Amorphous cross section. $w=$ {width}, $r=$ {r}')

fig2 = plt.figure(figsize=(6, 6))
gs = GridSpec(2, 4, figure=fig1, wspace=0.8)
ax2_1 = fig2.add_subplot(gs[:, :2])
ax2_2 = fig2.add_subplot(gs[:, 2:])

for k in bands.keys():
    kz = wire.kz[k] * np.ones((len(bands[k], )))
    ax2_1.plot(kz, bands[k], '.', color=color_list[8], markersize=0.2)
    ax2_2.plot(kz, bands[k], '.', color=color_list[8], markersize=0.2)

ax2_1.set_xlabel('$k/a$')
ax2_1.set_ylabel('$E(k)/t$')
ax2_1.set_xlim(-pi, pi)
ax2_1.tick_params(which='major', width=0.75, labelsize=10)
ax2_1.tick_params(which='major', length=6, labelsize=10)
ax2_1.set(xticks=[-pi, -pi/2, 0, pi/2, pi], xticklabels=['$-\pi$', '$-\pi/2$', '$0$', '$\pi/2$', '$\pi$'])

ax2_2.set_xlabel('$k/a$')
# ax2_2.set_ylabel('$E(k)/t$')
ax2_2.set_xlim(-pi, pi)
ax2_2.set_ylim(-0.5, 0.5)
ax2_2.tick_params(which='major', width=0.75, labelsize=10)
ax2_2.tick_params(which='major', length=6, labelsize=10)
ax2_2.set(xticks=[-pi, -pi/2, 0, pi/2, pi], xticklabels=['$-\pi$', '$-\pi/2$', '$0$', '$\pi/2$', '$\pi$'])
fig2.suptitle(f'Energy bands. $w=$ {width}, $r=$ {r}')
plt.show()