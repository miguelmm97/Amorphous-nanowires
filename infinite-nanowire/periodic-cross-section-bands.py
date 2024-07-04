#%% Modules setup

# Math and plotting
from numpy import pi
import numpy as np
import matplotlib.pyplot as plt

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

Nx, Ny    = 2, 2       # Number of sites in the cross-section
width     = 0.01       # Spread of the Gaussian distribution for the lattice sites
r         = 1.3        # Nearest-neighbour cutoff distance
t         = 1          # Hopping
eps       = 4 * t      # Onsite orbital hopping (in units of t)
lamb      = 0.5 * t    # Spin-orbit coupling in the cross-section (in units of t)
lamb_z    = 0.5 * t    # Spin-orbit coupling along z direction

#%% Main
wire = InfiniteNanowire_FuBerg(Nx=Nx, Ny=Ny, w=width, r=r, t=t, eps=eps, lamb=lamb, lamb_z=lamb_z)
wire.build_lattice()
# wire.get_boundary()
bands, eigenstates = wire.get_bands()



#%% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig1 = plt.figure(figsize=(6, 6))
ax1 = fig1.gca()
wire.plot_lattice(ax1)
ax1.set_title(f'Amorphous cross section. $w=$ {width}, $r=$ {r}')
plt.show()

fig2 = plt.figure(figsize=(6, 6))
ax2 = fig2.gca()
for k in bands.keys():
    kz = wire.kz[k] * np.ones((len(bands[k], )))
    ax2.plot(kz, bands[k], '.', color='royalblue')