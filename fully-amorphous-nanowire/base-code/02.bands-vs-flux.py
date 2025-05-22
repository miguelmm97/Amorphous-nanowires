#%% modules set up

# Math and plotting
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Kwant
import kwant

# modules
from modules.functions import *
from modules.FullyAmorphousWire_kwant import infinite_nanowire_kwant

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

Nx, Ny           = 7, 7                                     # Number of sites in the cross-section
r                = 1.3                                        # Nearest-neighbour cutoff distance
t                = 1                                          # Hopping
eps              = 4 * t                                      # Onsite orbital hopping (in units of t)
lamb             = 1 * t                                      # Spin-orbit coupling in the cross-section (in units of t)
lamb_z           = 1.8 * t                                    # Spin-orbit coupling along z direction
mu_leads         = - 0 * t                                    # Chemical potential at the leads
width            = 0.1                                          # Amorphous width 0.0001, 0.02, 0.05,
Nz               = 100                                        # Length of the wire
flux             = np.linspace(0, 1.5, 9)                     # Flux
kz               = np.linspace(-pi, pi, 1001)
params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}


bands_array = []
#%% Main

wire_kwant = infinite_nanowire_kwant(Nx, Ny, params_dict, mu_leads=mu_leads).finalized()
for i, phi in enumerate(flux):
    loger_main.info(f'Flux: {i} / {len(flux) - 1}')
    bands = kwant.physics.Bands(wire_kwant, params=dict(flux=phi))
    bands_array.append([bands(k) for k in kz])



#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


fig1 = plt.figure(figsize=(15, 15))
gs = GridSpec(3, 3, figure=fig1, wspace=0.3, hspace=0.4)
ax1_1 = fig1.add_subplot(gs[0, 0])
ax1_2 = fig1.add_subplot(gs[0, 1])
ax1_3 = fig1.add_subplot(gs[0, 2])
ax1_4 = fig1.add_subplot(gs[1, 0])
ax1_5 = fig1.add_subplot(gs[1, 1])
ax1_6 = fig1.add_subplot(gs[1, 2])
ax1_7 = fig1.add_subplot(gs[2, 0])
ax1_8 = fig1.add_subplot(gs[2, 1])
ax1_9 = fig1.add_subplot(gs[2, 2])
ax_vec = [ax1_1, ax1_2, ax1_3, ax1_4, ax1_5, ax1_6, ax1_7, ax1_8, ax1_9]


for i, phi in enumerate(flux):
    ax = ax_vec[i]
    ax.plot(kz, bands_array[i], color='dodgerblue', linewidth=0.5)
    ax.plot(kz, 0.02 * np.ones(kz.shape), '--', color='Black', alpha=0.2)

    ax.set_xlabel('$k/a$')
    ax.set_ylabel('$E(k)/t$')
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.2)
    ax.tick_params(which='major', width=0.75, labelsize=10)
    ax.tick_params(which='major', length=6, labelsize=10)
    # ax.set(xticks=[-pi, -pi/2, 0, pi/2, pi], xticklabels=['$-\pi$', '$-\pi/2$', '$0$', '$\pi/2$', '$\pi$'])
    ax.set_title(f'$\phi / \phi_0={phi :.2f}$')

# fig1.savefig(f'figures/bands-vs-flux.pdf', format='pdf', backend='pgf')
plt.show()

