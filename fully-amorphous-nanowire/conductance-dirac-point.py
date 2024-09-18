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
We check that the fully amorphous wire reduces to the translation invariant one.
"""

Nx, Ny, Nz       = 5, 5, 20                   # Number of sites in the cross-section
r                = 1.3                        # Nearest-neighbour cutoff distance
t                = 1                          # Hopping
eps              = 4 * t                      # Onsite orbital hopping (in units of t)
lamb             = 1 * t                      # Spin-orbit coupling in the cross-section (in units of t)
lamb_z           = 1.8 * t                    # Spin-orbit coupling along z direction
mu_leads         = 0 * t                      # Chemical potential at the leads
flux             = np.linspace(0, 3, 1000)    # Magnetic flux
width            = [0.0001, 0.02, 0.05, 0.1]  # Amorphous width
params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}

# Preallocation
G1 = np.zeros(flux.shape)
G2 = np.zeros(flux.shape)
G3 = np.zeros(flux.shape)
G4 = np.zeros(flux.shape)

#%% Main

# Fully amorphous wire
loger_main.info('Generating fully amorphous lattice...')
lattice1 = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=Nz, w=width[0], r=r)
lattice1.build_lattice()
nanowire1 = promote_to_kwant_nanowire3d(lattice1, params_dict, mu_leads=mu_leads).finalized()
loger_main.info('Nanowire promoted to Kwant successfully.')

loger_main.info('Generating fully amorphous lattice...')
lattice2 = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=Nz, w=width[1], r=r)
lattice2.build_lattice()
nanowire2 = promote_to_kwant_nanowire3d(lattice2, params_dict, mu_leads=mu_leads).finalized()
loger_main.info('Nanowire promoted to Kwant successfully.')

loger_main.info('Generating fully amorphous lattice...')
lattice3 = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=Nz, w=width[2], r=r)
lattice3.build_lattice()
nanowire3 = promote_to_kwant_nanowire3d(lattice3, params_dict, mu_leads=mu_leads).finalized()
loger_main.info('Nanowire promoted to Kwant successfully.')

loger_main.info('Generating fully amorphous lattice...')
lattice4 = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=Nz, w=width[3], r=r)
lattice4.build_lattice()
nanowire4 = promote_to_kwant_nanowire3d(lattice4, params_dict, mu_leads=mu_leads).finalized()
loger_main.info('Nanowire promoted to Kwant successfully.')

dirac_point = 0.1
for i, phi in enumerate(flux):
    loger_main.info(f'Flux: {i} / {len(flux) - 1}')
    S1 = kwant.smatrix(nanowire1, dirac_point, params=dict(flux=phi))
    S2 = kwant.smatrix(nanowire2, dirac_point, params=dict(flux=phi))
    S3 = kwant.smatrix(nanowire3, dirac_point, params=dict(flux=phi))
    S4 = kwant.smatrix(nanowire4, dirac_point, params=dict(flux=phi))
    G1[i] = S1.transmission(1, 0)
    G2[i] = S2.transmission(1, 0)
    G3[i] = S3.transmission(1, 0)
    G4[i] = S4.transmission(1, 0)


#%% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125', '#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']

fig1 = plt.figure(figsize=(10, 10))
gs = GridSpec(1, 1, figure=fig1)
ax1_1 = fig1.add_subplot(gs[0, 0])

ax1_1.plot(flux, G1, color=color_list[8], label=f'$w= {width[0]}$')
ax1_1.plot(flux, G2, color=color_list[7], label=f'$w= {width[1]}$')
ax1_1.plot(flux, G3, color=color_list[6], label=f'$w= {width[2]}$')
ax1_1.plot(flux, G4, color=color_list[5], label=f'$w= {width[3]}$')
ax1_1.legend(ncol=1, frameon=False, fontsize=16)

ylim = 1
for ax in [ax1_1]:
    ax.set_xlim(flux[0], flux[-1])
    ax.set_ylim(0, ylim)
    ax.tick_params(which='major', width=0.75, labelsize=10)
    ax.tick_params(which='major', length=6, labelsize=10)
    ax.set_xlabel("$\phi$", fontsize=10)
    ax.set_ylabel("$G[2e^2/h]$",fontsize=10)

fig1.suptitle(f'$\mu_l= {mu_leads}$, $r= {r}$, $N_x= {Nx}$, $N_y = {Ny}$, $N_z= {Nz}$', y=0.93, fontsize=20)
fig1.savefig('conductance-dirac-point.pdf', format='pdf', backend='pgf')
plt.show()