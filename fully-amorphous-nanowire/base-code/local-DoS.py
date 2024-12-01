#%% modules set up

# Math and plotting
import numpy as np
import sys
from datetime import date

# Kwant
import kwant

# modules
from modules.functions import *
from modules.AmorphousLattice_3d import AmorphousLattice_3d
from modules.FullyAmorphousWire_kwant import promote_to_kwant_nanowire3d, select_perfect_transmission_flux

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm

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


#%% Loading data
file_list = ['Exp20.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-cond-vs-flux-fully-amorphous')

# Parameters
Ef           = data_dict[file_list[0]]['Parameters']['Ef']
Nx           = data_dict[file_list[0]]['Parameters']['Nx']
Ny           = data_dict[file_list[0]]['Parameters']['Ny']
Nz           = data_dict[file_list[0]]['Parameters']['Nz']
r            = data_dict[file_list[0]]['Parameters']['r']
t            = data_dict[file_list[0]]['Parameters']['t']
eps          = data_dict[file_list[0]]['Parameters']['eps']
lamb         = data_dict[file_list[0]]['Parameters']['lamb']
lamb_z       = data_dict[file_list[0]]['Parameters']['lamb_z']
mu_leads     = data_dict[file_list[0]]['Parameters']['mu_leads']
params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}

# Simulation data
flux          = data_dict[file_list[0]]['Simulation']['flux']
G_array       = data_dict[file_list[0]]['Simulation']['G_array']
width         = data_dict[file_list[0]]['Simulation']['width']
x             = data_dict[file_list[0]]['Simulation']['x']
y             = data_dict[file_list[0]]['Simulation']['y']
z             = data_dict[file_list[0]]['Simulation']['z']
#%% Main

loger_main.info('Generating fully amorphous lattice...')
lattice = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=Nz, w=width, r=r)
lattice.set_configuration(x, y, z)
lattice.build_lattice(restrict_connectivity=False)
lattice.generate_disorder(K_hopp=0., K_onsite=0.)
nanowire = promote_to_kwant_nanowire3d(lattice, params_dict, mu_leads=mu_leads).finalized()
loger_main.info('Nanowire promoted to Kwant successfully.')

# Calculating the scattering wavefunctions at certain energies
loger_main.info('Calculating scattering wave functions...')
top_state = kwant.wave_function(nanowire, energy=Ef[0], params=dict(flux=0.5))
triv_state = kwant.wave_function(nanowire, energy=Ef[0], params=dict(flux=0.))
loger_main.info('Scattering wave functions calculated successfully')

# Calculating local density of states
loger_main.info('Calculating local Dos...')
local_density = kwant.operator.Density(nanowire)
top_density = local_density(top_state(0)[0])
triv_density = local_density(triv_state(0)[0])
max_value = np.max(np.array([top_density, triv_density]))
site_pos = np.array([site.pos for site in nanowire.id_by_site])


#%% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange']
fontsize = 20

# Defining a colormap without negatives
color_map = plt.get_cmap("gist_heat_r")
colors = color_map(np.linspace(0, 1, 20))
colors[0] = [1, 1, 1, 1]
color_map = LinearSegmentedColormap.from_list("custom_colormap", colors)
colormap = cm.ScalarMappable(norm=Normalize(vmin=0, vmax=max_value), cmap=color_map)


fig1 = plt.figure()
gs = GridSpec(1, 3, figure=fig1, wspace=0.2, hspace=0.0)
ax1 = fig1.add_subplot(gs[0, 0], projection='3d')
ax2 = fig1.add_subplot(gs[0, 1], projection='3d')

ax1.scatter(site_pos[:, 0], site_pos[:, 1], site_pos[:, 2], c=top_density, cmap=color_map, vmin=0, vmax=max_value)
ax1.get_proj = lambda: np.dot(Axes3D.get_proj(ax1), np.diag([0.5, 0.5, 1, 1]))
ax1.set_axis_off()

ax2.scatter(site_pos[:, 0], site_pos[:, 1], site_pos[:, 2], c=triv_density, cmap=color_map, vmin=0, vmax=max_value)
ax2.get_proj = lambda: np.dot(Axes3D.get_proj(ax2), np.diag([0.5, 0.5, 1, 1]))
ax2.set_axis_off()


cbar_ax = fig1.add_subplot(gs[0, 2])
divider = make_axes_locatable(cbar_ax)
cax = divider.append_axes("left", size="10%", pad=0)
cbar = fig1.colorbar(colormap, cax=cax, orientation='vertical')
cbar_ax.set_axis_off()
cbar.set_label(label='$\\vert \psi \\vert ^2$', fontsize=20)




# Figure 1: Definition
fig2 = plt.figure(figsize=(20, 10))
gs = GridSpec(1, 1, figure=fig2)
ax1 = fig2.add_subplot(gs[0, 0])
ax_vec = [ax1]

# Figure 1: Plots
for i in range(G_array.shape[1]):
    for j in range(len(Ef)):
        ax = ax_vec[j]
        label = None if (j % 2 != 0) else f'$w= {width[i]}$'
        ax.plot(flux, G_array[j, i, :], color=color_list[i], linestyle='solid', label=label)
        ax.plot(flux, 1 * np.ones(flux.shape), '--', color='Black', alpha=0.2)
        ax.text(0.5, 1.1, f'$E_f= {Ef[j]}$', fontsize=fontsize)
        ax.plot(flux[20], G_array[j, i, 20], 'ob')

# Figure 1: Format
ax1.legend(ncol=5, frameon=False, fontsize=20)
fig2.suptitle(f'$\mu_l= {mu_leads}$, $r= {r}$, $N_x= {Nx}$, $N_y = {Ny}$, $N_z= {Nz}$', y=0.93, fontsize=20)
ylim = np.max(G_array)
for ax in ax_vec:
    ax.set_xlim(flux[0], flux[-1])
    ax.set_ylim(0, ylim)
    ax.tick_params(which='major', width=0.75, labelsize=10)
    ax.tick_params(which='major', length=6, labelsize=10)
    ax.set_xlabel("$\phi$", fontsize=fontsize)
    ax.set_ylabel("$G[2e^2/h]$", fontsize=fontsize)



plt.show()