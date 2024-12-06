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
file_list = ['Exp23.h5']
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

# Variables
idx1, idx2 = 0, 33 # 100, 144
flux1 = flux[idx1]
flux2 = flux[idx2]
#%% Main

loger_main.info('Generating fully amorphous lattice...')
lattice = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=Nz, w=width, r=r)
lattice.set_configuration(x, y, z)
lattice.build_lattice(restrict_connectivity=False)
lattice.generate_disorder(K_hopp=0., K_onsite=0.)
nanowire = promote_to_kwant_nanowire3d(lattice, params_dict, mu_leads=mu_leads).finalized()
site_pos = np.array([site.pos for site in nanowire.id_by_site])
loger_main.info('Nanowire promoted to Kwant successfully.')



# Calculating the scattering wavefunctions at certain energies
loger_main.info('Calculating scattering wave functions...')
state1 = kwant.wave_function(nanowire, energy=Ef[0], params=dict(flux=flux1))
state2 = kwant.wave_function(nanowire, energy=Ef[0], params=dict(flux=flux2))
loger_main.info('Scattering wave functions calculated successfully')



# Total local density through a cut
# N_total = np.linspace(2, Nx / 2, 10)
R = np.linspace(2, np.sqrt(2 * (0.5 * Nx) ** 2), 10)
bulk_tot_density1 = np.zeros(R.shape)
bulk_tot_density2 = np.zeros(R.shape)

loger_main.info('Calculating total local bulk DoS...')
for i, n in enumerate(R):
    def bulk(site):
        x, y = site.pos[0] - 0.5 * Nx, site.pos[1] - 0.5 * Ny
        return (x ** 2 + y ** 2) < n ** 2

    loger_main.info(f'Section {i} / {len(R)}')
    total_density_operator = kwant.operator.Density(nanowire, where=bulk, sum=True)
    density_operator = kwant.operator.Density(nanowire, where=bulk, sum=False)
    bulk_tot_density1[i] = total_density_operator(state1(0)[0])
    bulk_tot_density2[i] = total_density_operator(state2(0)[0])

bulk_tot_density1 = bulk_tot_density1 / bulk_tot_density1[-1]
bulk_tot_density2 = bulk_tot_density2 / bulk_tot_density2[-1]



# Local density through a cut
R_local = np.linspace(2, np.sqrt(2 * (0.5 * Nx) ** 2), 3)
bulk_density1 = {}
bulk_density2 = {}
cut_pos = {}

def bulk(syst, rad):

    new_sites = tuple([site for site in syst.id_by_site if
                       (site.pos[0] - 0.5 * Nx) ** 2 + (site.pos[1] - 0.5 * Ny) ** 2 < rad ** 2])
    new_sites_pos = np.array([site.pos for site in syst.id_by_site if
                              (site.pos[0] - 0.5 * Nx) ** 2 + (site.pos[1] - 0.5 * Ny) ** 2 < rad ** 2])
    return new_sites, new_sites_pos

loger_main.info('Calculating local bulk DoS...')
for i, n in enumerate(R_local):
    cut_sites, cut_pos[i] = bulk(nanowire, n)
    loger_main.info(f'Section {i} / {len(R_local)}')
    density_operator = kwant.operator.Density(nanowire, where=cut_sites, sum=False)
    bulk_density1[i] = density_operator(state1(0)[0])
    bulk_density2[i] = density_operator(state2(0)[0])

for key in bulk_density1.keys():
    bulk_density1[key] = bulk_density1[key] / np.sum(bulk_density1[len(R_local) - 1])
    bulk_density2[key] = bulk_density2[key] / np.sum(bulk_density2[len(R_local) - 1])



# Normalisation for the plots
sigmas = 1
mean_value = np.mean(np.array([bulk_density1[len(R_local) - 1], bulk_density2[len(R_local) - 1]]))
std_value = np.std(np.array([bulk_density1[len(R_local) - 1], bulk_density2[len(R_local) - 1]]))
max_value, min_value = mean_value + sigmas * std_value, 0




#%% Figure 1
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['dodgerblue', 'limegreen', 'm', 'r', 'orange']
fontsize = 15


# Figure 1
fig1 = plt.figure()
gs = GridSpec(2, 1, figure=fig1, wspace=0.1, hspace=0.6)
ax1 = fig1.add_subplot(gs[0, 0])
ax2 = fig1.add_subplot(gs[1, 0])

ax2.plot(R, bulk_tot_density1, marker='o', linestyle='solid', color='limegreen', label='(a)')
ax2.plot(R, bulk_tot_density2, marker='o', linestyle='solid', color='dodgerblue', label='(b)')
ax2.set_xlim([R[0], R[-1]])
ax2.set_xlabel('$n_x$', fontsize=fontsize)
ax2.set_ylabel('Cumulative local DoS (%)', fontsize=fontsize)
ax2.legend(frameon=False, fontsize=15)


ax_vec = [ax1]
for i in range(G_array.shape[1]):
    for j in range(len(Ef)):
        ax = ax_vec[j]
        label = None if (j % 2 != 0) else f'$w= {width[i]}$'
        ax.plot(flux, G_array[j, i, :], color=color_list[i], linestyle='solid', label=label)
        ax.plot(flux, 1 * np.ones(flux.shape), '--', color='Black', alpha=0.2)
        ax.text(flux1, G_array[j, i, idx1] - 0.1, f'$(a)$', fontsize=fontsize)
        ax.text(flux2, G_array[j, i, idx2] - 0.1, f'$(b)$', fontsize=fontsize)
        ax.plot(flux2, G_array[j, i, idx2], 'ob')
        ax.plot(flux1, G_array[j, i, idx1], 'ob')

# Figure 1: Format
# ax1.legend(ncol=5, frameon=False, fontsize=20)
ylim = np.max(G_array)
for ax in ax_vec:
    ax.set_xlim(flux[0], flux[-1])
    ax.set_ylim(0, ylim)
    ax.tick_params(which='major', width=0.75, labelsize=10)
    ax.tick_params(which='major', length=6, labelsize=10)
    ax.set_xlabel("$\phi$", fontsize=fontsize)
    ax.set_ylabel("$G[2e^2/h]$", fontsize=fontsize)

fig1.suptitle(f'$\mu_l= {mu_leads}$, $r= {r}$, $N_x= {Nx}$, $N_y = {Ny}$, $N_z= {Nz}$', y=0.93, fontsize=20)

#%% Figure 2

# Defining a colormap
color_map = plt.get_cmap("magma").reversed()
colors = color_map(np.linspace(0, 1, 20))
colors[0] = [1, 1, 1, 1]
color_map = LinearSegmentedColormap.from_list("custom_colormap", colors)
colormap = cm.ScalarMappable(norm=Normalize(vmin=min_value, vmax=max_value), cmap=color_map)


# Figure 1
fig2 = plt.figure(figsize=(12, 10))
gs = GridSpec(3, 3, figure=fig2, wspace=0.1, hspace=0.1)
ax1 = fig2.add_subplot(gs[0, 0], projection='3d')
ax2 = fig2.add_subplot(gs[0, 1], projection='3d')
ax3 = fig2.add_subplot(gs[0, 2], projection='3d')
ax4 = fig2.add_subplot(gs[1, 0], projection='3d')
ax5 = fig2.add_subplot(gs[1, 1], projection='3d')
ax6 = fig2.add_subplot(gs[1, 2], projection='3d')


for i, ax in enumerate(fig2.axes):
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([0.5, 0.5, 2.5, 1.5]))
    ax.set_axis_off()
    if i<3:
        ax.set_title(f'$r < {R_local[i] :.2f}$')
        ax.scatter(cut_pos[i][:, 0], cut_pos[i][:, 1], cut_pos[i][:, 2], facecolor='white', edgecolor='black')
        ax.scatter(cut_pos[i][:, 0], cut_pos[i][:, 1], cut_pos[i][:, 2], c=bulk_density1[i], cmap=color_map, vmin=min_value,
                vmax=max_value)
    else:
        ax.set_title(f'$r < {R_local[i - 3] :.2f}$')
        ax.scatter(cut_pos[i - 3][:, 0], cut_pos[i - 3][:, 1], cut_pos[i - 3][:, 2], facecolor='white', edgecolor='black')
        ax.scatter(cut_pos[i - 3][:, 0], cut_pos[i - 3][:, 1], cut_pos[i - 3][:, 2], c=bulk_density2[i - 3], cmap=color_map,
                   vmin=min_value, vmax=max_value)

cbar_ax = fig2.add_subplot(gs[-1, :])
divider = make_axes_locatable(cbar_ax)
cax = divider.append_axes('bottom', size="10%", pad=0)
cbar = fig1.colorbar(colormap, cax=cax, orientation='horizontal')
cbar_ax.set_axis_off()
cbar.set_label(label='$\\vert \psi (r)\\vert ^2$', fontsize=20)


plt.show()