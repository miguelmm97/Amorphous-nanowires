#%% Modules and setup

# Plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm

# Modules
from modules.functions import *
from modules.colorbar_marker import *


#%% Loading data
file_list = ['Exp11.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-local-marker')

# Parameters
Nx           = data_dict[file_list[0]]['Parameters']['Nx']
Ny           = data_dict[file_list[0]]['Parameters']['Ny']
Nz           = data_dict[file_list[0]]['Parameters']['Nz']
r            = data_dict[file_list[0]]['Parameters']['r ']
t            = data_dict[file_list[0]]['Parameters']['t ']
eps          = data_dict[file_list[0]]['Parameters']['eps']
lamb         = data_dict[file_list[0]]['Parameters']['lamb']
lamb_z       = data_dict[file_list[0]]['Parameters']['lamb_z']
flux         = data_dict[file_list[0]]['Parameters']['flux']
width        = data_dict[file_list[0]]['Parameters']['width']


# Simulation data
x            = data_dict[file_list[0]]['Simulation']['x']
y            = data_dict[file_list[0]]['Simulation']['y']
z            = data_dict[file_list[0]]['Simulation']['z']
local_marker = data_dict[file_list[0]]['Simulation']['local_marker']


#%% Defining different cuts
def bulk(x, y, z, local_marker, cutoff, include_last_layer=True):
    x_pos, y_pos = x - 0.5 * Nx, y - 0.5 * Ny
    cond1 = np.abs(x_pos) < cutoff
    cond2 = np.abs(y_pos) < cutoff
    cond = cond1 * cond2
    if not include_last_layer:
        cond3 = 4 < z
        cond4 = z < Nz - 4
        cond = cond1 * cond2 * cond3 * cond4
    return x[cond], y[cond], z[cond], local_marker[cond]

N = np.linspace(0.1, Nx / 2 + 1, 3)
x1, y1, z1, marker1 = bulk(x, y, z, local_marker, N[0], include_last_layer=False)
x2, y2, z2, marker2 = bulk(x, y, z, local_marker, N[1], include_last_layer=False)
x3, y3, z3, marker3 = bulk(x, y, z, local_marker, N[2])
avg_marker1 = np.mean(marker1)
avg_marker2 = np.mean(marker2)
avg_marker3 = np.mean(marker3)
avg_marker = [avg_marker1, avg_marker2, avg_marker3]


N_array = np.linspace(2, Nx / 2 + 1, 10)
marker_per_cut = np.zeros(N_array.shape)
for i, n in enumerate(N_array):
    include_last_layer = True if n >= (Nx / 2) else False
    _, _, _, marker = bulk(x, y, z, local_marker, n, include_last_layer=include_last_layer)
    marker_per_cut[i] = np.mean(marker)


#%% Figure 1
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['dodgerblue', 'limegreen', 'm', 'r', 'orange']
fontsize = 15


# Defining a colormap
divnorm = mcolors.TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=max(marker3))
hex_list = ['#ff416d', '#ff7192', '#ffa0b6', '#ffd0db', '#ffffff', '#cfdaff', '#9fb6ff', '#6f91ff', '#3f6cff']

# Figure 1
fig1 = plt.figure(figsize=(12, 10))
gs = GridSpec(1, 3, figure=fig1, wspace=0.0, hspace=0.02)
ax1 = fig1.add_subplot(gs[0, 0], projection='3d')
ax2 = fig1.add_subplot(gs[0, 1], projection='3d')
ax3 = fig1.add_subplot(gs[0, 2], projection='3d')

ax1.scatter(x1, y1, z1, facecolor='white', edgecolor='black')
ax2.scatter(x2, y2, z2, facecolor='white', edgecolor='black')
ax3.scatter(x3, y3, z3, facecolor='white', edgecolor='black')
scatters1 = ax1.scatter(x1, y1, z1, c=marker1, norm=divnorm, cmap=get_continuous_cmap(hex_list),  linewidths=2.5)
scatters2 = ax2.scatter(x2, y2, z2, c=marker2, norm=divnorm, cmap=get_continuous_cmap(hex_list),  linewidths=2.5)
scatters3 = ax3.scatter(x3, y3, z3, c=marker3, norm=divnorm, cmap=get_continuous_cmap(hex_list),  linewidths=2.5)

for i, ax in enumerate(fig1.axes):
    ax.set_title(f'$n_x, n_y < {N[i] :.2f}$, $\\nu= {avg_marker[i] :.2f}$ ')
    ax.set_box_aspect((1, 1, 10))
    ax.set_axis_off()


# cbar_ax = fig1.add_subplot(gs[0, -1])
# divider = make_axes_locatable(cbar_ax)
# cax = divider.append_axes("left", size="10%", pad=0)
# cbar = fig1.colorbar(colormap, cax=cax, orientation='vertical')
# cbar_ax.set_axis_off()
# cbar.set_label(label='$\\vert \psi (r)\\vert ^2$', labelpad=10, fontsize=20)



# Figure 1
fig2 = plt.figure()
gs = GridSpec(1, 1, figure=fig2, wspace=0.0, hspace=0.02)
ax1 = fig2.add_subplot(gs[0, 0])

ax1.plot(N_array, marker_per_cut, marker='o', linestyle='solid', color='dodgerblue')
ax1.set_xlabel('$n_x, n_y$ included', fontsize=fontsize)
ax1.set_ylabel('$\overline{\\nu}$', fontsize=fontsize)
ax1.set_ylim([-1, 0.1])




fig1.savefig(f'../figures/{file_list[0]}-local-marker.pdf', format='pdf')
fig2.savefig(f'../figures/{file_list[0]}-total-marker.pdf', format='pdf')
plt.show()