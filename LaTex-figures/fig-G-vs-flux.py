#%% modules setup

# Math and plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter, LogLocator
# modules
from modules.functions import *

L = [2, 5]

#%% Loading data
file_list = ['draft-fig3-G-vs-flux.h5', 'draft-fig3-high-Ef.h5', 'draft-fig3-DoS.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-latex-figures')


# Simulation data for conductance
flux          = data_dict[file_list[0]]['Simulation']['flux']
G_low_Ef      = data_dict[file_list[0]]['Simulation']['G_array']
width         = data_dict[file_list[0]]['Simulation']['width']
Ef1           = data_dict[file_list[0]]['Parameters']['Ef']
G_high_Ef     = data_dict[file_list[1]]['Simulation']['G_array']
flux_2        = data_dict[file_list[1]]['Simulation']['flux']
Ef2           = data_dict[file_list[1]]['Parameters']['Ef']

# Simulation data for DoS
bulk_tot_density = data_dict[file_list[2]]['Simulation']['bulk_tot_density']
DoS_top          = data_dict[file_list[2]]['Simulation']['DoS_top']
DoS_loc          = data_dict[file_list[2]]['Simulation']['DoS_loc']
DoS_bound        = data_dict[file_list[2]]['Simulation']['DoS_bound']
cuts_top         = data_dict[file_list[2]]['Simulation']['cuts_top']
cuts_loc         = data_dict[file_list[2]]['Simulation']['cuts_loc']
cuts_bound       = data_dict[file_list[2]]['Simulation']['cuts_bound']
N                = data_dict[file_list[2]]['Simulation']['N']
nx               = data_dict[file_list[2]]['Simulation']['nx']

# States
flux_top = flux[39]
G_top = G_low_Ef[0, 2, 39]
flux_loc = flux[116]
G_loc = G_low_Ef[0, 1, 116]
flux_bound = flux[161]
G_bound = G_low_Ef[0, -1, 161]


#%% Figures

# Style sheet
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange']
markersize = 5
fontsize=20
fontsize_in = 20
site_size  = 0.1
site_lw    = 0.01
site_color = 'm'
hop_color  = 'royalblue'
hop_lw     = 0.05
lead_color = 'r'

# Key style modifications for dark theme
# plt.rcParams.update({
#     'axes.facecolor': 'black',
#     'figure.facecolor': 'black',
#     'savefig.facecolor': 'black',
#     'axes.edgecolor': 'white',
#     'axes.labelcolor': 'white',
#     'xtick.color': 'white',
#     'ytick.color': 'white',
#     'text.color': 'white',
#     'legend.edgecolor': 'white',
#     'legend.facecolor': 'black',
# })

# Colormap
sigmas = 3
mean_value = np.mean(DoS_top['2'])
std_value = np.std(DoS_top['2'])
max_value, min_value = mean_value + sigmas * std_value, 0
color_map = plt.get_cmap("magma").reversed()
colors = color_map(np.linspace(0, 1, 20))
colors[0] = [1, 1, 1, 1]
color_map = LinearSegmentedColormap.from_list("custom_colormap", colors)
colormap = cm.ScalarMappable(norm=Normalize(vmin=min_value, vmax=max_value), cmap=color_map)
palette = seaborn.color_palette(palette='viridis_r', n_colors=200)
palette = [palette[0], palette[50], palette[100], palette[130], palette[-1]]


# Figure 1: Definition
fig1 = plt.figure(figsize=(8, 6))
gs = GridSpec(2, 5, figure=fig1, wspace=0.5, hspace=0.3)
ax1 = fig1.add_subplot(gs[0, :])
ax2 = fig1.add_subplot(gs[1, 0])
ax3 = fig1.add_subplot(gs[1, 1], projection='3d')
ax4 = fig1.add_subplot(gs[1, 2], projection='3d')
ax5 = fig1.add_subplot(gs[1, 3], projection='3d')


# Figure 1: Conductance plots
ax_vec = [ax1]
for i in range(G_low_Ef.shape[1]):
    label = f'${width[i] :.1f}$'
    ax1.plot(flux, G_low_Ef[0, i, :], color=palette[i], linestyle='solid', label=label)

ax1.plot(flux, 1 * np.ones(flux.shape), '--', color='Black', alpha=0.2)
# ax1.text(4.1, 1.3, '$E_F^{nw}=$' + f'${Ef1[0]}$', fontsize=fontsize)
# ax1.text(3, 1.3,  '$E_F^{lead}=$' + f'${Ef1[0] + 1}$', fontsize=fontsize)
ax1.plot(flux_top, G_top, marker='o', color=palette[2], markersize=10, markeredgecolor='black')
ax1.plot(flux_loc, G_loc+0.05, marker='^', color=palette[1], markersize=10, markeredgecolor='black')
ax1.plot(flux_bound, G_bound, marker='*', color=palette[-1], markersize=12, markeredgecolor='black')
# ax1.text(flux_top, G_top, f'$(a)$', fontsize=fontsize - 2)


# Figure 1: Format
ax1.text(0.1, 1.3, '$(a)$', fontsize=fontsize)
ax1.text(0.6, 1.25, '$\\underline{w}$', fontsize=fontsize)
ax1.legend(loc='upper left', ncol=5, alignment='left', frameon=False, fontsize=fontsize, columnspacing=0.6, handlelength=1, labelspacing=0.2, bbox_to_anchor=(0.15, 1.02))
ax1.set_xlim(flux[0], flux[-1])
ax1.set_ylim(0, 1.5)
ax1.set(xticks=[0, 1, 2, 3, 4, 5])
majorsy = [0, 0.5, 1, 1.5]
ax1.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
ax1.set_ylabel("$G(2e^2/h)$", fontsize=fontsize)
ax1.set_xlabel("$\phi/\phi_0$", fontsize=fontsize, labelpad=-10)
ax1.tick_params(which='major', width=0.75, labelsize=fontsize)
ax1.tick_params(which='major', length=6, labelsize=fontsize)
ax1.tick_params(which='minor', width=0.75, labelsize=fontsize)
ax1.tick_params(which='minor', length=3, labelsize=fontsize)




ax3.scatter(np.round(cuts_top['2'][:, 0], 2), np.round(cuts_top['2'][:, 1], 2), np.round(cuts_top['2'][:, 2], 2), facecolor='white', edgecolor='black', rasterized=True)
ax3.scatter(np.round(cuts_top['2'][:, 0], 2), np.round(cuts_top['2'][:, 1], 2), np.round(cuts_top['2'][:, 2], 2), c=DoS_top['2'], cmap=color_map, vmin=min_value, vmax=max_value, rasterized=True)
ax3.set_box_aspect((3, 3, 10))
ax3.set_axis_off()
pos3 = ax3.get_position()
ax3.set_position([pos3.x0 - 0.02, pos3.y0 - 0.15, 0.25, 0.4])

ax4.scatter(np.round(cuts_loc['2'][:, 0], 2), np.round(cuts_loc['2'][:, 1], 2), np.round(cuts_loc['2'][:, 2], 2), facecolor='white', edgecolor='black', rasterized=True)
ax4.scatter(np.round(cuts_loc['2'][:, 0], 2), np.round(cuts_loc['2'][:, 1], 2), np.round(cuts_loc['2'][:, 2], 2), c=DoS_loc['2'], cmap=color_map, vmin=min_value, vmax=max_value, rasterized=True)
ax4.set_box_aspect((3, 3, 10))
ax4.set_axis_off()
pos4 = ax4.get_position()
ax4.set_position([pos4.x0, pos4.y0 - 0.15, 0.25, 0.4])

ax5.scatter(np.round(cuts_bound['2'][:, 0], 2), np.round(cuts_bound['2'][:, 1], 2), np.round(cuts_bound['2'][:, 2], 2), facecolor='white', edgecolor='black', rasterized=True)
ax5.scatter(np.round(cuts_bound['2'][:, 0], 2), np.round(cuts_bound['2'][:, 1], 2), np.round(cuts_bound['2'][:, 2], 2), c=DoS_bound['2'], cmap=color_map, vmin=min_value, vmax=max_value, rasterized=True)
ax5.set_box_aspect((3, 3, 10))
ax5.set_axis_off()
pos5 = ax5.get_position()
ax5.set_position([pos5.x0  + 0.02, pos5.y0 - 0.15, 0.25, 0.4])

# scatter_ax3 = fig1.add_axes([0.315, 0.3, 0.05, 0.05])
scatter_ax3 = fig1.add_axes([0.3, 0.38, 0.05, 0.05])
scatter_ax3.scatter([0], [0], color=palette[2], s=100, marker='o', edgecolor='black')
scatter_ax3.set_xticks([])
scatter_ax3.set_yticks([])
scatter_ax3.set_axis_off()
scatter_ax4 = fig1.add_axes([0.49, 0.38, 0.05, 0.05])
scatter_ax4.scatter([0], [0], color=palette[1], s=100, marker='^', edgecolor='black')
scatter_ax4.set_xticks([])
scatter_ax4.set_yticks([])
scatter_ax4.set_axis_off()
scatter_ax5 = fig1.add_axes([0.68, 0.38, 0.05, 0.05])
scatter_ax5.scatter([0], [0], color=palette[-1], s=200, marker='*', edgecolor='black')
scatter_ax5.set_xticks([])
scatter_ax5.set_yticks([])
scatter_ax5.set_axis_off()


cbar_ax = fig1.add_subplot(gs[1, 4])
divider = make_axes_locatable(cbar_ax)
cax = divider.append_axes("right", size="10%", pad=-2)
cbar = fig1.colorbar(colormap, cax=cax, orientation='vertical', ticks=[0., 0.0001, 0.0002, 0.0003])
cbar_ax.set_axis_off()
cbar.set_label(label='$\\vert \psi (r)\\vert ^2$', labelpad=10, fontsize=20)
cbar.ax.tick_params(which='major', width=0.75, labelsize=fontsize)
cbar.ax.set_yticklabels(['0', '1', '2', '3'])
cbar.ax.set_title(r'$\quad \quad \times 10^{-4}$', x=0.8, fontsize=fontsize-10)
#

scatter_ax3.text(-0.13, -0.02, '$(c)$', fontsize=fontsize)
# ax2.text(12, 0.85, '$(d)$', fontsize=fontsize)
# ax2.text(17, 0.85, '$(e)$', fontsize=fontsize)
ax2.text(2.1, 0.7, '$(b)$', fontsize=fontsize)
ax2.plot(N, bulk_tot_density[0, :], marker='o', linestyle='solid', color=palette[2])
ax2.plot(N, bulk_tot_density[1, :], marker='^', linestyle='solid', color=palette[1])
ax2.plot(N, bulk_tot_density[2, :], marker='*', linestyle='solid', color=palette[-1])
ax2.set_xlabel('$R$', fontsize=fontsize, labelpad=-1)
ax2.set_ylabel('DoS', fontsize=fontsize, labelpad=-10)
label = ax2.yaxis.get_label()
x, y = label.get_position()
label.set_position((x, y + 0.2))
# ax2.set_ylim(0, 1)
ax2.tick_params(which='major', width=0.75, labelsize=fontsize)
ax2.tick_params(which='major', length=6, labelsize=fontsize)
ax2.tick_params(which='minor', width=0.75, labelsize=fontsize)
ax2.tick_params(which='minor', length=3, labelsize=fontsize)
# majorsy = [0, 0.5, 1]
# minorsy = [0.25, 0.75]
# ax2.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
# ax2.yaxis.set_minor_locator(ticker.FixedLocator(minorsy))
ax2.set_yscale('log')
#majorsx = [0.5, 1]
#minorsx = [0.75]
#ax2.xaxis.set_major_locator(ticker.FixedLocator(majorsx))
#ax2.xaxis.set_minor_locator(ticker.FixedLocator(minorsx))

fig1.savefig('fig-G-vs-flux.pdf', format='pdf')
plt.show()

