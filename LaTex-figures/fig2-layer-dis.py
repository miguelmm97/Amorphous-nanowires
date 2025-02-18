#%% modules setup

# Math and plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn

# modules
from modules.functions import *


#%% Loading data
file_list = ['draft-fig2-avg-osc-w-002.h5', 'draft-fig2-avg-osc-w-005.h5', 'draft-fig2-avg-osc-w-01.h5',
             'draft-fig2-G-vs-flux.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-latex-figures')

# Parameters
width1 = data_dict[file_list[0]]['Parameters']['width']
width2 = data_dict[file_list[1]]['Parameters']['width']
width3 = data_dict[file_list[2]]['Parameters']['width']
flux0 = data_dict[file_list[2]]['Parameters']['flux0']
flux_half = data_dict[file_list[2]]['Parameters']['flux_half']

# Simulation data
fermi         = data_dict[file_list[0]]['Simulation']['fermi']
G0_1          = data_dict[file_list[0]]['Simulation']['avg_G0']
G_half_1      = data_dict[file_list[0]]['Simulation']['avg_G_half']
std_G0_1      = data_dict[file_list[0]]['Simulation']['std_G0']
std_G_half_1  = data_dict[file_list[0]]['Simulation']['std_G_half']

G0_2          = data_dict[file_list[1]]['Simulation']['avg_G0']
G_half_2      = data_dict[file_list[1]]['Simulation']['avg_G_half']
std_G0_2      = data_dict[file_list[1]]['Simulation']['std_G0']
std_G_half_2  = data_dict[file_list[1]]['Simulation']['std_G_half']

G0_3          = data_dict[file_list[2]]['Simulation']['avg_G0']
G_half_3      = data_dict[file_list[2]]['Simulation']['avg_G_half']
std_G0_3      = data_dict[file_list[2]]['Simulation']['std_G0']
std_G_half_3  = data_dict[file_list[2]]['Simulation']['std_G_half']

flux         = data_dict[file_list[3]]['Simulation']['flux']
G_array      = data_dict[file_list[3]]['Simulation']['G_array']
gap_array    = data_dict[file_list[3]]['Simulation']['gap_array']
width        = data_dict[file_list[3]]['Simulation']['width']
Ef           = data_dict[file_list[3]]['Parameters']['Ef']

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
palette = seaborn.color_palette(palette='viridis_r', n_colors=200)
palette = [palette[0], palette[50], palette[100] , palette[130], palette[-1]]


# Figure 1: Definition
fig1 = plt.figure(figsize=(8, 6))
gs = GridSpec(2, 3, figure=fig1, wspace=0.2, hspace=0.4)
ax1 = fig1.add_subplot(gs[0, 0])
ax2 = fig1.add_subplot(gs[0, 1])
ax3 = fig1.add_subplot(gs[0, 2])
ax4 = fig1.add_subplot(gs[1, :])


# Upper panel: Plots
ax1.plot(fermi, G0_2, color='#9A32CD', )
ax1.plot(fermi, G_half_2, color='#3F6CFF', alpha=0.5)
ax1.text(0.2, 1, f'$N_s=300$', fontsize=fontsize_in)
ax1.text(0.05, 5.3, f'$w= {width1}$',  fontsize=fontsize_in)
ax1.set(yticks=[0, 2, 4, 6])

ax2.plot(fermi, G0_2, label=f'$\phi / \phi_0= {flux0}$', color='#9A32CD')
ax2.plot(fermi, G_half_2, label=f'$\phi / \phi_0= {flux_half}$', color='#3F6CFF', alpha=0.5)
ax2.text(0.2, 1, '$\Delta E_F=1$', fontsize=fontsize_in)
ax2.set(yticks=[0, 2, 4, 6], yticklabels=[])
ax2.text(0.05, 5.3, f'$w= {width2}$', fontsize=fontsize)

ax3.plot(fermi, G0_3, label=f'$ {flux0}$', color='#9A32CD')
ax3.plot(fermi, G_half_3, label=f'${flux_half}$', color='#3F6CFF', alpha=0.5)
ax3.text(0.05, 5.3, f'$w= {width3}$', fontsize=fontsize_in)
ax3.set(yticks=[0, 2, 4, 6], yticklabels=[])
ax3.legend(ncol=1, frameon=False, fontsize=fontsize, loc='upper left', columnspacing=0.3, handlelength=0.75, labelspacing=0.2, bbox_to_anchor=(0.35, 0.45))
ax3.text(0.32, 3, '$\\underline{\phi/\phi_0}$', fontsize=fontsize_in)

# Upper panel: Format
# y_axis_ticks = [i for i in range(0, 8, 2)]
# y_axis_labels = [str(i) for i in range(0, 8, 2)]
ax1.set_ylabel("$G(2e^2/h)$",fontsize=fontsize)
# ax1.set(yticks=y_axis_ticks, yticklabels=y_axis_labels)
ax_vec = [ax1, ax2, ax3]
for ax in ax_vec:
    ax.set_xlim(fermi[0], 0.5)
    ax.set_ylim(0, 7)
    ax.set(xticks=[0, 0.25, 0.5], xticklabels=['0', '0.25', '0.5'])
    ax.tick_params(which='major', width=0.75, labelsize=fontsize)
    ax.tick_params(which='major', length=6, labelsize=fontsize)
    ax.set_xlabel("$E_F$", fontsize=fontsize)



# Lower panel: Plots
for i in range(G_array.shape[1]):
    label = f'${width[i] :.1f}$'
    ax4.plot(flux, G_array[0, i, :], color=palette[i], linestyle='solid', label=label)
    ax4.plot(flux, gap_array[i, :], color=palette[i], linestyle='dashed', label=None, alpha=0.3)
ax4.plot(flux, 1 * np.ones(flux.shape), '--', color='Black', alpha=0.2)

ax4.text(0.05, 1.2, '$\\underline{w}$', fontsize=fontsize)
ax4.legend(loc='upper left', ncol=3, alignment='left', frameon=False, fontsize=fontsize, columnspacing=0.4, handlelength=1, labelspacing=0.2,bbox_to_anchor=(0.02, 1.1))
ax4.text(4.1, 1.2, '$E_F^{nw}=$' + f'${Ef[0]}$', fontsize=fontsize)
ax4.text(3, 1.2,  '$E_F^{lead}=$' + f'${Ef[0] + 1}$', fontsize=fontsize)


ax4.set_ylim([0, 1.5])
ax4.set_xlim(0, 5)
ax4.set_ylabel("$G(2e^2/h)$", fontsize=fontsize)
ax4.set_xlabel("$\phi$", fontsize=fontsize, labelpad=-10)
ax4.tick_params(which='major', width=0.75, labelsize=fontsize)
ax4.tick_params(which='major', length=6, labelsize=fontsize)
ax4.tick_params(which='minor', width=0.75, labelsize=fontsize)
ax4.tick_params(which='minor', length=3, labelsize=fontsize)


fig1.savefig('fig2-layer-dis.pdf', format='pdf', backend='pgf')
plt.show()

