#%% modules setup

# Math and plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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


# Figure 1: Definition
fig1 = plt.figure(figsize=(10, 10))
gs = GridSpec(3, 3, figure=fig1, wspace=0.2, hspace=0.5)
ax1 = fig1.add_subplot(gs[0, 0])
ax2 = fig1.add_subplot(gs[0, 1])
ax3 = fig1.add_subplot(gs[0, 2])
ax4 = fig1.add_subplot(gs[1, :])
ax5 = fig1.add_subplot(gs[2, :])

# Upper panel: Plots
ax1.plot(fermi, G0_2, color='#9A32CD', )
ax1.plot(fermi, G_half_2, color='#3F6CFF', alpha=0.5)
ax1.text(0.25, 1, f'$w= {width1}$',  fontsize=fontsize_in)
ax1.text(0.05, 5, f'$N=300$',  fontsize=fontsize_in)

ax2.plot(fermi, G0_2, label=f'$\phi / \phi_0= {flux0}$', color='#9A32CD')
ax2.plot(fermi, G_half_2, label=f'$\phi / \phi_0= {flux_half}$', color='#3F6CFF', alpha=0.5)
ax2.text(0.25, 1, f'$w= {width2}$', fontsize=fontsize_in)
ax2.legend(ncol=2, frameon=False, fontsize=fontsize, loc='upper center', bbox_to_anchor=(0.5, 1.4))

ax3.plot(fermi, G0_3, color='#9A32CD')
ax3.plot(fermi, G_half_3, color='#3F6CFF', alpha=0.5)
ax3.text(0.25, 1, f'$w= {width3}$', fontsize=fontsize_in)

# Upper panel: Format
y_axis_ticks = [i for i in range(0, 8, 2)]
y_axis_labels = [str(i) for i in range(0, 8, 2)]
ax1.set_ylabel("$G(2e^2/h)$",fontsize=fontsize)
ax1.set(yticks=y_axis_ticks, yticklabels=y_axis_labels)
ax_vec = [ax1, ax2, ax3]
for ax in ax_vec:
    ax.set_xlim(fermi[0], fermi[-1])
    ax.set_ylim(0, 7)
    ax.set_xlim(0, 0.5)
    ax.tick_params(which='major', width=0.75, labelsize=fontsize)
    ax.tick_params(which='major', length=6, labelsize=fontsize)
    ax.set_xlabel("$E_F / t$", fontsize=fontsize)



# Lower panel: Plots
ax_vec = [ax4, ax5]
for i in range(G_array.shape[1]):
    for j in range(len(Ef)):
        ax = ax_vec[j]
        label = f'$w= {width[i]}$'
        ax.plot(flux, G_array[j, i, :], color=color_list[i], linestyle='solid', label=label)
        ax.plot(flux, gap_array[i, :], color=color_list[i], linestyle='dashed', label=None, alpha=0.3)
ax4.plot(flux, 1 * np.ones(flux.shape), '--', color='Black', alpha=0.2)
ax5.plot(flux, 1 * np.ones(flux.shape), '--', color='Black', alpha=0.2)
ax4.text(0.8, 0.6, f'$E_f= {Ef[0]}$', fontsize=fontsize_in)
ax5.text(0.8, 0.8, f'$E_f= {Ef[1]}$', fontsize=fontsize_in)
ax5.text(2.9, 0.1, f'$E_g$', fontsize=fontsize_in)


# Lower panel: Format
ax5.legend(loc='upper center', ncol=3, alignment='center', frameon=False, fontsize=fontsize, bbox_to_anchor=(0.5, 1.5))
ax5.set_xlabel("$\phi$", fontsize=fontsize)
ylim = 1.1
ax4.set(xticks=[0, 1, 2, 3, 4, 5], xticklabels=[])
for ax in ax_vec:
    ax.set_xlim(flux[0], flux[-1])
    ax.set_ylim(0, ylim)
    ax.tick_params(which='major', width=0.75, labelsize=fontsize)
    ax.tick_params(which='major', length=6, labelsize=fontsize)
    ax.set_ylabel("$G(2e^2/h)$", fontsize=fontsize)

fig1.savefig('draft-fig2.pdf', format='pdf', backend='pgf')
plt.show()

