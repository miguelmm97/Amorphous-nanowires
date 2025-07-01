#%% modules setup

# Math and plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn

# modules
from functions import *


#%% Loading data
file_list = ['fig2-avg-osc-w-002.h5', 'fig2-avg-osc-w-005.h5', 'fig2-avg-osc-w-01.h5',
             'fig2-GvsW.h5', 'fig2-G-highW.h5']
data_dict = load_my_data(file_list, '../data')

# Parameters
width1    = data_dict[file_list[0]]['Parameters']['width']
width2    = data_dict[file_list[1]]['Parameters']['width']
width3    = data_dict[file_list[2]]['Parameters']['width']
flux0     = data_dict[file_list[2]]['Parameters']['flux0']
flux_half = data_dict[file_list[2]]['Parameters']['flux_half']

# Simulation data
fermi         = data_dict[file_list[0]]['Simulation']['fermi']
G0_1          = data_dict[file_list[0]]['Simulation']['avg_G0']
G_half_1      = data_dict[file_list[0]]['Simulation']['avg_G_half']
std_G0_1      = data_dict[file_list[0]]['Simulation']['std_G0']
std_G_half_1  = data_dict[file_list[0]]['Simulation']['std_G_half']
error10_top, error10_bottom = G0_1 + 0.5 * std_G0_1, G0_1 - 0.5 * std_G0_1
error11_top, error11_bottom = G_half_1 + 0.5 * std_G_half_1, G_half_1 - 0.5 * std_G_half_1

G0_2          = data_dict[file_list[1]]['Simulation']['avg_G0']
G_half_2      = data_dict[file_list[1]]['Simulation']['avg_G_half']
std_G0_2      = data_dict[file_list[1]]['Simulation']['std_G0']
std_G_half_2  = data_dict[file_list[1]]['Simulation']['std_G_half']
error20_top, error20_bottom = G0_2 + 0.5 * std_G0_2, G0_2 - 0.5 * std_G0_2
error21_top, error21_bottom = G_half_2 + 0.5 * std_G_half_2, G_half_2 - 0.5 * std_G_half_2

G0_3          = data_dict[file_list[2]]['Simulation']['avg_G0']
G_half_3      = data_dict[file_list[2]]['Simulation']['avg_G_half']
std_G0_3      = data_dict[file_list[2]]['Simulation']['std_G0']
std_G_half_3  = data_dict[file_list[2]]['Simulation']['std_G_half']
error30_top, error30_bottom = G0_3 + 0.5 * std_G0_3, G0_3 - 0.5 * std_G0_3
error31_top, error31_bottom = G_half_3 + 0.5 * std_G_half_3, G_half_3 - 0.5 * std_G_half_3

flux         = data_dict[file_list[3]]['Simulation']['flux']
G_array      = data_dict[file_list[3]]['Simulation']['G_array']
gap_array    = data_dict[file_list[3]]['Simulation']['gap_array']
width        = data_dict[file_list[3]]['Simulation']['width']
Ef           = data_dict[file_list[3]]['Parameters']['Ef']
G_high_w     = data_dict[file_list[4]]['Simulation']['G_array']
gap_high_w   = data_dict[file_list[4]]['Simulation']['gap_array']

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
palette = seaborn.color_palette(palette='viridis_r', n_colors=250)
palette = [palette[0], palette[50], palette[100], palette[150], palette[200], palette[-1]]


# Definition
fig1 = plt.figure(figsize=(8, 8))
gs = GridSpec(5, 3, figure=fig1, wspace=0.2, hspace=1.5)
ax1 = fig1.add_subplot(gs[0:2, 0])
ax2 = fig1.add_subplot(gs[0:2, 1])
ax3 = fig1.add_subplot(gs[0:2, 2])
gs2 = gs[2:, :].subgridspec(3, 3, hspace=0.3)
ax4 = fig1.add_subplot(gs2[:2, :])
ax5 = fig1.add_subplot(gs2[2, :])


# Upper panel: Plots
ax1.plot(fermi, G0_1, color='#9A32CD')
ax1.plot(fermi, G_half_1, color='#3F6CFF', alpha=0.5)
ax1.fill_between(fermi, error10_bottom, error10_top, color='#9A32CD', alpha=0.2)
ax1.fill_between(fermi, error11_bottom, error11_top, color='#3F6CFF', alpha=0.2)
ax1.text(0.02, 4.8, f'$w= {width1}$',  fontsize=fontsize_in)
ax1.text(0.02, 6, f'$(a)$',  fontsize=fontsize_in)
ax1.set(yticks=[0, 2, 4, 6])

ax2.plot(fermi, G0_2, label=f'$\phi / \phi_0= {flux0}$', color='#9A32CD')
ax2.plot(fermi, G_half_2, label=f'$\phi / \phi_0= {flux_half}$', color='#3F6CFF', alpha=0.5)
ax2.fill_between(fermi, error20_bottom, error20_top, color='#9A32CD', alpha=0.2)
ax2.fill_between(fermi, error21_bottom, error21_top, color='#3F6CFF', alpha=0.2)
ax2.set(yticks=[0, 2, 4, 6], yticklabels=[])
ax2.text(0.02, 4.8, f'$w= {width2}$',  fontsize=fontsize_in)
ax2.text(0.02, 6, f'$(b)$',  fontsize=fontsize_in)

ax3.plot(fermi, G0_3, label=f'$ {flux0}$', color='#9A32CD')
ax3.plot(fermi, G_half_3, label=f'${flux_half}$', color='#3F6CFF', alpha=0.5)
ax3.fill_between(fermi, error30_bottom, error30_top, color='#9A32CD', alpha=0.2)
ax3.fill_between(fermi, error31_bottom, error31_top, color='#3F6CFF', alpha=0.2)
ax3.text(0.02, 4.8, f'$w= {width3}$',  fontsize=fontsize_in)
ax3.text(0.02, 6, f'$(c)$',  fontsize=fontsize_in)
ax3.set(yticks=[0, 2, 4, 6], yticklabels=[])
ax3.legend(ncol=1, frameon=False, fontsize=fontsize, loc='upper left', columnspacing=0.3, handlelength=0.75, labelspacing=0.2, bbox_to_anchor=(0.35, 0.45))
ax3.text(0.34, 3, '$\\underline{\phi/\phi_0}$', fontsize=fontsize_in)

# Upper panel: Format
ax1.set_ylabel("$\\overline{G}(e^2/h)$",fontsize=fontsize)
ax_vec = [ax1, ax2, ax3]
for ax in ax_vec:
    ax.set_xlim(fermi[0], 0.5)
    ax.set_ylim(0, 7)
    ax.set(xticks=[0, 0.25, 0.5], xticklabels=['0', '0.25', '0.5'])
    ax.tick_params(which='major', width=0.75, labelsize=fontsize)
    ax.tick_params(which='major', length=6, labelsize=fontsize)
    ax.set_xlabel("$E_F$", fontsize=fontsize)



# Lower panel: Conductance/ gap vs magnetic flux and
for i in range(G_array.shape[1]):
    label = f'${width[i] :.2f}$'
    ax4.plot(flux, G_array[0, i, :], color=palette[i], linestyle='solid', label=label)
    ax5.plot(flux, gap_array[i, :], color=palette[i], linestyle='solid', label=None)
ax4.plot(flux, 1 * np.ones(flux.shape), '--', color='Black', alpha=0.2)
ax4.plot(flux, G_high_w[0, 0, :], color='Purple', linestyle='dotted', label='$0.25$', alpha=0.5)
ax5.plot(flux, gap_high_w[0, :], color='Purple', linestyle='dotted', label=None)

ax4.set_ylabel("$G(e^2/h)$", fontsize=fontsize)
ax5.set_ylabel("$E_g$", fontsize=fontsize, labelpad=-20)
ax5.set_xlabel("$\phi/\phi_0$", fontsize=fontsize, labelpad=-10)
ax4.set_ylim([0, 1.75])
ax4.set_xlim(0, 5)
ax5.set_ylim([0, 0.15])
ax5.set_xlim(0, 5)

ax4.text(1.8, 1.4, '$\\underline{w}$', fontsize=fontsize)
ax4.legend(loc='upper left', ncol=3, alignment='left', frameon=False, fontsize=fontsize, columnspacing=0.4, handlelength=1, labelspacing=0.2,bbox_to_anchor=(0.4, 1.05))
ax4.text(0.3, 1.5, f'$(d)$',  fontsize=fontsize_in)
ax5.text(0.3, 0.1, f'$(e)$',  fontsize=fontsize_in)

ax4.tick_params(which='major', width=0.75, labelsize=fontsize)
ax4.tick_params(which='major', length=6, labelsize=fontsize)
ax4.tick_params(which='minor', width=0.75, labelsize=fontsize)
ax4.tick_params(which='minor', length=3, labelsize=fontsize)
ax4.set(xticks=[0, 1, 2, 3, 4, 5], xticklabels=[])
ax5.tick_params(which='major', width=0.75, labelsize=fontsize)
ax5.tick_params(which='major', length=6, labelsize=fontsize)
ax5.tick_params(which='minor', width=0.75, labelsize=fontsize)
ax5.tick_params(which='minor', length=3, labelsize=fontsize)
ax5.set(yticks=[0, 0.15])


fig1.savefig('fig2.pdf', format='pdf')
plt.show()

