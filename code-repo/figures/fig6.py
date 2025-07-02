#%% modules setup

# Math and plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn

# modules
from functions import *


#%% Loading data
file_list = ['fig6-GvsEf.h5', 'fig6-G-N10.h5',  'fig6-G-N15.h5', 'fig6-G-N20.h5']
data_dict = load_my_data(file_list, '../data')

# Parameters
Nx           = data_dict[file_list[0]]['Parameters']['Nx']
Ny           = data_dict[file_list[0]]['Parameters']['Ny']
Nz            = data_dict[file_list[0]]['Parameters']['Nz']
r            = data_dict[file_list[0]]['Parameters']['r']
t            = data_dict[file_list[0]]['Parameters']['t']
eps          = data_dict[file_list[0]]['Parameters']['eps']
lamb         = data_dict[file_list[0]]['Parameters']['lamb']
lamb_z       = data_dict[file_list[0]]['Parameters']['lamb_z'] # referred to as eta in the main text
mu_leads     = data_dict[file_list[0]]['Parameters']['mu_leads']

# Simulation data
G0            = data_dict[file_list[0]]['Simulation']['G_0']
Ghalf         = data_dict[file_list[0]]['Simulation']['G_half']
fermi         = data_dict[file_list[0]]['Simulation']['fermi']
width         = data_dict[file_list[0]]['Simulation']['width']
G1_inset      = data_dict[file_list[1]]['Simulation']['G_0']
G2_inset      = data_dict[file_list[2]]['Simulation']['G_0']
G3_inset      = data_dict[file_list[3]]['Simulation']['G_0']
fermi_inset   = data_dict[file_list[3]]['Simulation']['fermi']
N1            = data_dict[file_list[1]]['Parameters']['Nx']
N2            = data_dict[file_list[2]]['Parameters']['Nx']
N3            = data_dict[file_list[3]]['Parameters']['Nx']


#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange']
palette0 = seaborn.color_palette(palette='viridis_r', n_colors=100)
palette = [palette0[0], palette0[33], palette0[66], palette0[-1]]
palette2 = [palette0[10], palette0[43], palette0[76], 'k']
palette3 = seaborn.color_palette(palette='magma_r', n_colors=5)
marker_list=['o', 's', 'd', 'p', '*', 'h', '>', '<', 'X']
markersize = 5
fontsize=20

# Figure 1: Definition
fig1 = plt.figure(figsize=(8, 6))
gs = GridSpec(1, 1, figure=fig1, wspace=0.2, hspace=0.1)
ax1 = fig1.add_subplot(gs[0, 0])
ax1_inset = ax1.inset_axes([0.1, 0.72, 0.3, 0.25])


# Conductance vs Fermi energy
for i in range(G0.shape[-1]):
    ax1.plot(fermi, G0[:, i], color=palette[i], label=f'${width[i] :.2f}$')
    ax1.plot(fermi, Ghalf[:, i], color=palette[i], linestyle='dashed', alpha=0.7)


ax1.text(0.09, 6.05, '$\\underline{w}$', fontsize=fontsize)
ax1.text(0.7, 6, '$\phi_{\mathrm{max}}$', fontsize=fontsize)
ax1.text(0.7, 5.2, '$\phi=0$', fontsize=fontsize)
ax1.plot(0.65, 6, marker='_', color='k', markersize=20)
ax1.legend(loc='upper left', frameon=False, fontsize=fontsize, bbox_to_anchor=(0.0, 0.6), handlelength=0.8,
           columnspacing=0.3, labelspacing=0.2, handletextpad=0.4)


ax1.set_xlabel("$E_F$", fontsize=fontsize, labelpad=-1)
ax1.set_ylabel("$G(e^2/h)$", fontsize=fontsize)
ax1.set_xlim(fermi[0], 0.8)
ax1.set_ylim(0, np.max(G0[:, 1]))

ax1.tick_params(which='major', width=0.75, labelsize=fontsize)
ax1.tick_params(which='major', length=6, labelsize=fontsize)

# Inset: G for different N
ax1_inset.plot(fermi_inset * N1, G1_inset, color=palette3[0], label=f'${N1}$', linestyle='solid', alpha=0.8)
ax1_inset.plot(fermi_inset * N2, G2_inset, color=palette3[1], label=f'${N2}$', linestyle='solid', alpha=0.8)
ax1_inset.plot(fermi_inset * N3, G3_inset, color=palette3[2], label=f'${N3}$', linestyle='solid', alpha=0.8)

ax1_inset.set_xlabel("$E_FN$", fontsize=fontsize, labelpad=-15)
ax1_inset.set_ylabel("$G(e^2/h)$", fontsize=fontsize-2, labelpad=-10)
ax1.text(0.375, 9.7, '$\\underline{N}$', fontsize=fontsize)
ax1_inset.legend(loc='upper left', ncol=1, frameon=False, fontsize=fontsize, columnspacing=0.3, handlelength=0.75,
                 labelspacing=0.2, handletextpad=0.4,  bbox_to_anchor=(0.93, 0.9))

y_axis_ticks = [0, 14]
x_axis_ticks = [0, 20]
ax1_inset.set_xlim(0, 20)
ax1_inset.set_ylim(0, 14)
ax1_inset.tick_params(which='major', width=0.75, labelsize=fontsize)
ax1_inset.tick_params(which='major', length=6, labelsize=fontsize)
ax1_inset.set(yticks=y_axis_ticks)
ax1_inset.set(xticks=x_axis_ticks)

fig1.savefig('fig6.pdf', format='pdf', backend='pgf')
plt.show()
