#%% modules setup

# Math and plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn

# modules
from modules.functions import *


#%% Loading data
file_list = ['Exp2.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-cond-vs-Nxy')

# Parameters
Ef           = data_dict[file_list[0]]['Parameters']['Ef']
Nz           = data_dict[file_list[0]]['Parameters']['Nz']
r            = data_dict[file_list[0]]['Parameters']['r']
t            = data_dict[file_list[0]]['Parameters']['t']
eps          = data_dict[file_list[0]]['Parameters']['eps']
lamb         = data_dict[file_list[0]]['Parameters']['lamb']
lamb_z       = data_dict[file_list[0]]['Parameters']['lamb_z']
mu_leads     = data_dict[file_list[0]]['Parameters']['mu_leads']

# Simulation data
flux          = data_dict[file_list[0]]['Simulation']['flux']
width         = data_dict[file_list[0]]['Simulation']['width']
Nx            = data_dict[file_list[0]]['Simulation']['Nx']
Ny            = data_dict[file_list[0]]['Simulation']['Ny']
G_array       = data_dict[file_list[0]]['Simulation']['G_array']
K             = data_dict[file_list[0]]['Simulation']['K']

#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange']
palette = seaborn.color_palette(palette='magma', n_colors=G_array.shape[3])
markersize = 5
fontsize=20

# Figure 1: Definition
fig1 = plt.figure(figsize=(20, 10))
gs = GridSpec(3, 1, figure=fig1, wspace=0.13, hspace=0.3)
ax1_1 = fig1.add_subplot(gs[0, 0])
ax1_2 = fig1.add_subplot(gs[1, 0])
ax1_3 = fig1.add_subplot(gs[2, 0])
ax_vec1 = [ax1_1, ax1_2, ax1_3]

# Figure 2: Definition
fig2 = plt.figure(figsize=(20, 10))
gs = GridSpec(3, 1, figure=fig2, wspace=0.13, hspace=0.3)
ax2_1 = fig2.add_subplot(gs[0, 0])
ax2_2 = fig2.add_subplot(gs[1, 0])
ax2_3 = fig2.add_subplot(gs[2, 0])
ax_vec2 = [ax2_1, ax2_2, ax2_3]

# Figure 3: Definition
fig3 = plt.figure(figsize=(20, 10))
gs = GridSpec(3, 1, figure=fig3, wspace=0.13, hspace=0.3)
ax3_1 = fig3.add_subplot(gs[0, 0])
ax3_2 = fig3.add_subplot(gs[1, 0])
ax3_3 = fig3.add_subplot(gs[2, 0])
ax_vec3 = [ax3_1, ax3_2, ax3_3]

# Figure 4: Definition
fig4 = plt.figure(figsize=(20, 10))
gs = GridSpec(4, 1, figure=fig3, wspace=0.13, hspace=0.3)
ax4_1 = fig4.add_subplot(gs[0, 0])
ax4_2 = fig4.add_subplot(gs[1, 0])
ax4_3 = fig4.add_subplot(gs[2, 0])
ax_vec4 = [ax4_1, ax4_2, ax4_3]

ax_vec = [ax_vec1, ax_vec2, ax_vec3, ax_vec4]
fig_list = [fig1, fig2, fig3, fig4]

sample_K = 0
for sample_w in range(len(width)):

    # Figure: Plots
    for i in range(len(Ef)):
        ax = ax_vec[sample_w][i]
        ax.plot(flux, 1 * np.ones(flux.shape), '--', color='Black', alpha=0.2)
        ax.set_title(f'$w= {width[sample_w]}$, $E_f = {Ef[i]}$', fontsize=fontsize)
        for j in range(len(Nx)):
            label = f'$N_x= {Nx[j]}$'
            ax.plot(flux, G_array[:, sample_w, sample_K, j, i], color=palette[j], linestyle='solid', label=label)

    # Figure 1: Format
    ax1_1.legend(ncol=2, frameon=False, fontsize=10)
    fig_list[sample_w].suptitle(f'$\mu_l= {mu_leads}$, $r= {r}$, $N_z= {Nz}$', y=0.93, fontsize=20)
    for ax in ax_vec[sample_w]:
        ax.set_xlim(flux[0], flux[-1])
        ax.set_ylim(0, np.max(G_array))
        ax.tick_params(which='major', width=0.75, labelsize=10)
        ax.tick_params(which='major', length=6, labelsize=10)
        ax.set_xlabel("$\phi$", fontsize=fontsize)
        ax.set_ylabel("$G[2e^2/h]$", fontsize=fontsize)
    fig_list[sample_w].savefig(f'../figures/{file_list[0]}-G-vs-Nxy-w1.pdf', format='pdf', backend='pgf')


plt.show()
