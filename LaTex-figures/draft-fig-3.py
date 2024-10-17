#%% modules setup

# Math and plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# modules
from modules.functions import *


#%% Loading data
file_list = ['draft-fig3-G-vs-flux-1.h5', 'draft-fig3-G-vs-flux-2.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-latex-figures')


# Simulation data
flux         = data_dict[file_list[0]]['Simulation']['flux']
G_array_1      = data_dict[file_list[0]]['Simulation']['G_array']
width        = data_dict[file_list[0]]['Simulation']['width']
Ef           = data_dict[file_list[0]]['Parameters']['Ef']
G_array_2    = data_dict[file_list[0]]['Simulation']['G_array']


G_array = np.zeros((2, 5, 300))
G_array[:, [0, 2, 3, 4], :] = G_array_1[:, [0, 1, 2, 3], :]
G_array[:, 1, :] = G_array_1[:, 0, :]
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
fig1 = plt.figure(figsize=(10, 7))
gs = GridSpec(2, 1, figure=fig1, wspace=0.2, hspace=0.1)
ax1 = fig1.add_subplot(gs[0, 0])
ax2 = fig1.add_subplot(gs[1, 0])


ax_vec = [ax1, ax2]
for i in range(G_array.shape[1]):
    for j in range(len(Ef)):
        ax = ax_vec[j]
        label = f'$w= {width[i]}$'
        ax.plot(flux, G_array[j, i, :], color=color_list[i], linestyle='solid', label=label)
ax1.plot(flux, 1 * np.ones(flux.shape), '--', color='Black', alpha=0.2)
ax2.plot(flux, 1 * np.ones(flux.shape), '--', color='Black', alpha=0.2)
ax1.text(0.9, 0.6, f'$E_f= {Ef[0]}$', fontsize=fontsize)
ax2.text(0.8, 0.8, f'$E_f= {Ef[1]}$', fontsize=fontsize)


# Figure 1: Format
ax1.legend(loc='upper center', ncol=3, alignment='center', frameon=False, fontsize=fontsize, bbox_to_anchor=(0.5, 1.4))
ax2.set_xlabel("$\phi$", fontsize=fontsize)
ylim = 1.1
ax1.set(xticks=[0, 1, 2, 3, 4, 5], xticklabels=[])
ax1.set(yticks=[0, 0.5, 1])
ax2.set(yticks=[0, 0.5, 1])
for ax in ax_vec:
    ax.set_xlim(flux[0], flux[-1])
    ax.set_ylim(0, ylim)
    ax.tick_params(which='major', width=0.75, labelsize=fontsize)
    ax.tick_params(which='major', length=6, labelsize=fontsize)
    ax.set_ylabel("$G(2e^2/h)$", fontsize=fontsize)

fig1.savefig('draft-fig3.pdf', format='pdf', backend='pgf')
plt.show()

