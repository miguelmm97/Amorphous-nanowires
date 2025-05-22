#%% modules setup

# Math and plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn

# modules
from modules.functions import *


#%% Loading data
file_list = ['data-cluster-Gmax-L=100.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-cluster')

# Simulation data
N            = data_dict[file_list[0]]['Simulation']['N']
Nz           = data_dict[file_list[0]]['Simulation']['Nz']
Ef           = data_dict[file_list[0]]['Simulation']['Ef']
flux         = data_dict[file_list[0]]['Simulation']['flux']
width        = data_dict[file_list[0]]['Simulation']['width']
Gmax         = data_dict[file_list[0]]['Simulation']['Gmax']
Gmax_std     = data_dict[file_list[0]]['Simulation']['Gmax_std']
deltaG       = data_dict[file_list[0]]['Simulation']['deltaG']
deltaG_std   = data_dict[file_list[0]]['Simulation']['deltaG_std']
Nsamples     = data_dict[file_list[0]]['Simulation']['sample']

# Parameters
K_onsite     = data_dict[file_list[0]]['Parameters']['K_onsite']
r            = data_dict[file_list[0]]['Parameters']['r']
t            = data_dict[file_list[0]]['Parameters']['t']
eps          = data_dict[file_list[0]]['Parameters']['eps']
lamb         = data_dict[file_list[0]]['Parameters']['lamb']
lamb_z       = data_dict[file_list[0]]['Parameters']['lamb_z']
mu_leads     = data_dict[file_list[0]]['Parameters']['mu_leads']

G_error_top, G_error_bottom = Gmax + 0.5 * Gmax_std, Gmax - 0.5 * Gmax_std
deltaG_error_top, deltaG_error_bottom = deltaG + 0.5 * deltaG_std, deltaG - 0.5 * deltaG_std
#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange']
marker_list=['o', 's', 'd', 'p', '*', 'h', '>', '<', 'X']
markersize = 5
fontsize=20

# Figure 1: Definition
fig1 = plt.figure(figsize=(15, 8))
gs = GridSpec(1, 2, figure=fig1, wspace=0.2, hspace=0.3)
ax1 = fig1.add_subplot(gs[0, 0])
ax2 = fig1.add_subplot(gs[0, 1])
fig1.suptitle(f'$\mu_l= {mu_leads}$, $E_f= {Ef}$, $K= {K_onsite:.2f}$, $L= {Nz}$, $w= {width}$, $N_s= {Nsamples}$', y=0.93, fontsize=20)

# Figure 1: Plots
ax1.plot(N, Gmax, color='limegreen', marker='o', linestyle='solid')
ax1.fill_between(N, G_error_bottom, G_error_top, color='limegreen', alpha=0.2)
ax1.tick_params(which='major', width=0.75, labelsize=10)
ax1.tick_params(which='major', length=6, labelsize=10)
ax1.set_xlabel("$N$", fontsize=fontsize)
ax1.set_ylabel("$G[2e^2/h]$", fontsize=fontsize)
ax1.set_ylim(0, 1.1)

ax2.plot(N, deltaG, color='limegreen', marker='o', linestyle='solid')
ax2.fill_between(N, deltaG_error_bottom, deltaG_error_top, color='limegreen', alpha=0.2)
ax2.tick_params(which='major', width=0.75, labelsize=10)
ax2.tick_params(which='major', length=6, labelsize=10)
ax2.set_xlabel("$N$", fontsize=fontsize)
ax2.set_ylabel("$(G^* - G) / G^*$", fontsize=fontsize)
ax2.set_ylim(-0.2, 0.2)



# fig1.savefig(f'../figures/{file_list[0]}-cond-vs-N.pdf', format='pdf', backend='pgf')
plt.show()
