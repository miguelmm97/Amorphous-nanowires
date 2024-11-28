#%% Modules setup

# Math and plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# modules
from modules.functions import *

#%% Loading data
file_list = ['Exp27.h5', 'Exp22.h5', 'Exp26.h5', 'Exp23.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-cond-vs-Ef')

# Parameters
Nx           = data_dict[file_list[0]]['Parameters']['Nx']
Ny           = data_dict[file_list[0]]['Parameters']['Ny']
Nz           = data_dict[file_list[0]]['Parameters']['Nz']
r            = data_dict[file_list[0]]['Parameters']['r']
t            = data_dict[file_list[0]]['Parameters']['t']
eps          = data_dict[file_list[0]]['Parameters']['eps']
lamb         = data_dict[file_list[0]]['Parameters']['lamb']
lamb_z       = data_dict[file_list[0]]['Parameters']['lamb_z']
mu_leads     = data_dict[file_list[0]]['Parameters']['mu_leads']
params_dict  = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}

# File 1
flux_half_1    = data_dict[file_list[0]]['Simulation']['flux_max']
G_0_1          = data_dict[file_list[0]]['Simulation']['G_0']
G_half_1       = data_dict[file_list[0]]['Simulation']['G_half']
fermi_1        = data_dict[file_list[0]]['Simulation']['fermi']
width_1        = data_dict[file_list[0]]['Simulation']['width']

# File 2
flux_half_2    = data_dict[file_list[1]]['Simulation']['flux_max']
G_0_2          = data_dict[file_list[1]]['Simulation']['G_0']
G_half_2       = data_dict[file_list[1]]['Simulation']['G_half']
fermi_2        = data_dict[file_list[1]]['Simulation']['fermi']
width_2        = data_dict[file_list[1]]['Simulation']['width']

# File 3
flux_half_3    = data_dict[file_list[2]]['Simulation']['flux_max']
G_0_3          = data_dict[file_list[2]]['Simulation']['G_0']
G_half_3       = data_dict[file_list[2]]['Simulation']['G_half']
fermi_3        = data_dict[file_list[2]]['Simulation']['fermi']
K_onsite_3     = data_dict[file_list[2]]['Simulation']['K_onsite']

# File 4
flux_half_4    = data_dict[file_list[3]]['Simulation']['flux_max']
G_0_4          = data_dict[file_list[3]]['Simulation']['G_0']
G_half_4       = data_dict[file_list[3]]['Simulation']['G_half']
fermi_4        = data_dict[file_list[3]]['Simulation']['fermi']
K_onsite_4     = data_dict[file_list[3]]['Simulation']['K_onsite']


#%% Figures

font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange']
marker_list=['o', 's', 'd', 'p', '*', 'h', '>', '<', 'X']
line_list = ['solid', 'dashed', 'dashdot', 'dotted']
markersize = 5
fontsize=20
site_size  = 0.1
site_lw    = 0.01
site_color = 'm'
hop_color  = 'royalblue'
hop_lw     = 0.05
lead_color = 'r'

fig1 = plt.figure(figsize=(25, 7))
gs = GridSpec(1, 4, figure=fig1, wspace=0.1, hspace=0.0)
ax1 = fig1.add_subplot(gs[0, 0])
ax2 = fig1.add_subplot(gs[0, 1])
ax3 = fig1.add_subplot(gs[0, 2])
ax4 = fig1.add_subplot(gs[0, 3])

ax_amor = [ax1, ax2]
ax_dis = [ax3, ax4]

# Amorphous plots
for i in range(len(G_0_1[0, :-1])):

    label1 = '$\phi_{max}$' if i==0 else None
    label2 = '$\phi=0$' if i==0 else None
    ax1.plot(fermi_1, G_0_1[:, i],    color=color_list[i], label=f'$w= {width_1[i]}$', linestyle='solid')
    ax1.plot(fermi_1, G_half_1[:, i], color=color_list[i], alpha=0.5, linestyle='dotted')
    ax2.plot(fermi_2, G_0_2[:, i],    color=color_list[i], label=label1, linestyle='solid')
    ax2.plot(fermi_2, G_half_2[:, i], color=color_list[i], label=label2, alpha=0.5, linestyle='dotted')

for ax in ax_amor:
    y_axis_ticks = [i for i in range(0, 11, 2)]
    y_axis_labels = ['' for i in range(0, 11, 2)]
    ax.set_xlim(fermi_1[0], fermi_1[-1])
    ax.set_ylim(0, 10)
    ax.tick_params(which='major', width=0.75, labelsize=fontsize)
    ax.tick_params(which='major', length=6, labelsize=fontsize)
    ax.set_xlabel("$E_F / t$", fontsize=fontsize)
    ax.set(yticks=y_axis_ticks, yticklabels=y_axis_labels)


y_axis_ticks = [i for i in range(0, 11, 2)]
y_axis_labels = [str(i) for i in range(0, 11, 2)]
ax1.set_ylabel("$G(2e^2/h)$", fontsize=fontsize)
ax1.set(yticks=y_axis_ticks, yticklabels=y_axis_labels)
ax1.legend(ncol=1, frameon=False, fontsize=fontsize)
ax2.legend(ncol=1, frameon=False, fontsize=fontsize)
ax1.text(0.05, 5.5, '$\mu=0$', fontsize=fontsize)
ax2.text(0.05, 7.2, '$\mu=-t$', fontsize=fontsize)


# Disorder plots
for i in range(len(G_0_3[0, :-1])):
    ax3.plot(fermi_3, G_0_3[:, i],    color=color_list[i],  label=f'$K= {K_onsite_3[i]}$', linestyle='solid')
    ax3.plot(fermi_3, G_half_3[:, i], color=color_list[i], alpha=0.5, linestyle='dotted')
    ax4.plot(fermi_4, G_0_4[:, i],    color=color_list[i],  label=f'$K= {K_onsite_4[i]}$', linestyle='solid')
    ax4.plot(fermi_4, G_half_4[:, i], color=color_list[i], alpha=0.5, linestyle='dotted')

for ax in ax_dis:
    y_axis_ticks = [i for i in range(0, 11, 2)]
    y_axis_labels = ['' for i in range(0, 11, 2)]
    ax.set_xlim(fermi_1[0], fermi_1[-1])
    ax.set_ylim(0, 10)
    ax.tick_params(which='major', width=0.75, labelsize=fontsize)
    ax.tick_params(which='major', length=6, labelsize=fontsize)
    ax.set_xlabel("$E_F / t$", fontsize=fontsize)
    ax.set(yticks=y_axis_ticks, yticklabels=y_axis_labels)
ax3.legend(ncol=1, frameon=False, fontsize=fontsize)
ax3.text(0.05, 5.5, '$\mu=0$', fontsize=fontsize)
ax4.text(0.05, 7, '$\mu=-t$', fontsize=fontsize)
ax4.text(0.05, 8, f'$N_x = N_y = {Nx}$', fontsize=fontsize)
ax4.text(0.05, 9, f'$L = {Nz}$', fontsize=fontsize)


fig1.savefig(f'draft-fig6.pdf', format='pdf', backend='pgf')
plt.show()











# ax1.plot(fermi, G_0[:, i], color=color_list[i], label=f'$K= {K_onsite[i]}$', linestyle='solid')
# # Nanowire structure
# lattice = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=Nz, w=width[0], r=r)
# lattice.set_configuration(x, y, z)
# lattice.build_lattice()
# nanowire = promote_to_kwant_nanowire3d(lattice, params_dict, mu_leads=mu_leads).finalized()
#
# fig2 = plt.figure()
# gs = GridSpec(1, 1, figure=fig2)
# ax1 = fig1.add_subplot(gs[0, 0], projection='3d')
# kwant.plot(nanowire, site_size=site_size, site_lw=site_lw, site_color=site_color, hop_lw=hop_lw, hop_color=hop_color,
#            lead_site_size=site_size, lead_color=lead_color, lead_site_lw=site_lw, lead_hop_lw=hop_lw,
#            ax=ax1)
# ax1.set_axis_off()
# ax1.margins(-0.49, -0.49, -0.49)
#
#





