#%% modules setup

# Math and plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn

# Kwant
import kwant

# modules
from modules.functions import *
from modules.AmorphousLattice_2d import AmorphousLattice_2d
from modules.AmorphousWire_kwant import promote_to_kwant_nanowire

#%% Loading data
file_list = ['draft-fig1-oscillations.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-latex-figures')

Nx           = data_dict[file_list[0]]['Parameters']['Nx']
Ny           = data_dict[file_list[0]]['Parameters']['Ny']
n_layers     = data_dict[file_list[0]]['Parameters']['n_layers']
width        = data_dict[file_list[0]]['Parameters']['width']
r            = data_dict[file_list[0]]['Parameters']['r']
t            = data_dict[file_list[0]]['Parameters']['t']
eps          = data_dict[file_list[0]]['Parameters']['eps']
lamb         = data_dict[file_list[0]]['Parameters']['lamb']
lamb_z       = data_dict[file_list[0]]['Parameters']['lamb_z']
mu_leads     = data_dict[file_list[0]]['Parameters']['mu_leads']
flux0        = data_dict[file_list[0]]['Parameters']['flux0']
flux_half    = data_dict[file_list[0]]['Parameters']['flux_half']
params_dict  = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}

# Simulation data
fermi             = data_dict[file_list[0]]['Simulation']['fermi']
G0                = data_dict[file_list[0]]['Simulation']['G0']
G_half            = data_dict[file_list[0]]['Simulation']['G_half']
#%% Wire example

cross_section = AmorphousLattice_2d(Nx=5, Ny=5, w=width, r=r)
cross_section.build_lattice()
nanowire = promote_to_kwant_nanowire(cross_section, 10, params_dict, mu_leads=mu_leads).finalized()


#%% Figures

# Style sheet
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange']
markersize = 5
fontsize=20
site_size  = 0.1
site_lw    = 0.01
site_color = 'm'
hop_color  = 'royalblue'
hop_lw     = 0.05
lead_color = 'r'


# Figure 1: Definition
fig1 = plt.figure(figsize=(8, 7))
gs = GridSpec(1, 1, figure=fig1, wspace=0.2, hspace=0.3)
ax1 = fig1.add_subplot(gs[0, 0])
ax1_inset1 = ax1.inset_axes([0.05, 0.45, 0.3, 0.3], )
ax1_inset2 = ax1.inset_axes([0.6, -0.12, 0.35, 0.8], projection='3d')

# Figure 1: Plots
cross_section.plot_lattice(ax1_inset1)
ax1_inset1.axis('equal')
ax1_inset1.set_axis_off()

kwant.plot(nanowire, site_size=site_size, site_lw=site_lw, site_color=site_color, hop_lw=hop_lw, hop_color=hop_color,
           lead_site_size=site_size, lead_color=lead_color, lead_site_lw=site_lw, lead_hop_lw=hop_lw,
           site_edgecolor=None, lead_site_edgecolor=None, ax=ax1_inset2)
ax1_inset2.set_axis_off()
ax1_inset2.margins(-0.49, -0.49, -0.49)
ax1_inset2.set_xlim(-2.5, 6)
ax1_inset2.set_ylim(-2.5, 6)
ax1_inset2.set_zlim(0, 9)

ax1.plot(fermi, G0, color='#9A32CD', label=f'$\phi / \phi_0= {flux0}$')
ax1.plot(fermi, G_half, color='#3F6CFF', alpha=0.5, label=f'$\phi / \phi_0= {flux_half}$ ')
ax1.legend(ncol=1, frameon=False, fontsize=16)

y_axis_ticks = [i for i in range(0, 14, 2)]
y_axis_labels = [str(i) for i in range(0, 14, 2)]
ax1.set_xlim(fermi[0], fermi[-1])
ax1.set_ylim(0, 14)
ax1.tick_params(which='major', width=0.75, labelsize=fontsize)
ax1.tick_params(which='major', length=6, labelsize=fontsize)
ax1.set_xlabel("$E_F / t$", fontsize=fontsize)
ax1.set_ylabel("$G(2e^2/h)$",fontsize=fontsize)
ax1.set(yticks=y_axis_ticks, yticklabels=y_axis_labels)

fig1.savefig('draft-fig1.pdf', format='pdf', backend='pgf')
plt.show()

