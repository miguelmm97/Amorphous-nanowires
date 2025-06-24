#%% modules setup

# Math and plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn
from matplotlib.patches import FancyArrowPatch

# Kwant
import kwant

# modules
from modules.functions import *
from modules.AmorphousLattice_2d import AmorphousLattice_2d
from modules.AmorphousWire_kwant import promote_to_kwant_nanowire
from modules.AmorphousLattice_3d import AmorphousLattice_3d
from modules.FullyAmorphousWire_kwant import promote_to_kwant_nanowire3d


#%% Loading data
file_list = ['Exp67.h5', 'Exp74.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-latex-figures')

Nx           = data_dict[file_list[0]]['Parameters']['Nx']
Ny           = data_dict[file_list[0]]['Parameters']['Ny']
n_layers     = data_dict[file_list[0]]['Parameters']['Nz']
width        = data_dict[file_list[0]]['Simulation']['width']
r            = data_dict[file_list[0]]['Parameters']['r']
t            = data_dict[file_list[0]]['Parameters']['t']
eps          = data_dict[file_list[0]]['Parameters']['eps']
lamb         = data_dict[file_list[0]]['Parameters']['lamb']
lamb_z       = data_dict[file_list[0]]['Parameters']['lamb_z']
mu_leads     = data_dict[file_list[0]]['Parameters']['mu_leads']
# flux0        = data_dict[file_list[1]]['Parameters']['flux0']
flux_half    = data_dict[file_list[1]]['Simulation']['flux_max']
params_dict  = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}

# Simulation data
fermi_cryst       = data_dict[file_list[0]]['Simulation']['fermi']
G0_cryst          = data_dict[file_list[0]]['Simulation']['G_0'][:, 0]
G_half_cryst      = data_dict[file_list[0]]['Simulation']['G_half'][:, 0]
fermi             = data_dict[file_list[1]]['Simulation']['fermi']
G0                = data_dict[file_list[1]]['Simulation']['G_0']
G_half            = data_dict[file_list[1]]['Simulation']['G_half']

#%% Wire example

cross_section = AmorphousLattice_2d(Nx=5, Ny=5, w=0.1, r=r)
cross_section.build_lattice(restrict_connectivity=True)
nanowire = promote_to_kwant_nanowire(cross_section, 10, params_dict).finalized()
cross_section2 = AmorphousLattice_2d(Nx=5, Ny=5, w=0.000001, r=r)
cross_section2.build_lattice(restrict_connectivity=True)
nanowire2 = promote_to_kwant_nanowire(cross_section, 10, params_dict).finalized()
lattice = AmorphousLattice_3d(Nx=5, Ny=5, Nz=20,  w=0.1, r=r)
lattice.build_lattice()
nanowire3 = promote_to_kwant_nanowire3d(lattice, params_dict).finalized()
nanowire4 = promote_to_kwant_nanowire3d(lattice, params_dict, attach_leads=False).finalized()


#%% Figures

# Style sheet
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange']
palette = seaborn.color_palette(palette='viridis_r', n_colors=200)
palette = [palette[0], palette[50], palette[100] , palette[130], palette[-1]]
markersize = 5
fontsize=20
site_size  = 0.2
site_lw    = 0.00
site_color = 'lightskyblue'
hop_color  = 'lightskyblue'
hop_lw     = 0.07
lead_color = 'fuchsia' #'orangered'


# Figure 1: Definition
fig1 = plt.figure(figsize=(8, 6))
gs = GridSpec(1, 1, figure=fig1, wspace=0.2, hspace=0.3)
ax1 = fig1.add_subplot(gs[0, 0])
ax1_inset1 = ax1.inset_axes([0.02, 0.6, 0.3, 0.3])
ax1_inset2 = ax1.inset_axes([0.6, -0.12, 0.35, 0.8], projection='3d', zorder=1, )

# Figure 1: Plots
cross_section.plot_lattice(ax1_inset1, sitecolor='lightskyblue', linkcolor='lightskyblue', alpha_link=1, boundarycolor='lightskyblue',
                           alpha_boundary=1, alpha_site=1)
cross_section2.plot_lattice(ax1_inset1, sitecolor='grey', linkcolor='grey', alpha_link=0.25, boundarycolor='grey',
                          alpha_boundary=0, alpha_site=0.25)
ax1_inset1.axis('equal')
ax1_inset1.set_axis_off()
arrow1 = FancyArrowPatch((0.015, 8), (0.31, 8), arrowstyle='->', color='black', linewidth=1, mutation_scale=20)
arrow2 = FancyArrowPatch((0.04, 7.5), (0.04, 13), arrowstyle='->', color='black', linewidth=1, mutation_scale=20)
arrow3 = FancyArrowPatch((0.83, 0.63), (0.67, 1.6), arrowstyle='->', color='black', linewidth=1, mutation_scale=20, zorder=10)
arrow4 = FancyArrowPatch((0.8, 0.53), (0.89, 2.1), arrowstyle='->', color='black', linewidth=1, mutation_scale=20, zorder=10)
arrow5 = FancyArrowPatch((0.8119, 0.65), (0.8119, 7.5), arrowstyle='->', color='black', linewidth=1, mutation_scale=20, zorder=10)
ax1.add_patch(arrow1)
ax1.add_patch(arrow2)
ax1.add_patch(arrow3)
ax1.add_patch(arrow4)
ax1.add_patch(arrow5)
ax1.text(0.04, 13.2, f'$y$', fontsize=fontsize, color='black')
ax1.text(0.31, 8.1, f'$x$', fontsize=fontsize, color='black')
ax1.text(0.67, 2, f'$y$', fontsize=fontsize,color='black')
ax1.text(0.88, 2.2, f'$x$', fontsize=fontsize, color='black')
ax1.text(0.8119, 7.7, f'$z$', fontsize=fontsize, color='black')

kwant.plot(nanowire2, site_size=site_size, site_lw=site_lw, site_color=site_color, hop_lw=hop_lw, hop_color=hop_color,
           lead_site_size=site_size, lead_color=lead_color, lead_site_lw=site_lw, lead_hop_lw=hop_lw,
           site_edgecolor=None, lead_site_edgecolor=None, ax=ax1_inset2, num_lead_cells=2)
ax1_inset2.set_axis_off()
ax1_inset2.margins(-0.49, -0.49, -0.49)
ax1_inset2.set_xlim(-2.5, 6)
ax1_inset2.set_ylim(-2.5, 6)
ax1_inset2.set_zlim(0, 9)

# ax1_inset2.set_box_aspect((1, 1, 3))


# for i in range(G0.shape[-1]):
#     ax1.plot(fermi, G0[:, i], color=palette[i], label=f'${width[i] :.2f}$')#, alpha=0.75)
#     ax1.plot(fermi, G_half[:, i], color=palette[i], linestyle='dotted', linewidth=2)
ax1.plot(fermi_cryst, G0_cryst, color='#9A32CD', linestyle='dashed',  alpha=0.7)#, label=f'${flux0}$')
ax1.plot(fermi_cryst, G_half_cryst, color='#3F6CFF', linestyle='dashed',  alpha=0.7)#, label=f'${flux_half}$ ')
ax1.plot(fermi, G0[:, 0], color='#9A32CD', label=f'$0$')
ax1.plot(fermi, G_half[:, 0], color='#3F6CFF', label=f'${flux_half[0] :.2f}$ ')

ax1.legend(ncol=1, loc='upper left', frameon=False, fontsize=20, columnspacing=0.3, handlelength=0.75, labelspacing=0.2,  bbox_to_anchor=(0.35, 0.9))
ax1.text(0.46, 12.4, '$\\underline{\phi/\phi_0}$', fontsize=fontsize)
# ax1.text(0.4, 1.5, '$\Delta E_F=0$', fontsize=fontsize)
ax1.text(0.4, 2.5, f'$w= 0.1$', fontsize=fontsize)

y_axis_ticks = [i for i in range(0, 14, 2)]
y_axis_labels = [str(i) for i in range(0, 14, 2)]
ax1.set_xlim(fermi[0], fermi[-1])
ax1.set_ylim(0, 14)
ax1.tick_params(which='major', width=0.75, labelsize=fontsize, color='black')
ax1.tick_params(which='major', length=6, labelsize=fontsize, color='black')
ax1.set_xlabel("$E_F$", fontsize=fontsize, labelpad=-1, color='black')
ax1.set_ylabel("$G(e^2/h)$",fontsize=fontsize, labelpad=-1, color='black')
ax1.set(yticks=y_axis_ticks, yticklabels=y_axis_labels)
ax1.tick_params(axis='both', colors='black')
fig1.savefig('fig1-layer-dis.pdf', format='pdf')

fig2 = plt.figure(figsize=(6, 6))
gs = GridSpec(1, 1, figure=fig2, wspace=0.2, hspace=0.3)
ax1 = fig2.add_subplot(gs[0, 0], facecolor='black', projection='3d')
kwant.plot(nanowire3, site_size=site_size, site_lw=site_lw, site_color=site_color, hop_lw=hop_lw, hop_color=hop_color,
           lead_site_size=site_size, lead_color=lead_color, lead_site_lw=site_lw, lead_hop_lw=hop_lw,
           site_edgecolor=None, lead_site_edgecolor=None, ax=ax1, num_lead_cells=5)
ax1.set_axis_off()

fig3 = plt.figure(figsize=(6, 6))
gs = GridSpec(1, 1, figure=fig3, wspace=0.2, hspace=0.3)
ax2 = fig3.add_subplot(gs[0, 0], projection='3d')
kwant.plot(nanowire4, site_size=site_size, site_lw=site_lw, site_color=site_color, hop_lw=hop_lw, hop_color=hop_color,
           site_edgecolor=None, ax=ax2)
ax2.set_axis_off()
ax2.set_xlim(ax1.get_xlim())
ax2.set_ylim(ax1.get_ylim())
ax2.set_zlim(ax1.get_zlim())



fig4 = plt.figure(figsize=(6, 6), facecolor='black')
gs = GridSpec(1, 1, figure=fig4, wspace=0.2, hspace=0.3)
ax2 = fig4.add_subplot(gs[0, 0], facecolor='black', projection='3d')
kwant.plot(nanowire4, site_size=site_size, site_lw=site_lw, site_color=site_color, hop_lw=hop_lw, hop_color='black',
           site_edgecolor=None, ax=ax2)
ax2.set_axis_off()
ax2.set_xlim(ax1.get_xlim())
ax2.set_ylim(ax1.get_ylim())
ax2.set_zlim(ax1.get_zlim())


# cross_section.plot_lattice(ax1, sitecolor='lightskyblue', linkcolor='lightskyblue', alpha_link=1, boundarycolor='lightskyblue',
#                            alpha_boundary=1, alpha_site=1)
# cross_section2.plot_lattice(ax1, sitecolor='grey', linkcolor='grey', alpha_link=0.25, boundarycolor='grey',
#                            alpha_boundary=0, alpha_site=0.25)
# arrow1 = FancyArrowPatch((0.015, 8), (0.31, 8), arrowstyle='->', color='white', linewidth=1, mutation_scale=20)
# arrow2 = FancyArrowPatch((0.04, 7.5), (0.04, 13), arrowstyle='->', color='white', linewidth=1, mutation_scale=20)
# arrow3 = FancyArrowPatch((0.83, 0.63), (0.67, 1.6), arrowstyle='->', color='white', linewidth=1, mutation_scale=20, zorder=10)
# arrow4 = FancyArrowPatch((0.8, 0.53), (0.89, 2.1), arrowstyle='->', color='white', linewidth=1, mutation_scale=20, zorder=10)
# arrow5 = FancyArrowPatch((0.8119, 0.65), (0.8119, 7.5), arrowstyle='->', color='white', linewidth=1, mutation_scale=20, zorder=10)
# ax1.add_patch(arrow1)
# ax1.add_patch(arrow2)
# ax1.add_patch(arrow3)
# ax1.add_patch(arrow4)
# ax1.add_patch(arrow5)
# ax1.text(0.04, 13.2, f'$y$', fontsize=fontsize, color='white')
# ax1.text(0.31, 8.1, f'$x$', fontsize=fontsize, color='white')
# ax1.text(0.67, 2, f'$y$', fontsize=fontsize, color='white')
# ax1.text(0.88, 2.2, f'$x$', fontsize=fontsize, color='white')
# ax1.text(0.8119, 7.7, f'$z$', fontsize=fontsize, color='white')
#
fig2.savefig('nanowire_leads.pdf', format='pdf')
fig3.savefig('nanowire.pdf', format='pdf')
fig4.savefig('nanowire_sites.pdf', format='pdf')
plt.show()


