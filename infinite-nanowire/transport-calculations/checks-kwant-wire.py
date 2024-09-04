#%% Modules set up

# Math
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Kwant
import kwant

# Modules
from modules.functions import *
from modules.AmorphousLattice_2d import AmorphousLattice_2d
from modules.AmorphousWire_kwant import promote_to_kwant_nanowire, crystal_nanowire_kwant, infinite_nanowire_kwant, FuBerg_model_bands
from modules.InfiniteNanowire import InfiniteNanowire_FuBerg

#%% Logging setup
loger_main = logging.getLogger('main')
loger_main.setLevel(logging.INFO)

stream_handler = colorlog.StreamHandler()
formatter = ColoredFormatter(
    '%(black)s%(asctime) -5s| %(blue)s%(name) -10s %(black)s| %(cyan)s %(funcName) '
    '-40s %(black)s|''%(log_color)s%(levelname) -10s | %(message)s',
    datefmt=None,
    reset=True,
    log_colors={
        'TRACE': 'black',
        'DEBUG': 'purple',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)
stream_handler.setFormatter(formatter)
loger_main.addHandler(stream_handler)

#%% Variables
"""
We compare the structure and bands of the kwant nanowire and the one generated with our own class.
"""

Nx, Ny     = 5, 5                    # Number of sites in the cross-section
n_layers   = 10                      # Number of cross-section layers
cryst_area = Nx * Ny                 # Area of the crystalline cross-section
flux       = 0.56                    # Flux
t          = 1                       # Hopping
eps        = 4 * t                   # Onsite orbital hopping (in units of t)
lamb       = 1 * t                   # Spin-orbit coupling in the cross-section (in units of t)
lamb_z     = 1.8 * t                 # Spin-orbit coupling along z direction
kz = np.linspace(-pi, pi, 101)      # Momentum along the regular direction
fermi = np.linspace(0, 2, 50)

params_dict = {
    't': t,
    'eps': eps,
    'lamb': lamb,
    'lamb_z': lamb_z,
    'flux': flux
}

#%% Comparison of conductance

# Crystalline wire using our Amorphous module
loger_main.info('Generating amorphous cross section:')
cross_section = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=0.000000001, r=1.3)
cross_section.build_lattice()
nanowire = promote_to_kwant_nanowire(cross_section, n_layers, params_dict).finalized()
loger_main.info('Nanowire promoted to Kwant successfully.')

# Crystalline wire using Kwant
nanowire_kwant = crystal_nanowire_kwant(Nx, Ny, n_layers + 2, params_dict).finalized()


# Conductance calculation
G_module_0flux = np.zeros(fermi.shape)
G_module_halfflux = np.zeros(fermi.shape)
G_kwant_0flux = np.zeros(fermi.shape)
G_kwant_halfflux = np.zeros(fermi.shape)

for i, Ef in enumerate(fermi):
    loger_main.info(f'Calculating conductance for Ef: {i} / {fermi.shape[0] - 1}...')

    # Module nanowire
    S1 = kwant.smatrix(nanowire, Ef, params=dict(flux=0.))
    G_module_0flux[i] = S1.transmission(1, 0)

    S2 = kwant.smatrix(nanowire, Ef, params=dict(flux=flux))
    G_module_halfflux[i] = S2.transmission(1, 0)

    # Kwant nanowire
    S3 = kwant.smatrix(nanowire_kwant, Ef, params=dict(flux=0.))
    G_kwant_0flux[i] = S3.transmission(1, 0)

    S4 = kwant.smatrix(nanowire_kwant, Ef, params=dict(flux=flux))
    G_kwant_halfflux[i] = S4.transmission(1, 0)


#%% Comparison of band structures

# Infinite crystalline wire using our Amorphous module
loger_main.info('Getting band structures of the module nanowires...')
wire_module_0flux = InfiniteNanowire_FuBerg(lattice=cross_section, t=t, eps=eps, lamb=lamb, lamb_z=lamb_z, flux=0.)
bottom_bands_0flux = wire_module_0flux.get_bands(k_0=0, k_end=0, Nk=1, extract=True)[0]
wire_module_0flux.get_bands()

wire_module_halfflux = InfiniteNanowire_FuBerg(lattice=cross_section, t=t, eps=eps, lamb=lamb, lamb_z=lamb_z, flux=flux)
bottom_bands_halfflux = wire_module_halfflux.get_bands(k_0=0, k_end=0, Nk=1, extract=True)[0]
wire_module_halfflux.get_bands()

# Infinite wire directly from the Fu Berg Model
bands_model_0flux = FuBerg_model_bands(Nx, Ny, kz, 0., params_dict)[0]
bands_model_halfflux = FuBerg_model_bands(Nx, Ny, kz, flux, params_dict)[0]

# Infinite crystalline wire using kwant
loger_main.info('Getting band structures of the kwant nanowires...')
wire_kwant = infinite_nanowire_kwant(Nx, Ny, params_dict).finalized()
bands1= kwant.physics.Bands(wire_kwant, params=dict(flux=0.))
bands2= kwant.physics.Bands(wire_kwant, params=dict(flux=flux))
bands_kwant_0flux = [bands1(k) for k in kz]
bands_kwant_halfflux = [bands2(k) for k in kz]

#%% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
axcolour = ['#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125']

site_size  = 0.1
site_lw    = 0.01
site_color = 'm'
hop_color  = 'royalblue'
hop_lw     = 0.05
lead_color = 'r'


fig0 = plt.figure()
ax0 = fig0.gca()
cross_section.plot_lattice(ax0)


fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
kwant.plot(nanowire, site_size=site_size, site_lw=site_lw, site_color=site_color, hop_lw=hop_lw, hop_color=hop_color,
           lead_site_size=site_size, lead_color=lead_color, lead_site_lw=site_lw, lead_hop_lw=hop_lw,
           ax=ax1)
ax1.set_axis_off()


fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
kwant.plot(nanowire_kwant, site_size=site_size, site_lw=site_lw, site_color=site_color, hop_lw=hop_lw, hop_color=hop_color,
           lead_site_size=site_size, lead_color=lead_color, lead_site_lw=site_lw, lead_hop_lw=hop_lw,
           ax=ax2)
ax2.set_axis_off()


fig3 = plt.figure(figsize=(4, 6))
gs = GridSpec(1, 3, figure=fig3)
ax3_1 = fig3.add_subplot(gs[0, 0])
ax3_2 = fig3.add_subplot(gs[0, 1])
ax3_3 = fig3.add_subplot(gs[0, 2])

# 0 flux
ax3_1.plot(fermi, G_module_0flux, color='#9A32CD', label='module $\phi / \phi_0=0$')
ax3_1.plot(fermi, G_kwant_0flux, color='#3F6CFF', alpha=0.5, label=f'kwant $\phi / \phi_0=0$ ')
for i in bottom_bands_0flux.keys():
    ax3_1.plot(bottom_bands_0flux[i] * np.ones((10, )), np.linspace(0, 100, 10), '--', color='Grey', alpha=0.1)

# half flux
ax3_2.plot(fermi, G_module_halfflux, color='#9A32CD', label='module $\phi / \phi_0=0.5$')
ax3_2.plot(fermi, G_kwant_halfflux, color='#3F6CFF', alpha=0.5, label=f'kwant $\phi / \phi_0=0.5$ ')
for i in bottom_bands_0flux.keys():
    ax3_2.plot(bottom_bands_halfflux[i] * np.ones((10, )), np.linspace(0, 100, 10), '--', color='Grey', alpha=0.1)

# Module comparison
ax3_3.plot(fermi, G_module_0flux, color='#FF7256', label='module $\phi / \phi_0=0$')
ax3_3.plot(fermi, G_module_halfflux, color='#9A32CD', label='module $\phi / \phi_0=0.5$')

y_axis_ticks = [i for i in range(0, 20, 2)]
y_axis_labels = [str(i) for i in range(0, 20, 2)]
ax3_1.set_xlim(fermi[0], fermi[-1])
ax3_1.set_ylim(0, np.max(G_module_0flux))
ax3_1.tick_params(which='major', width=0.75, labelsize=10)
ax3_1.tick_params(which='major', length=6, labelsize=10)
ax3_1.set_xlabel("$E_F$ [$t$]", fontsize=10)
ax3_1.set_ylabel("$G[2e^2/h]$",fontsize=10)
ax3_1.legend(ncol=1, frameon=False, fontsize=16)
ax3_1.set(yticks=y_axis_ticks, yticklabels=y_axis_labels)

ax3_2.set_xlim(fermi[0], fermi[-1])
ax3_2.set_ylim(0, np.max(G_module_0flux))
ax3_2.tick_params(which='major', width=0.75, labelsize=10)
ax3_2.tick_params(which='major', length=6, labelsize=10)
ax3_2.set_xlabel("$E_F$ [$t$]", fontsize=10)
ax3_2.set_ylabel("$G[2e^2/h]$",fontsize=10)
ax3_2.legend(ncol=1, frameon=False, fontsize=16)
ax3_2.set(yticks=y_axis_ticks, yticklabels=y_axis_labels)

ax3_3.set_xlim(fermi[0], fermi[-1])
ax3_3.set_ylim(0, np.max(G_module_0flux))
ax3_3.tick_params(which='major', width=0.75, labelsize=10)
ax3_3.tick_params(which='major', length=6, labelsize=10)
ax3_3.set_xlabel("$E_F$ [$t$]", fontsize=10)
ax3_3.set_ylabel("$G[2e^2/h]$",fontsize=10)
ax3_3.legend(ncol=1, frameon=False, fontsize=16)
ax3_3.set(yticks=y_axis_ticks, yticklabels=y_axis_labels)





fig4 = plt.figure(figsize=(6, 6))
gs = GridSpec(1, 2, figure=fig4)
ax4_1 = fig4.add_subplot(gs[0, 0])
ax4_2 = fig4.add_subplot(gs[0, 1])

# Bands for the module
for i in wire_module_0flux.energy_bands.keys():
    ax4_1.plot(wire_module_0flux.kz, wire_module_0flux.energy_bands[i], color='#3F6CFF', linewidth=0.5)
for i in wire_module_halfflux.energy_bands.keys():
    ax4_2.plot(wire_module_halfflux.kz, wire_module_halfflux.energy_bands[i], color='#3F6CFF', linewidth=0.5)

# Bands for Kwant
ax4_1.plot(kz, bands_kwant_0flux, '.', color='#FF7256', markersize=0.5)
ax4_2.plot(kz, bands_kwant_halfflux, '.', color='#FF7256', markersize=0.5)

# Bands for the Fu Berg model
for i in bands_model_0flux.keys():
    ax4_1.plot(kz, bands_model_0flux[i], 'o', color='#9A32CD', markersize=1)
    ax4_2.plot(kz, bands_model_halfflux[i], 'o', color='#9A32CD', markersize=1)

ax4_1.set_xlabel('$k/a$')
ax4_1.set_ylabel('$E(k)/t$')
ax4_1.set_xlim(-pi, pi)
ax4_1.tick_params(which='major', width=0.75, labelsize=10)
ax4_1.tick_params(which='major', length=6, labelsize=10)
ax4_1.set(xticks=[-pi, -pi/2, 0, pi/2, pi], xticklabels=['$-\pi$', '$-\pi/2$', '$0$', '$\pi/2$', '$\pi$'])
ax4_1.set_title(f'$\phi / \phi_0=0$')

ax4_2.set_xlabel('$k/a$')
ax4_2.set_ylabel('$E(k)/t$')
ax4_2.set_xlim(-pi, pi)
ax4_2.tick_params(which='major', width=0.75, labelsize=10)
ax4_2.tick_params(which='major', length=6, labelsize=10)
ax4_2.set(xticks=[-pi, -pi/2, 0, pi/2, pi], xticklabels=['$-\pi$', '$-\pi/2$', '$0$', '$\pi/2$', '$\pi$'])
ax4_2.set_title(f'$\phi / \phi_0=0.5$')
plt.show()