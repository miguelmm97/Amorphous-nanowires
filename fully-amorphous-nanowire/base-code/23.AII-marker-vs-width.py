#%% modules set up

# Math and plotting
import numpy as np
from numpy.linalg import eigh
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn

# Kwant
import kwant

# modules
from modules.functions import *
from modules.AmorphousLattice_3d import AmorphousLattice_3d, take_cut_from_parent_wire
from modules.FullyAmorphousWire_kwant import promote_to_kwant_nanowire3d, spectrum, local_marker

import sys
from datetime import date

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

Nz         = 8
Nx         = 8
Ny         = 8
r          = 1.3
width      = np.linspace(0.000001, 0.8, 20)
t          = 1
eps        = 4 * t
lamb       = 1 * t
lamb_z     = 1.8 * t
fermi      = 0.5
params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}
cutoff_bulk = 0.4 * 0.5 * Nx
cutoff_z = cutoff_bulk

# Preallocation
Nsites = int(Nx * Ny * Nz)
gap_R = np.zeros((len(width), ))
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
marker_AII = np.zeros((len(width), Nsites))
marker_bulk_avg_AII = np.zeros((len(width), ))
marker_AIII = np.zeros((len(width), Nsites))
marker_bulk_avg_AIII = np.zeros((len(width), ))

# Auxiliary operators and functions
chiral_sym = np.kron(np.eye(Nsites), np.kron(sigma_z, sigma_z))
R = np.kron(np.eye(Nsites), 0.5 * (np.eye(4) - np.kron(sigma_z, sigma_z)))
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
TRS = np.kron(np.eye(Nsites), 1j * np.kron(np.eye(2), sigma_y))
def bulk(x, y, z, local_marker, cutoff_xy, cutoff_z, nx, ny, full_z=True, full_xy=True):

    # Coordinates
    x_pos, y_pos = x - 0.5 * nx, y - 0.5 * ny
    cond1 = np.abs(x_pos) < cutoff_xy
    cond2 = np.abs(y_pos) < cutoff_xy
    cond3 = (0.5 * Nz - cutoff_z) < z
    cond4 = (0.5 * Nz + cutoff_z) > z

    # Cutoff conditions
    if full_xy and full_z:
        return x, y, z, local_marker
    elif full_z:
        cond = cond1 * cond2
    else:
        cond = cond1 * cond2 * cond3 * cond4
    return x[cond], y[cond], z[cond], local_marker[cond]

#%% Main

# Gap as a function of Ef
loger_main.info(f'Calculating AII marker vs amorphicity...')
for i, w in enumerate(width):

    # Generating lattice structure of the wire
    lattice = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=Nz, w=w, r=r)
    lattice.build_lattice()
    lattice.generate_onsite_disorder(K_onsite=0.3)
    nanowire = promote_to_kwant_nanowire3d(lattice, params_dict, attach_leads=False).finalized()

    # Spectrum of the closed system
    H = nanowire.hamiltonian_submatrix(params=dict(flux=0., mu=-fermi))
    eps, _, rho = spectrum(H)

    # Local marker
    site_pos = np.array([site.pos for site in nanowire.id_by_site])
    x, y, z = site_pos[:, 0], site_pos[:, 1], site_pos[:, 2]
    Q_aux = 1j * (rho @ R - R @ rho)
    vals, vecs = eigh(Q_aux)
    Q = 0.5 * vecs @ np.diag(np.ones((4 * Nsites, )) - np.sign(vals)) @ np.conj(vecs.T)
    vals, vecs = eigh(Q)


    marker_AII[i, :] = local_marker(x, y, z, Q, chiral_sym)
    marker_AIII[i, :] = local_marker(x, y, z, rho, chiral_sym)
    loger_main.info(f'T H* T^\dagger = H: {np.allclose(TRS @ H.conj() @ TRS.T.conj(), H)}')
    loger_main.info(f'HS + SH = S: {np.allclose(H @ chiral_sym + chiral_sym @ H, np.zeros((Nsites * 4, Nsites * 4)))}')
    loger_main.info(f'T rho* T^\dagger = rho: {np.allclose(TRS @ rho.conj() @ TRS.T.conj(), rho)}')
    loger_main.info(f'rhoS + Srho = S: {np.allclose(rho @ chiral_sym + chiral_sym @ rho, chiral_sym)}')
    loger_main.info(f'Q² = 1: {np.allclose(Q @ Q, Q)}')
    loger_main.info(f'QS + SQ = S: {np.allclose(Q @ chiral_sym + chiral_sym @ Q, chiral_sym)}')

    # Bulk marker
    _, _, _, marker_bulk_AII = bulk(x, y, z, marker_AII[i, :], cutoff_bulk, cutoff_z, Nx, Ny, full_z=False, full_xy=False)
    marker_bulk_avg_AII[i] = np.mean(marker_bulk_AII)
    _, _, _, marker_bulk_AIII = bulk(x, y, z, marker_AIII[i, :], cutoff_bulk, cutoff_z, Nx, Ny, full_z=False, full_xy=False)
    marker_bulk_avg_AIII[i] = np.mean(marker_bulk_AIII)
    loger_main.info(f'width: {i}/{len(width) - 1}, bulk AII marker: {marker_bulk_avg_AII[i] :.4f}, bulk AIII marker: {marker_bulk_avg_AIII[i] :.4f}')



#%% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['dodgerblue', 'limegreen', 'm', 'r', 'orange']
fontsize = 20

# Figure 1
fig1 = plt.figure(figsize=(8, 6))
gs = GridSpec(1, 1, figure=fig1, wspace=0.1)
ax1 = fig1.add_subplot(gs[0, 0])

# Plot
ax1.plot(np.linspace(0, np.max(width), 10), np.zeros((10, )), '--', color='Black', alpha=0.2)
ax1.plot(width, marker_bulk_avg_AII, marker='o', linestyle='solid', color='red', label='AII')
ax1.plot(width, marker_bulk_avg_AIII, marker='o', linestyle='solid', color='blue', label='AIII')

ax1.legend()
ax1.set_xlabel('$w$', fontsize=fontsize)
ax1.set_ylabel('$\overline{\\nu}$', fontsize=fontsize, labelpad=-5)
ax1.set_ylim([-1, 0.2])
ax1.set_xlim([0, 0.8])

plt.show()
