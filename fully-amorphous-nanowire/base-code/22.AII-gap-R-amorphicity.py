#%% modules set up

# Math and plotting
import numpy as np
from numpy.linalg import eigh
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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
width      = np.linspace(0.000001, 0.5, 10)
t          = 1
eps        = 4 * t
lamb       = 1 * t
lamb_z     = 1.8 * t
fermi      = 0.5
params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}

# Preallocation
Nsites = int(Nx * Ny * Nz)
gap_R = np.zeros((len(width), ))
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

#%% Main

# Gap as a function of Ef
loger_main.info(f'Calculating gap vs amorphicity...')
for i, w in enumerate(width):

    # Generating lattice structure of the wire
    lattice = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=Nz, w=w, r=r)
    lattice.build_lattice()
    lattice.generate_disorder(K_onsite=0., K_hopp=0)
    nanowire = promote_to_kwant_nanowire3d(lattice, params_dict, attach_leads=False).finalized()

    # Spectrum of the closed system
    H = nanowire.hamiltonian_submatrix(params=dict(flux=0., mu=-fermi))
    eps, _, rho = spectrum(H)

    # Local marker
    site_pos = np.array([site.pos for site in nanowire.id_by_site])
    x, y, z = site_pos[:, 0], site_pos[:, 1], site_pos[:, 2]
    chiral_sym = np.kron(np.eye(len(x)), np.kron(sigma_z, sigma_z))
    R = 0.5 * (np.eye(Nsites * 4) - chiral_sym)
    Q_aux = 1j * (rho @ R - R @ rho)
    spectrum_R = np.linalg.eig(Q_aux)[0]
    if np.max(np.imag(spectrum_R)) > 1e-10:
        raise ValueError('R i[rho, R] is not a hermitian operator.')
    else:
        spectrum_R = np.sort(np.real(spectrum_R))
    gap_R[i] = spectrum_R[int(len(spectrum_R) / 2)] - spectrum_R[int(len(spectrum_R) / 2) - 1]
    loger_main.info(f'width: {i}/{len(width) - 1}, gap: {gap_R[i] :.2f}')

    # vals, vecs = eigh(Q_aux)
    # Q = 0.5 * vecs @ (np.eye(4 * Nsites) - np.sign(vals)) @ np.conj(vecs.T)
    # print(np.allclose(chiral_sym @ H @ chiral_sym, -H))
    # print(np.allclose(rho @ chiral_sym + chiral_sym @ rho, chiral_sym))


#%% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange']
marker_list=['o', 's', 'd', 'p', '*', 'h', '>', '<', 'X']
markersize = 5
fontsize=20

# Figure 1: Definition
fig1 = plt.figure(figsize=(8, 6))
gs = GridSpec(1, 1, figure=fig1, wspace=0.2, hspace=0.1)
ax1 = fig1.add_subplot(gs[0, 0])

ax1.plot(width, gap_R, marker='o', linestyle='solid')
plt.show()