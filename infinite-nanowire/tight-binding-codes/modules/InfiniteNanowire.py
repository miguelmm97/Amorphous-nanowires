#%% modules setup

# Math and plotting
from numpy import pi
import numpy as np
from scipy.integrate import quad

# Managing classes
from dataclasses import dataclass, field
from .AmorphousLattice_2d import AmorphousLattice_2d

# Managing logging
import logging
import colorlog
from colorlog import ColoredFormatter

#%% Logging setup
loger_wire = logging.getLogger('nanowire')
loger_wire.setLevel(logging.INFO)

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
loger_wire.addHandler(stream_handler)

#%% Module
"""
We set up a class to model the electronic structure of a nanowire with a particular amorphous cross-section.
"""

sigma_0 = np.eye(2, dtype=np.complex128)
sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
tau_0, tau_x, tau_y, tau_z  = sigma_0, sigma_x, sigma_y, sigma_z


# Functions for the Hamiltonian
def displacement2D(x1, y1, x2, y2):

    v = np.zeros((2,))
    v[0] = (x2 - x1)
    v[1] = (y2 - y1)

    # Norm of the vector between sites 2 and 1
    r = np.sqrt(v[0] ** 2 + v[1] ** 2)

    # Phi angle of the vector between sites 2 and 1 (angle in the XY plane)
    if v[0] == 0:                                    # Pathological case, separated to not divide by 0
        if v[1] > 0:
            phi = pi / 2                             # Hopping in y
        else:
            phi = 3 * pi / 2                         # Hopping in -y
    else:
        if v[1] > 0:
            phi = np.arctan2(v[1], v[0])             # 1st and 2nd quadrants
        else:
            phi = 2 * pi + np.arctan2(v[1], v[0])    # 3rd and 4th quadrants

    return r, phi

def Peierls(x1, y1, x2, y2, flux, area):
    def integrand(x, m, x0, y0):
        return m * (x - x0) + y0

    m = (y2 - y1) / (x2 - x1)
    I = quad(integrand, x1, x2, args=(m, x1, y1))[0]
    return np.exp(2 * pi * 1j * flux * I / area)

@dataclass
class InfiniteNanowire_FuBerg:
    """ Infinite amorphous cross-section nanowire based on the crystalline Fu and Berg model"""

    # Underlying lattice
    lattice: AmorphousLattice_2d  # Lattice for the nanowires cross-section

    # Electronic parameters
    eps:    float      # Onsite energy coupling different orbitals
    t:      float      # Isotropic hopping strength
    lamb:   float      # Spin orbit coupling in the cross-section
    lamb_z: float      # Spin orbit coupling along z
    flux:   float      # Magnetic flux threaded through the cross-section

    # Class fields
    dimH:                 int = field(init=False)    # Dimension of the single-particle Hilbert Space
    H:             np.ndarray = field(init=False)    # Hamiltonian
    kz:            np.ndarray = field(init=False)    # Momentum along z direction
    energy_bands:        dict = field(init=False)    # Energy bands
    eigenstates:         dict = field(init=False)    # Eigenstates

    def __post_init__(self):
        self.dimH = int(self.lattice.Nsites * 4)

    # Methods for calculating the Hamiltonian
    def H_onsite(self, k):
        return (self.eps - 2 * self.t * np.cos(k)) * np.kron(sigma_x, tau_0) + \
                 self.lamb_z * np.sin(k) * np.kron(sigma_y, tau_0)

    def H_offdiag(self, d, phi):
        f_cutoff = np.heaviside(self.lattice.r - d, 1) * np.exp(-d + 1)
        return - self.t * f_cutoff * np.kron(sigma_x, tau_0) + \
            1j * 0.5 * self.lamb * f_cutoff * np.kron(sigma_z, np.cos(phi) * tau_y - np.sin(phi) * tau_x)

    def get_Hamiltonian(self, k_0=-pi, k_end=pi, Nk=1001, debug=False):

        # Preallocation
        self.kz   = np.linspace(k_0, k_end, Nk)
        H_offdiag = np.zeros((self.dimH, self.dimH), dtype=np.complex128)
        self.H    = np.zeros((len(self.kz), self.dimH, self.dimH), dtype=np.complex128)

        # Off-diagonal terms
        loger_wire.info('Generating off-diagonal Hamiltonian')
        for i in range(self.lattice.Nsites):
            for n in self.lattice.neighbours[i]:
                loger_wire.trace(f'site: {i}, neighbour: {n}')
                d, phi = displacement2D(self.lattice.x[i], self.lattice.y[i], self.lattice.x[n], self.lattice.y[n])
                peierls_phase = Peierls(self.lattice.x[i], self.lattice.y[i], self.lattice.x[n], self.lattice.y[n],
                                        self.flux, self.lattice.area)
                H_offdiag[i * 4: i * 4 + 4, n * 4: n * 4 + 4] = self.H_offdiag(d, phi) * peierls_phase
        self.H = np.tile(H_offdiag, (len(self.kz), 1, 1))

        # Debug
        if debug:
            loger_wire.debug('Checking hermiticity of H...')
            if not np.allclose(H_offdiag, H_offdiag.T.conj(), atol=1e-15):
                error = np.abs(np.sum(H_offdiag - H_offdiag.T.conj()))
                loger_wire.error(f'Off-diagonal Hamiltonian is not hermitian. sum(H - H^\dagger): {error}')
                raise ValueError('Hamiltonian is not hermitian!')


        # Onsite terms
        loger_wire.info('Generating onsite Hamiltonian')
        for j, k in enumerate(self.kz):
            loger_wire.trace(f'kz: {j}/ {len(self.kz)}')
            self.H[j, :, :] += np.kron(np.eye(self.lattice.Nsites, dtype=np.complex128), self.H_onsite(k))

            # Debug
            if debug:
                loger_wire.debug('Checking hermiticity of H...')
                if not np.allclose(self.H[j, :, :], self.H[j, :, :].T.conj(), atol=1e-15):
                    error = np.abs(np.sum(self.H[j, :, :] - self.H[j, :, :].T.conj()))
                    loger_wire.error(f'Hamiltonian is not hermitian. sum(H - H^\dagger): {error}, kz: {j}')
                    raise ValueError('Hamiltonian is not hermitian!')

    def get_bands(self, k_0=-pi, k_end=pi, Nk=1001):

        # Calculating Hamiltonian
        self.energy_bands, self.eigenstates = {}, {}
        aux_bands = np.zeros((Nk, self.dimH))
        aux_eigenstates = np.zeros((Nk, self.dimH, self.dimH), dtype=np.complex128)
        self.get_Hamiltonian(k_0=k_0, k_end=k_end, Nk=Nk)

        # Diagonalising Hamiltonian
        loger_wire.info('Diagonalising Hamiltonian...')
        for j in range(len(self.kz)):
            loger_wire.trace(f'kz: {j}/ {len(self.kz)}')
            bands_k, eigenstates_k = np.linalg.eigh(self.H[j, :, :])
            idx = bands_k.argsort()
            aux_bands[j, :], aux_eigenstates[j, :, :] = bands_k[idx], eigenstates_k[:, idx]

        # Ordering bands
        for i in range(self.dimH):
            self.energy_bands[i] = aux_bands[:, i]
            self.eigenstates[i] = aux_eigenstates[:, :, i]

    def get_gap(self):
        return np.min(self.energy_bands[int(self.dimH / 2)] - self.energy_bands[int(self.dimH / 2) - 1])

