"""
This file contains the functions related to the topological characterization of amorphous nanowires.
Parts of this code have been adapted from https://doi.org/10.5281/zenodo.3741828.

The full repository for the project is public in https://github.com/miguelmm97/Amorphous-nanowires.git
For any questions, typos/errors or further data please write to mfmm@kth.se or miguelmartinezmiquel@gmail.com.
"""


# %% Modules setup

# Math and plotting
from numpy import pi
import numpy as np
from functools import partial
from scipy.sparse import diags


# Kwant
import kwant
from kwant.kpm import jackson_kernel

# Managing logging
import logging
import colorlog
from colorlog import ColoredFormatter

# %% Logging setup
loger_kwant = logging.getLogger('kwant')
loger_kwant.setLevel(logging.INFO)

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
loger_kwant.addHandler(stream_handler)


#%% Local chiral marker through ED

def spectrum(H, Nsp=None):
    """
    Input:
    H -> nd.array: Hamiltonian of the nanowire
    Nsp -> int: filling fraction

    Output:
    energy -> np.ndarray: sorted eigenvalues of H
    eigenstates -> np.ndarray: corresponding eigenvectors of H
    P -> np.ndarray: One particle density matrix (projector onto the filled bands)
    """
    if Nsp is None:
        Nsp = int(len(H) / 2)

    # Spectrum
    energy, eigenstates = np.linalg.eigh(H)
    idx = energy.argsort()
    energy = energy[idx]
    eigenstates = eigenstates[:, idx]

    # OPDM
    U = np.zeros((len(H), len(H)), dtype=np.complex128)
    U[:, 0: Nsp] = eigenstates[:, 0: Nsp]
    P = U @ np.conj(np.transpose(U))

    return energy, eigenstates, P

def local_marker(x, y, z, P, S):
    """
    Input:
    x, y, z -> np.ndarray: coordinates of the nanowire sites
    P -> np.ndarray: One particle density matrix (projector onto the filled bands)
    S -> np.ndarray: Chiral symmetry (kronecker product of onsite chiral symmetry)

    Output:
    local_marker -> np.ndarray: local chiral marker at each site
    """
    # Operators for calculating the marker
    X, Y, Z = np.repeat(x, 4), np.repeat(y, 4), np.repeat(z, 4)
    X = np.reshape(X, (len(X), 1))
    Y = np.reshape(Y, (len(Y), 1))
    Z = np.reshape(Z, (len(Z), 1))
    PS = P @ S
    XP = X * P
    YP = Y * P
    ZP = Z * P

    # Local chiral marker
    local_marker = np.zeros((len(x), ))
    M = PS @ XP @ YP @ ZP + PS @ ZP @ XP @ YP + PS @ YP @ ZP @ XP - PS @ XP @ ZP @ YP - PS @ ZP @ YP @ XP - PS @ YP @ XP @ ZP
    for i in range(len(x)):
        idx = 4 * i
        local_marker[i] = (8 * pi / 3) * np.imag(np.trace(M[idx: idx + 4, idx: idx + 4]))

    return local_marker


#%% Local chiral marker through the Kernel Polynomial Method
def kpm_vector_generator(H, state, max_moments):
    """
    Input:
    H -> np.ndarray: Hamiltonian
    state -> np.ndarray: state to which we apply the kpm expansion
    max_moments -> int: number of moments to use in the KPM expansion

    Output:
    np.ndarray: yields the state after each order of the expansion
    """

    # 0th moment in the expansion: Just the quantum state to which we are applying the operator
    alpha = state
    n = 0
    yield alpha

    # 1st moment in the expansion: Applying the Hamiltonian
    n += 1
    alpha_prev = alpha.copy()
    alpha = H @ alpha
    yield alpha

    # nth moments of the expansion: Follows by the recurrence of the Chebyshev polynomials
    n += 1
    while n < max_moments:
        alpha_save = alpha.copy()
        alpha = 2 * H @ alpha - alpha_prev
        alpha_prev = alpha_save
        yield alpha
        n += 1

def OPDM_KPM(state, num_moments, H, Ef=0, bounds=None):
    """
    Input:
    state -> np.ndarray: state to which we apply the kpm expansion
    num_moments -> int: number of moments to use in the KPM expansion
    H -> np.ndarray: Hamiltonian
    Ef -> float: Fermi energy at which to calculate the filled band projector (OPDM)
    bounds -> tuple of floats: Bounds for the spectrum, estimated if not provided

    Output:
    P_vec -> np.ndarray: OPDM applied onto the input state
    """

    # Rescaling of H and energies for the Kernel Polynomial Expansion
    num_moments = num_moments
    H_rescaled, (a, b) = kwant.kpm._rescale(H, 0.05, None, bounds)
    phi_f = np.arccos((Ef - b) / a)

    # Calculation of the coefficients in the expansion using the Jackson Kernel
    g = jackson_kernel(np.ones(num_moments))
    g[0] = 0
    m = np.arange(num_moments)
    m[0] = 1
    coefs = -2 * g * (np.sin(m * phi_f) / (m * np.pi))

    # Calculation of the OPDM (projector) applied onto vector as described in PRR 2, 013229 (2020)
    P_vec = (1 - phi_f/np.pi) * state + sum(c * vec for c, vec
                             in zip(coefs, kpm_vector_generator(H_rescaled, state, num_moments)))
    return P_vec

def local_marker_KPM_bulk(syst, S, Nx, Ny, Nz, Ef=0., num_moments=1000, num_vecs=5, bounds=None, cutoff=0.2):
    """
    Input:
    syst -> kwant.builder.Builder: Kwant closed system (no leads) for the nanowire
    S -> scipy.sparse.csr_matrix: Chiral symmetry of the nanowire
    Nx, Ny, Nz -> int: Number of sites in each direction
    Ef -> float: Fermi energy at which to calculate the OPDM
    num_moments -> int: number of moments to use on the KPM expansion
    num_vecs -> int: number of random vectors to use in the stochastic trace evaluation
    bounds -> tuple of floats: Bounds for the spectrum, estimated if not provided
    cutoff -> float: fraction of the nanowire we average over, taking the origin at the center of the nanowire (explained
                     in the main text)

    Output:
    np.complex128: Average bulk marker
    """

    # Region where we calculate the local marker
    project_to_region = partial(bulk_state, syst, rx=cutoff * Nx, ry=cutoff * Ny, rz=cutoff * Nz, Nx=Nx, Ny=Ny, Nz=Nz)

    # Operators involved in the calculation of the local marker
    H = syst.hamiltonian_submatrix(params=dict(flux=0., mu=0.), sparse=True).tocsr()
    P = partial(OPDM_KPM, num_moments=num_moments, H=H, Ef=Ef, bounds=bounds)
    [X, Y, Z] = position_operator_OBC(syst, Nx, Ny, Nz)[0]

    # Calculation using the stochastic trace + KPM algorithm
    M = 0.
    for i in range(num_vecs):

        # Random initial state supported in the region that we trace over
        loger_kwant.info(f'Random vector {i}/ {num_vecs - 1}')
        state, Nsites = project_to_region(state=np.exp(2j * np.pi * np.random.random((H.shape[0]))))

        # Calculation of the invariant
        P_psi = P(state)
        SP_psi = S @ P_psi
        PXP_psi, PYP_psi, PZP_psi = P(X @ P_psi),  P(Y @ P_psi),  P(Z @ P_psi)
        PXSP_psi, PYSP_psi, PZSP_psi = P(X @ SP_psi), P(Y @ SP_psi),  P(Z @ SP_psi)
        M +=  (Y @ PXSP_psi).T.conj() @ PZP_psi + (X @ PZSP_psi).T.conj() @ PYP_psi + (Z @ PYSP_psi).T.conj() @ PXP_psi
        M += -(Z @ PXSP_psi).T.conj() @ PYP_psi - (Y @ PZSP_psi).T.conj() @ PXP_psi - (X @ PYSP_psi).T.conj() @ PZP_psi

    return (8 * pi / 3) * np.imag(M) / (num_vecs * Nsites)

def local_marker_per_site_cross_section_KPM(syst, S, Nx, Ny, Nz, z0, z1, Ef=0., num_moments=500, bounds=None):
    """
    Input:
    syst -> kwant.builder.Builder: Kwant closed system (no leads) for the nanowire
    S -> scipy.sparse.csr_matrix: Chiral symmetry of the nanowire
    Nx, Ny, Nz -> int: Number of sites in each direction
    z0, z1 -> float: z positions between which we calculate the local marker on every site
    Ef -> float: Fermi energy at which to calculate the OPDM
    num_moments -> int: number of moments to use on the KPM expansion
    bounds -> tuple of floats: Bounds for the spectrum, estimated if not provided
    cutoff -> float: fraction of the nanowire we average over, taking the origin at the center of the nanowire (explained
                     in the main text)

    Output:
    local_marker -> np.ndarray: local marker for every site in the considered cut of the nanowire
    np.ndarray: x position of the sites where the marker is calculated
    np.ndarray: y position of the sites where the marker is calculated
    np.ndarray: z position of the sites where the marker is calculated
    """

    # Operators involved in the calculation of the local marker
    H = syst.hamiltonian_submatrix(params=dict(flux=0., mu=0.), sparse=True).tocsr()
    P = partial(OPDM_KPM, num_moments=num_moments, H=H, Ef=Ef, bounds=bounds)
    [X, Y, Z], pos = position_operator_OBC(syst, Nx, Ny, Nz)

    # Cross-section we are interested in
    cond1 = pos[:, 2] < z1
    cond2 = z0 < pos[:, 2]
    cond = cond1 * cond2
    indices = [i for i in range(int(Nx * Ny * Nz)) if cond[i]]
    local_marker = np.zeros((len(indices), ), dtype=np.complex128)

    # Calculation using the stochastic trace + KPM algorithm
    for i, idx in enumerate(indices):
        loger_kwant.info(f'site: {i}/ {len(indices)}')

        for j in range(4):
            # States localised in the site
            state = np.zeros((Nx * Ny * Nz * 4, ), dtype=np.complex128)
            state[idx * 4 + j] = 1.

            # Calculation of the invariant
            P_psi = P(state)
            SP_psi = S @ P_psi
            PXP_psi, PYP_psi, PZP_psi = P(X @ P_psi),  P(Y @ P_psi),  P(Z @ P_psi)
            PXSP_psi, PYSP_psi, PZSP_psi = P(X @ SP_psi), P(Y @ SP_psi),  P(Z @ SP_psi)
            local_marker[i] +=  (Y @ PXSP_psi).T.conj() @ PZP_psi + (X @ PZSP_psi).T.conj() @ PYP_psi + (Z @ PYSP_psi).T.conj() @ PXP_psi
            local_marker[i] += -(Z @ PXSP_psi).T.conj() @ PYP_psi - (Y @ PZSP_psi).T.conj() @ PXP_psi - (X @ PYSP_psi).T.conj() @ PZP_psi

        local_marker[i] = (8 * pi / 3) * np.imag(local_marker[i])
        loger_kwant.info(f'marker: {local_marker[i]}')

    return local_marker, pos[:, 0][cond], pos[:, 1][cond], pos[:, 2][cond]

def position_operator_OBC(syst, Nx, Ny, Nz):
    """
    Input:
      syst -> kwant.builder.Builder: Kwant closed system (no leads) for the nanowire
    Nx, Ny, Nz -> int: Number of sites in each direction

    Output:
    operators -> list of np.ndarrays: Position operators for the nanowire
    pos -> np.ndarray: position of the sites in the nanowire
    """
    operators = []
    norbs = syst.sites[0].family.norbs
    pos = np.array([s.pos for s in syst.sites])
    for c in range(pos.shape[1]):
        if c==0:
            N = Nx
        elif c==1:
            N = Ny
        else:
            N = Nz
        operators.append(diags(np.repeat(pos[:, c] - 0.5 * (N - 1), norbs), format='csr'))
    return operators, pos

def bulk_state(syst, rx, ry, rz, Nx, Ny, Nz, state):
    """
    Input:
    syst -> kwant.builder.Builder: Kwant closed system (no leads) for the nanowire
    rx, ry, rz -> float: cutoff distances delimiting the bulk
    Nx, Ny, Nz -> int: Number of sites in each direction
    state -> np.ndarray: state to be restricted to the bulk sites

    Output:
    state -> np.ndarray: state with only support in the bulk sites
    Nsites -> int: number sites in the considered bulk
    """

    # Selecting a region on the bulk
    pos = np.array([s.pos for s in syst.sites])
    x_pos, y_pos, z_pos = pos[:, 0] - 0.5 * (Nx-1), pos[:, 1] - 0.5 * (Ny-1), pos[:, 2] - 0.5 * (Nz-1)
    cond1 = np.abs(x_pos) < rx
    cond2 = np.abs(y_pos) < ry
    cond3 = np.abs(z_pos) < rz
    cond = cond1 * cond2 * cond3
    Nsites = len(cond[cond])
    cond = np.repeat(cond, 4)

    # Weighted state on the bulk region
    state[~cond] = 0.
    return state, Nsites



