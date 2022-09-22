# FUNCTIONS
# Function directory for the stacked chern amorphous nanowire

import numpy as np
from numpy import e, pi
from random import gauss

# Pauli matrices
sigma_0 = np.eye(2)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
sigma_p = 0.5 * (sigma_x + 1j * sigma_y)
sigma_m = 0.5 * (sigma_x - 1j * sigma_y)

# %%  Definition of the RPS, distance, angle and boundary conditions

def GaussianPointSet2D(x, y, width):
    # Generate a gaussian point set with the specified width
    # x, y, z: Positions for the crystalline case
    # width: Specified with for the gaussian distribution

    x = np.random.normal(x, width, len(x))
    y = np.random.normal(y, width, len(y))

    return x, y


def displacement2D(x1, y1, x2, y2, L_x, L_y, boundary):
    # Calculates the displacement vector between sites 2 and 1.
    # x1, y1, z1, x2, y2, z2: Coordinates of the sites
    # L_x, L_y, L_z: System sizes
    # boundary: String containing "Open" or "Closed"

    # Definition of the vector between sites 2 and 1 (from st.1 to st.2)
    v = np.zeros((2,))
    if boundary == "Closed":
        v[0] = (x2 - x1) - L_x * np.sign(x2 - x1) * np.heaviside(abs(x2 - x1) - L_x / 2, 0)
        v[1] = (y2 - y1) - L_y * np.sign(y2 - y1) * np.heaviside(abs(y2 - y1) - L_y / 2, 0)

    elif boundary == "Open":
        v[0] = (x2 - x1)
        v[1] = (y2 - y1)

    # Module of the vector between sites 2 and 1
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


# %% Hopping functions and hamiltonian

def xtranslation2D(x, y, n_x, n_y):
    # Translates the vector x one site in direction x
    # x, y: Vectors with the position of the lattice sites
    # n_x, n_y: Dimension s of the lattice grid
    transx = ((x + 1) % n_x) + n_x * y
    return transx

def ytranslation2D(x, y, n_x, n_y):
    # Translates the vector y one site in direction y
    # x, y: Vectors with the position of the lattice sites
    # n_x, n_y: Dimension s of the lattice grid
    transy = x + n_x * ((y + 1) % n_y)
    return transy

def spectrum(H):
    # Calculates the spectrum a of the given Hamiltonian
    # H: Hamiltonian for the model
    # n_particles: Number of particles we want (needed for the projector)

    energy, eigenstates = np.linalg.eigh(H)  # Diagonalise H
    idx = energy.argsort()                   # Indexes from lower to higher energy
    energy = energy[idx]                     # Ordered energy eigenvalues
    eigenstates = eigenstates[:, idx]        # Ordered eigenstates

    return energy, eigenstates

def H_onsite(M, t1, t2, lamb, k):
    # Calculates the diagonal block associated with the onsite energies and onsite orbital/spin hopping
    # M: Mass parameter of the model
    # lamb: Chiral spin orbit coupling/ pairing potential
    # t1: Chiral diagonal hopping
    # t2: Non-chiral spin orbit coupling
    # k: momentum in z direction

    H_on = - (M + 2 * t1 * np.cos(k)) * np.kron(sigma_0, sigma_z) + 2 * t2 * np.cos(k) *  np.kron(sigma_z, sigma_y) +\
                  - 2 * lamb * np.sin(k) * np.kron(sigma_z, sigma_x)

    return H_on

def RadialHopp(r):
    # Calculates the radial hopping strength for two sites at a distance r
    # r : Distance between sites (in units of a)

    hop_amplitude = e * np.exp(- r)  # Hopping amplitude

    return hop_amplitude

def H_hopp(r, phi, t1, t2, lamb):
    # Calculates the angular hopping hamiltonian
    # phi:  Relative angle between sites in XY cross-section
    # M: Mass parameter of the model
    # lamb: Chiral spin orbit coupling/ pairing potential
    # t1: Chiral diagonal hopping
    # t2: Non-chiral spin orbit coupling
    # k: momentum in z direction

    # Hamiltonian
    H_off = 1j * lamb * np.kron(sigma_x * np.cos(phi) + sigma_y * np.sin(phi), sigma_x) +\
         +  t2 * np.kron(sigma_x * np.cos(phi) + sigma_y * np.sin(phi), sigma_y) - t1 * np.kron(sigma_0, sigma_z)

    return RadialHopp(r) * 0.5 * H_off

def H_offdiag(n_sites, n_orb, n_neighbours, L_x, L_y, x, y, t1, t2, lamb, boundary):
    # Generates the Hamiltonian for a 3D insulator with the parameters specified
    # t2 = 0  we are in the DIII class (S = - sigma0 x sigma_y), if t2 is non-zero we are in the AII class
    # n_sites: Number of sites in the lattice
    # n_orb: Number of orbitals in the model
    # n_neighbours: Number of fixed neighbours
    # x: x position of the sites in the RPS
    # y: y position of the sites in the RPS
    # M, t1, t2, lamb: Parameters of the AII model
    # Boundary: String with values "Open" or "Closed" which selects the boundary we want

    # Declarations
    H = np.zeros((n_sites * n_orb, n_sites * n_orb), complex)              # Declaration of the matrix hamiltonian
    matrix_neighbours = np.tile(np.arange(n_sites), (n_sites, 1))          # Declaration matrix of neighbours
    matrix_dist = np.zeros((n_sites, n_sites))                             # Declaration matrix of dists
    matrix_phis = np.zeros((n_sites, n_sites))                             # Declaration matrix of phis

    # Off-diagonal hopping terms and neighbour structure
    cont = 1  # Starts at 1 to avoid calculating diagonal properties which we do not need
    for index1 in range(0, n_sites):
        for index2 in range(cont, n_sites):
            r, phi = displacement2D(x[index1], y[index1], x[index2], y[index2], L_x, L_y, boundary)
            matrix_dist[index1, index2], matrix_phis[index1, index2], = r, phi
            matrix_dist[index2, index1], matrix_phis[index2, index1], = r, phi + pi

        # Sorting
        idx = matrix_dist[index1, :].argsort()                         # Sorting the distances from minimum to maximum
        matrix_neighbours[index1, :] = matrix_neighbours[index1, idx]  # Ordered neighbours
        matrix_dist[index1, :] = matrix_dist[index1, idx]              # Ordered distances
        matrix_phis[index1, :] = matrix_phis[index1, idx]              # Ordered phis

        # Hamiltonian for the nearest neighbour contributions
        for index2 in range(1, n_neighbours + 1):
            row = index1 * n_orb
            column = int(matrix_neighbours[index1, index2]) * n_orb
            r, phi = matrix_dist[index1, index2], matrix_phis[index1, index2]
            H[row: row + n_orb, column: column + n_orb] = H_hopp(r, phi, t1, t2, lamb)
            H[column: column + n_orb, row: row + n_orb] = np.conj(H[row: row + n_orb, column: column + n_orb]).T

        cont = cont + 1

    return H


