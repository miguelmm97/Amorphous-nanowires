# FUNCTIONS
# Function directory for the stacked chern amorphous nanowire

import numpy as np
from numpy import e, pi
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, MultiPoint
from shapely.iterops import intersects

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
    if v[0] == 0:  # Pathological case, separated to not divide by 0
        if v[1] > 0:
            phi = pi / 2  # Hopping in y
        else:
            phi = 3 * pi / 2  # Hopping in -y
    else:
        if v[1] > 0:
            phi = np.arctan2(v[1], v[0])  # 1st and 2nd quadrants
        else:
            phi = 2 * pi + np.arctan2(v[1], v[0])  # 3rd and 4th quadrants

    return r, phi


def angle_x(v):
    # Computes the angle of a vector with respect to axis x
    # v: Vector in question

    if v[0] == 0:  # Pathological case, separated to not divide by 0
        if v[1] > 0:
            phi = pi / 2  # Hopping in y
        else:
            phi = 3 * pi / 2  # Hopping in -y
    else:
        if v[1] > 0:
            phi = np.arctan2(v[1], v[0])  # 1st and 2nd quadrants
        else:
            phi = 2 * pi + np.arctan2(v[1], v[0])  # 3rd and 4th quadrants

    return phi


def angle(v, ax):
    # Computes the angle between a vector and a particular axis with correct cuadrants from 0 to 2pi
    # v: Vector in question
    # ax: Axis to measure angle
    alpha = - angle_x(ax)
    R = np.array([[np.cos(alpha), -np.sin(alpha)],
                  [np.sin(alpha), np.cos(alpha)]])
    phi = angle_x(R @ v)
    return phi


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
    idx = energy.argsort()  # Indexes from lower to higher energy
    energy = energy[idx]  # Ordered energy eigenvalues
    eigenstates = eigenstates[:, idx]  # Ordered eigenstates

    return energy, eigenstates


def H_onsite(M, t1, t2, lamb, k):
    # Calculates the diagonal block associated with the onsite energies and onsite orbital/spin hopping
    # M: Mass parameter of the model
    # lamb: Chiral spin orbit coupling/ pairing potential
    # t1: Chiral diagonal hopping
    # t2: Non-chiral spin orbit coupling
    # k: momentum in z direction

    H_on = - (M + 2 * t1 * np.cos(k)) * np.kron(sigma_0, sigma_z) + 2 * t2 * np.cos(k) * np.kron(sigma_z, sigma_y) + \
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
    H_off = 1j * lamb * np.kron(sigma_x * np.cos(phi) + sigma_y * np.sin(phi), sigma_x) + \
            +  t2 * np.kron(sigma_x * np.cos(phi) + sigma_y * np.sin(phi), sigma_y) - t1 * np.kron(sigma_0, sigma_z)

    return RadialHopp(r) * 0.5 * H_off


def Peierls(x0, x1, y0, y1, B):
    # Calculates the Peierls phase betwee sites 0 and 1.
    # x0, x1, y0, y1: Coordinates of the sites
    # Magnetic field through the cross-section
    if x1 != x0:
        phase = B * ((y1 - y0) / (x1 - x0)) * 0.5 * (x1 ** 2 + x1 * x0) + B * y0 * (x1 - x0)
    else:
        phase = 0
    return np.exp(1j * 2 * pi * phase)


def H_offdiag(n_sites, n_orb, L_x, L_y, x, y, t1, t2, lamb, B, r_cutoff, boundary):
    # Generates the Hamiltonian for a 3D insulator with the parameters specified
    # t2 = 0  we are in the DIII class (S = - sigma0 x sigma_y), if t2 is non-zero we are in the AII class
    # n_sites: Number of sites in the lattice
    # n_orb: Number of orbitals in the model
    # x: x position of the sites in the RPS
    # y: y position of the sites in the RPS
    # M, t1, t2, lamb: Parameters of the AII model
    # B: Magnetic field through the cross-section of the wire
    # Boundary: String with values "Open" or "Closed" which selects the boundary we want

    H = np.zeros((n_sites * n_orb, n_sites * n_orb), complex)  # Declaration of the matrix hamiltonian

    # Loop through the different sites of the lattice
    cont = 1  # Starts at 1 to avoid calculating diagonal properties which we do not need
    for site1 in range(0, n_sites):
        for site2 in range(cont, n_sites):
            r, phi = displacement2D(x[site1], y[site1], x[site2], y[site2], L_x, L_y, boundary)
            if r < r_cutoff:
                row, column = site1 * n_orb, site2 * n_orb
                H[row: row + n_orb, column: column + n_orb] = H_hopp(r, phi, t1, t2, lamb) * Peierls(x[site1], x[site2],
                                                                                                     y[site1], y[site2],
                                                                                                     B)
                H[column: column + n_orb, row: row + n_orb] = np.conj(H[row: row + n_orb, column: column + n_orb]).T
        cont = cont + 1

    return H


def Check_blocks(H, len, site_dim, spin=None):
    # Checks the block structure of a Hamiltonian matrix
    # H: Hamiltonian matrix
    # len: Dimension of the full hilbert space
    # site_dim: Dimension of the onsite Hilbert space
    # spin : Spin value of the system if applicable

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.spy(H)
    for j in np.arange(0, len, site_dim):
        aux = j - 0.5
        ax.plot(aux * np.ones((len,)), np.arange(0, len, 1), 'b', linewidth=0.2)
        ax.plot(np.arange(0, len, 1), aux * np.ones((len,)), 'b', linewidth=0.2)

    if spin is not None:
        spin_dim = int(2 * spin + 1)
        for j in np.arange(0, len, spin_dim):
            aux = j - 0.5
            ax.plot(aux * np.ones((len,)), np.arange(0, len, 1), '--b', linewidth=0.1)
            ax.plot(np.arange(0, len, 1), aux * np.ones((len,)), '--b', linewidth=0.1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def list_neighbours(x, y, L_x, L_y, boundary):
    # Function that gives back a matrix of neighbours, their relative distance, angle, and vector.
    # x, y: x, y positions of each site
    # L_xy : Length of the system in each direction
    # Boundary: "Open" or "Closed"
    # n_neighbours: (Optional) Fixed number of nearest neighbours

    # Declarations
    L = len(x)
    matrix_neighbours = np.tile(np.arange(L), (L, 1))  # Declaration matrix of neighbours
    matrix_dist = np.zeros((L, L))  # Declaration matrix of dists
    matrix_phis = np.zeros((L, L))  # Declaration matrix of phis
    matrix_vecs_x = np.zeros((L, L))  # Declaration  matrix of vectors connecting neighbours
    matrix_vecs_y = np.zeros((L, L))  # Declaration  matrix of vectors connecting neighbours

    # Loop through the different sites of the lattice
    cont = 1
    for site1 in range(0, L):
        for site2 in range(cont, L):
            r, phi = displacement2D(x[site1], y[site1], x[site2], y[site2], L_x, L_y, boundary)
            matrix_dist[site1, site2], matrix_phis[site1, site2], = r, phi
            matrix_dist[site2, site1], matrix_phis[site2, site1], = r, phi + pi
            matrix_vecs_x[site1, site2], matrix_vecs_y[site1, site2] = np.cos(phi), np.sin(phi)
            matrix_vecs_x[site2, site1], matrix_vecs_y[site2, site1] = -np.cos(phi), -np.sin(phi)

    # Sorting
    idx = np.argsort(matrix_dist, axis=1)  # Sorting distances from minimum to maximum
    matrix_neighbours = np.take_along_axis(matrix_neighbours, idx, axis=1)  # Ordered neighbours
    matrix_dist = np.take_along_axis(matrix_dist, idx, axis=1)  # Ordered distances
    matrix_phis = np.take_along_axis(matrix_phis, idx, axis=1)  # Ordered phis
    matrix_vecs_x = np.take_along_axis(matrix_vecs_x, idx, axis=1)  # Ordered vectors connecting neighbours
    matrix_vecs_y = np.take_along_axis(matrix_vecs_y, idx, axis=1)  # Ordered vectors connecting neighbours

    # Delete the first column containing the same site so that we get only info about the neighbours
    neighbours = matrix_neighbours[:, 1:]
    dist = matrix_dist[:, 1:]
    phis = matrix_phis[:, 1:]
    vecs_x, vecs_y = matrix_vecs_x[:, 1:], matrix_vecs_y[:, 1:]

    return neighbours, dist, phis, vecs_x, vecs_y


def No_Boundary_intersection(boundary_line, boundary_points, line_neigh):

    # Calculates the intersection of a boundary line with a neighbour line, without taking the boundaries points
    # into account.
    # boundary_line: LineString containing the boundary
    # boundary_points: MultiPoint containing the points generating the boundary
    # line_neigh: Line connecting the two neighbours in question

    inters_point = line_neigh.intersection(boundary_line)

    return boundary_points.contains(inters_point)


def physical_boundary(x, y, neighbours, dist, vecs_x, vecs_y, neighbour_cutoff):

    # Calculates the "physical" boundary of an amorphous point set, that is, the outermost loop one can create by
    # connecting neighbouring sites (or at least it calculates one of the many).
    # x, y: x, y positions of the point set sites
    # neighbours: Matrix whose nth row contains the sites sorted by distance with respect to the nth site
    # dist: Matrix whose nth row contains the distances sorted by distance with respect to the nth site
    # vecs_x,y: Matrix whose nth row contains the x,y component of the directive vector sorted by distance with respect to the nth site
    # neighbour cutoff: Cutoff scale for the nearest neighbour couplings

    # Declarations
    site0, vec0, count = None, np.array([1, 0]), 0
    new_point = []
    pts_boundary = []
    sites_boundary = []

    # Choosing initial site
    site_left = np.where(x == min(x))[0][0]

    # Algorithm to find the boundary
    while site0 != site_left:

        # Parameters for each iteration
        dif = 2 * pi - 0.01  # Minimum angle between the neighbours and the last neighbour connection
        return_needed = 1    # 0 if we find a neighbour that does not intersect the boundary, 1 otherwise

        # Initial point
        if count == 0:
            site0 = site_left
            new_point = Point(x[site0], y[site0])
            pts_boundary = [new_point]                      # List of Point objects at the boundary
            multipts_boundary = MultiPoint(pts_boundary)    # Multipoint object forming the boundary
            sites_boundary.append(site0)                    # List of sites at the boundary

        # Neighbours admissible for the boundary
        ind1 = np.where(dist[site0, :] < neighbour_cutoff)[0]                  # "Nearest" neighbours
        list_neigh = neighbours[site0, ind1]                                   # List of nearest neighbours
        vec_neigh_x, vec_neigh_y = vecs_x[site0, ind1], vecs_y[site0, ind1]    # Vectors connecting the nearest neighbours

        # Choosing the neighbour that minimises the angle with vec0 and does not intersect the boundary
        for j, neigh in enumerate(list_neigh):

            # Select possible neighbour
            neigh_point = Point(x[neigh], y[neigh])               # Point object for the neighbour
            line_neigh = LineString([new_point, neigh_point])     # LineString connecting to the neighbour
            vec1 = np.array([vec_neigh_x[j], vec_neigh_y[j]])     # Vector to the neighbour
            ang = angle(vec1, -vec0)                              # Angle between neighbour and previous connection

            # Minimise the angle for the first point
            if count == 0:
                if ang < dif:
                    dif = ang
                    vec2 = vec1
                    site0 = neigh
                    return_needed = 0

            # Minimise the angle for the other points
            else:
                if ang < dif and No_Boundary_intersection(line_boundary, multipts_boundary, line_neigh) is True:
                    dif = ang
                    vec2 = vec1
                    site0 = neigh
                    return_needed = 0

        # In case returning is the only option...
        if return_needed == 1:
            for j, neigh in enumerate(list_neigh):
                vec2 = np.array([vec_neigh_x[j], vec_neigh_y[j]])
                site0 = neigh

        # New point at the boundary
        vec0 = vec2
        new_point = Point(x[site0], y[site0])
        pts_boundary.append(new_point)
        multipts_boundary = MultiPoint(pts_boundary)
        line_boundary = LineString(pts_boundary)
        sites_boundary.append(site0)
        count = count + 1

    return line_boundary, sites_boundary


def lattice_graph(L_x, L_y, n_sites, x, y, r_cutoff, pts_boundary=None):
    # Generates the  graph of the RPS
    # L_x, L_y: Dimensions of the RPS grid
    # n_sites: Number of sites in the RPS
    # x: x position of the sites in the RPS
    # y: y position of the sites in the RPS
    # r_cutoff: Distance cut-off for the hopping amplitudes
    # pts_boundary : List of sites at the boundary

    cont = 0
    for index1 in range(0, n_sites):
        for index2 in range(cont, n_sites):

            r, angle = displacement2D(x[index1], y[index1], x[index2], y[index2], L_x, L_y,
                                      "Open")  # Distance between sites
            if r < r_cutoff:  # Hopping between sites
                plt.plot([x[index1], x[index2]], [y[index1], y[index2]], 'tab:gray', linewidth=1, alpha=0.3)
        cont = cont + 1  # Update the counter so that we skip through index2 = previous indexes 1

    if pts_boundary is not None:
        for j in range(0, len(pts_boundary)):

            if j == len(pts_boundary) - 1:
                site1 = pts_boundary[j]
                site2 = pts_boundary[0]
            else:
                site1 = pts_boundary[j]
                site2 = pts_boundary[j + 1]

            plt.plot([x[site1], x[site2]], [y[site1], y[site2]], 'tab:Green', linewidth=1, alpha=1)

    plt.scatter(x, y)
    plt.xlim(-1, L_x + 1)
    plt.ylim(-1, L_y + 1)
    plt.xticks(color="w")
    plt.yticks(color="w")
    plt.show()

