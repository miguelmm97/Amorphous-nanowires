# FUNCTIONS
# Function directory for the stacked chern amorphous nanowire

import numpy as np
from numpy import e, pi
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, MultiPoint, Polygon
from shapely import intersects

# Pauli matrices
sigma_0 = np.eye(2)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
sigma_p = 0.5 * (sigma_x + 1j * sigma_y)
sigma_m = 0.5 * (sigma_x - 1j * sigma_y)

def spectrum(H):
    # Calculates the spectrum a of the given Hamiltonian
    # H: Hamiltonian for the model
    # n_particles: Number of particles we want (needed for the projector)

    energy, eigenstates = np.linalg.eigh(H)  # Diagonalise H
    idx = energy.argsort()  # Indexes from lower to higher energy
    energy = energy[idx]  # Ordered energy eigenvalues
    eigenstates = eigenstates[:, idx]  # Ordered eigenstates

    return energy, eigenstates

# %%  Definition of the RPS, distance, angle and boundary conditions

def GaussianPointSet2D(x, y, width):
    """
    Generates a gaussian point set with the specified width

    Params:
    ------
    x, y, z:  {np.array(float)}Positions for the crystalline lattice
    width:    {float} Specified with for the gaussian distribution

    Returns:
    -------
    x, y:     {np.array(float)} Positions for the gaussian point set
    """
    x = np.random.normal(x, width, len(x))
    y = np.random.normal(y, width, len(y))

    return x, y

def displacement2D(x1, y1, x2, y2, L_x, L_y, boundary):
    """
    Calculates the displacement vector between sites 2 and 1

    Params:
    ------
    x1:        {float}   Coordinates of the sites
    y1:        {float}   Coordinates of the sites
    x2:        {float}   Coordinates of the sites
    y2:        {float}   Coordinates of the sites
    L_x:       {int}     System sizes
    L_y:       {int}     System sizes
    boundary:  {string}  'Open' or  'Closed'

    Returns:
    -------
    r:    {float} Distance between sites
    phi:  {float} Angle measured from +x between the sites
    """

    # Definition of the vector between sites 2 and 1 (from st.1 to st.2)
    v = np.zeros((2,))
    if boundary == "Closed":
        v[0] = (x2 - x1) - L_x * np.sign(x2 - x1) * np.heaviside(abs(x2 - x1) - L_x / 2, 0)
        v[1] = (y2 - y1) - L_y * np.sign(y2 - y1) * np.heaviside(abs(y2 - y1) - L_y / 2, 0)

    elif boundary == "Open":
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

def angle_x(v):
    """
    Computes the angle of a vector with respect to axis x

    Params:
    ------
    v:   {np.array(2,)} Vector in question

    Returns:
    -------
    phi: {float} Angle
    """

    if v[0] == 0:                                  # Pathological case, separated to not divide by 0
        if v[1] > 0:
            phi = pi / 2                           # Hopping in y
        else:
            phi = 3 * pi / 2                       # Hopping in -y
    else:
        if v[1] > 0:
            phi = np.arctan2(v[1], v[0])           # 1st and 2nd quadrants
        else:
            phi = 2 * pi + np.arctan2(v[1], v[0])  # 3rd and 4th quadrants

    return phi

def angle(v, ax):
    """
        Computes the angle of a vector and a particular axis with correct quadrants from 0 to 2pi

        Params:
        ------
        v:   {np.array(2, )} Vector in question
        ax:  {np.array(2, }  Axis to measure angle

        Returns:
        -------
        phi: {float} Angle
        """

    alpha = - angle_x(ax)
    R = np.array([[np.cos(alpha), -np.sin(alpha)],
                  [np.sin(alpha), np.cos(alpha)]])
    phi = angle_x(R @ v)
    return phi

def xtranslation2D(x, y, n_x, n_y):
    """
    Translates the vector x one site in x direction
    Params:
    ------
    x:   {np.array}     Vectors with the position of the lattice sites
    y:   {np.array}     Vectors with the position of the lattice sites
    n_x: {int}          Dimensions of the lattice grid
    n_y: {int}          Dimensions of the lattice grid

    Returns:
    -------
    transx:  {np.array} Translated positions of the sites
    """

    transx = ((x + 1) % n_x) + n_x * y
    return transx

def ytranslation2D(x, y, n_x, n_y):
    """
    Translates the vector y one site in y direction
    Params:
    ------
    x:   {np.array}     Vectors with the position of the lattice sites
    y:   {np.array}     Vectors with the position of the lattice sites
    n_x: {int}          Dimensions of the lattice grid
    n_y: {int}          Dimensions of the lattice grid

    Returns:
    -------
    transy:  {np.array} Translated positions of the sites
    """
    transy = x + n_x * ((y + 1) % n_y)
    return transy



# %% Hopping functions and hamiltonian


def RadialHopp(r):
    """
    Calculates the radial hopping strength for two sites at a distance r

    Params:
    ------
    r: {float} Distance between the sites

    Returns:
    -------
    hop_amplitude: {float} Hopping amplitude
    """

    hop_amplitude = e * np.exp(- r)  # Hopping amplitude

    return hop_amplitude


def H_hopp(r, phi, t1, t2, lamb):
    """
    Calculates the angular hopping hamiltonian on the inner degrees of freedom

    Params:
    ------
    r:     {float} Distance between sites
    phi:   {float} Relative angle between sites in XY cross-section
    t1:    {float} Chiral diagonal hopping
    t2:    {float} Non-chiral spin orbit coupling
    lamb:  {float} Chiral spin orbit coupling/ pairing potential

    Returns:
    -------
    {np.array(4, 4)} Matrix hamiltonian
    """

    H_off = 1j * lamb * np.kron(sigma_x * np.cos(phi) - sigma_y * np.sin(phi), sigma_x) + \
            +  t2 * np.kron(sigma_x * np.cos(phi) - sigma_y * np.sin(phi), sigma_y) - t1 * np.kron(sigma_0, sigma_z)

    return RadialHopp(r) * 0.5 * H_off


def Peierls(x0, x1, y0, y1, B):
    """
    Calculates the Peierls phase between sites 0 and 1.

    Params:
    ------
    x0: {float} Coordinates of the sites
    x1: {float} Coordinates of the sites
    y0: {float} Coordinates of the sites
    y1: {float} Coordinates of the sites
    B:  {float} Magnetic field through the cross-section

    Returns:
    -------
    {float} Peierls phase
    """

    if x1 != x0:
        phase = B * ((y1 - y0) / (x1 - x0)) * 0.5 * (x1 ** 2 + x1 * x0) + B * (y0 * x1 - y1 * x0)
    else:
        phase = 0
    return np.exp(1j * 2 * pi * phase)


def H_onsite(M, t1, t2, lamb, k):
    """
    Calculates the onsite hamiltonian on the inner degrees of freedom

    Params:
    ------
    M:     {float} Mass term
    t1:    {float} Chiral diagonal hopping
    t2:    {float} Non-chiral spin orbit coupling
    lamb:  {float} Chiral spin orbit coupling/ pairing potential
    k:     {float} Momentum along the translationally invariant direction

    Returns:
    -------
    {np.array(4, 4)} Matrix hamiltonian
    """

    H_on = - (M + 2 * t1 * np.cos(k)) * np.kron(sigma_0, sigma_z) + 2 * t2 * np.cos(k) * np.kron(sigma_z, sigma_y) + \
           - 2 * lamb * np.sin(k) * np.kron(sigma_z, sigma_x)

    return H_on


def H_offdiag(n_sites, n_orb, x, y, t1, t2, lamb, B, neighbour_cutoff, neighbours, dist, phis):
    """
    Generates the Hamiltonian for a 3D insulator with the parameters specified.
    We enforce at least two neighbours per site.

    Params:
    ------
    n_sites:                   {int}         Number of sites in the lattice
    n_orb:                     {int}         Number of orbitals in the model
    x:                         {np.array()}  x position of the sites in the RPS
    y:                         {np.array()}  y position of the sites in the RPS
    t1, t2, lamb:              {float}       Parameters of the model
    B:                         {float}       Magnetic field through the cross-section of the wire
    neighbour_cutoff:          {float}       Scale of the allowed neighbour hoppings
    neighbours, dist, phis:    {np.array()}  Matrix with neighbour sites, distances and phis sorted by distance along every row

    Returns:
    -------
    {np.array} Matrix hamiltonian
    """

    H = np.zeros((n_sites * n_orb, n_sites * n_orb), complex)  # Definition of the matrix hamiltonian

    # Loop through the different sites of the lattice
    for site1 in range(0, n_sites):

        # Neighbours associated to this site, enforcing at least two neighbours
        ind1 = np.where(dist[site1, :] < neighbour_cutoff)[0]
        if len(ind1) < 2:
            list_neigh = neighbours[site1, np.array([0, 1])]
        else:
            list_neigh = neighbours[site1, ind1]

        # Coupling with the neighbours
        for j, site2 in enumerate(list_neigh):
            r, phi = dist[site1, j], phis[site1, j]
            row, column = site1 * n_orb, site2 * n_orb
            H[row: row + n_orb, column: column + n_orb] = H_hopp(r, phi, t1, t2, lamb) * Peierls(x[site1], x[site2],
                                                                                                 y[site1], y[site2], B)
            H[column: column + n_orb, row: row + n_orb] = np.conj(H[row: row + n_orb, column: column + n_orb]).T

    return H



#%% Neighbours and boundary

def list_neighbours(x, y, L_x, L_y, boundary, sorted=None):
    """
    Gives back a matrix of neighbours, their relative distance, angle, and vector.

    Params:
    ------
    x, y:                 {np.array()}        x, y positions of each site
    L_x, L_y :            {int}               Length of the system in each direction
    Boundary:             {string}            "Open" or "Closed"
    sorted:               {string, optional}  "sorted" gives the neighbours sorted by distance

    Returns:
    -------
    neighbours, dist, phis, vecs_x, vecs_y: {np.arrays}
    """

    # Declarations
    L = len(x)
    matrix_neighbours = np.tile(np.arange(L), (L, 1))  # Declaration matrix of neighbours
    matrix_dist = np.zeros((L, L))                     # Declaration matrix of dists
    matrix_phis = np.zeros((L, L))                     # Declaration matrix of phis
    matrix_vecs_x = np.zeros((L, L))                   # Declaration  matrix of vectors connecting neighbours
    matrix_vecs_y = np.zeros((L, L))                   # Declaration  matrix of vectors connecting neighbours

    # Loop through the different sites of the lattice
    cont = 1
    for site1 in range(0, L):
        for site2 in range(cont, L):
            r, phi = displacement2D(x[site1], y[site1], x[site2], y[site2], L_x, L_y, boundary)
            matrix_dist[site1, site2], matrix_phis[site1, site2], = r, phi
            matrix_dist[site2, site1], matrix_phis[site2, site1], = r, phi + pi
            matrix_vecs_x[site1, site2], matrix_vecs_y[site1, site2] = np.cos(phi), np.sin(phi)
            matrix_vecs_x[site2, site1], matrix_vecs_y[site2, site1] = -np.cos(phi), -np.sin(phi)

    if sorted == "sorted":
        # Sorting
        idx = np.argsort(matrix_dist, axis=1)  # Sorting distances from minimum to maximum
        matrix_neighbours = np.take_along_axis(matrix_neighbours, idx, axis=1)  # Ordered neighbours
        matrix_dist = np.take_along_axis(matrix_dist, idx, axis=1)              # Ordered distances
        matrix_phis = np.take_along_axis(matrix_phis, idx, axis=1)              # Ordered phis
        matrix_vecs_x = np.take_along_axis(matrix_vecs_x, idx, axis=1)          # Ordered vectors connecting neighbours
        matrix_vecs_y = np.take_along_axis(matrix_vecs_y, idx, axis=1)          # Ordered vectors connecting neighbours

        # Delete the first column containing the same site so that we get only info about the neighbours
        neighbours = matrix_neighbours[:, 1:]
        dist = matrix_dist[:, 1:]
        phis = matrix_phis[:, 1:]
        vecs_x, vecs_y = matrix_vecs_x[:, 1:], matrix_vecs_y[:, 1:]

    return neighbours, dist, phis, vecs_x, vecs_y


def No_Boundary_intersection(boundary_line, boundary_points, line_neigh):
    """
    Calculates the intersection between boundary_line and line_neigh without counting the points included
    in boundary_points

    Params:
    ------
    boundary_line:    {shapely.geometry.LineString}   Boundary
    boundary_points:  {shapely.geometry.MultiPoints}  Vertices of the boundary line
    line_neigh:       {shapely.geometry.LineString}   Line connecting the two neighbours in question

    Returns:
    -------
    True or False
    """

    inters_point = line_neigh.intersection(boundary_line)

    return boundary_points.contains(inters_point)


def physical_boundary(x, y, neighbours, dist, vecs_x, vecs_y, neighbour_cutoff):
    """
    Calculates one of the possible allowed boundaries of an amorphous point set without self intersections.
    It enforces that every point is at least connected to the two nearest neighbours, as we require in the
    Hamiltonian as well. In the event of running into an infinite loop because of a patological configuration
    of the lattice, we discard the calculation but keep track of how may configurations we throw away.

    Params:
    ------
    x, y:                                  {np.array} Position of the sites of the point set
    neighbours, dist, vecs_x, vecs_y:      {np.array} Row=site, colums sorted by distance to site
    neighbour_cutoff:                         {float} Cutoff scale for the neighbour couplings

    Returns:
    ------
    line_boundary:     {shapely.geometry.LineString} Line containing the boundary of the lattice
    sites_boundary:                           {list} List of sites at the boundary
    area:                                     {float} Area enclosed by the boundary
    loop                                        {int} Counter of thrown-away configurations
    """

    # Definitions
    site0, vec0, count, loop = None, np.array([1, 0]), 0, 0
    new_point = []
    pts_boundary = []
    sites_boundary = []
    modified_dist = dist

    # Adding neighbours for the points that have less than 2
    for j in range(len(x)):

        # Neighbours
        ind = np.where(dist[j, :] < neighbour_cutoff)[0]

        # Case of no neighbours
        if len(ind) == 0:
            neigh0, neigh1 = neighbours[j, 0], neighbours[j, 1]  # First two neighbours
            ind1 = np.where(neighbours[neigh0, :] == j)[0]       # Index for j as a neighbour for neigh0
            ind2 = np.where(neighbours[neigh1, :] == j)[0]       # Index for j as a neighbour for neigh1

            # Modified distances
            modified_dist[j, 0] = neighbour_cutoff - 0.1
            modified_dist[j, 1] = neighbour_cutoff - 0.1
            modified_dist[neigh0, ind1] = neighbour_cutoff - 0.1
            modified_dist[neigh1, ind2] = neighbour_cutoff - 0.1

        # Case of 1 neighbour
        elif len(ind) == 1:
            neigh1 = neighbours[j, 1]                           # First two neighbours
            ind1 = np.where(neighbours[neigh1, :] == j)[0]      # Index for j as a neighbour for neigh1

            # Modified distances
            modified_dist[j, 1] = neighbour_cutoff - 0.1
            modified_dist[neigh1, ind1] = neighbour_cutoff - 0.1

    # Choosing initial site
    site_left = np.where(x == min(x))[0][0]

    # Algorithm to find the boundary
    while site0 != site_left:
        dif = 2 * pi - 0.01
        return_needed = 1

        # Initial point
        if count == 0:
            site0 = site_left
            new_point = Point(x[site0], y[site0])
            pts_boundary = [new_point]
            multipts_boundary = MultiPoint(pts_boundary)
            sites_boundary.append(site0)

        # Neighbours admissible for the boundary
        ind1 = np.where(modified_dist[site0, :] < neighbour_cutoff)[0]
        list_neigh = neighbours[site0, ind1]
        vec_neigh_x, vec_neigh_y = vecs_x[site0, ind1], vecs_y[site0, ind1]

        # Choosing neighbour that minimises the angle with vec0 and does not intersect the boundary
        for j, neigh in enumerate(list_neigh):

            # Select possible neighbour
            neigh_point = Point(x[neigh], y[neigh])
            line_neigh = LineString([new_point, neigh_point])
            vec1 = np.array([vec_neigh_x[j], vec_neigh_y[j]])
            ang = angle(vec1, -vec0)

            # First point
            if count == 0:
                if ang < dif:
                    dif = ang
                    vec2 = vec1
                    site0 = neigh
                    return_needed = 0

            # Case we go back to the initial neighbour when we can close the boundary
            elif count > 1 and sites_boundary[-1] == sites_boundary[1] and neigh != sites_boundary[-2]:
                if ang < dif:
                    dif = ang
                    vec2 = vec1
                    site0 = neigh
                    return_needed = 0

            # Next points
            else:
                if ang < dif and No_Boundary_intersection(line_boundary, multipts_boundary, line_neigh) is True:
                    dif = ang
                    vec2 = vec1
                    site0 = neigh
                    return_needed = 0

        # In case returning back is the only option we go back to the previous point
        if return_needed == 1:
            vec_x = x[sites_boundary[-2]] - x[sites_boundary[-3]]
            vec_y = y[sites_boundary[-2]] - y[sites_boundary[-3]]
            vec2 = np.array([vec_x, vec_y])
            site0 = sites_boundary[-2]
            index = np.where(neighbours[site0, :] == sites_boundary[-1])[0]
            modified_dist[site0, index] = neighbour_cutoff + 0.1

        # New point at the boundary
        vec0 = vec2
        new_point = Point(x[site0], y[site0])
        pts_boundary.append(new_point)
        multipts_boundary = MultiPoint(pts_boundary)
        line_boundary = LineString(pts_boundary)
        sites_boundary.append(site0)
        count = count + 1

        # In case we run into an infinite loop
        if len(sites_boundary) > len(x):
            loop = 1
            break

    area = Polygon(pts_boundary).area

    return line_boundary, sites_boundary, area, loop


def lattice_graph(L_x, L_y, n_sites, x, y, neighbour_cutoff, neighbours, dist, pts_boundary=None):
    """
    Plots the lattice graph

    Params:
    ------
    L_x, L_y:                 {int} Dimensions of the RPS grid
    n_sites:                  {int} Number of sites in the RPS
    x:                   {np.array} x position of the sites in the RPS
    y:                   {np.array} y position of the sites in the RPS
    neighbour_cutoff:       {float} Distance cut-off for the hopping amplitudes
    neighbours, dist:    {np.array} Matrices with neighbours and distances on each colum sorted by distance
    pts_boundary:            {list} (Optional) List of sites at the boundary

    Returns:
    -------
    None, but plots the graph
    """

    modified_dist = dist

    # Adding neighbours for the points that have less than 2
    for j in range(len(x)):

        # Neighbours
        ind = np.where(dist[j, :] < neighbour_cutoff)[0]

        # Case of no neighbours
        if len(ind) == 0:
            neigh0, neigh1 = neighbours[j, 0], neighbours[j, 1]  # Firs two neighbours
            ind1 = np.where(neighbours[neigh0, :] == j)[0]  # Index for j as a neighbour for neigh0
            ind2 = np.where(neighbours[neigh1, :] == j)[0]  # Index for j as a neighbour for neigh1

            # Modified distances
            modified_dist[j, 0] = neighbour_cutoff - 0.1
            modified_dist[j, 1] = neighbour_cutoff - 0.1
            modified_dist[neigh0, ind1] = neighbour_cutoff - 0.1
            modified_dist[neigh1, ind2] = neighbour_cutoff - 0.1

        # Case of 1 neighbour
        elif len(ind) == 1:
            neigh1 = neighbours[j, 1]  # Firs two neighbours
            ind1 = np.where(neighbours[neigh1, :] == j)[0]  # Index for j as a neighbour for neigh1

            # Modified distances
            modified_dist[j, 1] = neighbour_cutoff - 0.1
            modified_dist[neigh1, ind1] = neighbour_cutoff - 0.1

    # Loop through the different sites of the lattice
    for site1 in range(0, n_sites):

        # Label each site
        plt.text(x[site1] + 0.1, y[site1] + 0.1, str(site1))

        # Neighbours associated to this site, enforcing at least two neighbours
        ind1 = np.where(modified_dist[site1, :] < neighbour_cutoff)[0]
        list_neigh = neighbours[site1, ind1]

        # Coupling with the neighbours
        for j, site2 in enumerate(list_neigh):
            plt.plot([x[site1], x[site2]], [y[site1], y[site2]], 'tab:gray', linewidth=1, alpha=0.3)

    # Plot the boundaary if it exists
    if pts_boundary is not None:
        for j in range(0, len(pts_boundary)):

            if j == len(pts_boundary) - 1:
                site1 = pts_boundary[j]
                site2 = pts_boundary[0]
            else:
                site1 = pts_boundary[j]
                site2 = pts_boundary[j + 1]

            plt.plot([x[site1], x[site2]], [y[site1], y[site2]], 'tab:Green', linewidth=1, alpha=1)

    # PLot the sites
    plt.scatter(x, y)

    plt.xlim(-1, L_x + 1)
    plt.ylim(-1, L_y + 1)
    plt.xticks(color="w")
    plt.yticks(color="w")
    plt.show()


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