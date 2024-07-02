#%% Modules setup

# Math and plotting
from numpy import pi
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, MultiPoint, Polygon
from shapely import intersects

# Managing classes
from dataclasses import dataclass, field

# Tracking time
import time

# Managing logging
# import logging
# import colorlog
# from colorlog import ColoredFormatter

#%% Module
def gaussian_point_set_2D(x, y, width):
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

def random_point_set_2D(x, y, d):
    x = x + d * (2 * np.random.rand(x.shape[0]) - 1)
    y = y + d * (2 * np.random.rand(y.shape[0]) - 1)
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

@dataclass
class InfiniteNanowire_FuBerg:
    """ Infinite amorphous cross-section nanowire based on the crystalline Fu and Berg model"""

    # Lattice parameters
    Nx: int           # Number of lattice sites along x direction
    Ny: int           # Number of lattice sites along y direction
    w:  float         # Width of the Gaussian distribution
    r:  float         # Cutoff distance to consider neighbours

    # Electronic parameters
    eps:    float     # Onsite energy coupling different orbitals
    t:      float     # Isotropic hopping strength
    lamb:   float     # Spin orbit coupling in the cross-section
    lamb_z: float     # Spin orbit coupling along z

    # Class fields
    dimH:          int = field(init=False)         # Dimension of the single-particle Hilbert Space
    Nsites:        int = field(init=False)         # Number of sites in the cross-section
    sites:         KDTree = field(init=False)      # Tree of sites
    neighbours:    KDTree = field(init=False)      # Neighbour list for each site
    H:             np.ndarray = field(init=False)  # Hamiltonian
    x:             np.ndarray = field(init=False)  # x position of the sites
    y:             np.ndarray = field(init=False)  # y position of the sites


    # Methods
    def build_lattice(self):
        self.Nsites = int(self.Nx * self.Ny)
        list_sites = np.arange(0, self.Nsites)
        # self.x, self.y = random_point_set_2D(list_sites % self.Nx, list_sites // self.Nx, self.d)
        self.x, self.y = gaussian_point_set_2D(list_sites % self.Nx, list_sites // self.Nx, self.w)
        self.sites = KDTree(np.array([self.x, self.y]).T)
        self.neighbours = self.sites.query_ball_point(np.array([self.x, self.y]).T, self.r, return_sorted=True)

    def get_boundary(self):

        # Initial parameters of the algorithm
        vec0 = np.array([1, 0])
        count, loop, avoid_site = 0, 0, None

        # Initial site of the boundary
        site0 = np.where(self.x == min(self.x))[0][0]
        boundary_points = [Point(self.x[site0], self.y[site0])]
        multipts_boundary = MultiPoint(boundary_points)
        boundary_line = LineString(boundary_points)
        boundary.append(site0)
        start_site = site0

        # Algorithm to find the boundary
        while site0 != start_site and count > 0:
            dif, eps, return_needed = 2 * pi - 0.01, 0.1, True

            # Add neighbours if there are none
            while len(self.neighbours[site0].remove(avoid_site)) < 2:
                self.neighbours[site0] = self.sites.query_ball_point(np.array(self.x[site0], self.y[site0]), self.r + eps,
                                                                     return_sorted=True)
                eps += 0.1

            # Scan for the neighbour at the boundary
            for n in self.neighbours[site0].remove(avoid_site):
                line_neigh = LineString([Point(self.x[site0], self.y[site0]), Point(x[n], y[n])])
                r, phi = displacement2D(x[site0], y[site0], x[n], y[n], L_x, L_y, boundary='Open')
                vec1 = np.array([np.cos(phi), np.sin(phi)])
                ang = angle(vec1, -vec0)
                if ang < dif and No_Boundary_intersection(boundary_line, multipts_boundary, line_neigh):
                    dif, vec2, site0, return_needed = ang, vec1, n, Fals

            if return_needed:
                vec2 = np.array([self.x[boundary[-2]] - self.x[boundary[-3]], self.y[boundary[-2]] - self.y[boundary[-3]]])
                avoid_site = site0
                site0 = boundary[-2]

            # New point at the boundary
            boundary_points.append(Point(x[site0], y[site0]))
            multipts_boundary = MultiPoint(pts_boundary)
            boundary_line = LineString(boundary_points)
            boundary.append(site0)
            count = count + 1
            vec0 = vec2

            # In case we run into an infinite loop
            if len(sites_boundary) > len(x):
                loop = 1
                break

        return boundary, Polygon(boundary_points).area

    def plot_lattice(self, ax):
        ax.scatter(self.x, self.y)
        for site in np.arange(0, self.Nsites):
            for n in self.neighbours[site]:
                plt.plot([self.x[site], self.x[n]], [self.y[site], self.y[n]], 'tab:Green', linewidth=1, alpha=1)
                plt.text(self.x[n] + 0.1, self.y[n] + 0.1, str(n))


wire = InfiniteNanowire_FuBerg(Nx=6, Ny=6, w=0.3, r=1.4, eps=0.1, t=0., lamb=0., lamb_z=0.)
wire.build_lattice()
boundary, area = wire.get_boundary()
fig1 = plt.figure(figsize=(8, 8))
wire.plot_lattice(fig1.gca())
plt.show()




# # First point or closing point
# if count == 0 or (count > 1 and boundary[-1] == boundary[1] and n != boundary[-2]):
#     if ang < dif:
#         dif, vec2, site0, return_needed = ang, vec1, n, Fal         # Next points
# elif ang < dif and No_Boundary_intersection(boundary_line, multipts_boundary, line_neigh):
#     dif, vec2, site0, return_needed = ang, vec1, n, False


# def H_hoppx(self):
#
#
# def get_Hamiltonian(self):
#     self.dimH = self.Nx * self.Ny * 4
#     self.H = np.zeros((self.dimH, self.dimH), dtype=np.complex128)
#     for i in sites.keys():
#
#         # States for the i-th site and nearest neighbours
#         psi_i = np.zeros((self.dimH, ), dtype=np.complex128)
#         psi_i_x = np.zeros((self.dimH, ), dtype=np.complex128)
#         psi_i_y = np.zeros((self.dimH,), dtype=np.complex128)
#         psi_i[i] = 1.
#         psi_i_x[i + 1] = 1. if sites[i]['x'] != (self.Nx - 1) else 0.
#         psi_i_y[i + Nx] = 1. if sites[i]['y'] != (self.Ny - 1) else 0.
#
#         # Hopping operators
#         onsite_block = np.outer(psi_i.conj(), psi_i)
#         x_block = np.outer(psi_i.conj(), psi_i_x)
#         y_block = np.outer(psi_i.conj(), psi_i_y)
#
#         # Hamiltonian
#         self.H += np.kron(H_onsite, onsite_block)
#         self.H += np.kron(H_hoppx, x_block)
#         self.H += np.kron(H_hoppy, y_block)
