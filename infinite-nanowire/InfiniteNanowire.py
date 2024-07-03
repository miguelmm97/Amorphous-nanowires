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

    x = np.random.normal(x, width, len(x))
    y = np.random.normal(y, width, len(y))
    return x, y

def random_point_set_2D(x, y, d):
    x = x + d * (2 * np.random.rand(x.shape[0]) - 1)
    y = y + d * (2 * np.random.rand(y.shape[0]) - 1)
    return x, y

def displacement2D(x1, y1, x2, y2, L_x, L_y, boundary):

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

def no_intersection(boundary_line, boundary_points, line_neigh):
    inters_point = line_neigh.intersection(boundary_line)
    return boundary_points.contains(inters_point)

def remove_site(list, site):
    try:
        return list.remove(site)
    except ValueError:
        return list


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
    dimH:          int = field(init=False)           # Dimension of the single-particle Hilbert Space
    Nsites:        int = field(init=False)           # Number of sites in the cross-section
    neighbours:    np.ndarray = field(init=False)    # Neighbours list for each site
    boundary:      list = field(init=False)          # List of sites forming the boundary
    area:          float = field(init=False)         # Area of the wire's cross-section
    H:             np.ndarray = field(init=False)    # Hamiltonian
    x:             np.ndarray = field(init=False)    # x position of the sites
    y:             np.ndarray = field(init=False)    # y position of the sites


    # Methods
    def build_lattice(self):
        self.Nsites = int(self.Nx * self.Ny)
        list_sites = np.arange(0, self.Nsites)
        self.x, self.y = gaussian_point_set_2D(list_sites % self.Nx, list_sites // self.Nx, self.w)
        self.neighbours = KDTree(np.array([self.x, self.y]).T).query_ball_point(np.array([self.x, self.y]).T, self.r)
        for i in range(self.Nsites):
            self.neighbours[i].remove(i)
            if len(self.neighbours[i]) < 2:
                raise ValueError('Connectivity of the lattice too low.')

    def get_boundary(self):

        # Initial parameters of the algorithm
        aux_vector = np.array([1, 0])
        count, loop, avoid_site = 0, 0, None

        # Initial site of the boundary
        site0 = np.where(self.x == min(self.x))[0][0]
        boundary_points = [Point(self.x[site0], self.y[site0])]
        multipts_boundary = MultiPoint(boundary_points)
        boundary_line = LineString([])
        self.boundary = [site0]
        start_site = site0

        # Algorithm to find the boundary
        while site0 != start_site or (site0 == start_site and count == 0):
            dif, return_needed = 2 * pi - 0.01, True
            current_x, current_y = self.x[site0], self.y[site0]
            current_point = Point(current_x, current_y)

            # Scan for the neighbour at the boundary
            list_neighbours = remove_site(self.neighbours[site0], avoid_site)
            for n in list_neighbours:
                line_neigh = LineString([current_point, Point(self.x[n], self.y[n])])
                r, phi = displacement2D(current_x, current_y, self.x[n], self.y[n], self.Nx, self.Nx, boundary='Open')
                vector_neigh = np.array([np.cos(phi), np.sin(phi)])
                ang = angle(vector_neigh, -aux_vector)
                if ang < dif:
                    if count == 0:
                        dif, new_aux_vector, site0, return_needed = ang, vector_neigh, n, False
                    elif no_intersection(boundary_line, multipts_boundary, line_neigh):
                        dif, new_aux_vector, site0, return_needed = ang, vector_neigh, n, False

            if return_needed:
                new_aux_vector = np.array([self.x[self.boundary[-2]] - self.x[self.boundary[-3]],
                                 self.y[self.boundary[-2]] - self.y[self.boundary[-3]]])
                avoid_site = site0
                site0 = self.boundary[-2]

            # New point at the boundary
            boundary_points.append(Point(self.x[site0], self.y[site0]))
            multipts_boundary = MultiPoint(boundary_points)
            boundary_line = LineString(boundary_points)
            self.boundary.append(site0)
            count += 1
            aux_vector = new_aux_vector

            # In case we run into an infinite loop
            if len(self.boundary) > self.Nsites:
                raise OverflowError('Algorithm caught in an infinite loop.')

    def plot_lattice(self, ax):

        # Lattice sites
        ax.scatter(self.x, self.y, color='deepskyblue', s=50)

        # Neighbour links
        for site in np.arange(0, self.Nsites):
            for n in self.neighbours[site]:
                plt.plot([self.x[site], self.x[n]], [self.y[site], self.y[n]], 'royalblue', linewidth=1, alpha=0.2)
                plt.text(self.x[n] + 0.1, self.y[n] + 0.1, str(n))

        # Boundary
        if self.boundary is not None:
            for j in range(0, len(self.boundary)):
                if j == len(self.boundary) - 1:
                    site1, site2 = self.boundary[j], self.boundary[0]
                else:
                    site1, site2 = self.boundary[j], self.boundary[j + 1]
                plt.plot([self.x[site1], self.x[site2]], [self.y[site1], self.y[site2]], 'm', linewidth=2, alpha=1)







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
