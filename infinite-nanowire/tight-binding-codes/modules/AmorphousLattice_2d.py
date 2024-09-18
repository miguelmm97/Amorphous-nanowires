#%% modules setup

# Math and plotting
from numpy import pi
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, MultiPoint, Polygon
from shapely import intersects
from scipy.integrate import quad


# Managing classes
from dataclasses import dataclass, field

# Tracking time
import time

# Managing logging
import logging
import colorlog
from colorlog import ColoredFormatter

#%% Logging setup
loger_amorphous = logging.getLogger('amorphous')
loger_amorphous.setLevel(logging.INFO)

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
loger_amorphous.addHandler(stream_handler)

#%% Module
"""
We set up a class to model an amorphous lattice in 2d, with proper connectivity and excluding edge cases
that are not relevant for nanowire physics.
"""

# Functions for creating the lattice
def gaussian_point_set_2D(x, y, width):
    x = np.random.normal(x, width, len(x))
    y = np.random.normal(y, width, len(y))
    return x, y

def random_point_set_2D(x, y, d):
    x = x + d * (2 * np.random.rand(x.shape[0]) - 1)
    y = y + d * (2 * np.random.rand(y.shape[0]) - 1)
    return x, y

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

def angle_x(v):

    if v[0] == 0:                # Pathological case, separated to not divide by 0
        if v[1] > 0:
            phi = pi / 2         # Hopping in y
        else:
            phi = 3 * pi / 2     # Hopping in -y
    else:
        if v[1] > 0:
            phi = np.arctan2(v[1], v[0])           # 1st and 2nd quadrants
        else:
            phi = 2 * pi + np.arctan2(v[1], v[0])  # 3rd and 4th quadrants

    return phi

def angle(v, ax):

    alpha = - angle_x(ax)
    R = np.array([[np.cos(alpha), -np.sin(alpha)],
                  [np.sin(alpha), np.cos(alpha)]])
    phi = angle_x(R @ v)
    return phi


# Functions for calculating the boundary
def no_intersection(boundary_line, boundary_points, line_neigh):
    inters_point = line_neigh.intersection(boundary_line)
    return boundary_points.contains(inters_point)

def remove_site(list, site):
    try:
        return list.remove(site)
    except ValueError:
        return list


@dataclass
class AmorphousLattice_2d:
    """ Infinite amorphous cross-section nanowire based on the crystalline Fu and Berg model"""

    # Lattice parameters
    Nx: int   # Number of lattice sites along x direction
    Ny: int   # Number of lattice sites along y direction
    w: float  # Width of the Gaussian distribution
    r: float  # Cutoff distance to consider neighbours

    # Class fields
    Nsites: int = field(init=False)             # Number of sites in the cross-section
    neighbours: np.ndarray = field(init=False)  # Neighbours list for each site
    boundary: list = field(init=False)          # List of sites forming the boundary
    area: float = field(init=False)             # Area of the wire's cross-section
    x: np.ndarray = field(init=False)           # x position of the sites
    y: np.ndarray = field(init=False)           # y position of the sites


    # Methods for building the lattice
    def generate_configuration(self, from_x=None, from_y=None):

        loger_amorphous.info('Generating lattice and neighbour tree...')

        # Dimension of the Hilbert space and sites
        self.Nsites = int(self.Nx * self.Ny)
        list_sites = np.arange(0, self.Nsites)
        x_crystal = list_sites % self.Nx
        y_crystal = list_sites // self.Nx

        # Positions of x and y coordinates on the amorphous lattice
        if from_x is not None and from_y is not None:
            self.x, self.y = from_x, from_y
        else:
            self.x, self.y = gaussian_point_set_2D(x_crystal, y_crystal, self.w)
        coords = np.array([self.x, self.y])

        # Neighbour tree and accepting/discarding the configuration
        self.neighbours = KDTree(coords.T).query_ball_point(coords.T, self.r)
        for i in range(self.Nsites):
            self.neighbours[i].remove(i)
            if len(self.neighbours[i]) < 2:
                raise ValueError('Connectivity of the lattice too low. Trying a different configuration...')

    def build_lattice(self, from_x=None, from_y=None, n_tries=0):

        if n_tries > 100:
            loger_amorphous.error('Loop. Parameters might not allow an acceptable configuration.')

        try:
            self.generate_configuration(from_x=from_x, from_y=from_y)
            self.get_boundary()
            loger_amorphous.info('Configuration accepted!')
        except Exception as error:
            loger_amorphous.warning(f'{error}')
            try:
                self.build_lattice(from_x=None, from_y=None, n_tries=n_tries + 1)
            except RecursionError:
                loger_amorphous.error('Recursion error. Infinite loop. Terminating...')
                exit()

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
        loger_amorphous.info('Constructing the boundary of the lattice...')
        while site0 != start_site or (site0 == start_site and count == 0):
            dif, return_needed = 2 * pi - 0.01, True
            current_x, current_y = self.x[site0], self.y[site0]
            current_point = Point(current_x, current_y)

            # Scan for the neighbour at the boundary
            list_neighbours = remove_site(self.neighbours[site0], avoid_site)
            if list_neighbours is None:
                raise TypeError('Boundary too complicated to handle. Trying a different configuration...')

            for n in list_neighbours:
                line_neigh = LineString([current_point, Point(self.x[n], self.y[n])])
                r, phi = displacement2D(current_x, current_y, self.x[n], self.y[n])
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
            loger_amorphous.trace(f'Boundary given by: {self.boundary}')

        loger_amorphous.trace(f'Boundary given by: {self.boundary}')
        self.area = Polygon(boundary_points).area
        if self.area < 0.25 * (self.Nx * self.Ny):
            loger_amorphous.warning(
                'Area too small, this configuration might have an isolated island. Plotting for a check...')
            fig1 = plt.figure(figsize=(6, 6))
            ax1 = fig1.gca()
            self.plot_lattice(ax1)
            plt.show()

    def plot_lattice(self, ax):

        # Neighbour links
        for site in range(self.Nsites):
            for n in self.neighbours[site]:
                plt.plot([self.x[site], self.x[n]], [self.y[site], self.y[n]], 'royalblue', linewidth=1, alpha=0.2)
                plt.text(self.x[n] + 0.1, self.y[n] + 0.1, str(n))

        # Boundary
        try:
            for j in range(0, len(self.boundary)):
                if j == len(self.boundary) - 1:
                    site1, site2 = self.boundary[j], self.boundary[0]
                else:
                    site1, site2 = self.boundary[j], self.boundary[j + 1]
                plt.plot([self.x[site1], self.x[site2]], [self.y[site1], self.y[site2]], 'm', linewidth=2, alpha=1)
        except AttributeError:
            loger_amorphous.warning('Boundary has not been calculated before plotting')
            pass

        # Lattice sites
        ax.scatter(self.x, self.y, color='deepskyblue', s=50)