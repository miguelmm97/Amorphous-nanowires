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
We set up a class to model an amorphous lattice in 3d, with proper connectivity and excluding edge cases
that are not relevant for nanowire physics.
"""

# Functions for creating the lattice
def gaussian_point_set_3D(x, y, z, width):
    x = np.random.normal(x, width, len(x))
    y = np.random.normal(y, width, len(y))
    z = np.random.normal(z, width, len(z))
    return x, y, z

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

def displacement(x1, y1, z1, x2, y2, z2, L_x, L_y, L_z):
    # Definition of the vector between sites 2 and 1 (from st.1 to st.2)
    v = np.zeros((3,))
    v[0] = (x2 - x1)
    v[1] = (y2 - y1)
    v[2] = (z2 - z1)
    r = np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)

    # Phi angle of the vector between sites 2 and 1 (angle in the XY plane)
    if v[0] == 0:
        if v[1] > 0:  # Pathological case, separated to not divide by 0
            phi = pi / 2
        else:
            phi = 3 * pi / 2
    else:
        if v[1] > 0:  # We take arctan2 because we have 4 quadrants
            phi = np.arctan2(v[1], v[0])  # 1st and 2nd quadrants
        else:
            phi = 2 * pi + np.arctan2(v[1], v[0])  # 3rd and 4th quadrants

    # Theta angle of the vector between sites 2 and 1 (angle from z)
    r_plane = np.sqrt(v[0] ** 2 + v[1] ** 2)  # Auxiliary radius for the xy plane

    if r_plane == 0:  # Pathological case, separated to not divide by 0
        if v[2] > 0:  # Hopping in z
            theta = 0
        elif v[2] < 0:  # Hopping in -z
            theta = pi
        else:
            theta = pi / 2  # XY planes
    else:
        theta = pi / 2 - np.arctan(v[2] / r_plane)  # 1st and 2nd quadrants

    return r, phi, theta

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


# Functions for generating instances of the amorphous class
def take_cut_from_parent_wire(parent, Nx_new=None, Ny_new=None, Nz_new=None, keep_disorder=True):

    # Sites from the new and parent lattice
    Nx, Ny, Nz = parent.Nx, parent.Ny, parent.Nz
    if Nx_new is None:
        Nx_new = parent.Nx
    if Ny_new is None:
        Ny_new = parent.Ny
    if Nz_new is None:
        Nz_new = parent.Nz

    # Selecting sites from the parent lattice
    Nsites1, Nxy = int(Nx * Ny * Nz_new), int(Nx * Ny)
    x1, y1, z1 = parent.x[:Nsites1], parent.y[:Nsites1], parent.z[:Nsites1]
    x2 = np.concatenate([x1[i * Nxy: i * Nxy + int(Nx * Ny_new)] for i in range(Nz)])
    y2 = np.concatenate([y1[i * Nxy: i * Nxy + int(Nx * Ny_new)] for i in range(Nz)])
    z2 = np.concatenate([z1[i * Nxy: i * Nxy + int(Nx * Ny_new)] for i in range(Nz)])
    x = np.concatenate([x2[i * Nx: i * Nx + Nx_new] for i in range(int(Ny_new * Nz_new))])
    y = np.concatenate([y2[i * Nx: i * Nx + Nx_new] for i in range(int(Ny_new * Nz_new))])
    z = np.concatenate([z2[i * Nx: i * Nx + Nx_new] for i in range(int(Ny_new * Nz_new))])

    # Generate child lattice
    lattice = AmorphousLattice_3d(Nx=Nx_new, Ny=Ny_new, Nz=Nz_new, w=parent.w, r=parent.r)
    lattice.set_configuration(x, y, z)
    lattice.build_lattice()
    if keep_disorder:
        lattice.K_hopp, lattice.K_onsite = parent.K_hopp, parent.K_onsite
        lattice.set_disorder(onsite_disorder=parent.onsite_disorder, disorder=parent.disorder)
    return lattice

@dataclass
class AmorphousLattice_3d:
    """ Infinite amorphous cross-section nanowire based on the crystalline Fu and Berg model"""

    # Class fields set upon instantiation
    Nx:  int                                          # Number of lattice sites along x direction
    Ny:  int                                          # Number of lattice sites along y direction
    Nz:  int                                          # Number of lattice sites along y direction
    w:   float                                        # Width of the Gaussian distribution
    r:   float                                        # Cutoff distance to consider neighbours

    # Class fields that can be set externally
    x: np.ndarray         = None                      # x position of the sites
    y: np.ndarray         = None                      # y position of the sites
    z: np.ndarray         = None                      # z position of the sites
    K_onsite: float       = None                      # Strength of the onsite disorder distribution
    K_hopp:   float       = None                      # Strength of the hopping disorder distribution
    disorder: np.ndarray  = None                      # Disorder matrix
    onsite_disorder: np.ndarray = None                # Disorder array for the only onsite case (much more efficient)

    # Class fields that can only be set internally
    Nsites: int = field(init=False)                         # Number of sites in the cross-section
    neighbours: np.ndarray = field(init=False)              # Neighbours list for each site
    neighbours_projection: np.ndarray = field(init=False)   # Neighbours list for each site on the 2d projection
    boundary: list = field(init=False)                      # List of sites forming the boundary
    area: float = field(init=False)                         # Area of the wire's cross-section


    # Methods for building the lattice

    def build_lattice(self, n_tries=0, restrict_connectivity=False):

        if n_tries > 100:
            loger_amorphous.error('Loop. Parameters might not allow an acceptable configuration.')
        if self.w  < 1e-10:
            loger_amorphous.error('The amorphicity cannot be strictly 0')
            exit()

        # Restricting to only connected lattice configurations
        if restrict_connectivity:
            try:
                self.generate_configuration(restrict_connectivity=True)
                self.get_boundary()
                loger_amorphous.trace('Configuration accepted!')
            except Exception as error:
                loger_amorphous.warning(f'{error}')
                try:
                    self.erase_configuration()
                    self.erase_disorder()
                    self.build_lattice(n_tries=n_tries + 1, restrict_connectivity=True)
                except RecursionError:
                    loger_amorphous.error('Recursion error. Infinite loop. Terminating...')
                    exit()
        else:
            # Accepting literally anything
            self.generate_configuration()
            self.area = (self.Nx - 1) * (self.Ny - 1)

    def generate_configuration(self, restrict_connectivity=False):
        loger_amorphous.trace('Generating lattice and neighbour tree...')

        # Positions of x and y coordinates on the amorphous lattice
        self.Nsites = int(self.Nx * self.Ny * self.Nz)
        if self.x is None and self.y is None and self.z is None:
            list_sites = np.arange(0, self.Nsites)
            x_crystal, y_crystal, z_crystal = list_sites % self.Nx, (list_sites // self.Nx) % self.Ny, list_sites // (self.Nx * self.Ny)
            self.x, self.y, self.z = gaussian_point_set_3D(x_crystal, y_crystal, z_crystal, self.w)
        coords = np.array([self.x, self.y, self.z])

        # Neighbour tree and accepting/discarding the configuration
        self.neighbours = KDTree(coords.T).query_ball_point(coords.T, self.r)
        for i in range(self.Nsites):
            self.neighbours[i].remove(i)
            if restrict_connectivity and len(self.neighbours[i]) < 2:
                raise ValueError('Connectivity of the lattice too low. Trying a different configuration...')

        self.K_onsite, self.K_hopp = 0., 0.

    def generate_onsite_disorder(self, K_onsite):
        loger_amorphous.trace('Generating disorder configuration...')
        self.K_onsite = K_onsite
        self.onsite_disorder = np.random.uniform(-self.K_onsite, self.K_onsite, self.Nsites)

    def generate_full_disorder(self, K_onsite=0., K_hopp=0.):

        loger_amorphous.trace('Generating disorder configuration...')
        self.K_onsite, self.K_hopp = K_onsite, K_hopp

        # Generate a matrix with diagonal onsite disorder and symmetric (hermitian) hopping disorder
        aux_diag = np.random.uniform(-self.K_onsite, self.K_onsite, self.Nsites)
        aux_matrix = np.random.uniform(-self.K_hopp, self.K_hopp, (self.Nsites, self.Nsites))
        disorder_matrix = np.tril(aux_matrix, k=-1)
        disorder_matrix = disorder_matrix + disorder_matrix.T
        disorder_matrix = disorder_matrix + np.diag(aux_diag)
        self.disorder = disorder_matrix
        self.onsite_disorder = aux_diag

    def get_boundary(self):

        # Collapse of the full lattice into a 2d plane
        coords = np.array([self.x, self.y])
        self.neighbours_projection = KDTree(coords.T).query_ball_point(coords.T, self.r)
        for i in range(self.Nsites):
            self.neighbours_projection[i].remove(i)
            if len(self.neighbours_projection[i]) < 2:
                raise ValueError('Connectivity of the lattice too low. Trying a different configuration...')

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
        loger_amorphous.trace('Constructing the boundary of the lattice...')
        while site0 != start_site or (site0 == start_site and count == 0):
            dif, return_needed = 2 * pi - 0.01, True
            current_x, current_y = self.x[site0], self.y[site0]
            current_point = Point(current_x, current_y)

            # Scan for the neighbour at the boundary
            list_neighbours = remove_site(self.neighbours_projection[site0], avoid_site)
            if list_neighbours is None:
                raise TypeError('2d boundary too complicated to handle. Trying a different configuration...')

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

    def plot_lattice(self, ax, numbering=False):

        # Neighbour links
        for site in range(self.Nsites):
            for n in self.neighbours[site]:
                ax.plot([self.x[site], self.x[n]], [self.y[site], self.y[n]], [self.z[site], self.z[n]], 'royalblue', linewidth=1, alpha=0.2)
                if numbering:
                    ax.text(self.x[n] + 0.1, self.y[n] + 0.1, self.z[n] + 0.1, str(n))

        # Lattice sites
        ax.scatter(self.x, self.y, self.z, color='deepskyblue', s=50)
        ax.set_axis_off()

    def plot_lattice_projection(self, ax, numbering=False):

        # Neighbour links
        for site in range(self.Nsites):
            for n in self.neighbours_projection[site]:
                ax.plot([self.x[site], self.x[n]], [self.y[site], self.y[n]], 'royalblue', linewidth=1, alpha=0.2)
                if numbering:
                    ax.text(self.x[n] + 0.1, self.y[n] + 0.1, str(n))

        # Boundary
        try:
            for j in range(0, len(self.boundary)):
                if j == len(self.boundary) - 1:
                    site1, site2 = self.boundary[j], self.boundary[0]
                else:
                    site1, site2 = self.boundary[j], self.boundary[j + 1]
                ax.plot([self.x[site1], self.x[site2]], [self.y[site1], self.y[site2]], 'm', linewidth=2, alpha=1)
        except AttributeError:
            loger_amorphous.warning('Boundary has not been calculated before plotting')
            pass

        # Lattice sites
        ax.scatter(self.x, self.y, color='deepskyblue', s=50)
        ax.set_axis_off()


    # Setters and erasers
    def set_configuration(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    def set_disorder(self, onsite_disorder, disorder):
        self.disorder = disorder
        self.onsite_disorder = onsite_disorder

    def erase_configuration(self):
        self.x, self.y, self.z = None, None, None

    def erase_disorder(self):
        self.disorder= None


     # Alternative way of getting the boundary
    def get_boundary2(self):

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
        loger_amorphous.trace('Constructing the boundary of the lattice...')
        while site0 != start_site or (site0 == start_site and count == 0):
            dif, return_needed = 2 * pi - 0.01, True
            current_x, current_y = self.x[site0], self.y[site0]
            current_point = Point(current_x, current_y)

            # Scan for the neighbour at the boundary
            list_neighbours = remove_site(self.neighbours[site0], avoid_site)
            if list_neighbours is None:
                raise TypeError('2d boundary too complicated to handle. Trying a different configuration...')

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

    def plot_lattice_projection2(self, ax):

        # Neighbour links
        for site in range(self.Nsites):
            for n in self.neighbours[site]:
                ax.plot([self.x[site], self.x[n]], [self.y[site], self.y[n]], 'royalblue', linewidth=1, alpha=0.2)
                ax.text(self.x[n] + 0.1, self.y[n] + 0.1, str(n))

        # Boundary
        try:
            for j in range(0, len(self.boundary)):
                if j == len(self.boundary) - 1:
                    site1, site2 = self.boundary[j], self.boundary[0]
                else:
                    site1, site2 = self.boundary[j], self.boundary[j + 1]
                ax.plot([self.x[site1], self.x[site2]], [self.y[site1], self.y[site2]], 'm', linewidth=2, alpha=1)
        except AttributeError:
            loger_amorphous.warning('Boundary has not been calculated before plotting')
            pass

        # Lattice sites
        ax.scatter(self.x, self.y, color='deepskyblue', s=50)
        ax.set_axis_off()