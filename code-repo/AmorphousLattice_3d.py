"""
This file contains the functions and classes used to create fully amorphous nanowires.

The full repository for the project is public in https://github.com/miguelmm97/Amorphous-nanowires.git
For any questions, typos/errors or further data please write to mfmm@kth.se or miguelmartinezmiquel@gmail.com.
"""

#%% Modules setup

# Math and plotting
import numpy as np
from scipy.spatial import KDTree

# Managing classes
from dataclasses import dataclass, field

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


#%% Functions
def gaussian_point_set_3D(x, y, z, width):
    """
    Input:
    x -> np.ndarray: x positions of the crystalline sites
    y -> np.ndarray: y positions of the crystalline sites
    z -> np.ndarray: z position of the crystalline sites
    width -> float: standard deviation of the Gaussian point set

    Output:
    x -> np.ndarray: x positions of the amorphous nanowire
    y -> np.ndarray: y positions of the amorphous nanowire
    z -> np.ndarray: z positions of the amorphous nanowire
    """
    x = np.random.normal(x, width, len(x))
    y = np.random.normal(y, width, len(y))
    z = np.random.normal(z, width, len(z))
    return x, y, z

def take_cut_from_parent_wire(parent, Nx_new=None, Ny_new=None, Nz_new=None, keep_disorder=True):
    """
    Input:
    parent -> kwant.builder.Builder: parent nanowire from which we want to make a cut
    Nx_new -> int: Number of sites in x direction after the cut
    Ny_new -> int: Number of sites in y direction after the cut
    Nz_new -> int: Number of sites in z direction after the cut
    keep_disorder -> bool: Specify if the cut should retain the disorder realization of the parent nanowire

    Output:
    lattice -> kwant.builder.Builder: resulting kwant system after the cut
    """

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
    x  = np.concatenate([x2[i * Nx: i * Nx + Nx_new] for i in range(int(Ny_new * Nz_new))])
    y  = np.concatenate([y2[i * Nx: i * Nx + Nx_new] for i in range(int(Ny_new * Nz_new))])
    z  = np.concatenate([z2[i * Nx: i * Nx + Nx_new] for i in range(int(Ny_new * Nz_new))])

    # Generate child lattice
    lattice = AmorphousLattice_3d(Nx=Nx_new, Ny=Ny_new, Nz=Nz_new, w=parent.w, r=parent.r)
    lattice.set_configuration(x, y, z)
    lattice.build_lattice()
    if keep_disorder:
        lattice.K_onsite = parent.K_onsite
        lattice.set_disorder(onsite_disorder=parent.onsite_disorder)
    return lattice

@dataclass
class AmorphousLattice_3d:

    # Class fields set upon instantiation
    Nx:  int                                        # Number of lattice sites along x direction
    Ny:  int                                        # Number of lattice sites along y direction
    Nz:  int                                        # Number of lattice sites along y direction
    w:   float                                      # Width of the Gaussian distribution
    r:   float                                      # Cutoff distance to consider neighbours

    # Class fields that can be set externally
    x: np.ndarray   = None                          # x position of the sites
    y: np.ndarray   = None                          # y position of the sites
    z: np.ndarray   = None                          # z position of the sites
    K_onsite: float = None                          # Strength of the onsite disorder distribution
    onsite_disorder: np.ndarray = None              # Disorder array for only the onsite case

    # Class fields that can only be set internally
    Nsites: int = field(init=False)                 # Number of sites in the cross-section
    neighbours: np.ndarray = field(init=False)      # Neighbours list for each site
    area: float = field(init=False)                 # Area of the wire's cross-section


    # Methods for building the lattice
    def build_lattice(self):
        if self.w  < 1e-10:
            loger_amorphous.error('The amorphicity cannot be strictly 0')
            exit()
        self.generate_configuration()
        self.area = (self.Nx - 1) * (self.Ny - 1)

    def generate_configuration(self):
        loger_amorphous.trace('Generating lattice and neighbour tree...')

        # Positions of x and y coordinates on the amorphous lattice
        self.Nsites = int(self.Nx * self.Ny * self.Nz)
        if self.x is None and self.y is None and self.z is None:
            list_sites = np.arange(0, self.Nsites)
            x_crystal = list_sites % self.Nx
            y_crystal = (list_sites // self.Nx) % self.Ny
            z_crystal = list_sites // (self.Nx * self.Ny)
            self.x, self.y, self.z = gaussian_point_set_3D(x_crystal, y_crystal, z_crystal, self.w)
        coords = np.array([self.x, self.y, self.z])

        # Neighbour tree and accepting/discarding the configuration
        self.neighbours = KDTree(coords.T).query_ball_point(coords.T, self.r)
        for i in range(self.Nsites):
            self.neighbours[i].remove(i)

        # Set up preliminary disorder
        self.K_onsite = 0.

    def generate_onsite_disorder(self, K_onsite):
        loger_amorphous.trace('Generating disorder configuration...')
        self.K_onsite = K_onsite
        self.onsite_disorder = np.random.uniform(-self.K_onsite, self.K_onsite, self.Nsites)


    # Setters and erasers
    def set_configuration(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    def set_disorder(self, onsite_disorder):
        self.onsite_disorder = onsite_disorder

    def erase_configuration(self):
        self.x, self.y, self.z = None, None, None

    def erase_disorder(self):
        self.onsite_disorder= None

