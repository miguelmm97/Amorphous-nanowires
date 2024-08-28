#%% Modules setup

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

# Kwant
import kwant
import tinyarray as ta

# Managing logging
import logging
import colorlog
from colorlog import ColoredFormatter


#%% Logging setup
loger_kwant = logging.getLogger('kwant')
loger_kwant.setLevel(logging.WARNING)

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



#%% Module
"""
Here we promote the infinite amorphous nanowire defined in class InfiniteNanowire.py 
into a kwant.system where to do transport calculations.
"""

sigma_0 = np.eye(2, dtype=np.complex128)
sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
tau_0, tau_x, tau_y, tau_z  = sigma_0, sigma_x, sigma_y, sigma_z


# Amorphous site family class from https://zenodo.org/records/4382484
class AmorphousWire_kwant(kwant.builder.SiteFamily):
    def __init__(self, norbs, coords, name=None):

        if norbs is not None:
            if int(norbs) != norbs or norbs <= 0:
                raise ValueError("The norbs parameter must be an integer > 0.")
            norbs = int(norbs)
        self.norbs = norbs
        self.coords = coords
        self.name = name
        # self.canonical_repr = str(self.__hash__())
        self.canonical_repr = "1" if name is None else name

    def pos(self, tag):
        return self.coords[tag[0], :]

    def normalize_tag(self, tag):
        if tag[0] >= len(self.coords):
            raise ValueError
        return ta.array(tag)

    def __hash__(self):
        return 1

def AmorphousCrossSection_ScatteringRegion(cross_section, n_layers, norbs=4, bonds=None):

    # Coordinates of the sites in the scattering region
    z_coords = np.arange(n_layers)
    coords = np.array([cross_section.x, cross_section.y, z_coords]).T

    # Initialise the system in Kwant
    syst = kwant.Builder()
    latt = AmorphousWire_kwant(norbs=norbs, coords=coords)
    syst[latt.neighbors()] = hopping_function

    def hopping_function(site1, site2):
        x1, y1, _ = site1.pos
        x2, y2, _ = site2.pos

def hopping(t, lamb, lamb_z, d, phi, theta, cutoff_dist):
    f_cutoff = np.heaviside(cutoff_dist - d, 1) * np.exp(-d + 1)
    normal_hopp = - t * np.kron(sigma_x, tau_0)
    spin_orbit_xy = 1j * 0.5 * lamb * np.kron(sigma_z  * np.sin(theta), np.cos(phi) * tau_y - np.sin(phi) * tau_x)
    spin_orbit_z = - 1j * 0.5 * lamb_z * np.cos(theta) * np.kron(sigma_y, tau_0)
    return f_cutoff * (normal_hopp + spin_orbit_xy + spin_orbit_z)

def onsite(eps):
    return eps * np.kron(sigma_x, tau_0)

