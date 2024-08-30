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
Here we promote the infinite amorphous nanowire defined in class InfiniteNanowire_FuBerg.py 
into a kwant.system where to do transport calculations.
"""

sigma_0 = np.eye(2, dtype=np.complex128)
sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
tau_0, tau_x, tau_y, tau_z  = sigma_0, sigma_x, sigma_y, sigma_z

class AmorphousCrossSectionWire_ScatteringRegion(kwant.builder.SiteFamily):
    def __init__(self, norbs, cross_section, name=None):

        if norbs is not None:
            if int(norbs) != norbs or norbs <= 0:
                raise ValueError("The norbs parameter must be an integer > 0.")
            norbs = int(norbs)

        # Class fields
        self.norbs = norbs
        self.coords = np.array([cross_section.x, cross_section.y]).T
        self.Nsites = len(self.coords[:, 0])
        self.Nx = len(cross_section.x)
        self.Ny = len(cross_section.y)
        self.name = name
        self.canonical_repr = "1" if name is None else name

    def pos(self, tag):
        return np.concatenate((self.coords[tag[0], :], np.array([tag[1]])))

    def normalize_tag(self, tag):
        return ta.array(tag)

    def __hash__(self):
        return 1


def promote_to_transport_nanowire(cross_section, n_layers, eps):

    latt = AmorphousCrossSectionWire_ScatteringRegion(norbs=4, cross_section=cross_section, name='scatt_region')

    syst = kwant.Builder()
    syst[(latt(i, z) for i in range(latt.Nsites) for z in range(n_layers))] = onsite(eps)

    # Hoppings
    for z in range(n_layers):
        # In the cross-section


        syst[((latt(i, z), latt(n, z)) for i in range(latt.Nsites) for n in cross_section.neighbours[i])] = \




        # Between cross-sections
        if z < n_layers - 1:
            syst[((latt(i, z), latt(i, z + 1)) for i in range(latt.Nsites))] = 2.
        else:
            pass


def hopping(t, lamb, lamb_z, d, phi, theta, cutoff_dist):
    f_cutoff = np.heaviside(cutoff_dist - d, 1) * np.exp(-d + 1)
    normal_hopp = - t * np.kron(sigma_x, tau_0)
    spin_orbit_xy = 1j * 0.5 * lamb * np.kron(sigma_z  * np.sin(theta), np.cos(phi) * tau_y - np.sin(phi) * tau_x)
    spin_orbit_z = - 1j * 0.5 * lamb_z * np.cos(theta) * np.kron(sigma_y, tau_0)
    return f_cutoff * (normal_hopp + spin_orbit_xy + spin_orbit_z)

def onsite(eps):
    return eps * np.kron(sigma_x, tau_0)

def displacement2D(pos0, pos1):

    x1, y1 = pos0[0], pos0[1]
    x2, y2 = pos1[0], pos1[1]

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

