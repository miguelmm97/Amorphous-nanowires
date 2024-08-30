# %% Modules setup

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


# %% Logging setup
loger_kwant = logging.getLogger('kwant')
loger_kwant.setLevel(logging.INFO)

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

# %% Module
"""
Here we promote the infinite amorphous nanowire defined in class InfiniteNanowire_FuBerg.py 
into a kwant.system where to do transport calculations.
"""

sigma_0 = np.eye(2, dtype=np.complex128)
sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
tau_0, tau_x, tau_y, tau_z = sigma_0, sigma_x, sigma_y, sigma_z


# Hopping functions
def hopping(t, lamb, lamb_z, d, phi, theta, cutoff_dist):
    f_cutoff = np.heaviside(cutoff_dist - d, 1) * np.exp(-d + 1)
    normal_hopp = - t * np.kron(sigma_x, tau_0)
    spin_orbit_xy = 1j * 0.5 * lamb * np.kron(sigma_z * np.sin(theta), np.cos(phi) * tau_y - np.sin(phi) * tau_x)
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

def Peierls(pos0, pos1, flux, area):
    def integrand(x, m, x0, y0):
        return m * (x - x0) + y0

    x1, y1 = pos0[0], pos0[1]
    x2, y2 = pos1[0], pos1[1]

    m = (y2 - y1) / (x2 - x1)
    I = quad(integrand, x1, x2, args=(m, x1, y1))[0]
    return np.exp(2 * pi * 1j * flux * I / area)

"""
Note that in the following hoppings are always defined down up, that is from x to x+1, y to y+1 z to z+1, so that we
get the angles correctly. Kwant then takes the complex conjugate for the reverse ones.

Note also that in kwant the hoppings are defined like (latt(), latt()) where the second entry refes to the site from 
which we hopp. In the displacement function is the opposite, (pos1, pos2) means we are hopping from 1 to 2.
"""

class AmorphousCrossSectionWire_ScatteringRegion(kwant.builder.SiteFamily):
    def __init__(self, norbs, cross_section, name=None):

        if norbs is not None:
            if int(norbs) != norbs or norbs <= 0:
                raise ValueError("The norbs parameter must be an integer > 0.")
            norbs = int(norbs)

        # Class fields
        loger_kwant.trace('Initialising cross section as a SiteFamily...')
        self.norbs = norbs
        self.coords = np.array([cross_section.x, cross_section.y]).T
        self.Nsites = cross_section.Nsites
        self.Nx = cross_section.Nx
        self.Ny = cross_section.Ny
        self.name = name
        self.canonical_repr = "1" if name is None else name

    def pos(self, tag):
        return np.concatenate((self.coords[tag[0], :], np.array([tag[1]])))

    def normalize_tag(self, tag):
        return ta.array(tag)

    def __hash__(self):
        return 1

def promote_to_transport_nanowire(cross_section, n_layers, param_dict):

    # Load parameters into the builder namespace
    try:
        t      = param_dict['t']
        eps    = param_dict['eps']
        lamb   = param_dict['lamb']
        lamb_z = param_dict['lamb_z']
        flux   = param_dict['flux']
    except KeyError as err:
        raise KeyError(f'Parameter error: {err}')

    # Create lattice structure for the scattering region from the amorphous cross-section
    latt = AmorphousCrossSectionWire_ScatteringRegion(norbs=4, cross_section=cross_section, name='scatt_region')

    # Initialise kwant system
    loger_kwant.info('Creating kwant scattering region...')
    syst = kwant.Builder()
    syst[(latt(i, z) for i in range(latt.Nsites) for z in range(n_layers))] = onsite(eps)

    # Hoppings
    hopp_z_up = hopping(t, lamb, lamb_z, 1., 0, 0, cross_section.r)
    for i in range(latt.Nsites):
        for n in cross_section.neighbours[i]:
            loger_kwant.trace(f'Defining hopping from site {i} to {n}.')

            # In the cross-section
            d, phi = displacement2D(latt(i, 0).pos, latt(n, 0).pos)
            peierls_phase = Peierls(latt(i, 0).pos, latt(n, 0).pos, flux, cross_section.area)
            syst[((latt(n, z), latt(i, z)) for z in range(n_layers))] = hopping(t, lamb, lamb_z, d, phi,
                                                                             pi / 2, cross_section.r) * peierls_phase
            # Between cross-sections
            loger_kwant.trace(f'Defining hopping of site {i} between cross-section layers.')
            syst[((latt(i, z + 1), latt(i, z)) for z in range(n_layers - 1))] = hopp_z_up

    complete_system = attach_cubic_leads(syst, cross_section, latt, n_layers, param_dict)
    return complete_system

def attach_cubic_leads(scatt_region, cross_section, latt, n_layers, param_dict):

    # Load parameters into the builder namespace
    try:
        t      = param_dict['t']
        eps    = param_dict['eps']
        lamb   = param_dict['lamb']
        lamb_z = param_dict['lamb_z']
    except KeyError as err:
        raise KeyError(f'Parameter error: {err}')

    # Fixed regular lattice hoppings
    hopp_z_up = hopping(t, lamb, lamb_z, 1., 0, 0, cross_section.r)
    hopp_x_up = hopping(t, lamb, lamb_z, 1., 0, pi / 2, cross_section.r)
    hopp_y_up = hopping(t, lamb, lamb_z, 1., pi / 2, pi / 2, cross_section.r)

    # Left lead: definition
    loger_kwant.info('Attaching left lead...')
    sym_left_lead = kwant.TranslationalSymmetry((0, 0, -1))
    left_lead = kwant.Builder(sym_left_lead)
    latt_lead = kwant.lattice.cubic(norbs=4)

    # Left lead: Hoppings
    loger_kwant.trace('Defining hoppings in the firs unit cell of the lead...')
    left_lead[(latt_lead(i, j, 0) for i in range(latt.Nx) for j in range(latt.Ny))] = onsite(eps)
    left_lead[kwant.builder.HoppingKind((1, 0, 0), latt_lead, latt_lead)] = hopp_x_up
    left_lead[kwant.builder.HoppingKind((0, 1, 0), latt_lead, latt_lead)] = hopp_y_up
    left_lead[kwant.builder.HoppingKind((0, 0, 1), latt_lead, latt_lead)] = hopp_z_up

    # Left lead: Attachment
    loger_kwant.trace('Defining the way to attach the lead to the system...')
    scatt_region[(latt_lead(i, j, -1) for i in range(latt.Nx) for j in range(latt.Ny))] = onsite(eps)
    scatt_region[kwant.builder.HoppingKind((1, 0, 0), latt_lead, latt_lead)] = hopp_x_up
    scatt_region[kwant.builder.HoppingKind((0, 1, 0), latt_lead, latt_lead)] = hopp_y_up
    scatt_region[(((latt(i + latt.Ny * j, 0), latt_lead(i, j, -1)) for i in range(latt.Nx) for j in range(latt.Ny)))] = hopp_z_up
    scatt_region.attach_lead(left_lead)

    # Right lead: definition
    loger_kwant.info('Attaching right lead...')
    sym_right_lead = kwant.TranslationalSymmetry((0, 0, 1))
    right_lead = kwant.Builder(sym_right_lead)
    latt_lead = kwant.lattice.cubic(norbs=4)

    # Right lead: Hoppings
    loger_kwant.trace('Defining hoppings in the firs unit cell of the lead...')
    right_lead[(latt_lead(i, j, 0) for i in range(latt.Nx) for j in range(latt.Ny))] = onsite(eps)
    right_lead[kwant.builder.HoppingKind((1, 0, 0), latt_lead, latt_lead)] = hopp_x_up
    right_lead[kwant.builder.HoppingKind((0, 1, 0), latt_lead, latt_lead)] = hopp_y_up
    right_lead[kwant.builder.HoppingKind((0, 0, 1), latt_lead, latt_lead)] = hopp_z_up

    # Right lead: Attachment
    loger_kwant.trace('Defining the way to attach the lead to the system...')
    scatt_region[(latt_lead(i, j, n_layers) for i in range(latt.Nx) for j in range(latt.Ny))] = onsite(eps)
    scatt_region[kwant.builder.HoppingKind((1, 0, 0), latt_lead, latt_lead)] = hopp_x_up
    scatt_region[kwant.builder.HoppingKind((0, 1, 0), latt_lead, latt_lead)] = hopp_y_up
    scatt_region[(((latt_lead(i, j, n_layers), latt(i + latt.Ny * j, n_layers - 1)) for i in range(latt.Nx)
                                                                for j in range(latt.Ny)))] = hopp_z_up
    scatt_region.attach_lead(right_lead)
    return scatt_region



