# %% modules setup

# Math and plotting
from numpy import pi
import numpy as np
from scipy.integrate import quad

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

def displacement3D_kwant(site1, site0):
    x1, y1, z1 = site0.pos[0], site0.pos[1], site0.pos[2]
    x2, y2, z2 = site1.pos[0], site1.pos[1], site1.pos[2]

    # Definition of the vector between sites 2 and 1 (from st.1 to st.2)
    v = np.zeros((3,))
    v[0] = (x2 - x1)
    v[1] = (y2 - y1)
    v[2] = (z2 - z1)

    # Module of the vector between sites 2 and 1
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

def Peierls_kwant(site1, site0, flux, area):
    def integrand(x, m, x0, y0):
        return m * (x - x0) + y0

    x1, y1 = site0.pos[0], site0.pos[1]
    x2, y2 = site1.pos[0], site1.pos[1]

    if x2 == x1:
        I = 0
    else:
        m = (y2 - y1) / (x2 - x1)
        I = quad(integrand, x1, x2, args=(m, x1, y1))[0]
    return np.exp(2 * pi * 1j * flux * I / area)


#%% Kwant classes
"""
Note that in the following hoppings are always defined down up, that is from x to x+1, y to y+1 z to z+1, so that we
get the angles correctly. Kwant then takes the complex conjugate for the reverse ones.

Note also that in kwant the hoppings are defined like (latt(), latt()) where the second entry refes to the site from 
which we hopp.
"""

class FullyAmorphousWire_ScatteringRegion(kwant.builder.SiteFamily):
    def __init__(self, norbs, lattice, name=None):

        if norbs is not None:
            if int(norbs) != norbs or norbs <= 0:
                raise ValueError("The norbs parameter must be an integer > 0.")
            norbs = int(norbs)

        # Class fields
        loger_kwant.trace('Initialising cross section as a SiteFamily...')
        self.norbs = norbs
        self.coords = np.array([lattice.x, lattice.y, lattice.z]).T
        self.Nsites = lattice.Nsites
        self.Nx = lattice.Nx
        self.Ny = lattice.Ny
        self.Nz = lattice.Nz
        self.name = name
        self.canonical_repr = "1" if name is None else name

    def pos(self, tag):
        return self.coords[tag, :][0, :]

    def normalize_tag(self, tag):
        return ta.array(tag)

    def __hash__(self):
        return 1

def promote_to_kwant_nanowire3d(lattice_tree, param_dict, attach_leads=True, mu_leads=0.):

    # Load parameters into the builder namespace
    try:
        t      = param_dict['t']
        eps    = param_dict['eps']
        lamb   = param_dict['lamb']
        lamb_z = param_dict['lamb_z']
    except KeyError as err:
        raise KeyError(f'Parameter error: {err}')

    # Create lattice structure for the scattering region from the amorphous cross-section
    latt = FullyAmorphousWire_ScatteringRegion(norbs=4, lattice=lattice_tree, name='scatt_region')

    # Initialise kwant system
    loger_kwant.info('Creating kwant scattering region...')
    syst = kwant.Builder()
    syst[(latt(i) for i in range(latt.Nsites))] = onsite(eps)

    # Hopping functions
    def hopp(site1, site0, flux):
        d, phi, theta = displacement3D_kwant(site1, site0)
        return hopping(t, lamb, lamb_z, d, phi, theta, lattice_tree.r) * Peierls_kwant(site1, site0, flux, lattice_tree.area)

    # Populate hoppings
    for i in range(latt.Nsites):
        for n in lattice_tree.neighbours[i]:
            loger_kwant.trace(f'Defining hopping from site {i} to {n}.')
            syst[(latt(n), latt(i))] = hopp

    if attach_leads:
        complete_system = attach_cubic_leads(syst, lattice_tree, latt, param_dict, mu_leads=mu_leads)
    else:
        complete_system = syst
    return complete_system

def attach_cubic_leads(scatt_region, lattice_tree, latt, param_dict, mu_leads=0.):

    # Load parameters into the builder namespace
    try:
        t      = param_dict['t']
        eps    = param_dict['eps']
        lamb   = param_dict['lamb']
        lamb_z = param_dict['lamb_z']
    except KeyError as err:
        raise KeyError(f'Parameter error: {err}')

    onsite_leads = onsite(eps) + mu_leads * np.kron(sigma_0, tau_0)

    # Hoppings
    def hopp_lead_wire(site1, site0, flux):
        d, phi, theta = displacement3D_kwant(site1, site0)
        return hopping(t, lamb, lamb_z, d, phi, theta, lattice_tree.r) * Peierls_kwant(site1, site0, flux, lattice_tree.area)
    def hopp_x_up(site1, site0, flux):
        return hopping(t, lamb, lamb_z, 1., 0, pi / 2, lattice_tree.r) * Peierls_kwant(site1, site0, flux, lattice_tree.area)
    hopp_z_up = hopping(t, lamb, lamb_z, 1., 0, 0, lattice_tree.r)
    hopp_y_up = hopping(t, lamb, lamb_z, 1., pi / 2, pi / 2, lattice_tree.r)

    # Left lead: definition
    loger_kwant.info('Attaching left lead...')
    sym_left_lead = kwant.TranslationalSymmetry((0, 0, -1))
    left_lead = kwant.Builder(sym_left_lead)
    latt_lead = kwant.lattice.cubic(norbs=4)

    # Left lead: Hoppings
    loger_kwant.trace('Defining hoppings in the firs unit cell of the lead...')
    left_lead[(latt_lead(i, j, 0) for i in range(latt.Nx) for j in range(latt.Ny))] = onsite_leads
    left_lead[kwant.builder.HoppingKind((1, 0, 0), latt_lead, latt_lead)] = hopp_x_up
    left_lead[kwant.builder.HoppingKind((0, 1, 0), latt_lead, latt_lead)] = hopp_y_up
    left_lead[kwant.builder.HoppingKind((0, 0, 1), latt_lead, latt_lead)] = hopp_z_up

    # Left lead: Attachment
    loger_kwant.trace('Defining the way to attach the lead to the system...')
    scatt_region[(latt_lead(i, j, -1) for i in range(latt.Nx) for j in range(latt.Ny))] = onsite_leads
    scatt_region[kwant.builder.HoppingKind((1, 0, 0), latt_lead, latt_lead)] = hopp_x_up
    scatt_region[kwant.builder.HoppingKind((0, 1, 0), latt_lead, latt_lead)] = hopp_y_up

    interface_left = []
    for i in range(lattice_tree.Nsites):
        if lattice_tree.z[i] < 0.5:
            interface_left.append(i)

    for site in interface_left:
        for i in range(latt.Nx):
            for j in range(latt.Ny):
                if displacement3D_kwant(latt_lead(i, j, -1), latt(site))[0] < lattice_tree.r:
                    scatt_region[(latt(site), latt_lead(i, j, -1))] = hopp_lead_wire
    scatt_region.attach_lead(left_lead)


    # # Right lead: definition
    loger_kwant.info('Attaching right lead...')
    sym_right_lead = kwant.TranslationalSymmetry((0, 0, 1))
    right_lead = kwant.Builder(sym_right_lead)
    latt_lead = kwant.lattice.cubic(norbs=4)
    #
    # # Right lead: Hoppings
    loger_kwant.trace('Defining hoppings in the first unit cell of the lead...')
    right_lead[(latt_lead(i, j, 0) for i in range(latt.Nx) for j in range(latt.Ny))] = onsite_leads
    right_lead[kwant.builder.HoppingKind((1, 0, 0), latt_lead, latt_lead)] = hopp_x_up
    right_lead[kwant.builder.HoppingKind((0, 1, 0), latt_lead, latt_lead)] = hopp_y_up
    right_lead[kwant.builder.HoppingKind((0, 0, 1), latt_lead, latt_lead)] = hopp_z_up
    #
    # # Right lead: Attachment
    loger_kwant.trace('Defining the way to attach the lead to the system...')
    scatt_region[(latt_lead(i, j, latt.Nz) for i in range(latt.Nx) for j in range(latt.Ny))] = onsite_leads
    # scatt_region[(latt_lead(i, j, latt.Nz + 1) for i in range(latt.Nx) for j in range(latt.Ny))] = onsite_leads
    scatt_region[kwant.builder.HoppingKind((1, 0, 0), latt_lead, latt_lead)] = hopp_x_up
    scatt_region[kwant.builder.HoppingKind((0, 1, 0), latt_lead, latt_lead)] = hopp_y_up

    interface_right = []
    for i in range(lattice_tree.Nsites):
        if lattice_tree.z[i] > (lattice_tree.Nz - 1) - 0.5:
            interface_right.append(i)

    for site in interface_right:
        for i in range(latt.Nx):
            for j in range(latt.Ny):
                if displacement3D_kwant(latt_lead(i, j, latt.Nz), latt(site))[0] < lattice_tree.r:
                    scatt_region[(latt(site), latt_lead(i, j, latt.Nz))] = hopp_lead_wire
    scatt_region.attach_lead(right_lead)

    return scatt_region


#%% Transport functions

def select_perfect_transmission_flux(nanowire, flux0=0.5, flux_end=0.7, Nflux=200):

    loger_kwant.info(f'Calculating flux that gives perfect conductance for this sample...')
    flux = np.linspace(flux0, flux_end, Nflux)

    Gmax = 0.
    flux_max = flux0
    for i, phi in enumerate(flux):
        S0 = kwant.smatrix(nanowire, 0.1, params=dict(flux=phi))
        G = S0.transmission(1, 0)
        loger_kwant.info(f'Flux: {i} / {Nflux - 1}, Conductance: {G}')
        if G > Gmax:
            Gmax = G
            flux_max = phi
        else:
            pass

    return flux_max, Gmax

def thermal_average(G0, Ef0, T, thermal_interval= None):

   # Definitions
    energy_factor = 150                  # t=150 meV in Bi2Se3
    k_B           = 8.617333262e-2       # [meV/K]
    beta          = 1 / (k_B * T)
    G_avg         = np.zeros(G0.shape)
    if thermal_interval is None:
        thermal_interval = int((1.5 * (k_B * T) * len(Ef0)) / (Ef0[:-1] - Ef0[0]))

    def df_FD(E, Ef, T):
        if T != 0:
            return - beta * np.exp(beta * (E - Ef) * energy_factor) / (np.exp(beta * (E - Ef)* energy_factor) + 1) ** 2
        else:
            raise ValueError("T=0 limit undefined unless inside an integral!")

    for i, Ef in enumerate(Ef0):
        delta_E = Ef0[i - thermal_interval: i + thermal_interval]
        delta_G = G0[i - thermal_interval: i + thermal_interval]
        integrand = - delta_G * df_FD(delta_E, Ef, T)
        G_avg[i] = np.trapezoid(integrand, delta_E)

    return G_avg
