# %% Modules setup

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

#%% Functions for setting up the Hamiltonian

# Sigma matrices
sigma_0 = np.eye(2, dtype=np.complex128)
sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
tau_0, tau_x, tau_y, tau_z = sigma_0, sigma_x, sigma_y, sigma_z

def hopping(t, lamb, eta, d, phi, theta, cutoff_dist):
    """
    Input:
    t -> float: hopping amplitude as described in the main text
    lamb -> float: spin-orbit coupling as described in the main text
    eta -> float: hopping amplitude as described in the main text
    d -> float: distance between the two sites connected by the hopping term
    phi -> float: azimuthal angle between the two sites connected by the hopping term
    theta -> float: spherical angle between the two sites connected by the hopping term
    cutoff_dist -> float: cutoff distance above which the hopping vanishes

    Output:
    np.ndarray (4x4): Hopping amplitude between the two sites
    """
    f_cutoff = np.heaviside(cutoff_dist - d, 1) * np.exp(-d + 1)
    normal_hopp = - t * np.kron(sigma_x, tau_0)
    spin_orbit_xy = 1j * 0.5 * lamb * np.kron(sigma_z * np.sin(theta), np.cos(phi) * tau_y - np.sin(phi) * tau_x)
    spin_orbit_z = - 1j * 0.5 * eta * np.cos(theta) * np.kron(sigma_y, tau_0)
    return f_cutoff * (normal_hopp + spin_orbit_xy + spin_orbit_z)

def onsite(eps):
    """
    Input:
    eps -> float: onsite energy as described in the main text

    Output:
    np.ndarray (4x4): Onsite energy contribution for the site
    """
    return eps * np.kron(sigma_x, tau_0)

def displacement2D_kwant(site1, site0):
    """
    Input:
    site1: kwant.builder.Site -> Site towards we hopp
    site0: kwant.builder.Site -> Site from which we hopp
    Output:
    r -> float: distance between the two sites
    phi -> float: azimuthal angle between sites
    """

    x1, y1 = site0.pos[0], site0.pos[1]
    x2, y2 = site1.pos[0], site1.pos[1]

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

def displacement3D_kwant(site1, site0):
    """
    Input:
    site1: kwant.builder.Site -> Site towards we hopp
    site0: kwant.builder.Site -> Site from which we hopp

    Output:
    r -> float: distance between sites
    phi -> float: azimuthal angle between sites
    theta -> float: spherical angle between sites
    """
    x1, y1, z1 = site0.pos[0], site0.pos[1], site0.pos[2]
    x2, y2, z2 = site1.pos[0], site1.pos[1], site1.pos[2]

    # Definition of the vector between sites 2 and 1 (from site1 to site2)
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

def Peierls_kwant(site1, site0, flux, area):
    """
    Input:
    site1: kwant.builder.Site -> Site towards we hopp
    site0: kwant.builder.Site -> Site from which we hopp
    flux -> float: magnetic flux threaded through the reference cross-section
    area -> float: area of the reference cross-section

    Output:
    float: Peierl's phase associated with the hopping from site site0 to site1
    """
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

#%% Transport functions
def select_perfect_transmission_flux(nanowire, flux0=0.8, flux_end=1.5, Nflux=100, Ef=0., mu_leads=0.):
    """
    Input:
    nanowire -> kwant.builder.Builder: nanowire system
    flux0 -> float: magnetic flux value from which to start scanning for perfect transmission
    flux_end -> float: magnetic flux value where to end the scann for perfect transmission
    Nflux -> int: number of points in the flux scann
    Ef -> float: Fermi energy at which to perform the scann
    mu_leads -> float: chemical potential in the leads

    Output:
    flux_max -> float: magnetic flux at which maximum transmission is achieved
    Gmax -> float: Maximum conductance achieved
    """

    loger_kwant.trace(f'Calculating flux that gives perfect conductance for this sample...')
    flux = np.linspace(flux0, flux_end, Nflux)

    Gmax = 0.
    flux_max = flux0
    for i, phi in enumerate(flux):
        S0 = kwant.smatrix(nanowire, 0., params=dict(flux=phi, mu=-Ef, mu_leads=mu_leads - Ef))
        G = S0.transmission(1, 0)
        loger_kwant.info(f'Flux: {i} / {Nflux - 1}, Conductance: {G :.2f}')
        if G > Gmax:
            Gmax = G
            flux_max = phi
            if Gmax > 0.98:
                break
        else:
            pass

    return flux_max, Gmax

#%% Kwant classes

class AmorphousCrossSectionWire_ScatteringRegion(kwant.builder.SiteFamily):
    """
    Class to build a kwant.builder.SiteFamily for a layered-amorphous nanowire
    """
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

class FullyAmorphousWire_ScatteringRegion(kwant.builder.SiteFamily):
    """
    Class to build a kwant.builder.SiteFamily for a fully amorphous nanowire
    """
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


def promote_to_kwant_nanowire_2d(cross_section, n_layers, param_dict, attach_leads=True):
    """
    Input:
    cross_section -> AmorphousLattice_2d: Cross-section of the layered-amorphous nanowre
    n_layers -> int: number of layers in z direction of the layered-amorphous nanowire
    param_dict -> dict: dictionary of parameters for the Hamiltonian of the nanowire
    attach_leads -> bool: Indicates if leads should be attached to the system (transport in kwant)
                          or it should be a closed system (calculate spectral properties)

    Output:
    complete_system -> kwant.builder.Builder: Nanowire kwant system
    """

    # Load parameters into the builder namespace
    try:
        t      = param_dict['t']
        eps    = param_dict['eps']
        lamb   = param_dict['lamb']
        lamb_z = param_dict['lamb_z']
    except KeyError as err:
        raise KeyError(f'Parameter error: {err}')

    # Create lattice structure for the scattering region from the amorphous cross-section
    latt = AmorphousCrossSectionWire_ScatteringRegion(norbs=4, cross_section=cross_section, name='scatt_region')

    def onsite_potential(site, mu):
        index = site.tag[0]
        if cross_section.K_onsite < 1e-12:
            return onsite(eps) + mu * np.kron(sigma_0, tau_0)
        else:
            return onsite(eps) + (mu + cross_section.onsite_disorder[index]) * np.kron(sigma_0, tau_0)

    # Initialise kwant system
    loger_kwant.info('Creating kwant scattering region...')
    syst = kwant.Builder()
    syst[(latt(i, z) for i in range(latt.Nsites) for z in range(n_layers))] = onsite_potential

    # Hopping functions
    hopp_z_up = hopping(t, lamb, lamb_z, 1., 0, 0, cross_section.r)
    def hopp_cross_section(site1, site0, flux):
        d, phi = displacement2D_kwant(site1, site0)
        return hopping(t, lamb, lamb_z, d, phi, pi / 2, cross_section.r) * Peierls_kwant(site1, site0, flux, cross_section.area)

    # Populate hoppings
    for i in range(latt.Nsites):
        for n in cross_section.neighbours[i]:
            loger_kwant.trace(f'Defining hopping from site {i} to {n}.')

            # In the cross-section
            syst[((latt(n, z), latt(i, z)) for z in range(n_layers))] = hopp_cross_section

            # Between cross-sections
            loger_kwant.trace(f'Defining hopping of site {i} between cross-section layers.')
            syst[((latt(i, z + 1), latt(i, z)) for z in range(n_layers - 1))] = hopp_z_up

    if attach_leads:
        complete_system = attach_cubic_leads_2d(syst, cross_section, latt, n_layers, param_dict)
    else:
        complete_system = syst
    return complete_system

def promote_to_kwant_nanowire_3d(lattice_tree, param_dict, attach_leads=True, interface_z=0.5, interface_r=1.3):
    """
    Input:
    lattice_tree -> AmorphousLattice_3d: Structure of the fully amorphous nanowire
    param_dict -> dict: dictionary of parameters for the Hamiltonian of the nanowire
    attach_leads -> bool: Indicates if leads should be attached to the system (transport in kwant)
                          or it should be a closed system (calculate spectral properties)
    interface_z -> float: maximum distance at which sites are considered in order to connect the scattering region to the leads
    interface_r -> float: cutoff distance for the hopping between the scattering region and the leads

    Output:
    complete_system -> kwant.builder.Builder: Nanowire kwant system
    """

    # Load parameters into the builder namespace
    try:
        t      = param_dict['t']
        eps    = param_dict['eps']
        lamb   = param_dict['lamb']
        eta    = param_dict['eta']
    except KeyError as err:
        raise KeyError(f'Parameter error: {err}')

    # Create SiteFamily for the scattering region from the amorphous lattice
    latt = FullyAmorphousWire_ScatteringRegion(norbs=4, lattice=lattice_tree, name='scatt_region')

    # Hopping and onsite functions
    def onsite_potential(site, mu):
        index = site.tag[0]
        if lattice_tree.K_onsite < 1e-12:
            return onsite(eps) + mu * np.kron(sigma_0, tau_0)
        else:
            return onsite(eps) + (mu + lattice_tree.onsite_disorder[index]) * np.kron(sigma_0, tau_0)

    def hopp(site1, site0, flux):
        d, phi, theta = displacement3D_kwant(site1, site0)
        return hopping(t, lamb, eta, d, phi, theta, lattice_tree.r) * Peierls_kwant(site1, site0, flux, lattice_tree.area)


    # Initialise kwant system
    loger_kwant.trace('Creating kwant scattering region...')
    syst = kwant.Builder()
    syst[(latt(i) for i in range(latt.Nsites))] = onsite_potential

    # Populate hoppings
    for i in range(latt.Nsites):
        for n in lattice_tree.neighbours[i]:
            loger_kwant.trace(f'Defining hopping from site {i} to {n}.')
            syst[(latt(n), latt(i))] = hopp

    if attach_leads:
        complete_system = attach_cubic_leads_3d(syst, lattice_tree, latt, param_dict, interface_z=interface_z, interface_r=interface_r)
    else:
        complete_system = syst
    return complete_system

def attach_cubic_leads_2d(scatt_region, cross_section, latt, n_layers, param_dict):
    """
    Input:
    scatt_region -> kwant.builder.Builder: Scattering region for the nanowire system
    cross_section -> AmorphousLattice_2d: Cross-section for the layered-amorphous nanowire
    latt -> kwant.builder.SiteFamily.AmorphousCrossSectionWire-ScatteringRegion: Site family of the scattering region
    n_layers -> int: number of layers in z direction of the layered-amorphous nanowire
    param_dict -> dict: dictionary of parameters for the Hamiltonian of the nanowire

    Output:
    scatt_region -> kwant.builder.Builder: lead-nanowire-lead kwant system
    """

    # Load parameters into the builder namespace
    try:
        t      = param_dict['t']
        eps    = param_dict['eps']
        lamb   = param_dict['lamb']
        lamb_z = param_dict['lamb_z']
    except KeyError as err:
        raise KeyError(f'Parameter error: {err}')

    def onsite_leads(site, mu_leads):
        return onsite(eps) + mu_leads * np.kron(sigma_0, tau_0)

    # Hoppings
    def hopp_x_up(site1, site0, flux):
        return hopping(t, lamb, lamb_z, 1., 0, pi / 2, cross_section.r) * Peierls_kwant(site1, site0, flux, cross_section.area)
    def hopp_lead_wire(site1, site0, flux):
        d, phi, theta = displacement3D_kwant(site1, site0)
        return hopping(t, lamb, lamb_z, d, phi, theta, cross_section.r) * Peierls_kwant(site1, site0, flux, cross_section.area)
    hopp_z_up = hopping(t, lamb, lamb_z, 1., 0, 0, cross_section.r)
    hopp_y_up = hopping(t, lamb, lamb_z, 1., pi / 2, pi / 2, cross_section.r)

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
    scatt_region[(((latt(i + latt.Ny * j, 0), latt_lead(i, j, -1)) for i in range(latt.Nx) for j in range(latt.Ny)))] = hopp_lead_wire
    scatt_region.attach_lead(left_lead)

    # Right lead: definition
    loger_kwant.info('Attaching right lead...')
    sym_right_lead = kwant.TranslationalSymmetry((0, 0, 1))
    right_lead = kwant.Builder(sym_right_lead)
    latt_lead = kwant.lattice.cubic(norbs=4)

    # Right lead: Hoppings
    loger_kwant.trace('Defining hoppings in the first unit cell of the lead...')
    right_lead[(latt_lead(i, j, 0) for i in range(latt.Nx) for j in range(latt.Ny))] = onsite_leads
    right_lead[kwant.builder.HoppingKind((1, 0, 0), latt_lead, latt_lead)] = hopp_x_up
    right_lead[kwant.builder.HoppingKind((0, 1, 0), latt_lead, latt_lead)] = hopp_y_up
    right_lead[kwant.builder.HoppingKind((0, 0, 1), latt_lead, latt_lead)] = hopp_z_up

    # Right lead: Attachment
    loger_kwant.trace('Defining the way to attach the lead to the system...')
    scatt_region[(latt_lead(i, j, n_layers) for i in range(latt.Nx) for j in range(latt.Ny))] = onsite_leads
    scatt_region[kwant.builder.HoppingKind((1, 0, 0), latt_lead, latt_lead)] = hopp_x_up
    scatt_region[kwant.builder.HoppingKind((0, 1, 0), latt_lead, latt_lead)] = hopp_y_up
    scatt_region[(((latt_lead(i, j, n_layers), latt(i + latt.Ny * j, n_layers - 1)) for i in range(latt.Nx)
                                                                for j in range(latt.Ny)))] = hopp_lead_wire
    scatt_region.attach_lead(right_lead)
    return scatt_region

def attach_cubic_leads_3d(scatt_region, lattice_tree, latt, param_dict, interface_z=0.5, interface_r=1.3, leads='topological'):
    """
    Input:
    scatt_region -> kwant.builder.Builder: Scattering region for the nanowire system
    lattice_tree -> AmorphousLattice_3d: Cross-section for the fully amorphous nanowire
    latt -> kwant.builder.SiteFamily.FullyAmorphousWire_ScatteringRegion: Site family of the scattering region
    param_dict -> dict: dictionary of parameters for the Hamiltonian of the nanowire
    interface_z -> float: maximum distance at which sites are considered in order to connect the scattering region to the leads
    interface_r -> float: cutoff distance for the hopping between the scattering region and the leads
    leads -> string: 'topological' if the leads are to have the same parameters as the nanowire, 'metallic' for the leads to be in a trivial phase

    Output:
    scatt_region -> kwant.builder.Builder: lead-nanowire-lead kwant system
    """
    # Load parameters into the builder namespace
    if leads == 'topological':
        try:
            t      = param_dict['t']
            eps    = param_dict['eps']
            lamb   = param_dict['lamb']
            eta = param_dict['eta']
        except KeyError as err:
            raise KeyError(f'Parameter error: {err}')
    elif leads == 'metallic':
        t = param_dict['t']
        eps = 0
        lamb = 0
        eta = 0

    # Hoppings and onsite
    def onsite_leads(site, mu_leads):
        return onsite(eps) + mu_leads * np.kron(sigma_0, tau_0)
    def hopp_lead_wire(site1, site0, flux):
        d, phi, theta = displacement3D_kwant(site1, site0)
        return hopping(t, lamb, eta, d, phi, theta, lattice_tree.r) * Peierls_kwant(site1, site0, flux, lattice_tree.area)
    def hopp_x_up(site1, site0, flux):
        return hopping(t, lamb, eta, 1., 0, pi / 2, lattice_tree.r) * Peierls_kwant(site1, site0, flux, lattice_tree.area)
    hopp_z_up = hopping(t, lamb, eta, 1., 0, 0, lattice_tree.r)
    hopp_y_up = hopping(t, lamb, eta, 1., pi / 2, pi / 2, lattice_tree.r)


    # Left lead: definition
    loger_kwant.trace('Attaching left lead...')
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
        if lattice_tree.z[i] < interface_z:
            interface_left.append(i)

    for site in interface_left:
        for i in range(latt.Nx):
            for j in range(latt.Ny):
                if displacement3D_kwant(latt_lead(i, j, -1), latt(site))[0] < interface_r:
                    scatt_region[(latt(site), latt_lead(i, j, -1))] = hopp_lead_wire
    scatt_region.attach_lead(left_lead)


    # # Right lead: definition
    loger_kwant.trace('Attaching right lead...')
    sym_right_lead = kwant.TranslationalSymmetry((0, 0, 1))
    right_lead = kwant.Builder(sym_right_lead)
    latt_lead = kwant.lattice.cubic(norbs=4)

    # # Right lead: Hoppings
    loger_kwant.trace('Defining hoppings in the first unit cell of the lead...')
    right_lead[(latt_lead(i, j, 0) for i in range(latt.Nx) for j in range(latt.Ny))] = onsite_leads
    right_lead[kwant.builder.HoppingKind((1, 0, 0), latt_lead, latt_lead)] = hopp_x_up
    right_lead[kwant.builder.HoppingKind((0, 1, 0), latt_lead, latt_lead)] = hopp_y_up
    right_lead[kwant.builder.HoppingKind((0, 0, 1), latt_lead, latt_lead)] = hopp_z_up

    # # Right lead: Attachment
    loger_kwant.trace('Defining the way to attach the lead to the system...')
    scatt_region[(latt_lead(i, j, latt.Nz) for i in range(latt.Nx) for j in range(latt.Ny))] = onsite_leads
    # scatt_region[(latt_lead(i, j, latt.Nz + 1) for i in range(latt.Nx) for j in range(latt.Ny))] = onsite_leads
    scatt_region[kwant.builder.HoppingKind((1, 0, 0), latt_lead, latt_lead)] = hopp_x_up
    scatt_region[kwant.builder.HoppingKind((0, 1, 0), latt_lead, latt_lead)] = hopp_y_up

    interface_right = []
    for i in range(lattice_tree.Nsites):
        if lattice_tree.z[i] > (lattice_tree.Nz - 1) - interface_z:
            interface_right.append(i)

    for site in interface_right:
        for i in range(latt.Nx):
            for j in range(latt.Ny):
                if displacement3D_kwant(latt_lead(i, j, latt.Nz), latt(site))[0] < interface_r:
                    scatt_region[(latt(site), latt_lead(i, j, latt.Nz))] = hopp_lead_wire
    scatt_region.attach_lead(right_lead)

    return scatt_region




