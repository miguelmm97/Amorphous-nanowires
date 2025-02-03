# %% modules setup

# Math and plotting
from numpy import pi
import numpy as np
from scipy.integrate import cumulative_trapezoid, quad

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

def displacement2D_kwant(site1, site0):
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



"""
Note that in the following hoppings are always defined down up, that is from x to x+1, y to y+1 z to z+1, so that we
get the angles correctly. Kwant then takes the complex conjugate for the reverse ones.

Note also that in kwant the hoppings are defined like (latt(), latt()) where the second entry refes to the site from 
which we hopp.
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

def promote_to_kwant_nanowire(cross_section, n_layers, param_dict, attach_leads=True, mu_leads=0.):

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

    # Initialise kwant system
    loger_kwant.info('Creating kwant scattering region...')
    syst = kwant.Builder()
    syst[(latt(i, z) for i in range(latt.Nsites) for z in range(n_layers))] = onsite(eps)

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
        complete_system = attach_cubic_leads(syst, cross_section, latt, n_layers, param_dict, mu_leads=mu_leads)
    else:
        complete_system = syst
    return complete_system

def attach_cubic_leads(scatt_region, cross_section, latt, n_layers, param_dict, mu_leads=0.):

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

def crystal_nanowire_kwant(Nx, Ny, n_layers, param_dict):

    # Load parameters into the builder namespace
    try:
        t      = param_dict['t']
        eps    = param_dict['eps']
        lamb   = param_dict['lamb']
        lamb_z = param_dict['lamb_z']
    except KeyError as err:
        raise KeyError(f'Parameter error: {err}')

    # Define lattice and initialise system and sites
    latt = kwant.lattice.cubic(1, norbs=4)
    syst = kwant.Builder()
    syst[(latt(i, j, k) for i in range(Nx) for j in range(Ny) for k in range(n_layers))] = onsite(eps)

    # Hoppings
    cutoff = 1.3
    def hopp_x_up(site1, site0, flux):
        return hopping(t, lamb, lamb_z, 1., 0., pi / 2, cutoff) * Peierls_kwant(site1, site0, flux, (Nx - 1) * (Ny - 1))
    hopp_z_up = hopping(t, lamb, lamb_z, 1., 0., 0, cutoff)
    hopp_y_up = hopping(t, lamb, lamb_z, 1., pi / 2, pi / 2, cutoff)
    syst[kwant.builder.HoppingKind((1, 0, 0), latt, latt)] = hopp_x_up
    syst[kwant.builder.HoppingKind((0, 1, 0), latt, latt)] = hopp_y_up
    syst[kwant.builder.HoppingKind((0, 0, 1), latt, latt)] = hopp_z_up

    # Lead
    lead = kwant.Builder(kwant.TranslationalSymmetry((0, 0, -1)))
    lead[(latt(i, j, 0) for i in range(Nx) for j in range(Ny))] = onsite(eps)
    lead[kwant.builder.HoppingKind((1, 0, 0), latt, latt)] = hopp_x_up
    lead[kwant.builder.HoppingKind((0, 1, 0), latt, latt)] = hopp_y_up
    lead[kwant.builder.HoppingKind((0, 0, 1), latt, latt)] = hopp_z_up
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    return syst

def infinite_nanowire_kwant(Nx, Ny, param_dict, mu_leads=0.):
    # Load parameters into the builder namespace
    try:
        t = param_dict['t']
        eps = param_dict['eps']
        lamb = param_dict['lamb']
        lamb_z = param_dict['lamb_z']
    except KeyError as err:
        raise KeyError(f'Parameter error: {err}')

    onsite_leads = onsite(eps) + mu_leads * np.kron(sigma_0, tau_0)

    # Define lattice and initialise system and sites
    latt = kwant.lattice.cubic(1, norbs=4)
    lead = kwant.Builder(kwant.TranslationalSymmetry((0, 0, 1)))
    lead[(latt(i, j, 0) for i in range(Nx) for j in range(Ny))] = onsite_leads

    # Hoppings
    cutoff = 1.3
    def hopp_x_up(site1, site0, flux):
        return hopping(t, lamb, lamb_z, 1., 0, pi / 2, cutoff) * Peierls_kwant(site1, site0, flux, (Nx - 1) * (Ny - 1))
    hopp_z_up = hopping(t, lamb, lamb_z, 1., 0, 0, cutoff)
    hopp_y_up = hopping(t, lamb, lamb_z, 1., pi / 2, pi / 2, cutoff)
    lead[kwant.builder.HoppingKind((1, 0, 0), latt, latt)] = hopp_x_up
    lead[kwant.builder.HoppingKind((0, 1, 0), latt, latt)] = hopp_y_up
    lead[kwant.builder.HoppingKind((0, 0, 1), latt, latt)] = hopp_z_up

    return lead

def FuBerg_model_bands(Nx, Ny, kz, flux, param_dict):

    # Load parameters into the builder namespace
    try:
        t = param_dict['t']
        eps = param_dict['eps']
        lamb = param_dict['lamb']
        lamb_z = param_dict['lamb_z']
    except KeyError as err:
        raise KeyError(f'Parameter error: {err}')

    # Lattice parameters
    Nsites = int(Nx * Ny)
    area = (Nx - 1) * (Ny - 1)
    dimH = Nsites * 4
    sites = np.arange(0, Nsites)
    x = sites % Nx
    y = sites // Nx

    # Hamiltonian parameters
    H_offdiag = np.zeros((dimH, dimH), dtype=np.complex128)
    H = np.zeros((len(kz), dimH, dimH), dtype=np.complex128)
    def peierls_x(y):
        return np.exp(2 * pi * 1j * flux * y / area)

    # Off-diagonal Hamiltonian
    for n in sites:
        # Jump operators along x and y direction
        state_n, state_nx, state_ny = np.zeros((Nsites,)), np.zeros((Nsites,)), np.zeros((Nsites,))
        state_n[n] = 1
        if x[n] != Nx - 1: state_nx[n + 1] = 1
        if y[n] != Ny - 1: state_ny[n + Nx] = 1
        jump_x, jump_y = np.outer(state_n, state_nx) * peierls_x(y[n]), np.outer(state_n, state_ny)

        # Off-diagonal Hamiltonian
        H_offdiag += -t * np.kron(jump_x, np.kron(sigma_x, tau_0)) - t * np.kron(jump_y, np.kron(sigma_x, tau_0)) + \
                     1j * 0.5 * lamb * (np.kron(jump_x, np.kron(sigma_z, tau_y)) - np.kron(jump_y, np.kron(sigma_z, tau_x)))
    H_offdiag += H_offdiag.T.conj()

    # Full Hamiltonian
    for i, k in enumerate(kz):
        H[i, :, :] = (eps - 2 * t * np.cos(k)) * np.kron(np.eye(Nsites), np.kron(sigma_x, tau_0)) + \
                     + lamb_z * np.sin(k) * np.kron(np.eye(Nsites), np.kron(sigma_y, tau_0)) + H_offdiag

    # Band structure
    energy_bands, eigenstates = {}, {}
    aux_bands = np.zeros((len(kz), dimH))
    aux_eigenstates = np.zeros((len(kz), dimH, dimH), dtype=np.complex128)
    for j in range(len(kz)):
        bands_k, eigenstates_k = np.linalg.eigh(H[j, :, :])
        idx = bands_k.argsort()
        aux_bands[j, :], aux_eigenstates[j, :, :] = bands_k[idx], eigenstates_k[:, idx]

    # Ordering bands
    for i in range(dimH):
        energy_bands[i] = aux_bands[:, i]
        eigenstates[i] = aux_eigenstates[:, :, i]

    return energy_bands, eigenstates

def thermal_average(G, Ef, kBT):

    # Derivative of the Fermi-Dirac distribution
    def df_FD(E, mu, kBT):
        beta = 1 / kBT    # kBT in units of t
        if kBT != 0:
            return - beta * np.exp(beta * (E - mu)) / (np.exp(beta * (E - mu)) + 1) ** 2
        else:
            raise ValueError("T=0 limit undefined unless inside an integral!")

    # Thermal range to average over
    dE = Ef[1] - Ef[0]
    sample_th = int(kBT / dE)
    if sample_th < 5:
        raise ValueError('Thermal interval to average over is too small!')
    Ef_th = Ef[sample_th: -sample_th]
    G_th = np.zeros((len(Ef_th),))

    # Average
    for i, E in enumerate(Ef_th):
        j = i + sample_th
        E_interval = Ef[j - sample_th: j + sample_th]
        G_interval = G[j - sample_th: j + sample_th]
        integrand = - G_interval * df_FD(E_interval, Ef[j], kBT)
        G_th[i] = cumulative_trapezoid(integrand, E_interval)[-1]
    return G_th, Ef_th