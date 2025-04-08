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

def promote_to_kwant_nanowire3d(lattice_tree, param_dict, attach_leads=True, interface_z=0.5, interface_r=1.3):

    # Load parameters into the builder namespace
    try:
        t      = param_dict['t']
        eps    = param_dict['eps']
        lamb   = param_dict['lamb']
        lamb_z = param_dict['lamb_z']
    except KeyError as err:
        raise KeyError(f'Parameter error: {err}')

    # Create SiteFamily for the scattering region from the amorphous lattice
    latt = FullyAmorphousWire_ScatteringRegion(norbs=4, lattice=lattice_tree, name='scatt_region')

    # Hopping and onsite functions
    def onsite_potential(site, mu):
        index = site.tag[0]
        return onsite(eps) + mu * np.kron(sigma_0, tau_0) + \
            np.kron(sigma_0, tau_0) * lattice_tree.disorder[index, index]

    def hopp(site1, site0, flux):
        index0, index1 = site0.tag[0], site1.tag[0]
        index_neigh = lattice_tree.neighbours[index0].index(index1)
        d, phi, theta = displacement3D_kwant(site1, site0)
        if lattice_tree.K_hopp < 1e-12:
            return hopping(t, lamb, lamb_z, d, phi, theta, lattice_tree.r) * Peierls_kwant(site1, site0, flux, lattice_tree.area)
        else:
            return (hopping(t, lamb, lamb_z, d, phi, theta, lattice_tree.r)  + np.kron(sigma_0, tau_0) *
                lattice_tree.disorder[index0, index_neigh]) * Peierls_kwant(site1, site0, flux, lattice_tree.area)

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
        complete_system = attach_cubic_leads(syst, lattice_tree, latt, param_dict, interface_z=interface_z, interface_r=interface_r)
    else:
        complete_system = syst
    return complete_system

def attach_cubic_leads(scatt_region, lattice_tree, latt, param_dict, interface_z=0.5, interface_r=1.3, leads='topological'):

    # Load parameters into the builder namespace
    if leads == 'topological':
        try:
            t      = param_dict['t']
            eps    = param_dict['eps']
            lamb   = param_dict['lamb']
            lamb_z = param_dict['lamb_z']
        except KeyError as err:
            raise KeyError(f'Parameter error: {err}')
    elif leads == 'metallic':
        t = param_dict['t']
        eps = 0
        lamb = 0
        lamb_z = 0

    # Hoppings and onsite
    def onsite_leads(site, mu_leads):
        return onsite(eps) + mu_leads * np.kron(sigma_0, tau_0)
    def hopp_lead_wire(site1, site0, flux):
        d, phi, theta = displacement3D_kwant(site1, site0)
        return hopping(t, lamb, lamb_z, d, phi, theta, lattice_tree.r) * Peierls_kwant(site1, site0, flux, lattice_tree.area)
    def hopp_x_up(site1, site0, flux):
        return hopping(t, lamb, lamb_z, 1., 0, pi / 2, lattice_tree.r) * Peierls_kwant(site1, site0, flux, lattice_tree.area)
    hopp_z_up = hopping(t, lamb, lamb_z, 1., 0, 0, lattice_tree.r)
    hopp_y_up = hopping(t, lamb, lamb_z, 1., pi / 2, pi / 2, lattice_tree.r)


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

def crystal_nanowire_kwant(Nx, Ny, n_layers, param_dict, mu_leads=0., from_disorder=None):

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
    for i in range(Nx):
        for j in range(Ny):
            for k in range(n_layers):
                index_site = i + Nx * j + (Nx * Ny) * k
                syst[latt(i, j, k)] = onsite(eps) + from_disorder[index_site] * np.kron(sigma_0, tau_0)

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
    onsite_leads = onsite(eps) + mu_leads * np.kron(sigma_0, tau_0)
    lead = kwant.Builder(kwant.TranslationalSymmetry((0, 0, -1)))
    lead[(latt(i, j, 0) for i in range(Nx) for j in range(Ny))] = onsite_leads
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

    def onsite_leads(K):
        return onsite(eps) + mu_leads * np.kron(sigma_0, tau_0) # + np.random.normal(-K, K)

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

#%% Transport functions

def select_perfect_transmission_flux(nanowire, flux0=0.8, flux_end=1.5, Nflux=100, Ef=0., mu_leads=0.):

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

def select_minimal_transmission_flux(nanowire, flux0=0.5, flux_end=1, Nflux=200):

    loger_kwant.trace(f'Calculating flux that gives minimal conductance for this sample...')
    flux = np.linspace(flux0, flux_end, Nflux)

    Gmin = 100
    flux_min = flux0
    for i, phi in enumerate(flux):
        S0 = kwant.smatrix(nanowire, 0.1, params=dict(flux=phi))
        G = S0.transmission(1, 0)
        loger_kwant.trace(f'Flux: {i} / {Nflux - 1}, Conductance: {G :.2e}')
        if G < Gmin:
            Gmin = G
            flux_min = phi
        else:
            pass

    return flux_min, Gmin

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

#%% Topology functions

def spectrum(H, Nsp=None):

    if Nsp is None:
        Nsp = int(len(H) / 2)

    # Spectrum
    energy, eigenstates = np.linalg.eigh(H)
    idx = energy.argsort()
    energy = energy[idx]
    eigenstates = eigenstates[:, idx]

    # OPDM
    U = np.zeros((len(H), len(H)), dtype=np.complex128)
    U[:, 0: Nsp] = eigenstates[:, 0: Nsp]
    rho = U @ np.conj(np.transpose(U))

    return energy, eigenstates, rho

def local_marker(x, y, z, P, S):

    # Operators for calculating the marker
    X, Y, Z = np.repeat(x, 4), np.repeat(y, 4), np.repeat(z, 4)
    X = np.reshape(X, (len(X), 1))
    Y = np.reshape(Y, (len(Y), 1))
    Z = np.reshape(Z, (len(Z), 1))
    PS = P @ S
    XP = X * P
    YP = Y * P
    ZP = Z * P

    # Local chiral marker
    local_marker = np.zeros((len(x), ))
    M = PS @ XP @ YP @ ZP + PS @ ZP @ XP @ YP + PS @ YP @ ZP @ XP - PS @ XP @ ZP @ YP - PS @ ZP @ YP @ XP - PS @ YP @ XP @ ZP
    for i in range(len(x)):
        idx = 4 * i
        local_marker[i] = (8 * pi / 3) * np.imag(np.trace(M[idx: idx + 4, idx: idx + 4]))

    return local_marker

def local_marker_1d(z, P, S, Nx, Ny, Nz):

    # Operators for calculating the marker
    Z = np.diag(np.repeat(z, 4))
    local_marker = np.zeros((Nz, ), dtype=np.complex128)
    M = P @ S @ Z @ P

    # Local marker
    for i in range(Nz):
        step = 4 * Nx * Ny
        idx = step * i
        local_marker[i] = - 2 * np.trace(M[idx: idx + step, idx: idx + step])
        if np.imag(local_marker[i]) > 1e-13:
            raise ValueError('Local marker is complex valued')
        else:
            local_marker[i] = np.real(local_marker[i])
    print(np.sum(local_marker))

    return local_marker

def local_marker_periodic(x_array, y_array, z_array, P, S, Nx, Ny, Nz):

    local_marker = np.zeros((len(x_array),))

    for i, (x, y, z) in enumerate(zip(x_array, y_array, z_array)):

        # Position operators (take the particular site to be as far from the branchcut as possible)
        halfx, halfy, halfz = np.floor(Nx / 2), np.floor(Ny / 2), np.floor(Nz / 2)
        deltax = np.heaviside(x - halfx, 0) * abs(x - (halfx + Nx)) + np.heaviside(halfx - x, 0) * abs(halfx - x)
        deltay = np.heaviside(y - halfy, 0) * abs(y - (halfy + Ny)) + np.heaviside(halfy - y, 0) * abs(halfy - y)
        deltaz = np.heaviside(z - halfz, 0) * abs(z - (halfz + Nz)) + np.heaviside(halfz - z, 0) * abs(halfz - z)
        x, y, z = (x + deltax) % Nx, (y + deltay) % Ny, (z + deltaz) % Nz
        X, Y, Z = np.repeat(x, 4), np.repeat(y, 4), np.repeat(z, 4)
        X = np.reshape(X, (len(X), 1))
        Y = np.reshape(Y, (len(Y), 1))
        Z = np.reshape(Z, (len(Z), 1))
        PS, XP, YP, ZP = P @ S, X * P, Y * P, Z * P

        # Local chiral marker
        M = PS @ XP @ YP @ ZP + PS @ ZP @ XP @ YP + PS @ YP @ ZP @ XP - PS @ XP @ ZP @ YP - PS @ ZP @ YP @ XP - PS @ YP @ XP @ ZP
        idx = 4 * i
        local_marker[i] = (8 * pi / 3) * np.imag(np.trace(M[idx: idx + 4, idx: idx + 4]))

    return local_marker

