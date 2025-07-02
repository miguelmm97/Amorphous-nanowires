"""
This code provides minimal functions for the simulations discussed in the main text. Each function takes in certain
parameters and produces an output data .h5 file with the data of the corresponding simulation. The functions can be
directly run provided the appropriated packages are in the virtual environment provided in environment.yml.

The functions are written to produce a simulation from scratch. In order to reproduce the data for the figures presented
in the main text, one can use these functions combined with the data from the different simulations given in the data folder.

The full repository for the project is public in https://github.com/miguelmm97/Amorphous-nanowires.git
For any questions, typos/errors or further data please write to mfmm@kth.se or miguelmartinezmiquel@gmail.com.
"""

# %% modules set up

# Math and plotting
import scipy.sparse

import sys
from datetime import date

# Kwant
import kwant

# modules
from functions import *
from AmorphousLattice_2d import AmorphousLattice_2d
from AmorphousLattice_3d import AmorphousLattice_3d, take_cut_from_parent_wire
from Amorphous_Nanowires_kwant import promote_to_kwant_nanowire_2d, promote_to_kwant_nanowire_3d, select_perfect_transmission_flux
from topology import local_marker_KPM_bulk, local_marker_per_site_cross_section_KPM

# %% Logging setup
loger_main = logging.getLogger('main')
loger_main.setLevel(logging.INFO)

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
loger_main.addHandler(stream_handler)

#%% Functions for transport

def G_vs_Ef_fully_amorphous(fermi, width, Nx, Ny, L, K_onsite, t, eps, lamb, eta, r, mu_leads, filename, datadir):
    """
    Input:
    fermi -> np.ndarray: Fermi energies at which to do the transport calculation
    width: -> np.ndarray: widths at which to do the transport calculation
    Nx -> int: Number of sites of the nanowire in x direction
    Ny -> int: Number of sites of the nanowire in y direction
    L -> int: Number of sites of the nanowire in z direction
    K_onsite -> float: onsite disorder strength
    t -> float: hopping amplitude as described in the main text
    eps -> float: onsite energy as described in the main text
    lamb -> float: spin-orbit coupling as described in the main text
    eta -> float: hopping amplitude as described in the main text
    r -> float: cutoff distance above which the hopping vanishes
    mu_leads -> float: chemical potential in the leads
    filename -> str: filename of the file where to store the simulations
    datadir -> str: data directory for storing the simulation

    Output:
    None, but produces a file with the data of the simulation.
    """

    # Preallocation
    params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'eta': eta}
    G_0        = np.zeros((len(fermi), len(width)))
    G_half     = np.zeros((len(fermi), len(width)))
    flux_max   = np.zeros((len(width),))
    X          = np.zeros((len(width), Nx * Ny * L))
    Y          = np.zeros((len(width), Nx * Ny * L))
    Z          = np.zeros((len(width), Nx * Ny * L))
    disorder   = np.zeros((len(width), Nx * Ny * L))

    # Calculation
    for i, w in enumerate(width):

        # Amorphous nanowire (the next 4 lines can be replaced with the create_fully_amorphous_nanowire_from_data() function
        # in order to reproduce any specific data files provided.)
        loger_main.info('Generating fully amorphous lattice...')
        lattice = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=L, w=w, r=r)
        lattice.build_lattice()
        lattice.generate_onsite_disorder(K_onsite=K_onsite)
        nanowire = promote_to_kwant_nanowire_3d(lattice, params_dict).finalized()
        X[i, :], Y[i, :], Z[i, :]  = lattice.x, lattice.y, lattice.z
        disorder[i, :] = lattice.onsite_disorder
        loger_main.info('Nanowire promoted to Kwant successfully.')

        # Scanning for flux that gives perfect transmission at the Dirac point
        flux_max[i], Gmax = select_perfect_transmission_flux(nanowire, flux0=0., flux_end=1, Ef=0.01, mu_leads=mu_leads)
        loger_main.info(f'Flux for perfect transmission: {flux_max[i]}, Conductance at the Dirac point: {Gmax}')

        # Conductance calculation as a function of Fermi energy
        for j, Ef in enumerate(fermi):
            S0 = kwant.smatrix(nanowire, 0., params=dict(flux=0., mu=-Ef, mu_leads=mu_leads - Ef))
            G_0[j, i] = S0.transmission(1, 0)
            S1 = kwant.smatrix(nanowire, 0., params=dict(flux=flux_max[i], mu=-Ef, mu_leads=mu_leads - Ef))
            G_half[j, i] = S1.transmission(1, 0)
            loger_main.info(f'Ef: {j} / {len(fermi) - 1}, width: {i} / {len(width) - 1} || G0: {G_0[j, i] :.2f} || Ghalf: {G_half[j, i] :.2f}')

    # Storing data
    filepath = os.path.join(datadir, filename)
    with h5py.File(filepath, 'w') as f:

        # Simulation folder
        simulation = f.create_group('Simulation')
        store_my_data(simulation, 'width',         width)
        store_my_data(simulation, 'K_onsite',      K_onsite)
        store_my_data(simulation, 'fermi',         fermi)
        store_my_data(simulation, 'flux_max',      flux_max)
        store_my_data(simulation, 'G_0',           G_0)
        store_my_data(simulation, 'G_half',        G_half)
        store_my_data(simulation, 'x',             X)
        store_my_data(simulation, 'y',             Y)
        store_my_data(simulation, 'z',             Z)
        store_my_data(simulation, 'disorder',      disorder)

        # Parameters folder
        parameters = f.create_group('Parameters')
        store_my_data(parameters, 'Nx',            Nx)
        store_my_data(parameters, 'Ny',            Ny)
        store_my_data(parameters, 'Nz',            L)
        store_my_data(parameters, 'r',             r)
        store_my_data(parameters, 't',             t)
        store_my_data(parameters, 'eps',           eps)
        store_my_data(parameters, 'lamb',          lamb)
        store_my_data(parameters, 'eta',           eta)
        store_my_data(parameters, 'mu_leads',      mu_leads)

        # Attributes
        attr_my_data(parameters, "Date", str(date.today()))
        attr_my_data(parameters, "Code_path", sys.argv[0])

    loger_main.info('Data saved correctly')


def G_vs_flux_fully_amorphous(flux, width, fermi, Nx, Ny, L, K_onsite, t, eps, lamb, eta, r, mu_leads, filename, datadir):
    """
    Input:
    flux -> np.ndarray: flux at which to do the transport calculation
    width: -> np.ndarray: widths at which to do the transport calculation
    fermi -> float: Fermi energy at which to do the transport calculation
    Nx -> int: Number of sites of the nanowire in x direction
    Ny -> int: Number of sites of the nanowire in y direction
    L -> int: Number of sites of the nanowire in z direction
    K_onsite -> float: onsite disorder strength
    t -> float: hopping amplitude as described in the main text
    eps -> float: onsite energy as described in the main text
    lamb -> float: spin-orbit coupling as described in the main text
    eta -> float: hopping amplitude as described in the main text
    r -> float: cutoff distance above which the hopping vanishes
    mu_leads -> float: chemical potential in the leads
    filename -> str: filename of the file where to store the simulations
    datadir -> str: data directory for storing the simulation

    Output:
    None, but produces a file with the data of the simulation.
    """

    # Preallocation
    params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'eta': eta}
    G        = np.zeros((len(flux), len(width)))
    X        = np.zeros((len(width), Nx * Ny * L))
    Y        = np.zeros((len(width), Nx * Ny * L))
    Z        = np.zeros((len(width), Nx * Ny * L))
    disorder = np.zeros((len(width), Nx * Ny * L))

    # Calculation
    for i, w in enumerate(width):

        # Amorphous nanowire (the next 4 lines can be replaced with the create_fully_amorphous_nanowire_from_data() function
        # in order to reproduce any specific data files provided.)
        loger_main.info('Generating fully amorphous lattice...')
        lattice = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=L, w=w, r=r)
        lattice.build_lattice()
        lattice.generate_onsite_disorder(K_onsite=K_onsite)
        nanowire = promote_to_kwant_nanowire_3d(lattice, params_dict).finalized()
        X[i, :], Y[i, :], Z[i, :] = lattice.x, lattice.y, lattice.z
        disorder[i, :] = lattice.onsite_disorder
        loger_main.info('Nanowire promoted to Kwant successfully.')

        # Conductance calculation as a function of flux
        for j, phi in enumerate(flux):
            S0 = kwant.smatrix(nanowire, 0., params=dict(flux=phi, mu=-fermi, mu_leads=mu_leads - fermi))
            G[j, i] = S0.transmission(1, 0)
            loger_main.info(f'Flux: {j} / {len(flux) - 1}, width: {i} / {len(width) - 1} || G: {G[j, i] :.2f}')

    # Storing data
    filepath = os.path.join(datadir, filename)
    with h5py.File(filepath, 'w') as f:

        # Simulation folder
        simulation = f.create_group('Simulation')
        store_my_data(simulation, 'width',        width)
        store_my_data(simulation, 'flux',         flux)
        store_my_data(simulation, 'K_onsite',     K_onsite)
        store_my_data(simulation, 'fermi',        fermi)
        store_my_data(simulation, 'G',            G)
        store_my_data(simulation, 'x',            X)
        store_my_data(simulation, 'y',            Y)
        store_my_data(simulation, 'z',            Z)
        store_my_data(simulation, 'disorder',     disorder)

        # Parameters folder
        parameters = f.create_group('Parameters')
        store_my_data(parameters, 'Nx',           Nx)
        store_my_data(parameters, 'Ny',           Ny)
        store_my_data(parameters, 'Nz',           L)
        store_my_data(parameters, 'r',            r)
        store_my_data(parameters, 't',            t)
        store_my_data(parameters, 'eps',          eps)
        store_my_data(parameters, 'lamb',         lamb)
        store_my_data(parameters, 'eta',          eta)
        store_my_data(parameters, 'mu_leads',     mu_leads)

        # Attributes
        attr_my_data(parameters, "Date", str(date.today()))
        attr_my_data(parameters, "Code_path", sys.argv[0])

    loger_main.info('Data saved correctly')


def G_vs_L_fully_amorphous(flux, width, fermi, Nx, Ny, L, K_onsite, t, eps, lamb, eta, r, mu_leads, filename, datadir):
    """
    Input:
    flux -> np.ndarray: flux at which to do the transport calculation
    width: -> float: width at which to do the transport calculation
    fermi -> float: Fermi energy at which to do the transport calculation
    Nx -> int: Number of sites of the nanowire in x direction
    Ny -> int: Number of sites of the nanowire in y direction
    L ->  np.ndarray: lengths of the nanowire for which to do the transport calculation
    K_onsite -> float: onsite disorder strength
    t -> float: hopping amplitude as described in the main text
    eps -> float: onsite energy as described in the main text
    lamb -> float: spin-orbit coupling as described in the main text
    eta -> float: hopping amplitude as described in the main text
    r -> float: cutoff distance above which the hopping vanishes
    mu_leads -> float: chemical potential in the leads
    filename -> str: filename of the file where to store the simulations
    datadir -> str: data directory for storing the simulation

    Output:
    None, but produces a file with the data of the simulation.
    """

    # Preallocation
    params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'eta': eta}
    G        = np.zeros((len(flux), len(L)))
    disorder = np.zeros((Nx * Ny * np.max(L), ))

    # Calculation
    # Amorphous parent lattice (the next 4 lines can be replaced with the create_fully_amorphous_nanowire_from_data() function
    # in order to reproduce any specific data files provided.)
    loger_main.info('Generating fully amorphous lattice...')
    parent_lattice = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=np.max(L), w=width, r=r)
    parent_lattice.build_lattice()
    parent_lattice.generate_onsite_disorder(K_onsite=K_onsite)
    X, Y, Z = parent_lattice.x, parent_lattice.y, parent_lattice.z

    for i, l in enumerate(L):

        # Selecting different cuts from the parent nanowire
        lattice = take_cut_from_parent_wire(parent_lattice, Nz_new=l, keep_disorder=True)
        nanowire = promote_to_kwant_nanowire_3d(lattice, params_dict).finalized()

        # Calculating conductance
        for j, phi in enumerate(flux):
            S = kwant.smatrix(nanowire, 0., params=dict(flux=phi, mu=-fermi, mu_leads=mu_leads - fermi))
            G[j, i] = S.transmission(1, 0)
            loger_main.info(f'L: {j} / {len(L) - 1}, flux: {j} / {len(flux) - 1} || G: {G[j, i] :.2f}')

    # Storing data
    filepath = os.path.join(datadir, filename)
    with h5py.File(filepath, 'w') as f:

        # Simulation folder
        simulation = f.create_group('Simulation')
        store_my_data(simulation, 'width',        width)
        store_my_data(simulation, 'flux',         flux)
        store_my_data(simulation, 'K_onsite',     K_onsite)
        store_my_data(simulation, 'fermi',        fermi)
        store_my_data(simulation, 'G',            G)
        store_my_data(simulation, 'L',            L)
        store_my_data(simulation, 'x',            X)
        store_my_data(simulation, 'y',            Y)
        store_my_data(simulation, 'z',            Z)
        store_my_data(simulation, 'disorder',     disorder)

        # Parameters folder
        parameters = f.create_group('Parameters')
        store_my_data(parameters, 'Nx',           Nx)
        store_my_data(parameters, 'Ny',           Ny)
        store_my_data(parameters, 'r',            r)
        store_my_data(parameters, 't',            t)
        store_my_data(parameters, 'eps',          eps)
        store_my_data(parameters, 'lamb',         lamb)
        store_my_data(parameters, 'eta',          eta)
        store_my_data(parameters, 'mu_leads',     mu_leads)

        # Attributes
        attr_my_data(parameters, "Date", str(date.today()))
        attr_my_data(parameters, "Code_path", sys.argv[0])

    loger_main.info('Data saved correctly')


def G_vs_Ef_layer_amorphous(fermi, width, Nx, Ny, L, K_onsite, t, eps, lamb, eta, r, mu_leads, filename, datadir):
    """
    Input:
    fermi -> np.ndarray: Fermi energies at which to do the transport calculation
    width: -> np.ndarray: widths at which to do the transport calculation
    Nx -> int: Number of sites of the nanowire in x direction
    Ny -> int: Number of sites of the nanowire in y direction
    L -> int: Number of sites of the nanowire in z direction
    K_onsite -> float: onsite disorder strength
    t -> float: hopping amplitude as described in the main text
    eps -> float: onsite energy as described in the main text
    lamb -> float: spin-orbit coupling as described in the main text
    eta -> float: hopping amplitude as described in the main text
    r -> float: cutoff distance above which the hopping vanishes
    mu_leads -> float: chemical potential in the leads
    filename -> str: filename of the file where to store the simulations
    datadir -> str: data directory for storing the simulation

    Output:
    None, but produces a file with the data of the simulation.
    """

    # Preallocation
    params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'eta': eta}
    G_0        = np.zeros((len(fermi), len(width)))
    G_half     = np.zeros((len(fermi), len(width)))
    flux_max   = np.zeros((len(width),))
    X          = np.zeros((len(width), Nx * Ny))
    Y          = np.zeros((len(width), Nx * Ny))
    disorder   = np.zeros((len(width), Nx * Ny))

    # Calculation
    for i, w in enumerate(width):

        # Amorphous nanowire (the next 4 lines can be replaced with the create_layer_amorphous_nanowire_from_data() function
        # in order to reproduce any specific data files provided.)
        loger_main.info('Generating layer amorphous lattice...')
        lattice = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=w, r=r)
        lattice.build_lattice()
        lattice.generate_onsite_disorder(K_onsite=K_onsite)
        nanowire = promote_to_kwant_nanowire_2d(lattice, L,  params_dict).finalized()
        X[i, :], Y[i, :]  = lattice.x, lattice.y
        disorder[i, :] = lattice.onsite_disorder
        loger_main.info('Nanowire promoted to Kwant successfully.')

        # Scanning for flux that gives perfect transmission at the Dirac point
        flux_max[i], Gmax = select_perfect_transmission_flux(nanowire, flux0=0., flux_end=1, Ef=0.01, mu_leads=mu_leads)
        loger_main.info(f'Flux for perfect transmission: {flux_max[i]}, Conductance at the Dirac point: {Gmax}')

        # Conductance calculation as a function of Fermi energy
        for j, Ef in enumerate(fermi):
            S0 = kwant.smatrix(nanowire, 0., params=dict(flux=0., mu=-Ef, mu_leads=mu_leads - Ef))
            G_0[j, i] = S0.transmission(1, 0)
            S1 = kwant.smatrix(nanowire, 0., params=dict(flux=flux_max[i], mu=-Ef, mu_leads=mu_leads - Ef))
            G_half[j, i] = S1.transmission(1, 0)
            loger_main.info(f'Ef: {j} / {len(fermi) - 1}, width: {i} / {len(width) - 1} || G0: {G_0[j, i] :.2f} || Ghalf: {G_half[j, i] :.2f}')

    # Storing data
    filepath = os.path.join(datadir, filename)
    with h5py.File(filepath, 'w') as f:

        # Simulation folder
        simulation = f.create_group('Simulation')
        store_my_data(simulation, 'width',         width)
        store_my_data(simulation, 'K_onsite',      K_onsite)
        store_my_data(simulation, 'fermi',         fermi)
        store_my_data(simulation, 'flux_max',      flux_max)
        store_my_data(simulation, 'G_0',           G_0)
        store_my_data(simulation, 'G_half',        G_half)
        store_my_data(simulation, 'x',             X)
        store_my_data(simulation, 'y',             Y)
        store_my_data(simulation, 'disorder',      disorder)

        # Parameters folder
        parameters = f.create_group('Parameters')
        store_my_data(parameters, 'Nx',            Nx)
        store_my_data(parameters, 'Ny',            Ny)
        store_my_data(parameters, 'Nz',            L)
        store_my_data(parameters, 'r',             r)
        store_my_data(parameters, 't',             t)
        store_my_data(parameters, 'eps',           eps)
        store_my_data(parameters, 'lamb',          lamb)
        store_my_data(parameters, 'eta',           eta)
        store_my_data(parameters, 'mu_leads',      mu_leads)

        # Attributes
        attr_my_data(parameters, "Date", str(date.today()))
        attr_my_data(parameters, "Code_path", sys.argv[0])

    loger_main.info('Data saved correctly')


def G_vs_flux_layer_amorphous(flux, width, fermi, Nx, Ny, L, K_onsite, t, eps, lamb, eta, r, mu_leads, filename, datadir):
    """
    Input:
    flux -> np.ndarray: flux at which to do the transport calculation
    width: -> np.ndarray: widths at which to do the transport calculation
    fermi -> float: Fermi energy at which to do the transport calculation
    Nx -> int: Number of sites of the nanowire in x direction
    Ny -> int: Number of sites of the nanowire in y direction
    L -> int: Number of sites of the nanowire in z direction
    K_onsite -> float: onsite disorder strength
    t -> float: hopping amplitude as described in the main text
    eps -> float: onsite energy as described in the main text
    lamb -> float: spin-orbit coupling as described in the main text
    eta -> float: hopping amplitude as described in the main text
    r -> float: cutoff distance above which the hopping vanishes
    mu_leads -> float: chemical potential in the leads
    filename -> str: filename of the file where to store the simulations
    datadir -> str: data directory for storing the simulation

    Output:
    None, but produces a file with the data of the simulation.
    """

    # Preallocation
    params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'eta': eta}
    G           = np.zeros((len(flux), len(width)))
    X           = np.zeros((len(width), Nx * Ny))
    Y           = np.zeros((len(width), Nx * Ny))
    disorder    = np.zeros((len(width), Nx * Ny))

    # Calculation
    for i, w in enumerate(width):

        # Amorphous nanowire(the next 4 lines can be replaced with the create_layer_amorphous_nanowire_from_data() function
        # in order to reproduce any specific data files provided.)
        loger_main.info('Generating layer amorphous lattice...')
        lattice = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=w, r=r)
        lattice.build_lattice()
        lattice.generate_onsite_disorder(K_onsite=K_onsite)
        nanowire = promote_to_kwant_nanowire_2d(lattice, L,  params_dict).finalized()
        X[i, :], Y[i, :]  = lattice.x, lattice.y
        disorder[i, :] = lattice.onsite_disorder
        loger_main.info('Nanowire promoted to Kwant successfully.')

        # Conductance calculation as a function of flux
        for j, phi in enumerate(flux):
            S0 = kwant.smatrix(nanowire, 0., params=dict(flux=phi, mu=-fermi, mu_leads=mu_leads - fermi))
            G[j, i] = S0.transmission(1, 0)
            loger_main.info(f'Flux: {j} / {len(flux) - 1}, width: {i} / {len(width) - 1} || G: {G[j, i] :.2f}')

    # Storing data
    filepath = os.path.join(datadir, filename)
    with h5py.File(filepath, 'w') as f:

        # Simulation folder
        simulation = f.create_group('Simulation')
        store_my_data(simulation, 'width',        width)
        store_my_data(simulation, 'flux',         flux)
        store_my_data(simulation, 'K_onsite',     K_onsite)
        store_my_data(simulation, 'fermi',        fermi)
        store_my_data(simulation, 'G',            G)
        store_my_data(simulation, 'x',            X)
        store_my_data(simulation, 'y',            Y)
        store_my_data(simulation, 'disorder',     disorder)

        # Parameters folder
        parameters = f.create_group('Parameters')
        store_my_data(parameters, 'Nx',           Nx)
        store_my_data(parameters, 'Ny',           Ny)
        store_my_data(parameters, 'Nz',           L)
        store_my_data(parameters, 'r',            r)
        store_my_data(parameters, 't',            t)
        store_my_data(parameters, 'eps',          eps)
        store_my_data(parameters, 'lamb',         lamb)
        store_my_data(parameters, 'eta',          eta)
        store_my_data(parameters, 'mu_leads',     mu_leads)

        # Attributes
        attr_my_data(parameters, "Date", str(date.today()))
        attr_my_data(parameters, "Code_path", sys.argv[0])

    loger_main.info('Data saved correctly')


def DoS(flux, width, fermi, Nx, Ny, L, x, y, z, disorder, t, eps, lamb, eta, r, mu_leads, filename, datadir):
    """
    Input:
    flux -> float: flux at which to calculate the DoS
    width: -> float: width at which to calculate the DoS
    fermi -> float: Fermi energy at which to calculate the DoS
    Nx -> int: Number of sites of the nanowire in x direction
    Ny -> int: Number of sites of the nanowire in y direction
    L -> int: Number of sites of the nanowire in z direction
    x -> np.ndarray: x coordinate of the nanowire sites
    y -> np.ndarray: y coordinate of the nanowire sites
    z -> np.ndarray: z coordinate of the nanowire sites
    K_onsite -> float: onsite disorder strength
    t -> float: hopping amplitude as described in the main text
    eps -> float: onsite energy as described in the main text
    lamb -> float: spin-orbit coupling as described in the main text
    eta -> float: hopping amplitude as described in the main text
    r -> float: cutoff distance above which the hopping vanishes
    mu_leads -> float: chemical potential in the leads
    filename -> str: filename of the file where to store the simulations
    datadir -> str: data directory for storing the simulation

    Output:
    None, but produces a file with the data of the simulation.
    """

    params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'eta': eta}

    # Amorphous nanowire
    loger_main.info('Generating lattice for the topological state...')
    lattice = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=L, w=width, r=r)
    lattice.set_configuration(x, y, z)
    lattice.set_disorder(disorder)
    lattice.build_lattice()
    nanowire = promote_to_kwant_nanowire_3d(lattice, params_dict).finalized()
    loger_main.info('Nanowire promoted to Kwant successfully.')

    # Calculation
    # Scattering state
    loger_main.info('Calculating scattering wave functions...')
    state = kwant.wave_function(nanowire, params=dict(flux=flux, mu=-fermi, mu_leads=mu_leads - fermi))

    # Total DoS through cuts
    R = np.linspace(2, Nx / 2, 10)
    DoS_R = np.zeros((len(R), ))
    for i, n in enumerate(R):
        def bulk(site):
            x, y = site.pos[0] - 0.5 * (Nx - 1), site.pos[1] - 0.5 * (Ny - 1)
            cond1 = np.abs(x) < n
            cond2 = np.abs(y) < n
            return cond1 * cond2
        total_density_operator = kwant.operator.Density(nanowire, where=bulk, sum=True)
        DoS_R[i] = total_density_operator(state(0)[0])
    DoS_R = DoS_R / DoS_R[-1]

    # Local DoS
    def bulk(syst, rad):
        new_sites_x = tuple([site for site in syst.id_by_site if 0 < (site.pos[0] - 0.5 * (Nx - 1)) < rad])
        new_sites = tuple([site for site in new_sites_x if 0 < (site.pos[1] - 0.5 * (Ny - 1)) < rad])
        new_sites_pos = np.array([site.pos for site in new_sites])
        return new_sites, new_sites_pos
    bulk_sites, bulk_pos = bulk(nanowire, Nx / 2 + 1)
    density_operator = kwant.operator.Density(nanowire, where=bulk_sites, sum=False)
    local_DoS = density_operator(state(0)[0])
    local_DoS = local_DoS / np.sum(local_DoS)


    # Storing data
    filepath = os.path.join(datadir, filename)
    with h5py.File(filepath, 'w') as f:
        # Simulation folder
        simulation = f.create_group('Simulation')
        store_my_data(simulation, 'R', R)
        store_my_data(simulation, 'DoS_R', DoS_R)
        store_my_data(simulation, 'bulk_sites', bulk_sites)
        store_my_data(simulation, 'bulk_pos', bulk_pos)
        store_my_data(simulation, 'local_DoS', local_DoS)
        store_my_data(simulation, 'flux', flux)
        store_my_data(simulation, 'width', width)
        store_my_data(simulation, 'fermi', fermi)


        # Parameters folder
        parameters = f.create_group('Parameters')
        store_my_data(parameters, 'Nx', Nx)
        store_my_data(parameters, 'Ny', Ny)
        store_my_data(parameters, 'Nz', L)
        store_my_data(parameters, 'r ', r)
        store_my_data(parameters, 't ', t)
        store_my_data(parameters, 'eps', eps)
        store_my_data(parameters, 'lamb', lamb)
        store_my_data(parameters, 'eta', eta)
        store_my_data(parameters, 'mu_leads', mu_leads)

        # Attributes
        attr_my_data(parameters, "Date", str(date.today()))
        attr_my_data(parameters, "Code_path", sys.argv[0])

    loger_main.info('Data saved correctly')


def marker_vs_width(width, fermi, N, L, K_onsite, t, eps, lamb, eta, r, num_moments, num_vecs, cutoff, filename, datadir):
    """
    Input:
    width: -> float: width of the nanowire
    fermi -> float: Fermi energy at which to calculate the local marker
    N -> int: Number of sites of the nanowire in x and y direction
    L -> int: Number of sites of the nanowire in z direction
    K_onsite -> float: onsite disorder strength
    t -> float: hopping amplitude as described in the main text
    eps -> float: onsite energy as described in the main text
    lamb -> float: spin-orbit coupling as described in the main text
    eta -> float: hopping amplitude as described in the main text
    r -> float: cutoff distance above which the hopping vanishes
    num_moments -> int: number of moments to use on the KPM expansion
    num_vecs -> int: number of random vectors to use in the stochastic trace evaluation
    cutoff -> float: fraction of the nanowire we average over, taking the origin at the center of the nanowire (explained
                     in the main text)
    filename -> str: filename of the file where to store the simulations
    datadir -> str: data directory for storing the simulation

    Output:
    None, but produces a file with the data of the simulation.
    """

    # Preallocation
    bulk_marker = np.zeros((len(width),))
    params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'eta': eta}
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    # Calculation
    for i, w in enumerate(width):
        loger_main.info(f'Generating lattice for w: {w}')
        lattice = AmorphousLattice_3d(Nx=N, Ny=N, Nz=L, w=w, r=r)
        lattice.build_lattice()
        lattice.generate_onsite_disorder(K_onsite=K_onsite)
        nanowire = promote_to_kwant_nanowire_3d(lattice, params_dict, attach_leads=False).finalized()
        S = scipy.sparse.kron(np.eye(N * N * L), np.kron(sigma_z, sigma_z), format='csr')

        # Local marker through KPM + Stochastic trace algorithm
        loger_main.info('Calculating bulk marker through KPM algorithm')
        bulk_marker[i] = local_marker_KPM_bulk(nanowire, S, N, N, L, Ef=fermi, num_moments=num_moments, num_vecs=num_vecs, cutoff=cutoff)
        loger_main.info(f'width: {i}/{len(width) - 1}, marker KPM: {bulk_marker[i] :.5f}')


    # Storing data
    filepath = os.path.join(datadir, filename)
    with h5py.File(filepath, 'w') as f:
        # Simulation folder
        simulation = f.create_group('Simulation')
        store_my_data(simulation, 'bulk_marker',    bulk_marker)
        store_my_data(simulation, 'width',          width)
        store_my_data(simulation, 'num_moments',    num_moments)
        store_my_data(simulation, 'num_vecs',       num_vecs)
        store_my_data(simulation, 'cutoff',         cutoff)

        # Parameters folder
        parameters = f.create_group('Parameters')
        store_my_data(parameters, 'N',              N)
        store_my_data(parameters, 'Nz',             L)
        store_my_data(parameters, 'r ',             r)
        store_my_data(parameters, 't ',             t)
        store_my_data(parameters, 'eps',            eps)
        store_my_data(parameters, 'lamb',           lamb)
        store_my_data(parameters, 'eta',            eta)
        store_my_data(parameters, 'cutoff',         cutoff)

        # Attributes
        attr_my_data(parameters, "Date", str(date.today()))
        attr_my_data(parameters, "Code_path", sys.argv[0])

    loger_main.info('Data saved correctly')


def marker_cross_section(width, fermi, N, L, K_onsite, t, eps, lamb, eta, r, num_moments, z0, z1, filename, datadir):
    """
    Input:
    width: -> float: width of the nanowire
    fermi -> float: Fermi energy at which to calculate the local marker
    N -> int: Number of sites of the nanowire in x and y direction
    L -> int: Number of sites of the nanowire in z direction
    K_onsite -> float: onsite disorder strength
    t -> float: hopping amplitude as described in the main text
    eps -> float: onsite energy as described in the main text
    lamb -> float: spin-orbit coupling as described in the main text
    eta -> float: hopping amplitude as described in the main text
    r -> float: cutoff distance above which the hopping vanishes
    num_moments -> int: number of moments to use on the KPM expansion
    z0, z1 -> float: z positions between which we calculate the local marker on every site
    filename -> str: filename of the file where to store the simulations
    datadir -> str: data directory for storing the simulation

    Output:
    None, but produces a file with the data of the simulation.
    """

    # Preallocation
    params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'eta': eta}
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    # Amorphous nanowire
    loger_main.info(f'Generating lattice')
    lattice = AmorphousLattice_3d(Nx=N, Ny=N, Nz=L, w=width, r=r)
    lattice.build_lattice()
    lattice.generate_onsite_disorder(K_onsite=K_onsite)
    nanowire = promote_to_kwant_nanowire_3d(lattice, params_dict, attach_leads=False).finalized()
    S = scipy.sparse.kron(np.eye(N * N * L), np.kron(sigma_z, sigma_z), format='csr')

    # Local marker through KPM + Stochastic trace algorithm
    loger_main.info('Calculating bulk marker through KPM algorithm')
    local_marker, x, y, z = local_marker_per_site_cross_section_KPM(nanowire, S, N, N, L, z0, z1, Ef=fermi, num_moments=num_moments)

    # Storing data
    filepath = os.path.join(datadir, filename)
    with h5py.File(filepath, 'w') as f:

        # Simulation folder
        simulation = f.create_group('Simulation')
        store_my_data(simulation, 'local_marker',    local_marker)
        store_my_data(simulation, 'width',           width)
        store_my_data(simulation, 'x',               x)
        store_my_data(simulation, 'y',               y)
        store_my_data(simulation, 'z',               z)
        store_my_data(simulation, 'z0',              z0)
        store_my_data(simulation, 'z1',              z1)

        # Parameters folder
        parameters = f.create_group('Parameters')
        store_my_data(parameters, 'Nx',              N)
        store_my_data(parameters, 'Ny',              N)
        store_my_data(parameters, 'Nz',              L)
        store_my_data(parameters, 'r ',              r)
        store_my_data(parameters, 't ',              t)
        store_my_data(parameters, 'eps',             eps)
        store_my_data(parameters, 'lamb',            lamb)
        store_my_data(parameters, 'eta',             eta)

        # Attributes
        attr_my_data(parameters, "Date", str(date.today()))
        attr_my_data(parameters, "Code_path", sys.argv[0])

    loger_main.info('Data saved correctly')


def create_fully_amorphous_nanowire_from_data(x, y, z, width, Nx, Ny, L, r):
    """
    Input:
    x, y, z -> np.ndarray: x, y, z positions of the stored nanowire
    width: -> float: width of the amorphous distribution
    Nx -> int: Number of sites of the nanowire in x direction
    Ny -> int: Number of sites of the nanowire in y direction
    L -> int: Number of sites of the nanowire in z direction
    r -> float: cutoff distance above which the hopping vanishes

    Output:
    lattice -> AmorphousLattice_3d: Structure for the nanowire
    """
    lattice = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=L, w=width, r=r)
    lattice.set_configuration(x, y, z)
    lattice.build_lattice()
    return lattice


def create_layer_amorphous_nanowire_from_data(x, y, width, Nx, Ny, L, r):
    """
    Input:
    x, y -> np.ndarray: x, y positions of the stored nanowire
    width: -> float: width of the amorphous distribution
    Nx -> int: Number of sites of the nanowire in x direction
    Ny -> int: Number of sites of the nanowire in y direction
    L -> int: Number of sites of the nanowire in z direction
    r -> float: cutoff distance above which the hopping vanishes

    Output:
    lattice -> AmorphousLattice_3d: Structure for the nanowire
    """
    lattice = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=width, r=r)
    lattice.set_configuration(x, y)
    lattice.build_lattice()
    return lattice

