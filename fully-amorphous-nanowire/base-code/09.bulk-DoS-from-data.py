#%% modules set up

# Math and plotting
import numpy as np

# Kwant
import kwant

# modules
from modules.functions import *
from modules.AmorphousLattice_3d import AmorphousLattice_3d
from modules.FullyAmorphousWire_kwant import promote_to_kwant_nanowire3d, select_perfect_transmission_flux

import sys
from datetime import date

#%% Logging setup
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


#%% Loading data
file_list = ['Exp25.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-cond-vs-flux-fully-amorphous')

# Parameters
Ef           = data_dict[file_list[0]]['Parameters']['Ef']
Nx           = data_dict[file_list[0]]['Parameters']['Nx']
Ny           = data_dict[file_list[0]]['Parameters']['Ny']
Nz           = data_dict[file_list[0]]['Parameters']['Nz']
r            = data_dict[file_list[0]]['Parameters']['r']
t            = data_dict[file_list[0]]['Parameters']['t']
eps          = data_dict[file_list[0]]['Parameters']['eps']
lamb         = data_dict[file_list[0]]['Parameters']['lamb']
lamb_z       = data_dict[file_list[0]]['Parameters']['lamb_z']
mu_leads     = data_dict[file_list[0]]['Parameters']['mu_leads']
params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}

# Simulation data
flux          = data_dict[file_list[0]]['Simulation']['flux']
G_array       = data_dict[file_list[0]]['Simulation']['G_array']
width         = data_dict[file_list[0]]['Simulation']['width']
x             = data_dict[file_list[0]]['Simulation']['x']
y             = data_dict[file_list[0]]['Simulation']['y']
z             = data_dict[file_list[0]]['Simulation']['z']

# Variables
flux_top = flux[39]
w_top = width[2]
index_top = 2
flux_loc = flux[116]
w_loc = width[1]
index_loc = 1
flux_bound = flux[161]
w_bound = width[-1]
index_bound = -1

#%% Main: Different nanowires
loger_main.info('Generating lattice for the topological state...')
lattice = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=Nz, w=w_top, r=r)
lattice.set_configuration(x[index_top, :], y[index_top, :], z[index_top, :])
lattice.build_lattice(restrict_connectivity=False)
lattice.generate_disorder(K_hopp=0., K_onsite=0.)
nanowire_top = promote_to_kwant_nanowire3d(lattice, params_dict, mu_leads=mu_leads).finalized()
site_pos_top = np.array([site.pos for site in nanowire_top.id_by_site])
loger_main.info('Nanowire promoted to Kwant successfully.')

loger_main.info('Generating lattice for the localised state...')
lattice = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=Nz, w=w_loc, r=r)
lattice.set_configuration(x[index_loc, :], y[index_loc, :], z[index_loc, :])
lattice.build_lattice(restrict_connectivity=False)
lattice.generate_disorder(K_hopp=0., K_onsite=0.)
nanowire_loc = promote_to_kwant_nanowire3d(lattice, params_dict, mu_leads=mu_leads).finalized()
site_pos_loc = np.array([site.pos for site in nanowire_loc.id_by_site])
loger_main.info('Nanowire promoted to Kwant successfully.')

loger_main.info('Generating lattice for the bound state...')
lattice = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=Nz, w=w_bound, r=r)
lattice.set_configuration(x[index_bound, :], y[index_bound, :], z[index_bound, :])
lattice.build_lattice(restrict_connectivity=False)
lattice.generate_disorder(K_hopp=0., K_onsite=0.)
nanowire_bound = promote_to_kwant_nanowire3d(lattice, params_dict, mu_leads=mu_leads).finalized()
site_pos_bound = np.array([site.pos for site in nanowire_bound.id_by_site])
loger_main.info('Nanowire promoted to Kwant successfully.')



#%% Main: Wavefunctions for the different states
loger_main.info('Calculating scattering wave functions...')
top_state = kwant.wave_function(nanowire_top, energy=Ef[0], params=dict(flux=flux_top))
loc_state = kwant.wave_function(nanowire_loc, energy=Ef[0], params=dict(flux=flux_loc))
bound_state = kwant.wave_function(nanowire_bound, energy=Ef[0], params=dict(flux=flux_bound))
loger_main.info('Scattering wave functions calculated successfully')

system_list = [nanowire_top, nanowire_loc, nanowire_bound]
state_list = [top_state, loc_state, bound_state]


#%% Main: Total DoS through cuts
N = np.linspace(2, Nx / 2, 10)
bulk_tot_density = np.zeros((len(state_list), len(N)))

loger_main.info('Calculating total local bulk DoS...')
for i, n in enumerate(N):
    loger_main.info(f'Section {i} / {len(N)}')
    def bulk(site):
        x, y = site.pos[0] - 0.5 * Nx, site.pos[1] - 0.5 * Ny
        z = site.pos[2]
        cond3 = 4 < z
        cond4 = z < Nz - 4
        if i < len(N) - 1:
            return ((np.abs(x) and np.abs(y)) < n) and cond3 and cond4
        else:
            return (np.abs(x) and np.abs(y)) < n

    for j, (nanowire, state) in enumerate(zip(system_list, state_list)):
        total_density_operator = kwant.operator.Density(nanowire, where=bulk, sum=True)
        bulk_tot_density[j, i] = total_density_operator(state(0)[0])

# Normalization in percentage
for i in range(len(state_list)):
    bulk_tot_density[i, :] = bulk_tot_density[i, :] / bulk_tot_density[i, -1]


#%% Main: Local density through a cut
nx = np.linspace(2, Nx / 2, 3)
bulk_density = {'top': {}, 'loc': {}, 'bound': {}}
cut_pos = {'top': {}, 'loc': {}, 'bound': {}}
key_list = ['top', 'loc', 'bound']

# Cuts of the wires
def bulk(syst, rad):
    new_sites_x = tuple([site for site in syst.id_by_site if np.abs(site.pos[0] - 0.5 * Nx) < rad])
    new_sites = tuple([site for site in new_sites_x if np.abs(site.pos[1] - 0.5 * Ny) < rad])
    new_sites_pos = np.array([site.pos for site in new_sites])
    return new_sites, new_sites_pos

# DoS
loger_main.info('Calculating local bulk DoS...')
for (key, nanowire, state) in zip(key_list, system_list, state_list):
    for i, n in enumerate(nx):
        loger_main.info(f'Section {i} / {len(nx)}')
        cut_sites, cut_pos[key][i] = bulk(nanowire, n)
        density_operator = kwant.operator.Density(nanowire, where=cut_sites, sum=False)
        bulk_density[key][i] = density_operator(state(0)[0])

# Normalisation
for state in bulk_density.keys():
    for cut in bulk_density[state].keys():
        bulk_density[state][cut] = bulk_density[state][cut] / np.sum(bulk_density[state][len(nx) - 1])



#%% Saving data
data_dir = '/home/mfmm/Projects/amorphous-nanowires/data/data-bulk-dos'
file_list = os.listdir(data_dir)
expID = get_fileID(file_list, common_name='Exp')
filename = '{}{}{}'.format('Exp', expID, '.h5')
filepath = os.path.join(data_dir, filename)


with h5py.File(filepath, 'w') as f:

    # Simulation folder
    simulation = f.create_group('Simulation')
    DoS_top = simulation.create_group('DoS_top')
    DoS_loc = simulation.create_group('DoS_loc')
    DoS_bound = simulation.create_group('DoS_bound')
    cuts_top = simulation.create_group('cuts_top')
    cuts_loc = simulation.create_group('cuts_loc')
    cuts_bound = simulation.create_group('cuts_bound')

    store_my_data(simulation,  'G_array',          G_array)
    store_my_data(simulation,    'flux',           flux)
    store_my_data(simulation,   'width',           width)
    store_my_data(simulation,       'N',           N)
    store_my_data(simulation,      'nx',           nx)
    store_my_data(simulation, 'bulk_tot_density', bulk_tot_density)

    store_my_dict(simulation['DoS_top'],  bulk_density['top'])
    store_my_dict(simulation['cuts_top'],  cut_pos['top'])
    store_my_dict(simulation['DoS_loc'], bulk_density['loc'])
    store_my_dict(simulation['cuts_loc'], cut_pos['loc'])
    store_my_dict(simulation['DoS_bound'], bulk_density['bound'])
    store_my_dict(simulation['cuts_bound'], cut_pos['bound'])


    # Parameters folder
    parameters = f.create_group('Parameters')
    store_my_data(parameters, 'flux_top',   flux_top)
    store_my_data(parameters, 'flux_loc',   flux_loc)
    store_my_data(parameters, 'flux_bound', flux_bound)
    store_my_data(parameters, 'idx_top',    index_top)
    store_my_data(parameters, 'idx_loc',    index_loc)
    store_my_data(parameters, 'idx_bound',  index_bound)
    store_my_data(parameters, 'Ef',         Ef)
    store_my_data(parameters, 'Nx',         Nx)
    store_my_data(parameters, 'Ny',         Ny)
    store_my_data(parameters, 'Nz',         Nz)
    store_my_data(parameters, 'r ',         r)
    store_my_data(parameters, 't ',         t)
    store_my_data(parameters, 'eps',        eps)
    store_my_data(parameters, 'lamb',       lamb)
    store_my_data(parameters, 'lamb_z',     lamb_z)
    store_my_data(parameters, 'mu_leads',   mu_leads)

    # Attributes
    attr_my_data(parameters, "Date",       str(date.today()))
    attr_my_data(parameters, "Code_path",  sys.argv[0])

loger_main.info('Data saved correctly')









