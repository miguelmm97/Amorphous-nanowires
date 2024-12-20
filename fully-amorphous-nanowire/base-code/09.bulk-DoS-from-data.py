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
file_list = ['Exp22.h5']
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
idx1, idx2 = 100, 144 # 25, 33
flux1 = flux[idx1]
flux2 = flux[idx2]
#%% Main

loger_main.info('Generating fully amorphous lattice...')
lattice = AmorphousLattice_3d(Nx=Nx, Ny=Ny, Nz=Nz, w=width, r=r)
lattice.set_configuration(x, y, z)
lattice.build_lattice(restrict_connectivity=False)
lattice.generate_disorder(K_hopp=0., K_onsite=0.)
nanowire = promote_to_kwant_nanowire3d(lattice, params_dict, mu_leads=mu_leads).finalized()
site_pos = np.array([site.pos for site in nanowire.id_by_site])
loger_main.info('Nanowire promoted to Kwant successfully.')

# Calculating the scattering wavefunctions at certain energies
loger_main.info('Calculating scattering wave functions...')
state1 = kwant.wave_function(nanowire, energy=Ef[0], params=dict(flux=flux1))
state2 = kwant.wave_function(nanowire, energy=Ef[0], params=dict(flux=flux2))
loger_main.info('Scattering wave functions calculated successfully')

# Total local density through a cut
N = np.linspace(2, Nx / 2, 10)
bulk_tot_density1 = np.zeros(N.shape)
bulk_tot_density2 = np.zeros(N.shape)

loger_main.info('Calculating total local bulk DoS...')
for i, n in enumerate(N):
    def bulk(site):
        x, y = site.pos[0] - 0.5 * Nx, site.pos[1] - 0.5 * Ny
        return (np.abs(x) and np.abs(y)) < n
    loger_main.info(f'Section {i} / {len(N)}')
    total_density_operator = kwant.operator.Density(nanowire, where=bulk, sum=True)
    density_operator = kwant.operator.Density(nanowire, where=bulk, sum=False)
    bulk_tot_density1[i] = total_density_operator(state1(0)[0])
    bulk_tot_density2[i] = total_density_operator(state2(0)[0])
bulk_tot_density1 = bulk_tot_density1 / bulk_tot_density1[-1]
bulk_tot_density2 = bulk_tot_density2 / bulk_tot_density2[-1]


# Local density through a cut
N_local = np.linspace(2, Nx / 2, 3)
bulk_density1 = {}
bulk_density2 = {}
cut_pos = {}

def bulk(syst, rad):
    new_sites_x = tuple([site for site in syst.id_by_site if np.abs(site.pos[0] - 0.5 * Nx) < rad])
    new_sites = tuple([site for site in new_sites_x if np.abs(site.pos[1] - 0.5 * Ny) < rad])
    new_sites_pos = np.array([site.pos for site in new_sites])
    return new_sites, new_sites_pos

loger_main.info('Calculating local bulk DoS...')
for i, n in enumerate(N_local):
    cut_sites, cut_pos[i] = bulk(nanowire, n)
    loger_main.info(f'Section {i} / {len(N_local)}')
    density_operator = kwant.operator.Density(nanowire, where=cut_sites, sum=False)
    bulk_density1[i] = density_operator(state1(0)[0])
    bulk_density2[i] = density_operator(state2(0)[0])

for key in bulk_density1.keys():
    bulk_density1[key] = bulk_density1[key] / np.sum(bulk_density1[len(N_local) - 1])
    bulk_density2[key] = bulk_density2[key] / np.sum(bulk_density2[len(N_local) - 1])

# Normalisation for the plots
sigmas = 1
mean_value = np.mean(np.array([bulk_density1[len(N_local) - 1], bulk_density2[len(N_local) - 1]]))
std_value = np.std(np.array([bulk_density1[len(N_local) - 1], bulk_density2[len(N_local) - 1]]))
max_value, min_value = mean_value + sigmas * std_value, 0



#%% Saving data
data_dir = '/home/mfmm/Projects/amorphous-nanowires/data/data-bulk-dos'
file_list = os.listdir(data_dir)
expID = get_fileID(file_list, common_name='Exp')
filename = '{}{}{}'.format('Exp', expID, '.h5')
filepath = os.path.join(data_dir, filename)


with h5py.File(filepath, 'w') as f:

    # Simulation folder
    simulation = f.create_group('Simulation')
    DoS1 = simulation.create_group('DoS1')
    DoS2 = simulation.create_group('DoS2')
    cuts = simulation.create_group('cuts')

    store_my_data(simulation,  'G_array',          G_array)
    store_my_data(simulation,    'flux',           flux)
    store_my_data(simulation,   'width',           width)
    store_my_data(simulation,       'N',           N)
    store_my_data(simulation, 'N_local',           N_local)
    store_my_data(simulation, 'bulk_tot_density1', bulk_tot_density1)
    store_my_data(simulation, 'bulk_tot_density2', bulk_tot_density2)

    store_my_dict(simulation['DoS1'],  bulk_density1)
    store_my_dict(simulation['DoS2'],  bulk_density2)
    store_my_dict(simulation['cuts'],  cut_pos)


    # Parameters folder
    parameters = f.create_group('Parameters')
    store_my_data(parameters, 'flux1', flux1)
    store_my_data(parameters, 'flux2', flux2)
    store_my_data(parameters, 'idx1',   idx1)
    store_my_data(parameters, 'idx2',   idx2)
    store_my_data(parameters, 'Ef',      Ef)
    store_my_data(parameters, 'Nx',      Nx)
    store_my_data(parameters, 'Ny',      Ny)
    store_my_data(parameters, 'Nz',      Nz)
    store_my_data(parameters, 'r ',      r)
    store_my_data(parameters, 't ',      t)
    store_my_data(parameters, 'eps',     eps)
    store_my_data(parameters, 'lamb',    lamb)
    store_my_data(parameters, 'lamb_z',  lamb_z)
    store_my_data(parameters, 'mu_leads', mu_leads)

    # Attributes
    attr_my_data(parameters, "Date",       str(date.today()))
    attr_my_data(parameters, "Code_path",  sys.argv[0])

loger_main.info('Data saved correctly')









