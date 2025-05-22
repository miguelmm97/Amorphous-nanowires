#%% Modules and setup

# Math
import numpy as np

# Modules
from modules.functions import *
from modules.colorbar_marker import *
import config

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


#%% Obtaining the data
loger_main.info('Loading variables from the .toml file')
variables = config.variables_cond_vs_N


N           = variables['N']
Nsamples    = variables['Nsamples']
width       = variables['width']
Nz          = variables['Nz']
r           = variables['r']
t           = variables['t']
mu_leads    = variables['mu_leads']
flux_max    = variables['flux_max']
flux_min    = variables['flux_min']
flux_L      = variables['flux_L']
Ef          = variables['Ef']
K_onsite    = variables['K_onsite']
eps         = 4 * t
lamb        = 1 * t
lamb_z      = 1.8 * t
mu_leads    = mu_leads * t
flux        = np.linspace(flux_min, flux_max, flux_L)
params_dict = {'t': t, 'eps': eps, 'lamb': lamb, 'lamb_z': lamb_z}


data_dir = ['data-15', 'data-14', 'data-13', 'data-12', 'data-10', 'data-9', 'data-8']
Gmax = np.zeros((len(N), ), dtype=np.real64)
Gmax_std = np.zeros((len(N), ), dtype=np.real64)
deltaG = np.zeros((len(N), ), dtype=np.real64)
deltaG_std = np.zeros((len(N), ), dtype=np.real64)

for i, directory in data_dir:
    for j, file in enumerate(os.listdir(directory)):
        if file.endswith('h5'):
            file_path = os.path.join(directory, file)
            with h5py.File(file_path, 'r') as f:
                if j == 0:
                    Gmax_N = np.zeros((Nsamples, ), dtype=np.real64)
                Gmax_N[j] =  f['Simulation']['Gmax'][()]
    Gmax[i] = np.mean(Gmax_N)
    Gmax_std[i] = np.std(Gmax_N)
    if i ==0:
        Gref = Gmax[i]
    else:
        deltaG[i] = np.mean((np.ones((Nsamples, )) * Gref - Gmax_N) / Gref)
        deltaG_std[i] = np.std((np.ones((Nsamples, )) * Gref - Gmax_N) / Gref)

# Saving the data
data_dir = '.'
filename = f'data-cluster-L={Nz}.h5'
filepath = os.path.join(data_dir, filename)

loger_main.info('Saving data...')
with h5py.File(filepath, 'w') as f:

    # Simulation folder
    simulation = f.create_group('Simulation')
    store_my_data(simulation, 'N',             N)
    store_my_data(simulation, 'Ef',            Ef)
    store_my_data(simulation, 'flux',          flux)
    store_my_data(simulation, 'width',         width)
    store_my_data(simulation, 'Gmax',          Gmax)
    store_my_data(simulation, 'Gmax_std',      Gmax_std)
    store_my_data(simulation, 'deltaG',        deltaG)
    store_my_data(simulation, 'deltaG_std',    deltaG_std)
    store_my_data(simulation, 'sample',        Nsamples)

    # Parameters folder
    parameters = f.create_group('Parameters')
    store_my_data(simulation, 'Nz',            Nz)
    store_my_data(parameters, 'K_onsite',      K_onsite)
    store_my_data(parameters, 'r',             r)
    store_my_data(parameters, 't',             t)
    store_my_data(parameters, 'eps',           eps)
    store_my_data(parameters, 'lamb',          lamb)
    store_my_data(parameters, 'lamb_z',        lamb_z)
    store_my_data(parameters, 'mu_leads',      mu_leads)

    # Attributes
    attr_my_data(parameters, "Date",       str(date.today()))
    attr_my_data(parameters, "Code_path",  sys.argv[0])

loger_main.info('Data saved correctly')