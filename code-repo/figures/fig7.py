#%% Modules and setup

# Plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn
import matplotlib.ticker as ticker

# Modules
from functions import *
from colorbar_marker import *


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

#%% Loading data marker
file_list = ['fig7-marker-vs-width.h5']
data_dict = load_my_data(file_list, '../data')

# Simulation data
N            = data_dict[file_list[0]]['Simulation']['N']
num_vecs     = data_dict[file_list[0]]['Simulation']['num_vecs']
num_moments  = data_dict[file_list[0]]['Simulation']['num_moments']
width        = data_dict[file_list[0]]['Simulation']['width']
avg_marker   = data_dict[file_list[0]]['Simulation']['avg_marker']
std_marker   = data_dict[file_list[0]]['Simulation']['std_marker']
med_marker   = data_dict[file_list[0]]['Simulation']['med_marker']
mode_marker   = data_dict[file_list[0]]['Simulation']['mode_marker']
marker       = data_dict[file_list[0]]['Simulation']['marker']
Nsamples     = data_dict[file_list[0]]['Simulation']['Nsamples']

# Parameters
L            = data_dict[file_list[0]]['Parameters']['Nz']
cutoff       = data_dict[file_list[0]]['Parameters']['cutoff']
t            = data_dict[file_list[0]]['Parameters']['t']
eps          = data_dict[file_list[0]]['Parameters']['eps']
lamb         = data_dict[file_list[0]]['Parameters']['lamb']
lamb_z       = data_dict[file_list[0]]['Parameters']['lamb_z']
error_bar_up    = avg_marker + 0.5 * std_marker
error_bar_down  = avg_marker - 0.5 * std_marker



#%% Figures

# Style
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['dodgerblue', 'limegreen', 'm', 'r', 'orange']
fontsize = 20
palette = seaborn.color_palette(palette='magma_r', n_colors=len(N))
palette2 = seaborn.color_palette(palette='viridis_r', n_colors=200)
palette2 = [palette2[0], palette2[40], palette2[80], palette2[120], palette2[160], palette2[-1]]


# Figure 1
fig1 = plt.figure(figsize=(8, 6))
gs = GridSpec(1, 1, figure=fig1, wspace=0.1)
ax1 = fig1.add_subplot(gs[0, 0])

# Plot
ax1.plot(np.linspace(0, np.max(width), 10), np.zeros((10, )), '--', color='Black', alpha=0.2)
for i in range(0, len(N) - 1):
    ax1.plot(width, avg_marker[i, :], marker='o', linestyle='solid', color=palette[i], label=f'${N[len(N)-i -1]}$')
    ax1.fill_between(width, error_bar_down[i, :], error_bar_up[i, :], color=palette[i], alpha=0.3)

lgnd = ax1.legend(loc='upper left', ncol=1, frameon=False, fontsize=fontsize, handlelength=1, columnspacing=0.5, labelspacing=0.2, bbox_to_anchor=(0.05, 0.77))
ax1.text(0.135, -0.17, '$\\underline{N}$', fontsize=fontsize)
ax1.text(0.5, -0.55, f'$N_s = {100}$', fontsize=fontsize)
ax1.text(0.5, -0.65, f'$\\vert x_i - x_i^0\\vert < 0.2 N_i$', fontsize=fontsize)
ax1.text(0.5, -0.75, f'$L = 150$', fontsize=fontsize)

# Labels and limits
ax1.set_xlabel('$w$', fontsize=fontsize)
ax1.set_ylabel('$\overline{\\nu}$', fontsize=fontsize, labelpad=-5)
ax1.set_ylim([-1, 0.1])
ax1.set_xlim([0, 0.8])

# Tick params
majorsy = [-1, - 0.75, -0.5, -0.25, 0]
ax1.yaxis.set_major_locator(ticker.FixedLocator(majorsy))
ax1.tick_params(which='major', width=0.75, labelsize=fontsize)
ax1.tick_params(which='major', length=6, labelsize=fontsize)
ax1.tick_params(which='minor', width=0.75, labelsize=fontsize)
ax1.tick_params(which='minor', length=3, labelsize=fontsize)


fig1.savefig('fig7.pdf', format='pdf')
plt.show()
