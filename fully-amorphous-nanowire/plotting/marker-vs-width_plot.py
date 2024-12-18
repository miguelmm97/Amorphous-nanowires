#%% Modules and setup

# Plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm

# Modules
from modules.functions import *
from modules.colorbar_marker import *


#%% Loading data
file_list = ['Exp1.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/data-marker-vs-width')

# Parameters
Nx           = data_dict[file_list[0]]['Parameters']['Nx']
Ny           = data_dict[file_list[0]]['Parameters']['Ny']
Nz           = data_dict[file_list[0]]['Parameters']['Nz']
r            = data_dict[file_list[0]]['Parameters']['r ']
t            = data_dict[file_list[0]]['Parameters']['t ']
eps          = data_dict[file_list[0]]['Parameters']['eps']
lamb         = data_dict[file_list[0]]['Parameters']['lamb']
lamb_z       = data_dict[file_list[0]]['Parameters']['lamb_z']
flux         = data_dict[file_list[0]]['Parameters']['flux']



# Simulation data
local_marker = data_dict[file_list[0]]['Simulation']['local_marker']
width = data_dict[file_list[0]]['Simulation']['width']


#%% Figure 1
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['dodgerblue', 'limegreen', 'm', 'r', 'orange']
fontsize = 15


# Figure 1
fig1 = plt.figure()
gs = GridSpec(1, 1, figure=fig1, wspace=0.1, hspace=0.35)
ax1 = fig1.add_subplot(gs[0, 0])


ax1.plot(width, local_marker, marker='o', linestyle='solid', color='dodgerblue')
ax1.set_xlabel('$w$', fontsize=fontsize)
ax1.set_ylabel('$\\nu$', fontsize=fontsize)
ax1.set_ylim([-1.1, 0.1])
plt.show()