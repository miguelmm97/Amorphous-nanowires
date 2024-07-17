#%% Modules setup

# Math and plotting
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn
import itertools

# Managing system, data and config files
from functions import load_my_data, load_my_attr

#%% Loading data
file_list = ['flux-gap1.h5']
data_dict = load_my_data(file_list, '../../../data-flux-gap')

# Parameters
Nsamples   = data_dict[file_list[0]]['Parameters']['Nsamples']
Nx         = data_dict[file_list[0]]['Parameters']['Nx']
Ny         = data_dict[file_list[0]]['Parameters']['Ny']
width      = data_dict[file_list[0]]['Parameters']['width']
r          = data_dict[file_list[0]]['Parameters']['r']
t          = data_dict[file_list[0]]['Parameters']['t']
eps        = data_dict[file_list[0]]['Parameters']['eps']
lamb       = data_dict[file_list[0]]['Parameters']['lamb']
lamb_z     = data_dict[file_list[0]]['Parameters']['lamb_z']

# Data
gap        = data_dict[file_list[0]]['Simulation']['gap']
flux       = data_dict[file_list[0]]['Simulation']['flux']
n_rejected = data_dict[file_list[0]]['Simulation']['n_rejected']

# Preallocation
gap_mean = np.array((len(flux), ))
gap_std  = np.array((len(flux), ))


#%% Postproduction
for i in range(len(gap[:, 0])):
    idx = np.where(np.abs(gap[i, :]) > 1e-14)[0]
    if len(idx) != (Nsamples - n_rejected[i]):
        raise ValueError('Number of 0s in the gap do not coincide with rejected configurations.')
    gap_mean[i] = np.mean(np.take(gap[i, :], idx))
    gap_std[i] = np.std(np.take(gap[i, :], idx))
error_bars = np.array([gap_mean + 0.5 * gap_std, gap_mean - 0.5 * gap_std])


#%% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125', '#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']

fig1 = plt.figure(figsize=(6, 6))
ax1 = fig1.gca()
ax1.plot(flux, gap_mean, '.', color=color_list[3], markersize=7)
ax1.plot(flux, gap_mean, color=color_list[8], linewidth=0.5)
ax1.errorbar(flux, gap_mean, yerr=error_bars, color=axcolour[8])
ax1.set_xlabel('$\phi/\phi_0$')
ax1.set_ylabel('$E_g$')
ax1.set_xlim(flux[0], flux[-1])
ax1.tick_params(which='major', width=0.75, labelsize=10)
ax1.tick_params(which='major', length=6, labelsize=10)
fig1.suptitle(f'$N_s =$ {Nsamples}, $N_x=N_y=$ {Nx}, $w=$ {width}, $r=$ {r}, $\epsilon=$ {eps}, $\lambda=$ {lamb}, $\lambda_z=$ {lamb_z}')
plt.show()



