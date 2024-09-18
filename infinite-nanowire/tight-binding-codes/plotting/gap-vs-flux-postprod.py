#%% modules setup

# Math and plotting
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools

# Managing data
import h5py
import os
import sys
from datetime import date

# modules
from functions import load_my_data, load_my_attr

#%% Loading data
file_list = ['flux-gap4.h5']
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
# flux       = data_dict[file_list[0]]['Simulation']['flux']

# Preallocation
flux = np.linspace(0., 4., 1000)
gap_mean = np.zeros((len(flux), ))
gap_std  = np.zeros((len(flux), ))



#%% Postproduction
for i in range(len(flux)):
    gap_mean[i] = np.mean(gap[i, :])
    gap_std[i] = np.std(gap[i, :])

error_bar_up = gap_mean + 0.5 * gap_std
error_bar_down = gap_mean - 0.5 * gap_std

#%% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['#FF7256', '#00BFFF', '#00C957', '#9A32CD', '#FFC125', '#FF7D66', '#FF416D', '#00B5A1', '#3F6CFF']

fig1 = plt.figure(figsize=(6, 6))
ax1 = fig1.gca()
# ax1.plot(flux, gap_mean, '.', color=color_list[3], markersize=2, alpha=0.3)
ax1.plot(flux, gap_mean, color=color_list[3], linewidth=0.5)
ax1.fill_between(flux, error_bar_down, error_bar_up, color=color_list[3], alpha=0.3)
# ax1.errorbar(flux, gap_mean, yerr=gap_std, color=color_list[3], alpha=0.3)
ax1.set_xlabel('$\phi/\phi_0$')
ax1.set_ylabel('$E_g$')
ax1.set_xlim(flux[0], flux[-1])
ax1.tick_params(which='major', width=0.75, labelsize=10)
ax1.tick_params(which='major', length=6, labelsize=10)
fig1.suptitle(f'$N_s =$ {Nsamples}, $N_x=N_y=$ {Nx}, $w=$ {width}, $r=$ {r}, $\epsilon=$ {eps}, $\lambda=$ {lamb}, $\lambda_z=$ {lamb_z}')
plt.show()
fig1.savefig("amorphous-AB-oscillations-average.pdf")



