#%% Modules and setup

# Plotting
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
from scipy.optimize import curve_fit


# Modules
from modules.functions import *
from modules.colorbar_marker import *


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
file_list = ['Exp10.h5', 'Exp11.h5', 'Exp12.h5', 'Exp13.h5', 'Exp14.h5', 'Exp16.h5', 'Exp15.h5']
data_dict = load_my_data(file_list, '/home/mfmm/Projects/amorphous-nanowires/data/local-simulations/data-OPDM')

# Load data
Nx           = data_dict[file_list[0]]['Parameters']['Nx']
Ny           = data_dict[file_list[0]]['Parameters']['Nx']
Nz           = data_dict[file_list[0]]['Parameters']['Nz']
r            = data_dict[file_list[0]]['Parameters']['r ']
t            = data_dict[file_list[0]]['Parameters']['t ']
eps          = data_dict[file_list[0]]['Parameters']['eps']
lamb         = data_dict[file_list[0]]['Parameters']['lamb']
lamb_z       = data_dict[file_list[0]]['Parameters']['lamb_z']
x            = data_dict[file_list[0]]['Simulation']['x']
y            = data_dict[file_list[0]]['Simulation']['y']
z            = data_dict[file_list[0]]['Simulation']['z']
OPDM1        = data_dict[file_list[0]]['Simulation']['OPDM_r']
OPDM2        = data_dict[file_list[1]]['Simulation']['OPDM_r']
OPDM3        = data_dict[file_list[2]]['Simulation']['OPDM_r']
OPDM4        = data_dict[file_list[3]]['Simulation']['OPDM_r']
OPDM5        = data_dict[file_list[4]]['Simulation']['OPDM_r']
OPDM6        = data_dict[file_list[5]]['Simulation']['OPDM_r']
OPDM7        = data_dict[file_list[6]]['Simulation']['OPDM_r']
radius1      = np.real(data_dict[file_list[0]]['Simulation']['r'])
radius2      = np.real(data_dict[file_list[1]]['Simulation']['r'])
radius3      = np.real(data_dict[file_list[2]]['Simulation']['r'])
radius4      = np.real(data_dict[file_list[3]]['Simulation']['r'])
radius5      = np.real(data_dict[file_list[4]]['Simulation']['r'])
radius6      = np.real(data_dict[file_list[5]]['Simulation']['r'])
radius7      = np.real(data_dict[file_list[6]]['Simulation']['r'])
width1       = data_dict[file_list[0]]['Simulation']['width']
width2       = data_dict[file_list[1]]['Simulation']['width']
width3       = data_dict[file_list[2]]['Simulation']['width']
width4       = data_dict[file_list[3]]['Simulation']['width']
width5       = data_dict[file_list[4]]['Simulation']['width']
width6       = data_dict[file_list[5]]['Simulation']['width']
width7       = data_dict[file_list[6]]['Simulation']['width']


# Histograms of the data
r_bins1, binned_opdm1 = bin_my_samples(radius1, OPDM1, num_bins=15)
r_bins2, binned_opdm2 = bin_my_samples(radius2, OPDM2, num_bins=15)
r_bins3, binned_opdm3 = bin_my_samples(radius3, OPDM3, num_bins=15)
r_bins4, binned_opdm4 = bin_my_samples(radius4, OPDM4, num_bins=15)
r_bins5, binned_opdm5 = bin_my_samples(radius5, OPDM5, num_bins=15)
r_bins6, binned_opdm6 = bin_my_samples(radius6, OPDM6, num_bins=15)
r_bins7, binned_opdm7 = bin_my_samples(radius7, OPDM7, num_bins=15)
OPDM_sum1 = np.real(np.array([np.sum(binned_opdm1[i]) for i in range(len(binned_opdm1))if len(binned_opdm1[i]) != 0]))
OPDM_sum2 = np.real(np.array([np.sum(binned_opdm2[i]) for i in range(len(binned_opdm2))if len(binned_opdm2[i]) != 0]))
OPDM_sum3 = np.real(np.array([np.sum(binned_opdm3[i]) for i in range(len(binned_opdm3))if len(binned_opdm3[i]) != 0]))
OPDM_sum4 = np.real(np.array([np.sum(binned_opdm4[i]) for i in range(len(binned_opdm4))if len(binned_opdm4[i]) != 0]))
OPDM_sum5 = np.real(np.array([np.sum(binned_opdm5[i]) for i in range(len(binned_opdm5))if len(binned_opdm5[i]) != 0]))
OPDM_sum6 = np.real(np.array([np.sum(binned_opdm6[i]) for i in range(len(binned_opdm6))if len(binned_opdm6[i]) != 0]))
OPDM_sum7 = np.real(np.array([np.sum(binned_opdm7[i]) for i in range(len(binned_opdm7))if len(binned_opdm7[i]) != 0]))
rad1 = np.array([r_bins1[i] for i in range(len(binned_opdm1)) if len(binned_opdm1[i]) != 0])
rad2 = np.array([r_bins2[i] for i in range(len(binned_opdm2)) if len(binned_opdm2[i]) != 0])
rad3 = np.array([r_bins3[i] for i in range(len(binned_opdm3)) if len(binned_opdm3[i]) != 0])
rad4 = np.array([r_bins4[i] for i in range(len(binned_opdm4)) if len(binned_opdm4[i]) != 0])
rad5 = np.array([r_bins5[i] for i in range(len(binned_opdm5)) if len(binned_opdm5[i]) != 0])
rad6 = np.array([r_bins6[i] for i in range(len(binned_opdm6)) if len(binned_opdm6[i]) != 0])
rad7 = np.array([r_bins7[i] for i in range(len(binned_opdm7)) if len(binned_opdm7[i]) != 0])


# Exponential fit
def funcfit(L, C, xi): return C * np.exp(- L / xi)
# def funcfit(L, C, xi): return C - L / xi
# fit1, covariance1 = curve_fit(funcfit, rad1, np.log(OPDM_sum1), p0=[np.log(OPDM_sum1[0]), 1])
# exp1 = np.exp(funcfit(rad1, fit1[0], fit1[1]))
fit1, covariance1 = curve_fit(funcfit, rad1, OPDM_sum1, p0=[OPDM_sum1[0], 1])
exp1 = funcfit(rad1, fit1[0], fit1[1])
cov1 = np.sqrt(np.diag(covariance1))
loger_main.info(f'C1: {fit1[0]}, std_C: {cov1[0]}, xi1: {fit1[1]}, std_xi: {cov1[1]}')

# fit2, covariance2 = curve_fit(funcfit, rad2, np.log(OPDM_sum2), p0=[np.log(OPDM_sum2[0]), 1])
# exp2 = np.exp(funcfit(rad2, fit2[0], fit2[1]))
fit2, covariance2 = curve_fit(funcfit, rad2, OPDM_sum2, p0=[OPDM_sum2[0], 1])
exp2 = funcfit(rad2, fit2[0], fit2[1])
cov2 = np.sqrt(np.diag(covariance2))
loger_main.info(f'C2: {fit2[0]}, std_C: {cov2[0]}, xi2: {fit2[1]}, std_xi: {cov2[1]}')

# fit3, covariance3 = curve_fit(funcfit, rad3, np.log(OPDM_sum3), p0=[np.log(OPDM_sum3[0]), 1])
# exp3 = np.exp(funcfit(rad3, fit3[0], fit3[1]))
fit3, covariance3 = curve_fit(funcfit, rad3, OPDM_sum3, p0=[OPDM_sum3[0], 1])
exp3 = funcfit(rad3, fit3[0], fit3[1])
cov3 = np.sqrt(np.diag(covariance3))
loger_main.info(f'C3: {fit3[0]}, std_C: {cov3[0]}, xi2: {fit3[1]}, std_xi: {cov3[1]}')

# fit4, covariance4 = curve_fit(funcfit, rad4, np.log(OPDM_sum4), p0=[np.log(OPDM_sum4[0]), 1])
# exp4 = np.exp(funcfit(rad4, fit4[0], fit4[1]))
fit4, covariance4 = curve_fit(funcfit, rad4, OPDM_sum4, p0=[OPDM_sum4[0], 1])
exp4 = funcfit(rad4, fit4[0], fit4[1])
cov4 = np.sqrt(np.diag(covariance4))
loger_main.info(f'C4: {fit4[0]}, std_C: {cov4[0]}, xi2: {fit4[1]}, std_xi: {cov4[1]}')

# fit5, covariance5 = curve_fit(funcfit, rad5, np.log(OPDM_sum5), p0=[np.log(OPDM_sum5[0]), 1])
# exp5 = np.exp(funcfit(rad5, fit5[0], fit5[1]))
fit5, covariance5 = curve_fit(funcfit, rad5, OPDM_sum5, p0=[OPDM_sum5[0], 1])
exp5 = funcfit(rad5, fit5[0], fit5[1])
cov5 = np.sqrt(np.diag(covariance5))
loger_main.info(f'C5: {fit5[0]}, std_C: {cov5[0]}, xi2: {fit5[1]}, std_xi: {cov5[1]}')

# fit5, covariance5 = curve_fit(funcfit, rad5, np.log(OPDM_sum5), p0=[np.log(OPDM_sum5[0]), 1])
# exp5 = np.exp(funcfit(rad5, fit5[0], fit5[1]))
fit6, covariance6 = curve_fit(funcfit, rad6, OPDM_sum6, p0=[OPDM_sum6[0], 1])
exp6 = funcfit(rad6, fit6[0], fit6[1])
cov6 = np.sqrt(np.diag(covariance6))
loger_main.info(f'C6: {fit6[0]}, std_C: {cov6[0]}, xi2: {fit6[1]}, std_xi: {cov6[1]}')





#%% Figures
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
color_list = ['limegreen', 'dodgerblue', 'm', 'r', 'orange']
marker_list=['o', 's', 'd', 'p', '*', 'h', '>', '<', 'X']
markersize = 5
fontsize=20
palette = seaborn.color_palette(palette='viridis_r', n_colors=300)
palette = [palette[0], palette[50], palette[100], palette[150], palette[200], palette[250], palette[-1]]


# Figure 2
fig1 = plt.figure(figsize=(8, 8))
gs = GridSpec(1, 1, figure=fig1, wspace=0.2, hspace=0.3)
ax1 = fig1.add_subplot(gs[0, 0])

ax1.plot(rad1, OPDM_sum1, marker='o', color=palette[0], label=f'$w= {width1}$')
ax1.plot(rad2, OPDM_sum2, marker='o', color=palette[1], label=f'$w= {width2}$')
ax1.plot(rad3, OPDM_sum3, marker='o', color=palette[2], label=f'$w= {width3}$')
ax1.plot(rad4, OPDM_sum4, marker='o', color=palette[3], label=f'$w= {width4}$')
ax1.plot(rad5, OPDM_sum5, marker='o', color=palette[4], label=f'$w= {width5}$')
ax1.plot(rad6, OPDM_sum6, marker='o', color=palette[5], label=f'$w= {width6}$')
ax1.plot(rad7, OPDM_sum7, marker='o', color=palette[6], label=f'$w= {width7}$')
# ax1.plot(rad1, exp1, marker='none', linestyle='dashed', color=palette[0])
# ax1.plot(rad2, exp2, marker='none', linestyle='dashed', color=palette[1])
# ax1.plot(rad3, exp3, marker='none', linestyle='dashed', color=palette[2])
# ax1.plot(rad4, exp4, marker='none', linestyle='dashed', color=palette[3])
# ax1.plot(rad5, exp5, marker='none', linestyle='dashed', color=palette[4])

ax1.set_xlabel('$\\vert x - y \\vert$', fontsize=fontsize)
ax1.set_ylabel('$\sum_{r, \\alpha} \\vert \\rho(x, y) \\vert^{2}$', fontsize=fontsize)
ax1.set_ylim(1e-8, 1)
# ax1.set_xlim(min(rad1), 5)
ax1.tick_params(which='major', width=0.75, labelsize=fontsize)
ax1.tick_params(which='major', length=6, labelsize=fontsize)
ax1.legend()
ax1.set_yscale('log')

fig1.savefig(f'../figures/opdm-localisation.pdf', format='pdf')
plt.show()