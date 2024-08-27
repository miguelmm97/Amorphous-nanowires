#%% Modules setup

# Math and plotting
from numpy import pi
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Modules
from AmorphousLattice_2d import AmorphousLattice_2d

#%% Main

n_runs, n_w, n_r   = 100, 100, 100
Nx, Ny             = 6, 6
widths             = np.linspace(0, 0.5, n_w)
rs                 = np.linspace(1.1, 3, n_r)
accepted_runs      = np.zeros((len(widths), len(rs)))

for i, w in enumerate(widths):
    print(f'Iter: {i}/ {n_w}')
    for j, r in enumerate(rs):
        n_declined = 0
        for k in range(n_runs):
            lattice = AmorphousLattice_2d(Nx=6, Ny=6, w=w, r=r)
            try:
                lattice.build_lattice()
                lattice.get_boundary()
            except Exception as e:
                n_declined += 1
        accepted_runs[i, j] = n_runs - n_declined
accepted_runs = accepted_runs / n_runs


#%% Statistics of the algorithm
font = {'family': 'serif', 'color': 'black', 'weight': 'normal', 'size': 22, }
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fontsize = 15
fig = plt.figure(figsize=(8, 8))
ax = fig.gca()

stats = ax.imshow(accepted_runs, origin='lower', aspect='auto')
ax.set_ylabel('width', fontsize=fontsize)
ax.set_xlabel('r', fontsize=fontsize)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(stats, cax=cax, orientation='vertical')
cbar.set_label(label='Probability of success', labelpad=0, fontsize=fontsize)
cbar.ax.tick_params(which='major', length=14, labelsize=15)

ax.set(yticks=[0, n_w / 4, n_w / 2, 3 * n_w / 4, n_w - 1],
       yticklabels=[0, widths[-1] / 4, widths[-1] / 2, 3 * widths[-1] / 4, widths[-1]])
ax.set(xticks=[0, n_r / 4, n_r / 2, 3 * n_r / 4, n_r - 1],
       xticklabels=[f'{rs[0] :.2f}', f'{rs[int(n_r / 4)]:.2f}', f'{rs[int(n_r / 2)] :.2f}',
                    f'{rs[int(3 * n_r / 4)] :.2f}', f'{rs[-1]:.2f}'])

ax.tick_params(which='major', width=0.75, labelsize=15)
ax.tick_params(which='major', length=14, labelsize=15)
ax.set_title(f'Probability of success for the boundary algorithm. Nx: {Nx}, Ny:  {Ny}, Runs: {n_runs}', fontsize=fontsize)
plt.show()
