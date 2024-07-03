#%% Modules setup

# Math and plotting
from numpy import pi
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Modules
from InfiniteNanowire import InfiniteNanowire_FuBerg

#%% Main
widths = np.linspace(0, 0.5, 100)
rs     = np.linspace(1.1, 3, 100)
accepted_runs = np.zeros((len(widths), len(rs)))
n_runs = 100

for i, w in enumerate(widths):
    print(i)
    for j, r in enumerate(rs):

        n_declined = 0
        for k in range(n_runs):
            wire = InfiniteNanowire_FuBerg(Nx=6, Ny=6, w=w, r=r, eps=0.1, t=0., lamb=0., lamb_z=0.)
            try:
                wire.build_lattice()
                wire.get_boundary()
            except Exception as e:
                n_declined += 1

        accepted_runs[i, j] = n_runs - n_declined


fig = plt.figure(figsize=(8, 8))
ax = fig.gca()
stats = ax.imshow(accepted_runs, origin='lower', aspect='auto')
ax.set_ylabel('width')
ax.set_xlabel('r')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(stats, cax=cax, orientation='vertical')
cbar.set_label(label='Accepted runs', labelpad=0, fontsize=10)
ax.set(yticks=[0, len(widths) / 4, len(widths) / 2, 3 * len(widths) / 4, len(widths) - 1],
         yticklabels=[0, widths[-1] / 4, widths[-1] / 2, 3 * widths[-1] / 4, widths[-1]])
ax.set(xticks=[0, len(rs) / 4, len(rs) / 2, 3 * len(rs) / 4, len(rs) - 1],
       xticklabels=[f'{rs[0] :.2f}', f'{rs[0] + (rs[-1] - rs[0]) / 4 :.2f}',
                      f'{rs[0] + (rs[-1] - rs[0]) / 2 :.2f}', f'{rs[0] + 3 * (rs[-1] - rs[0]) / 4 :.2f}',
                      f'{rs[-1]:.2f}'])

ax.tick_params(which='major', width=0.75, labelsize=15)
ax.tick_params(which='major', length=14, labelsize=15)
plt.show()
