#%% Modules set up

# Math
import numpy as np
import matplotlib.pyplot as plt

# Kwant
import kwant
import tinyarray as ta

# Modules
from modules.functions import *
from modules.AmorphousLattice_2d import AmorphousLattice_2d

#%% Main

class Scattering_CrossSection(kwant.builder.SiteFamily):
    def __init__(self, norbs, cross_section, name=None):

        if norbs is not None:
            if int(norbs) != norbs or norbs <= 0:
                raise ValueError("The norbs parameter must be an integer > 0.")
            norbs = int(norbs)

        # Class fields
        self.norbs = norbs
        self.coords = np.array([cross_section.x, cross_section.y]).T
        self.name = name
        self.canonical_repr = "1" if name is None else name
        self.Nsites = len(self.coords[:, 0])

    def pos(self, tag):
        return np.concatenate((self.coords[tag[0], :], np.array([tag[1]])))

    def normalize_tag(self, tag):
        return ta.array(tag)

    def __hash__(self):
        return 1

cross_section = AmorphousLattice_2d(Nx=3, Ny=3, w=0.1, r=1.4)
cross_section.build_lattice()
latt = Scattering_CrossSection(norbs=1, cross_section=cross_section, name='wire')

n_layers = 5
syst = kwant.Builder()
pos_try = latt(4, 0)

# Define sites and onsite energy at the same time
syst[(latt(i, z) for i in range(latt.Nsites) for z in range(n_layers))] = 1.

for z in range(n_layers - 1):
    syst[((latt(i, z), latt(i, z + 1)) for i in range(latt.Nsites))] = 2.
    syst[((latt(i, z), latt(n, z)) for i in range(latt.Nsites) for n in cross_section.neighbours[i])] = 5.




    # syst[((latt(i, z), latt(i, z + 1)) for z in range(n_layers - 1))] = 2.
    #
    #
    # # syst[latt(1, 0), latt(1, 1)] = 4.
    # for neigh in cross_section.neighbours[i]:
    #      syst[((latt(i, z), latt(neigh, z)) for z in range(n_layers))] = 4.




fig = plt.figure(figsize=(8, 5))
ax = fig.gca()
kwant.plot(syst, hop_color='royalblue', hop_lw=0.05, site_size=0.1, site_color='m', ax=ax)
