#%% Modules set up

# Math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Kwant
import kwant
import tinyarray as ta

# Modules
from modules.functions import *
from modules.AmorphousLattice_2d import AmorphousLattice_2d

#%% Main

"""
To create an amorphous lattice in Kwant we need to define a lattice ourselves. For that we define a class that
inherits the SiteFamily class from Kwant and we specify the method pos(self, tag) to identify each site uniquely.
In this case the tag system for an amorphous cross-section nanowire is (i, z), where i is the site number in the class
AmorphousLattice_2d with which we create the amorphous cross-section, and z is the number of the transversal layer along
the wire.
"""
class AmorphousCrossSectionWire_ScatteringRegion(kwant.builder.SiteFamily):
    def __init__(self, norbs, cross_section, name=None):

        if norbs is not None:
            if int(norbs) != norbs or norbs <= 0:
                raise ValueError("The norbs parameter must be an integer > 0.")
            norbs = int(norbs)

        # Class fields
        self.norbs = norbs
        self.coords = np.array([cross_section.x, cross_section.y]).T
        self.Nsites = len(self.coords[:, 0])
        self.name = name
        self.canonical_repr = "1" if name is None else name

    def pos(self, tag):
        return np.concatenate((self.coords[tag[0], :], np.array([tag[1]])))

    def normalize_tag(self, tag):
        return ta.array(tag)

    def __hash__(self):
        return 1

"""
We now create the amorphous cross section with our usual class and promote it to kwant through the new class.
"""

Nx, Ny = 5, 5
n_layers = 100
cross_section = AmorphousLattice_2d(Nx=Nx, Ny=Ny, w=0.1, r=1.4)
cross_section.build_lattice()
latt = AmorphousCrossSectionWire_ScatteringRegion(norbs=1, cross_section=cross_section, name='wire')

"""
We can now build the system as usual in kwant using the tags for our new SiteFamily.
"""

# Initialise and define onsite energies
syst = kwant.Builder()
syst[(latt(i, z) for i in range(latt.Nsites) for z in range(n_layers))] = 1.

# Hoppings
for z in range(n_layers):
    # In the cross-section
    syst[((latt(i, z), latt(n, z)) for i in range(latt.Nsites) for n in cross_section.neighbours[i])] = 5.
    # Between cross-sections
    if z < n_layers - 1:
        syst[((latt(i, z), latt(i, z + 1)) for i in range(latt.Nsites))] = 2.
    else:
        pass

"""
Once the scattering region is finalised we need to attach the leads. One way to do this is by attaching regular lattice
leads to the amorphous scattering region. In order to do that we create a lead as normal, with a cubic lattice and 
define the sites and hoppings in the first unit cell of the lead. After this we should define an extra layer in the
system that has the structure of the lead's unit cell, that is, we define an extra cross section layer that is going to
be cubic, not amorphous. We define the hoppings in that extra layer and also the hoppings from that layer to the
initial amorphous layer. This allows that the scattering region and lead have the same crystal structure and can be 
attached automatically.
"""
# Left lead
latt_lead = kwant.lattice.cubic(norbs=1)
sym_left_lead = kwant.TranslationalSymmetry((0, 0, -1))
left_lead = kwant.Builder(sym_left_lead)
left_lead[(latt_lead(i, j, 0) for i in range(Nx) for j in range(Ny))] = 1.
left_lead[latt_lead.neighbors()] = 3.

syst[(latt_lead(i, j, -1) for i in range(Nx) for j in range(Ny))] = 1.
syst[latt_lead.neighbors()] = 3.
syst[(((latt_lead(i, j, -1), latt(i + Ny * j, 0)) for i in range(Nx) for j in range(Ny)))] = 2.
syst.attach_lead(left_lead)


# Right lead
sym_right_lead = kwant.TranslationalSymmetry((0, 0, 1))
right_lead = kwant.Builder(sym_right_lead)
right_lead[(latt_lead(i, j, 0) for i in range(Nx) for j in range(Ny))] = 1.
right_lead[latt_lead.neighbors()] = 3.

syst[(latt_lead(i, j, n_layers) for i in range(Nx) for j in range(Ny))] = 1.
syst[latt_lead.neighbors()] = 3.
syst[(((latt_lead(i, j, n_layers), latt(i + Ny * j, n_layers - 1)) for i in range(Nx) for j in range(Ny)))] = 2.
syst.attach_lead(right_lead)



# Style of kwant plotting
site_size  = 0.1
site_lw    = 0.01
site_color = 'm'
hop_color  = 'royalblue'
hop_lw     = 0.05
lead_color = 'r'

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
kwant.plot(syst, site_size=site_size, site_lw=site_lw, site_color=site_color, hop_lw=hop_lw, hop_color=hop_color,
           lead_site_size=site_size, lead_color=lead_color, lead_site_lw=site_lw, lead_hop_lw=hop_lw,
           ax=ax)
ax.set_axis_off()
plt.show()