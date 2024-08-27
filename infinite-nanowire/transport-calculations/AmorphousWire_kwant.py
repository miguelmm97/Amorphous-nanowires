#%% Modules setup

# Math and plotting
from numpy import pi
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, MultiPoint, Polygon
from shapely import intersects
from scipy.integrate import quad

# Managing classes
from dataclasses import dataclass, field

# Tracking time
import time

# Kwant
import kwant
import tinyarray as ta

# Managing logging
import logging
import colorlog
from colorlog import ColoredFormatter


#%% Logging setup
loger_wire = logging.getLogger('kwant')
loger_wire.setLevel(logging.WARNING)

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
loger_wire.addHandler(stream_handler)



#%% Module
"""
Here we promote the infinite amorphous nanowire defined in class InfiniteNanowire.py 
into a kwant.system where to do transport calculations.
"""

# Amorphous site family class from https://zenodo.org/records/4382484
class AmorphousWire_kwant(kwant.builder.SiteFamily):
    def __init__(self, norbs, positions, name=None):

        if norbs is not None:
            if int(norbs) != norbs or norbs <= 0:
                raise ValueError("The norbs parameter must be an integer > 0.")
            norbs = int(norbs)
        self.norbs = norbs
        self.positions = positions
        self.name = name
        # self.canonical_repr = str(self.__hash__())
        self.canonical_repr = "1" if name is None else name

    def pos(self, tag):
        return self.positions[tag[0], :]

    def normalize_tag(self, tag):
        if tag[0] >= len(self.positions):
            raise ValueError
        return ta.array(tag)

    def __hash__(self):
        return 1

def amorphous_wire_kwant(coords, onsite, hopping, norbs=4, bonds=None):
    syst = kwant.Builder()
    latt = AmorphousWire_kwant(norbs=norbs, positions=coords)
