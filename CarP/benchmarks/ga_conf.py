import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
import networkx as nx

from pathlib import Path
import sys

from astropy.wcs.docstrings import cylfix



path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import CarP
from CarP.rmsd import rmsd_qcp
from CarP.ga_operations import *
import os

pwd = os.path.abspath('')
conf = CarP.Conformer(os.path.join(pwd,'glucose'))
conf.load_log(software='xyz')

cm = conf.connectivity_matrix()
cycles = (nx.cycle_basis(cm))

rd_list = conf.sort_ring_atoms(cycles, cm, conf)
Cspace = []

conf_int = conf

for conf_num in range(50):
    CarP.modify_ring(conf,0, rd_list, prob_model=[0.2,0.2,0.2,0.2,0.2])
    xyz = conf.xyz.tolist()
    Cspace.append(xyz)
    conf = conf_int

rmsd = np.zeros((len(Cspace), len(Cspace)))
for i, conf1 in enumerate(Cspace):
    for j, conf2 in enumerate(Cspace):
        rmsd[i,j] = rmsd_qcp(conf1, conf2)

def cluster(Cspace, rmsd_mat):

    Z = CarP.linkage(rmsd_mat, 'single', optimal_ordering=True)

    fig = plt.figure(figsize=(6,4))
    dn = dendrogram (Z)
    fig.tight_layout()

    return fig

fig = cluster(Cspace, rmsd)
fig.savefig('cluster.png', dpi=300)


