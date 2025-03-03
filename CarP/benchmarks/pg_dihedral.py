import numpy as np
import py3Dmol
import networkx as nx

from pathlib import Path
import sys

from astropy.wcs.docstrings import cylfix

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import CarP
import os

pwd = os.path.abspath('')
conf = CarP.Conformer(os.path.join(pwd,'glucose_protected'))
conf.load_log(software='xyz')

cm = conf.connectivity_matrix()
cycles = (nx.cycle_basis(cm))

rd_list = conf.sort_ring_atoms(cycles, cm, conf)
pg_list = conf.find_pg(rd_list,cm)

pg = pg_list[0]
pg_values = pg.values()
pg_values = list(pg_values)

dihedral = CarP.measure_dihedral(conf,pg_values)

print('Sorted Atoms:',rd_list)
print('Identified Protecting Group:',pg_list)
print('Dihedral Angle of Protecting Group:',dihedral[0])

conf_space = [conf]
for angle in range(360):
    CarP.set_dihedral(conf,pg_values,angle)
    conf_space.append(conf)

pg_R = pg['R']
rd = rd_list[0]
c1 = rd['C1']

dist_list = []
for conf in conf_space:
    dist = CarP.measure_distance(conf,[c1,pg_R])
    dist_list.append(dist)

min_dist = min(dist_list)
min_index = dist_list.index(min_dist)

min_conf = conf_space[min_index]
conf.save_xyz()
