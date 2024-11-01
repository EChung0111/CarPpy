import numpy as np
import matplotlib.pyplot as plt
import os

from matplotlib.pyplot import savefig
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
import networkx as nx

from pathlib import Path
import sys
import shutil

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import CarP
import os

pwd = os.path.abspath('')
conf = CarP.Conformer(os.path.join(pwd,'glucose'))
conf.load_log(software='g16')

conf.plot_ir(xmin=0,xmax=4000,save_fig=True)