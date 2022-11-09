import math
from . import rmsd, utilities
import numpy as np
import sys, copy
from scipy import interpolate
from scipy.linalg import expm
from optparse import OptionParser

"""
These functions are all genetic algorithm operations. Each operation modifies existing conformer objects.
"""

def modify_glyc(conf, bond, model = "random", Fmap = None):
    """Modifies the angle between the two rings attached by a specified glycosidic bond

    :param conf: a conformer object
    :param bond: (int) index of which edge in the list of edges of the conformer
    """
    edge = conf.graph.edges[bond]
    atoms = len(edge['linker_atoms']) ; angles = [] ; n=0

    if model == "Fmaps" and edge['linker_type'] not in Fmap.keys(): model = "random"

    if model == 'random': 

        while n < atoms-3: 
            angles.append((utilities.draw_random()*360)-180.0)
            n += 1 
        if atoms == 7: #NAc linkage, set two bonds linear
            angles[1] = (utilities.draw_random_int(top=2)+0.005)*179.0
            angles[2] = (utilities.draw_random_int(top=2)+0.005)*179.0

    if model == "Fmaps" : 

        Fmap = Fmap[edge['linker_type']]
        grid = Fmap.shape[0]

        Fmap_cum = np.ravel(np.cumsum(Fmap)) 
        rnd  = utilities.draw_random()
        Fmap_cum -= rnd 
        for i, p in enumerate(Fmap_cum):
            if p > 0: break
       
        a1, a2 = int(i/grid), int(i%grid)
        angles.append(a2*360.0/grid - 180.0)
        angles.append((grid-a1)*360.0/grid - 180.0)
        #print (edge['linker_type'], angles[0], angles[1])
        #Take care of x16 linkages
        if   edge['linker_type'] == 'a16' or edge['linker_type'] == 'b16':  
            angles.append(utilities.draw_random_int(top=2)*120.0-60.0)
            #print(angles[-1])

    if   atoms == 5:
        conf.set_glycosidic(bond, angles[0], angles[1])
    elif atoms == 6:
        conf.set_glycosidic(bond, angles[0], angles[1], angles[2])
    elif atoms == 7: #NAc linkage, set two bonds linear
        conf.set_glycosidic(bond, angles[0], angles[1], angles[2], angles[3])

def modify_c6(conf, ring):
    """Modifies the 6th carbon of a ring, randomly draws an integer and edits the dihedral angle

    :param conf: a conformer object
    :param ring: (int) index that specifies which ring of the conformer is being edited 
    """
    node = conf.graph.nodes[ring]
    if 'c6_atoms' in node:
        atoms = node['c6_atoms']
        dih = ((utilities.draw_random_int(top=3)-1)*120.0)+60.0
        conf.set_c6(ring, dih)

def modify_ring(conf, ring, prob_model = None):

    pucker = draw_random_pucker(prob_model)
    #print("setting ring {0:5d} to {1:5s}".format(ring, pucker))
    utilities.set_ring_pucker(conf, ring, pucker)

def draw_random_pucker(prob_model=None):

    pucker_list = [ 'Chair', 'Boat', 'Skew', 'Half', 'Env']
    puckers = {  'Chair': ['1C4' ,  '4C1'], 
                 'Boat' : ['1,4B', 'B1,4',  '2,5B', 'B2,5', '3,6B', 'B3,6'], 
                 'Half' : ['1H2' ,  '2H1',  '2H3' ,  '3H2', '3H4' ,  '4H3',  '4H5' ,  '5H4',  '5H6' ,  '6H5',  '6H1' ,  '1H6'],
                 'Skew' : ['1S3' ,  '3S1',  '5S1' ,  '1S5',  '6S2' ,  '2S6'],
                 'Env'  : ['1E'  ,  'E1' ,  '2E'  ,  'E2' ,  '3E'  ,  'E3' ,  '4E'  ,  'E4' ,   '5E'  ,  'E5' ,  '6E'  ,  'E6' ]
                 }

    if not prob_model: 
        prob_model = [ 0.5, 0.15, 0.15, 0.1, 0.1]
    else: 
        P = 0 
        for x in prob_model: P += x 
        if len(prob_model) != 5 or P != 1: 
            print("Probablity model must have 5 elements (Chair, Boat, Skew, Half, Env) and the probabliities must add to one")

    #for i,j  in zip(pucker_list, prob_model):
    n = utilities.draw_random()
    J =  0
    for i,j  in zip(pucker_list, prob_model):
        J += j
        if n < J: 
            pucker = i
            return puckers[pucker][utilities.draw_random_int(top=len(puckers[pucker]))]

def cross_over(conf1, conf2):
    """Swaps angle measures of two conformers

    :param conf1: the first conformer object
    :param conf2: the second conformer object
    """

    #Compare the edges.
    #if connectivity is identical, compare dihs angles, then exchange the information if different. 
    #if conf1.graph.edges == conf2.graph.edges: 
    #    for e1, e2 in zip(conf1.graph.edges, conf2.graph.edges):
    #        if bond_distance(l1, conf1.graph.edges[e1]['dih'], conf2.graph.edges[e2]['dih'], 'l1') < 5.0:
    #            pass
    #        else:
    #exchange glycosidic bond:
    bond = utilities.draw_random_int(len(conf1.dih))
    print("Modifying bond number {0:2d}".format(bond))
    phi1, psi1 = conf1.dih_angels[bond]
    phi2, psi2 = conf2.dih_angels[bond] 

    conf2.set_glycosidic(bond, phi1, psi1)
    conf1.set_glycosidic(bond, phi2, psi2)




