import numpy as np
import re, os
from subprocess import Popen, PIPE
import networkx as nx
from operator import itemgetter, attrgetter
import matplotlib.pyplot as plt
from utilities import *

class Conformer_Test:

    def sort_ring_atoms(cycles_in_graph, graph):
        rd_list = []

        for ring in cycles_in_graph:

            rd = {}
            oxygen_atoms = 0
            oxygen_atom_list = []
            for at in ring:
                if 'O' in at:
                    oxygen_atoms += 1
                    oxygen_atom_list.append(at)

            if oxygen_atoms == 0:
                continue
            
            def countn(graph, node, filter):
                counter = 0
                for nat in graph.neighbors(node):
                    if filter in nat:
                        counter += 1
                return counter
            
            def adjacent_atoms(graph, node):
                n_list = [nat for nat in graph.neighbors(node)]
                return n_list
            
            def pyranose_basis(sugar_basis):
                adj_atom_O = adjacent_atoms(graph, rd['O'])

                for atom in adj_atom_O:
                    if countn(graph, atom, 'H') == 2: #Reduced C5
                        rd['C5'] = atom
                    else:
                        for adj_at in adjacent_atoms(graph, atom):
                            if countn(graph, adj_at, 'H') == 2 and countn(graph, adj_at, 'O') == 1: #Normal Case
                                rd['C5'] = atom
                            elif countn(graph, adj_at, 'H') == 0 and countn(graph, adj_at, 'O') == 2: #C6 is Carboxylic Acid
                                rd['C5'] = atom
                            elif countn(graph, adj_at, 'H') == 3: #C6 is Methyl
                                rd['C5'] = atom

                if sugar_basis.index(rd['C5']) == 0: #Check to see if order needs to be flipped
                    sugar_basis = sugar_basis.reverse()
                    
                for atom in sugar_basis:
                    if 'O' in atom:
                        Carb_Oxygen = atom
                sugar_basis.remove(Carb_Oxygen)
                    
                for atom_index, atom in enumerate(sugar_basis):
                    rd[f"C{atom_index+1}"] = atom
                
                return rd
            
            def furanose_basis(sugar_basis):
                adj_atom_O = adjacent_atoms(graph, rd['O'])

                for atom in adj_atom_O:
                    if countn(graph, atom, 'H') == 2: #Reduced C5
                        rd['C4'] = atom
                    else:
                        for adj_at in adjacent_atoms(graph, atom):
                            if countn(graph, adj_at, 'H') == 2 and countn(graph, adj_at, 'O') == 1: #Normal Case
                                rd['C4'] = atom
                            elif countn(graph, adj_at, 'H') == 0 and countn(graph, adj_at, 'O') == 2: #C5 is Carboxylic Acid
                                rd['C4'] = atom
                            elif countn(graph, adj_at, 'H') == 3: #C5 is Reduced
                                rd['C4'] = atom

                if sugar_basis.index(rd['C4']) == 0: #Check to see if order needs to be flipped
                    sugar_basis = sugar_basis.reverse()
                    
                for atom in sugar_basis:
                    if 'O' in atom:
                        Carb_Oxygen = atom
                sugar_basis.remove(Carb_Oxygen)
                    
                for atom_index, atom in enumerate(sugar_basis):
                    if 'O' in atom:
                        sugar_basis.remove(atom)
                    rd[f"C{atom_index+1}"] = atom

                return rd

            if oxygen_atoms == 1: #normal case
                sugar_basis = list(nx.cycle_basis(graph, oxygen_atom_list[0])[0])
                rd['O'] = oxygen_atom_list[0]
                if len(ring) == 6: #6 membered sugar rings
                    rd = pyranose_basis(sugar_basis)
                
                elif len(ring) == 5: #5 membered sugar ring
                    rd = furanose_basis(sugar_basis)

            if oxygen_atoms == 3: #fused ring
                if len(ring) >= 7:

                    for oxygen_atom in oxygen_atom_list:
                        test_basis = nx.minimum_cycle_basis(graph, oxygen_atom)
                        for cycle in test_basis:
                            if len(cycle) == 6:
                                oxygen_atom_counter = 0
                                oxygen_atom_cycle_list = []
                                for cycle_atom in cycle:
                                    if 'O' in cycle_atom:
                                        oxygen_atom_counter += 1
                                        oxygen_atom_cycle_list.append(cycle_atom)
                                if oxygen_atom_counter == 1:
                                    rd['O'] = oxygen_atom_cycle_list[0]
                                    sugar_basis = list(cycle)
                                rd = pyranose_basis(sugar_basis)

                            elif len(cycle) == 5:
                                oxygen_atom_cycle_list = []
                                oxygen_atom_counter = 0
                                for cycle_atom in cycle:
                                    if cycle_atom == 'O':
                                        oxygen_atom_counter += 1
                                        oxygen_atom_cycle_list.append(cycle_atom)
                                if oxygen_atom_counter == 1:
                                    rd['O'] = oxygen_atom_cycle_list[0]
                                    sugar_basis = list(cycle)
                                    rd = furanose_basis(sugar_basis)

            rd_list.append(rd)
        for rd in rd_list:
            if rd == {}:
                rd_list.remove(rd)
        return rd_list

if __name__ == '__main__':
    
    Furanose_Graph = nx.Graph()
    nx.add_cycle(Furanose_Graph, ['C2', 'C4', 'C5', 'C6', 'O7'])
    Furanose_Graph.add_edge('C1', 'C2')
    Furanose_Graph.add_edge('O12', 'C1')
    Furanose_Graph.add_edge('O9', 'C5')
    Furanose_Graph.add_edge('O10', 'C4')
    Furanose_Graph.add_edge('O8', 'C6')
    Furanose_Graph.add_edge('H24', 'O12')
    Furanose_Graph.add_edge('H13', 'C1')
    Furanose_Graph.add_edge('H14', 'C1')
    Furanose_Graph.add_edge('H15', 'C2')
    Furanose_Graph.add_edge('H17', 'C4')
    Furanose_Graph.add_edge('H18', 'C5')
    Furanose_Graph.add_edge('H19', 'C6')
    fcycles_in_graph = nx.cycle_basis(Furanose_Graph)

    Pyranose_Graph = nx.Graph()
    nx.add_cycle(Pyranose_Graph, ['C2', 'C3', 'C4', 'C5', 'C6', 'O7'])
    Pyranose_Graph.add_edge('C1', 'C2')
    Pyranose_Graph.add_edge('O12', 'C1')
    Pyranose_Graph.add_edge('O11', 'C3')
    Pyranose_Graph.add_edge('O9', 'C5')
    Pyranose_Graph.add_edge('O10', 'C4')    

    Pyranose_Graph.add_edge('O8', 'C6')
    Pyranose_Graph.add_edge('H24', 'O12')
    Pyranose_Graph.add_edge('H13', 'C1')
    Pyranose_Graph.add_edge('H14', 'C1')
    Pyranose_Graph.add_edge('H15', 'C2')
    Pyranose_Graph.add_edge('H16', 'C3')
    Pyranose_Graph.add_edge('H17', 'C4')
    Pyranose_Graph.add_edge('H18', 'C5')
    Pyranose_Graph.add_edge('H19', 'C6')
    pcycles_in_graph = nx.cycle_basis(Pyranose_Graph)

    Benzene_Graph = nx.Graph()
    nx.add_cycle(Benzene_Graph, ['C1', 'C2', 'C3', 'C4', 'C5', 'C6'])
    bcycles_in_graph = nx.cycle_basis(Benzene_Graph)

    Fused_Graph = nx.Graph()
    nx.add_cycle(Fused_Graph, ['C2', 'C3', 'C4', 'C5', 'C6', 'O7'])
    nx.add_cycle(Fused_Graph, ['O7', 'C6', 'O32', 'C31', 'O9', 'C3', 'C2'])
    nx.add_cycle(Fused_Graph, ['C6', 'C5', 'C4', 'C3', 'O9', 'C31', 'O32'])
    Fused_Graph.add_edge('C1', 'C2')
    Fused_Graph.add_edge('O12', 'C1')
    Fused_Graph.add_edge('O18', 'C5')
    Fused_Graph.add_edge('O10', 'C4')
    Fused_Graph.add_edge('H24', 'O12')
    Fused_Graph.add_edge('H13', 'C1')
    Fused_Graph.add_edge('H14', 'C1')
    Fused_Graph.add_edge('H15', 'C2')
    Fused_Graph.add_edge('H16', 'C3')
    Fused_Graph.add_edge('H17', 'C4')
    Fused_Graph.add_edge('H18', 'C5')
    Fused_Graph.add_edge('H19', 'C6')
    fzcycles_in_graph = nx.cycle_basis(Fused_Graph)
    
    ring_dict_list = Conformer_Test.sort_ring_atoms(cycles_in_graph=fzcycles_in_graph, graph=Fused_Graph)
    print(ring_dict_list)