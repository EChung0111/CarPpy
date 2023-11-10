import numpy as np
import re, os
from subprocess import Popen, PIPE
import networkx as nx
from operator import itemgetter, attrgetter
import matplotlib.pyplot as plt
from utilities import *

class Conformer_Test:

    def sort_ring_atoms(self, cycles_in_graph, graph):
        rd_list = []

        for ring in cycles_in_graph:
            if len(ring) > 7 or len(ring) < 5:
                continue

            rd = {}
            oxygen_atoms = 0
            oxygen_atom_list = []
            for at in ring:
                if self.atoms[at] == 'O':
                    oxygen_atoms += 1
                    oxygen_atom_list.append(at)
            
            if oxygen_atoms == 0:
                continue

            def pyranose_basis():
                adj_atom_O = adjacent_atoms(self.conn_mat, rd['O'])

                for atom in adj_atom_O:
                    if (self.atoms[atom].count('H') == 2) or ([self.atoms[adj_at] for adj_at in adjacent_atoms(atom)].count('H') == 2): #Reduced C5
                        rd['C5'] = atom
                    elif [self.atoms[adj_at] for adj_at in adjacent_atoms(atom)].count('H') == 2 and [self.atoms[adj_at] for adj_at in adjacent_atoms(atom)].count('O') == 1: #Normal Case
                        rd['C5'] = atom
                    elif [self.atoms[adj_at] for adj_at in adjacent_atoms(atom)].count('H') == 0 and [self.atoms[adj_at] for adj_at in adjacent_atoms(atom)].count('O') == 2: #C6 is Carboxylic Acid
                        rd['C5'] = atom
                    elif [self.atoms[adj_at] for adj_at in adjacent_atoms(atom)].count('H') == 3: #C6 is Reduced
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
            
            def furanose_basis():
                adj_atom_O = adjacent_atoms(self.conn_mat, rd['O'])

                for atom in adj_atom_O:
                    if (self.atoms[atom].count('H') == 2) or ([self.atoms[adj_at] for adj_at in adjacent_atoms(atom)].count('H') == 2): #Reduced C4
                        rd['C4'] = atom
                    elif [self.atoms[adj_at] for adj_at in adjacent_atoms(atom)].count('H') == 2 and [self.atoms[adj_at] for adj_at in adjacent_atoms(atom)].count('O') == 1: #Normal Case
                        rd['C4'] = atom
                    elif [self.atoms[adj_at] for adj_at in adjacent_atoms(atom)].count('H') == 0 and [self.atoms[adj_at] for adj_at in adjacent_atoms(atom)].count('O') == 2: #C5 is Carboxylic Acid
                        rd['C4'] = atom
                    elif [self.atoms[adj_at] for adj_at in adjacent_atoms(atom)].count('H') == 3: #C5 is Reduced
                        rd['C4'] = atom

                if sugar_basis.index(rd['C4']) == 0: #Check to see if order needs to be flipped
                    sugar_basis = sugar_basis.reverse()
                   
                for atom in sugar_basis:
                    if 'O' in atom:
                        Carb_Oxygen = atom
                sugar_basis.remove(Carb_Oxygen)
                    
                for atom_index, atom in enumerate(sugar_basis):
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
                                    if self.atoms[cycle_atom] == 'O':
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
                                    if self.atoms[cycle_atom] == 'O':
                                        oxygen_atom_counter += 1
                                        oxygen_atom_cycle_list.append(cycle_atom)
                                if oxygen_atom_counter == 1:
                                    rd['O'] = oxygen_atom_cycle_list[0]
                                    sugar_basis = list(cycle)
                                    rd = furanose_basis(sugar_basis)

            rd_list.append(rd)
        print(rd_list)
        return rd_list

if __name__ == '__main__':
    
    Carb_Graph = nx.Graph()
    