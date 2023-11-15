import numpy as np
import networkx as nx
from utilities import adjacent_atoms

class ConformerTest:

    def __init__(self, conn_mat, atoms):
        self.conn_mat = conn_mat
        self.atoms = atoms
    
    def pyranose_basis(self, rd, sugar_basis):
        adj_atom_O = adjacent_atoms(self.conn_mat, rd['O'])

        for atom in adj_atom_O:
            if self.atoms[atom].count('H') == 2 or [self.atoms[adj_at] for adj_at in adjacent_atoms(atom)].count('H') == 2:
                rd['C5'] = atom
            elif (
                [self.atoms[adj_at] for adj_at in adjacent_atoms(atom)].count('H') == 2 and
                [self.atoms[adj_at] for adj_at in adjacent_atoms(atom)].count('O') == 1
            ):
                rd['C5'] = atom
            elif (
                [self.atoms[adj_at] for adj_at in adjacent_atoms(atom)].count('H') == 0 and
                [self.atoms[adj_at] for adj_at in adjacent_atoms(atom)].count('O') == 2 
            ):
                rd['C5'] = atom
            elif (
                [self.atoms[adj_at] for adj_at in adjacent_atoms(atom)].count('O') == 0 and
                [self.atoms[adj_at] for adj_at in adjacent_atoms(atom)].count('H') == 3
            ):
                rd['C5'] = atom

        if sugar_basis.index(rd['C5']) == 0:
            sugar_basis.reverse()

        Carb_Oxygen = [atom for atom in sugar_basis if 'O' in atom][0]
        sugar_basis.remove(Carb_Oxygen)

        for atom_index, atom in enumerate(sugar_basis):
            rd[f"C{atom_index + 1}"] = atom

        return rd

    def furanose_basis(self, rd, sugar_basis):
        adj_atom_O = adjacent_atoms(self.conn_mat, rd['O'])

        for atom in adj_atom_O:
            if self.atoms[atom].count('H') == 2 or [self.atoms[adj_at] for adj_at in adjacent_atoms(atom)].count('H') == 2:
                rd['C5'] = atom
            elif (
                [self.atoms[adj_at] for adj_at in adjacent_atoms(atom)].count('H') == 2 and
                [self.atoms[adj_at] for adj_at in adjacent_atoms(atom)].count('O') == 1 and
                atom.count('H') == 1
            ):
                rd['C5'] = atom
            elif (
                [self.atoms[adj_at] for adj_at in adjacent_atoms(atom)].count('H') == 0 and
                [self.atoms[adj_at] for adj_at in adjacent_atoms(atom)].count('O') == 2
            ):
                rd['C5'] = atom
            elif (
                [self.atoms[adj_at] for adj_at in adjacent_atoms(atom)].count('O') == 0 and
                [self.atoms[adj_at] for adj_at in adjacent_atoms(atom)].count('H') == 3
            ):
                rd['C5'] = atom

        if sugar_basis.index(rd['C5']) == 0:
            sugar_basis.reverse()

        Carb_Oxygen = [atom for atom in sugar_basis if 'O' in atom][0]
        sugar_basis.remove(Carb_Oxygen)

        for atom_index, atom in enumerate(sugar_basis):
            rd[f"C{atom_index + 2}"] = atom

        return rd

    def sort_ring_atoms(self, cycles_in_graph, graph):
        rd_list = []

        for ring in cycles_in_graph:
            if 5 <= len(ring) <= 7:
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

            if oxygen_atoms == 1:
                sugar_basis = list(nx.cycle_basis(graph, oxygen_atom_list[0])[0])
                rd['O'] = oxygen_atom_list[0]

                if len(ring) == 6:
                    rd = self.pyranose_basis(rd, sugar_basis)
                elif len(ring) == 5:
                    rd = self.furanose_basis(rd, sugar_basis)

            if oxygen_atoms == 3 and len(ring) >= 7:
                for oxygen_atom in oxygen_atom_list:
                    test_basis = nx.minimum_cycle_basis(graph, oxygen_atom)

                    for cycle in test_basis:
                        if len(cycle) == 6:
                            oxygen_atom_counter = sum(1 for cycle_atom in cycle if self.atoms[cycle_atom] == 'O')
                            oxygen_atom_cycle_list = [cycle_atom for cycle_atom in cycle if self.atoms[cycle_atom] == 'O']

                            if oxygen_atom_counter == 1:
                                rd['O'] = oxygen_atom_cycle_list[0]
                                sugar_basis = list(cycle)
                                rd = self.pyranose_basis(rd, sugar_basis)

                        elif len(cycle) == 5:
                            oxygen_atom_cycle_list = [cycle_atom for cycle_atom in cycle if self.atoms[cycle_atom] == 'O']
                            oxygen_atom_counter = len(oxygen_atom_cycle_list)

                            if oxygen_atom_counter == 1:
                                rd['O'] = oxygen_atom_cycle_list[0]
                                sugar_basis = list(cycle)
                                rd = self.furanose_basis(rd, sugar_basis)

            rd_list.append(rd)

        return rd_list
