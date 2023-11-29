import numpy as np
import networkx as nx
from utilities import adjacent_atoms
from itertools import zip_longest

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
    
    def glycosidic_link_check(self, conn_mat, rd, c1_list):
        glycosidic_link_list = []

        for ring_index in range(1, 7):
            if ring_index == 5:
                continue

            atom = rd[f"C{ring_index}"]

            for het_at in adjacent_atoms(self.atoms[atom]):
                if 'C' not in het_at and 'H' not in het_at:
                    adj_atom_list = ConformerTest.adjacent_atoms(conn_mat, het_at)

                    if [self.atoms[het_at]].count('H') == 2 and rd['C5'] not in adj_atom_list:
                        c1_count = sum(1 for adj_at in adj_atom_list if adj_at in c1_list)

                        if c1_count > 0:
                            glycosidic_link_list.append(f"C{ring_index}")

        return glycosidic_link_list
    
    def ring_dict_finder(atom, rd_list):
        for rd in rd_list:
            if atom in rd.values():
                return rd

    def find_red_end(self, c1_list, rd_list, conn_mat):
        for c1 in c1_list:
            ring_dict = ConformerTest.ring_dict_finder(c1, rd_list)

            for atom in adjacent_atoms(self.atoms[c1]):
                if 'C' not in atom and 'H' not in atom:
                    if atom in ring_dict.values():
                        continue

                    if ConformerTest.count_n(conn_mat=conn_mat, node=atom, filter='H') >= 1:
                        return rd_list.index(ring_dict)
                    else:
                        c1_count = sum(1 for adj_atom in ConformerTest.adjacent_atoms(conn_mat=conn_mat, node=atom)
                                       if adj_atom in c1_list)

                        if c1_count == 2 and len(ConformerTest.glycosidic_link_check(conn_mat, ring_dict, c1_list)) > 1:
                            return rd_list.index(ring_dict)
    
    def ring_connectivity_checker(rd1, rd2, conn_mat):
        edge_check_list = []

        atom1_list = list(range(1, len(rd1.keys())))
        atom2_list = list(range(1, len(rd2.keys())))

        for atom1_index, atom2_index in zip_longest(atom1_list, atom2_list):
            if atom1_index == 5 or atom2_index == 5:
                continue

            if len(rd1.keys()) == 7 and atom2_index is not None:
                edge_check_list.append([rd1['C1'], rd2[f"C{atom2_index}"]])
            elif len(rd1.keys()) == 6 and atom2_index is not None:
                edge_check_list.append([rd1['C2'], rd2[f"C{atom2_index}"]])
                edge_check_list.append([rd2['C1'], rd1[f"C{atom1_index}"]])

        connections = sum(1 for edge in edge_check_list if len(nx.shortest_path(conn_mat, edge[0], edge[1])) == 3)

        return connections > 0
    
    def ring_graph_maker(rd_list, conn_mat):
        ring_graph = nx.Graph()

        for rd1 in rd_list:
            for rd2 in rd_list:
                if rd1 != rd2 and ConformerTest.ring_connectivity_checker(rd1=rd1, rd2=rd2, conn_mat=conn_mat) \
                        and not ring_graph.has_edge(rd_list.index(rd1), rd_list.index(rd2)) \
                        and not ring_graph.has_edge(rd_list.index(rd2), rd_list.index(rd1)):
                    ring_graph.add_edge(rd_list.index(rd1), rd_list.index(rd2))

        return ring_graph
    
    def sort_rings(rd_list, conn_mat):
        c1_list = [rd['C1'] if 'C1' in rd else rd['C2'] for rd in rd_list]
        red_end = ConformerTest.find_red_end(c1_list=c1_list, rd_list=rd_list, conn_mat=conn_mat)
        ring_graph = ConformerTest.ring_graph_maker(rd_list=rd_list, conn_mat=conn_mat)
        tree = nx.dfs_tree(ring_graph, red_end)

        glyco_list = [ConformerTest.glycosidic_link_check(conn_mat=conn_mat, rd=rd, c1_list=c1_list) for rd in rd_list]

        return tree, glyco_list
