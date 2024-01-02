import numpy as np
import networkx as nx
from utilities import adjacent_atoms
from itertools import zip_longest
import math

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
        
        if (
            [self.atoms[rd['C6']]].count('H') >= 1 and
            [self.atoms[rd['C1']]].count('C') > 1):

            rd_index = 6
            while rd_index > 0:
                rd[f"C{rd_index +1}"] = rd[f"C{rd_index}"]
                rd_index -= 1
            
            for C2_adjaceent in adjacent_atoms(self.atoms[rd['C2']]):
                if C2_adjaceent not in sugar_basis and 'C' in C2_adjaceent:
                    rd['C1'] = C2_adjaceent

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

    def sort_ring_atoms(self, cycles_in_graph, conn_mat):
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
                sugar_basis = list(nx.cycle_basis(conn_mat, oxygen_atom_list[0])[0])
                rd['O'] = oxygen_atom_list[0]

                if len(ring) == 6:
                    rd = self.pyranose_basis(rd, sugar_basis)
                elif len(ring) == 5:
                    rd = self.furanose_basis(rd, sugar_basis)

            if oxygen_atoms == 3 and len(ring) >= 7:
                for oxygen_atom in oxygen_atom_list:
                    test_basis = nx.minimum_cycle_basis(conn_mat, oxygen_atom)

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
    
    def glycosidic_link_check(self, rd, c1_list):
        glycosidic_link_list = []

        for ring_index in range(1, len(list(rd.values()))):
            if ring_index == 5:
                continue

            atom = rd[f"C{ring_index}"]

            for het_at in adjacent_atoms(self.atoms[atom]):
                if 'C' not in het_at and 'H' not in het_at:
                    adj_atom_list = adjacent_atoms(self.atoms[het_at])

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

                    if [self.atoms[atom]].count('H') >= 1:
                        return rd_list.index(ring_dict)
                    else:
                        c1_count = sum(1 for adj_atom in adjacent_atoms(self.atoms(atom))
                                       if adj_atom in c1_list)

                        if c1_count == 2 and len(ConformerTest.glycosidic_link_check(rd=ring_dict, c1_list=c1_list)) > 1:
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
    
    def amide_check(self, rd):

        if len(list(rd.values())) == 7:
            C2 = rd['C2']
        elif len(list(rd.values())) == 8:
            C2 = rd['C3']
        else:
            C2 = None
        
        if C2 is not None:
            HC2_count = [self.atoms[C2]].count('H')
            NC2_count = [self.atoms[C2]].count('N')

            for C2_adj_at in adjacent_atoms(self.atoms[C2]):
                if 'N' in C2_adj_at:
                    HN_count = [self.atoms[C2_adj_at]].count('H')
                    CN_count = [self.atoms[C2_adj_at]].count('C')
                    
                    for N_adj_at in adjacent_atoms(self.atoms[C2_adj_at]):
                        if 'C' in N_adj_at and N_adj_at != C2:
                            OC_count = [self.atoms[N_adj_at]].count('O')
                            CC_count = [self.atoms[N_adj_at]].count('C')
            
                            if HC2_count == 1 and NC2_count == 1 and HN_count == 1 and CN_count == 2 and OC_count == 1 and CC_count == 2:
                                amide = True

                            else:
                                amide = False

            if 'amide' not in locals():
                amide = False

            return amide
    
    def ring_graph_maker(rd_list, conn_mat):
        ring_graph = nx.Graph()

        for rd1 in rd_list:
            for rd2 in rd_list:
                if rd1 != rd2 and ConformerTest.ring_connectivity_checker(rd1=rd1, rd2=rd2, conn_mat=conn_mat) \
                        and not ring_graph.has_edge(rd_list.index(rd1), rd_list.index(rd2)) \
                        and not ring_graph.has_edge(rd_list.index(rd2), rd_list.index(rd1)):
                    ring_graph.add_edge(rd_list.index(rd1), rd_list.index(rd2))
        
        if ConformerTest.amide_check(conn_mat=conn_mat,rd=rd1) == True:
                ring_graph.add_edge(f"Amide {rd_list.index(rd1)}", f"Ring {rd_list.index(rd1)}", weight=2)
        
        if ring_graph.number_of_edges() == 0:
            ring_graph.add_node('Ring 0')

        return ring_graph
    
    def sort_rings(rd_list, conn_mat):
        c1_list = [rd['C1'] if 'C1' in rd else rd['C2'] for rd in rd_list]
        red_end = ConformerTest.find_red_end(c1_list=c1_list, rd_list=rd_list, conn_mat=conn_mat)
        ring_graph = ConformerTest.ring_graph_maker(rd_list=rd_list, conn_mat=conn_mat)

        dfs_ring_list = list(nx.dfs_preorder_nodes(ring_graph, red_end))
        for dfs_index,node in enumerate(dfs_ring_list):
            if 'Amide' in node:
                dfs_ring_list.remove(node)
            else:
                rd_list_index = int(list(node.split())[-1])
                dfs_ring_list[dfs_index] = rd_list[rd_list_index]
                
        if dfs_ring_list == []:
            dfs_ring_list = rd_list

        glyco_list = [ConformerTest.glycosidic_link_check(conn_mat=conn_mat, rd=rd, c1_list=c1_list) for rd in dfs_ring_list]
        return dfs_ring_list,glyco_list
    
    def dihedral_angle(atom1, atom2, atom3, atom4, conf):

        atom1_index = list(conf[:, 0]).index(atom1)
        atom2_index = list(conf[:, 0]).index(atom2)
        atom3_index = list(conf[:, 0]).index(atom3)
        atom4_index = list(conf[:, 0]).index(atom4)

        atom1_coords = []
        atom2_coords = []
        atom3_coords = []
        atom4_coords = []

        for c_index in range(1, 4):
            atom1_coords.append(float(conf[atom1_index, c_index]))
            atom2_coords.append(float(conf[atom2_index, c_index]))
            atom3_coords.append(float(conf[atom3_index, c_index]))
            atom4_coords.append(float(conf[atom4_index, c_index]))

        vector_1 = np.array([coord2 - coord1 for coord1, coord2 in zip(atom1_coords, atom2_coords)])
        vector_2 = np.array([coord2 - coord1 for coord1, coord2 in zip(atom2_coords, atom3_coords)])
        vector_3 = np.array([coord2 - coord1 for coord1, coord2 in zip(atom3_coords, atom4_coords)])

        norm1 = np.cross(vector_1, vector_2)
        norm1_mag = math.sqrt(np.sum([n1 ** 2 for n1 in norm1]))
        norm1 = np.array([n1 / norm1_mag for n1 in norm1])

        norm2 = np.cross(vector_2, vector_3)
        norm2_mag = math.sqrt(np.sum([n2 ** 2 for n2 in norm2]))
        norm2 = np.array([n2 / norm2_mag for n2 in norm2])

        vector2_mag = math.sqrt(np.sum([vcoord ** 2 for vcoord in vector_2]))
        unit_vector_2 = np.array([vcoord / vector2_mag for vcoord in vector_2])
        frame_vector = np.cross(norm1, unit_vector_2)

        x = np.dot(norm1, norm2)
        y = np.dot(frame_vector, norm2)

        dihedral = math.atan2(y, x)

        return dihedral
    def sugar_stero(rd, conf):
        if len(rd.values()) == 7:
            dihedral_angle = ConformerTest.dihedral_angle(rd['O'], rd['C5'], rd['C4'], rd['C6'], conf)
        elif len(rd.values()) > 7:
            dihedral_angle = ConformerTest.dihedral_angle(rd['O'], rd['C6'], rd['C5'], rd['C7'], conf)
        else:
            dihedral_angle = None

        if dihedral_angle is not None and 0 > dihedral_angle:
            sugar_type = 'D'
        elif dihedral_angle is not None and dihedral_angle > 0:
            sugar_type = "L"
        else:
            sugar_type = 'None'

        return sugar_type
    
    def glycosidic_link_type(rd, sugar_type, conf, conn_mat):

        if len(rd.values()) == 7:

            enumeric_H = [adj_at for adj_at in ConformerTest.adjacent_atoms(conn_mat, rd['C1']) if
                          'H' in adj_at and adj_at not in rd][0]
            dihedral_angle = ConformerTest.dihedral_angle(rd['O'], rd['C1'], rd['C2'], enumeric_H, conf)

        elif len(rd.values()) > 7:

            enumeric_H = [adj_at for adj_at in ConformerTest.adjacent_atoms(conn_mat, rd['C2']) if
                          'H' in adj_at and adj_at not in rd][0]
            dihedral_angle = ConformerTest.dihedral_angle(rd['O'], rd['C2'], rd['C3'], enumeric_H, conf)

        else:
            dihedral_angle = None

        if dihedral_angle is not None and 0 > dihedral_angle:
            if sugar_type == 'D':
                link_type = 'B'
            elif sugar_type == "L":
                link_type = 'A'
            else:
                link_type = "None"

        elif dihedral_angle is not None and dihedral_angle > 0:
            if sugar_type == 'D':
                link_type = 'A'
            elif sugar_type == "L":
                link_type = 'B'
            else:
                link_type = "None"
        else:
            link_type = "None"

        return link_type
    def ring_stereo_compiler(self, conf, dfs_list, conn_mat):

        sugar_type_list = []
        glyco_type_list = []

        for rd in dfs_list:
            sugar_type = ConformerTest.sugar_stero(rd, conf)
            link_type = ConformerTest.glycosidic_link_type(rd, sugar_type, conf, conn_mat)
            sugar_type_list.append(sugar_type)
            glyco_type_list.append(link_type)

        return sugar_type_list, glyco_type_list
    
    