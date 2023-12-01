import networkx as nx
import numpy as np
import math
import matplotlib.pyplot as plt
from itertools import zip_longest

class ConformerTest:

    @staticmethod
    def xyztograph(path):
        file_open = open(path)
        xyz_array = []

        for line in file_open:
            if '   ' in line:
                line_list = line.split()
                xyz_array.append(line_list)
        
        xyz_array = np.array(xyz_array)

        distXH = 1.25
        distXX = 1.65
        
        Nat = len(xyz_array[:,0])
        Conn_Mat = nx.Graph()

        for atom1 in range(Nat):
            for atom2 in range(Nat):
                atom1_type = xyz_array[atom1,0]
                atom2_type = xyz_array[atom2,0]

                delx = float(xyz_array[atom2,1]) - float(xyz_array[atom1,1])
                dely = float(xyz_array[atom2,2]) - float(xyz_array[atom1,2])
                delz = float(xyz_array[atom2,3]) - float(xyz_array[atom1,3])

                distance = math.sqrt(delx**2 + dely**2 + delz**2)

                if atom1_type == 'H' and atom2_type == 'H':
                    continue
                elif atom1_type == 'H' or atom2_type == 'H':
                    if distance <= distXH:
                        if (Conn_Mat.has_edge(f"{atom1_type}{atom1+1}", f"{atom2_type}{atom2+1}") == False and
                            Conn_Mat.has_edge(f"{atom2_type}{atom2+1}", f"{atom1_type}{atom1+1}") == False
                            ):
                            Conn_Mat.add_edge(f"{atom1_type}{atom1+1}", f"{atom2_type}{atom2+1}")
                
                elif atom1_type != 'H' and atom2_type != 'H':
                    if distance <= distXX:
                        if (Conn_Mat.has_edge(f"{atom1_type}{atom1+1}", f"{atom2_type}{atom2+1}") == False and
                            Conn_Mat.has_edge(f"{atom2_type}{atom2+1}", f"{atom1_type}{atom1+1}") == False
                            ):
                            Conn_Mat.add_edge(f"{atom1_type}{atom1+1}", f"{atom2_type}{atom2+1}")

        return Conn_Mat

    @staticmethod
    def count_n(conn_mat, node, filter):
        counter = 0
        for neighbor in conn_mat.neighbors(node):
            if filter in neighbor:
                counter += 1
        return counter

    @staticmethod
    def adjacent_atoms(conn_mat, node):
        return [neighbor for neighbor in conn_mat.neighbors(node)]

    @staticmethod
    def pyranose_basis(conn_mat, oxygen_atom, sugar_basis, rd):
        adj_atom_O = ConformerTest.adjacent_atoms(conn_mat, oxygen_atom)

        for atom in adj_atom_O:
            if ConformerTest.count_n(conn_mat, atom, 'H') == 2:
                rd['C5'] = atom
            else:
                for adj_at in ConformerTest.adjacent_atoms(conn_mat, atom):
                    if (
                        ConformerTest.count_n(conn_mat, adj_at, 'H') == 2 and
                        ConformerTest.count_n(conn_mat, adj_at, 'O') == 1 and
                        ConformerTest.count_n(conn_mat, atom, 'H') == 1
                    ):
                        rd['C5'] = atom
                        rd['C6'] = adj_at
                    elif (
                        ConformerTest.count_n(conn_mat, adj_at, 'H') == 0 and
                        ConformerTest.count_n(conn_mat, adj_at, 'O') == 2
                    ):
                        rd['C5'] = atom
                        rd['C6'] = adj_at
                    elif ConformerTest.count_n(conn_mat, adj_at, 'H') == 3:
                        rd['C5'] = atom
                        rd['C6'] = adj_at

        if sugar_basis.index(rd['C5']) == 0:
            sugar_basis.reverse()

        Carb_Oxygen = [atom for atom in sugar_basis if 'O' in atom][0]
        sugar_basis.remove(Carb_Oxygen)

        for atom_index, atom in enumerate(sugar_basis):
            rd[f"C{atom_index + 1}"] = atom

        return rd

    @staticmethod
    def furanose_basis(conn_mat, oxygen_atom, sugar_basis, rd):
        adj_atom_O = ConformerTest.adjacent_atoms(conn_mat, oxygen_atom)

        for atom in adj_atom_O:
            if ConformerTest.count_n(conn_mat, atom, 'H') == 2:
                rd['C5'] = atom
            else:
                for adj_at in ConformerTest.adjacent_atoms(conn_mat, atom):
                    if (
                        ConformerTest.count_n(conn_mat, adj_at, 'H') == 2 and
                        ConformerTest.count_n(conn_mat, adj_at, 'O') == 1 and
                        ConformerTest.count_n(conn_mat, atom, 'H') == 1
                    ):
                        rd['C5'] = atom
                        rd['C6'] = adj_at
                    elif (
                        ConformerTest.count_n(conn_mat, adj_at, 'H') == 0 and
                        ConformerTest.count_n(conn_mat, adj_at, 'O') == 2
                    ):
                        rd['C5'] = atom
                        rd['C6'] = adj_at
                    elif ConformerTest.count_n(conn_mat, adj_at, 'H') == 3:
                        rd['C5'] = atom
                        rd['C6'] = adj_at

        if sugar_basis.index(rd['C5']) == 0:
            sugar_basis.reverse()

        Carb_Oxygen = [atom for atom in sugar_basis if 'O' in atom][0]
        sugar_basis.remove(Carb_Oxygen)

        for atom_index, atom in enumerate(sugar_basis):
            rd[f"C{atom_index + 2}"] = atom

        return rd

    @staticmethod
    def sort_ring_atoms(cycles_in_conn_mat, conn_mat):
        rd_list = []

        for ring in cycles_in_conn_mat:
            rd = {}
            oxygen_atoms = 0
            oxygen_atom_list = []

            for at in ring:
                if 'O' in at:
                    oxygen_atoms += 1
                    oxygen_atom_list.append(at)

            if oxygen_atoms == 0:
                continue

            if oxygen_atoms == 1:
                sugar_basis_list = list(nx.cycle_basis(conn_mat, oxygen_atom_list[0]))
                rd['O'] = oxygen_atom_list[0]
                for sugar_basis in sugar_basis_list:
                    if len(sugar_basis) == len(ring) and rd['O'] in sugar_basis:

                        if len(ring) == 6:
                            rd = ConformerTest.pyranose_basis(conn_mat, rd['O'], sugar_basis, rd)
                        elif len(ring) == 5:
                            rd = ConformerTest.furanose_basis(conn_mat, rd['O'], sugar_basis, rd)

            if oxygen_atoms == 3 and len(ring) >= 7:
                for oxygen_atom in oxygen_atom_list:
                    test_basis = nx.minimum_cycle_basis(conn_mat, oxygen_atom)
                    for cycle in test_basis:
                        if len(cycle) == 6:
                            oxygen_atom_counter = sum(1 for cycle_atom in cycle if 'O' in cycle_atom)
                            oxygen_atom_cycle_list = [cycle_atom for cycle_atom in cycle if 'O' in cycle_atom]

                            if oxygen_atom_counter == 1:
                                rd['O'] = oxygen_atom_cycle_list[0]
                                sugar_basis = list(cycle)
                                rd = ConformerTest.pyranose_basis(conn_mat, rd['O'], sugar_basis, rd)

                        elif len(cycle) == 5:
                            oxygen_atom_cycle_list = [cycle_atom for cycle_atom in cycle if cycle_atom == 'O']
                            oxygen_atom_counter = len(oxygen_atom_cycle_list)

                            if oxygen_atom_counter == 1:
                                rd['O'] = oxygen_atom_cycle_list[0]
                                sugar_basis = list(cycle)
                                rd = ConformerTest.furanose_basis(conn_mat, rd['O'], sugar_basis, rd)

            rd_list.append(rd)

        rd_list = [rd for rd in rd_list if len(rd.keys()) >= 5]
        return rd_list
    
    @staticmethod
    def glycosidic_link_check(conn_mat, rd, c1_list):
        glycosidic_link_list = []

        for ring_index in range(1, 7):
            if ring_index == 5:
                continue

            atom = rd[f"C{ring_index}"]

            for het_at in ConformerTest.adjacent_atoms(conn_mat, atom):
                if 'C' not in het_at and 'H' not in het_at:
                    adj_atom_list = ConformerTest.adjacent_atoms(conn_mat, het_at)

                    if ConformerTest.count_n(conn_mat, het_at, 'C') == 2 and rd['C5'] not in adj_atom_list:
                        c1_count = sum(1 for adj_at in adj_atom_list if adj_at in c1_list)

                        if c1_count > 0:
                            glycosidic_link_list.append(f"C{ring_index}")

        return glycosidic_link_list
    
    @staticmethod
    def amide_check(conn_mat, rd):
        C2 = rd['C2']
        HC2_count = ConformerTest.count_n(conn_mat=conn_mat, node=C2, filter='H')
        NC2_count = ConformerTest.count_n(conn_mat=conn_mat, node=C2, filter='N')

        for C2_adj_at in ConformerTest.adjacent_atoms(conn_mat=conn_mat, node=C2):
            if 'N' in C2_adj_at:
                HN_count = ConformerTest.count_n(conn_mat=conn_mat, node=C2_adj_at, filter='H')
                CN_count = ConformerTest.count_n(conn_mat=conn_mat, node=C2_adj_at, filter='C')
                
                for N_adj_at in ConformerTest.adjacent_atoms(conn_mat=conn_mat, node=C2_adj_at):
                    if 'C' in N_adj_at and N_adj_at != C2:
                        OC_count = ConformerTest.count_n(conn_mat=conn_mat, node=N_adj_at, filter='O')
                        CC_count = ConformerTest.count_n(conn_mat=conn_mat, node=N_adj_at, filter='C')
        
                        if HC2_count == 1 and NC2_count == 1 and HN_count == 1 and CN_count == 2 and OC_count == 1 and CC_count == 2:
                            amide = True

                        else:
                            amide = False
        if 'amide' not in locals():
            amide = False

        return amide

    @staticmethod
    def ring_dict_finder(atom, rd_list):
        for rd in rd_list:
            if atom in rd.values():
                return rd

    @staticmethod
    def find_red_end(c1_list, rd_list, conn_mat):
        for c1 in c1_list:
            ring_dict = ConformerTest.ring_dict_finder(c1, rd_list)

            for atom in ConformerTest.adjacent_atoms(conn_mat=conn_mat, node=c1):
                if 'C' not in atom and 'H' not in atom:
                    if atom in ring_dict.values():
                        continue

                    if ConformerTest.count_n(conn_mat=conn_mat, node=atom, filter='H') >= 1:
                        return f"Ring {rd_list.index(ring_dict)}"
                    else:
                        c1_count = sum(1 for adj_atom in ConformerTest.adjacent_atoms(conn_mat=conn_mat, node=atom)
                                       if adj_atom in c1_list)

                        if c1_count == 2 and len(ConformerTest.glycosidic_link_check(conn_mat, ring_dict, c1_list)) > 1:
                            return f"Ring {rd_list.index(ring_dict)}"

    @staticmethod
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

    @staticmethod
    def ring_graph_maker(rd_list, conn_mat):
        ring_graph = nx.Graph()

        for rd1 in rd_list:
            for rd2 in rd_list:
                if rd1 != rd2 and ConformerTest.ring_connectivity_checker(rd1=rd1, rd2=rd2, conn_mat=conn_mat) \
                        and not ring_graph.has_edge(f"Ring {rd_list.index(rd1)}", f"Ring {rd_list.index(rd2)}") \
                        and not ring_graph.has_edge(f"Ring {rd_list.index(rd2)}", f"Ring {rd_list.index(rd1)}"):
                    ring_graph.add_edge(f"Ring {rd_list.index(rd1)}", f"Ring {rd_list.index(rd2)}", weight=1)
            
            if ConformerTest.amide_check(conn_mat=conn_mat,rd=rd1) == True:
                ring_graph.add_edge(f"Amide {rd_list.index(rd1)}", f"Ring {rd_list.index(rd1)}", weight=2)

        return ring_graph

    @staticmethod
    def sort_rings(rd_list, conn_mat):
        c1_list = [rd['C1'] if 'C1' in rd else rd['C2'] for rd in rd_list]
        red_end = ConformerTest.find_red_end(c1_list=c1_list, rd_list=rd_list, conn_mat=conn_mat)
        ring_graph = ConformerTest.ring_graph_maker(rd_list=rd_list, conn_mat=conn_mat)
        tree = nx.dfs_tree(ring_graph, red_end)

        glyco_list = [ConformerTest.glycosidic_link_check(conn_mat=conn_mat, rd=rd, c1_list=c1_list) for rd in rd_list]
        
        dfs_ring_list = list(nx.dfs_preorder_nodes(ring_graph, red_end))
        for dfs_index,node in enumerate(dfs_ring_list):
            if 'Amide' in node:
                dfs_ring_list.remove(node)
            else:
                rd_list_index = int(list(node.split())[-1])
                dfs_ring_list[dfs_index] = rd_list[rd_list_index]
                

        return tree, glyco_list, dfs_ring_list

if __name__ == "__main__":
    #This section is just for testing (Will not be in final code)
    xyz_file = input()
    if '.xyz' in xyz_file:
        conn_mat = ConformerTest.xyztograph(xyz_file)
        cycles_in_graph = nx.cycle_basis(conn_mat)

    
        ring_dict_list = ConformerTest.sort_ring_atoms(cycles_in_conn_mat=cycles_in_graph, conn_mat=conn_mat)
        print(ring_dict_list)
        ring_tree, glyco_array, dfs_ring_list = ConformerTest.sort_rings(rd_list=ring_dict_list, conn_mat=conn_mat)
        print(glyco_array)
        print(dfs_ring_list)
        nx.draw(ring_tree, with_labels=True)
        plt.show()
        
