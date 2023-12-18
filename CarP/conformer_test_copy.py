import networkx as nx
import numpy as np
import math
import matplotlib.pyplot as plt
from itertools import zip_longest

class ConformerTest:

    @staticmethod
    def xyztoarray(path):
        file_open = open(path)
        xyz_array = []

        for line in file_open:
            if '   ' in line:
                line_list = line.split()
                xyz_array.append(line_list)
        
        xyz_array = np.array(xyz_array)

        for rownum,row in enumerate(xyz_array):
            xyz_array[rownum,0] = f"{xyz_array[rownum,0]}{rownum+1}"

        return xyz_array

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
            if filter in neighbor and neighbor != node:
                counter += 1
        return counter

    @staticmethod
    def adjacent_atoms(conn_mat, node):
        neighbor_list = [neighbor for neighbor in conn_mat.neighbors(node)]

        for neighbor in neighbor_list:
            if neighbor == node:
                neighbor_list.remove(neighbor)
        return neighbor_list

    @staticmethod
    def pyranose_basis(conn_mat, oxygen_atom, sugar_basis, rd):
        adj_atom_O = ConformerTest.adjacent_atoms(conn_mat, oxygen_atom)

        for atom in adj_atom_O:
            if ConformerTest.count_n(conn_mat, atom, 'H') == 2:
                rd['C5'] = atom
            else:
                for adj_at in ConformerTest.adjacent_atoms(conn_mat, atom):
                    if (
                        ConformerTest.count_n(conn_mat, adj_at, 'H') >= 1 and
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

        if (
            ConformerTest.count_n(conn_mat=conn_mat, node=rd['C6'], filter='H') >= 1 and
            ConformerTest.count_n(conn_mat=conn_mat, node=rd['C1'], filter='C') > 1):

            rd_index = 6
            while rd_index > 0:
                rd[f"C{rd_index +1}"] = rd[f"C{rd_index}"]
                rd_index -= 1
            
            for C2_adjaceent in ConformerTest.adjacent_atoms(conn_mat=conn_mat, node=rd['C2']):
                if C2_adjaceent not in sugar_basis and 'C' in C2_adjaceent:
                    rd['C1'] = C2_adjaceent

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
                        ConformerTest.count_n(conn_mat, adj_at, 'H') >= 1 and
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
    def dihedral_angle(atom1, atom2, atom3, atom4, xyz_array):

        atom1_index = list(xyz_array[:,0]).index(atom1)
        atom2_index = list(xyz_array[:,0]).index(atom2)
        atom3_index = list(xyz_array[:,0]).index(atom3)
        atom4_index = list(xyz_array[:,0]).index(atom4)

        atom1_coords = []
        atom2_coords = []
        atom3_coords = []
        atom4_coords = []
        
        for c_index in range(1,4):
            atom1_coords.append(float(xyz_array[atom1_index,c_index]))
            atom2_coords.append(float(xyz_array[atom2_index,c_index]))
            atom3_coords.append(float(xyz_array[atom3_index,c_index]))
            atom4_coords.append(float(xyz_array[atom4_index,c_index]))
        
        vector_1 = np.array([coord2-coord1 for coord1,coord2 in zip(atom1_coords,atom2_coords)])
        vector_2 = np.array([coord2-coord1 for coord1,coord2 in zip(atom2_coords,atom3_coords)])
        vector_3 = np.array([coord2-coord1 for coord1,coord2 in zip(atom3_coords,atom4_coords)])

        norm1 = np.cross(vector_1,vector_2)
        norm1_mag = math.sqrt(np.sum([n1**2 for n1 in norm1]))
        norm1 = np.array([n1/norm1_mag for n1 in norm1])

        norm2 = np.cross(vector_2,vector_3)
        norm2_mag = math.sqrt(np.sum([n2**2 for n2 in norm2]))
        norm2 = np.array([n2/norm2_mag for n2 in norm2])

        vector2_mag =  math.sqrt(np.sum([vcoord**2 for vcoord in vector_2]))
        unit_vector_2 = np.array([vcoord/vector2_mag for vcoord in vector_2])
        frame_vector = np.cross(norm1,unit_vector_2)

        x =  np.dot(norm1,norm2)
        y =  np.dot(frame_vector,norm2)

        dihedral = math.atan2(y,x)

        return dihedral

    @staticmethod
    def sugar_stero(rd, xyz_array):
        if len(rd.values()) == 7:
            dihedral_angle = ConformerTest.dihedral_angle(rd['O'], rd['C1'], rd['C2'], rd['C6'], xyz_array)
        elif len(rd.values()) > 7:
            dihedral_angle = ConformerTest.dihedral_angle(rd['O'], rd['C2'], rd['C3'], rd['C7'], xyz_array)
        else:
            dihedral_angle = None

        if dihedral_angle is not None and 0 > dihedral_angle:
            sugar_type = 'D'
        elif dihedral_angle is not None and dihedral_angle > 0:
            sugar_type = "L"
        else:
            sugar_type = 'None'
        
        return sugar_type
    
    @staticmethod
    def glycosidic_link_type(rd, sugar_type, xyz_array, conn_mat):

        if len(rd.values()) == 7:
            
            enumeric_H = [adj_at for adj_at in ConformerTest.adjacent_atoms(conn_mat, rd['C1']) if 'H' in adj_at and adj_at not in rd][0]
            dihedral_angle = ConformerTest.dihedral_angle(rd['O'], rd['C1'], rd['C2'], enumeric_H, xyz_array)
        
        elif len(rd.values()) > 7:

            enumeric_H = [adj_at for adj_at in ConformerTest.adjacent_atoms(conn_mat, rd['C2']) if 'H' in adj_at and adj_at not in rd][0]
            dihedral_angle = ConformerTest.dihedral_angle(rd['O'], rd['C2'], rd['C3'], enumeric_H, xyz_array)
       
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
        
    @staticmethod
    def glycosidic_link_check(conn_mat, rd,  c1_list):
        glycosidic_link_list = []

        for ring_index in range(1, len(list(rd.values()))):
            if ring_index == 5:
                continue

            atom = rd[f"C{ring_index}"]

            for het_at in ConformerTest.adjacent_atoms(conn_mat, atom):
                if 'C' not in het_at and 'H' not in het_at and het_at not in rd.values():
                    adj_atom_list = ConformerTest.adjacent_atoms(conn_mat, het_at)

                    if ConformerTest.count_n(conn_mat, het_at, 'C') == 2 and rd['C5'] not in adj_atom_list:
                        c1_count = sum(1 for adj_at in adj_atom_list if adj_at in c1_list)

                        if c1_count > 0:
                            glycosidic_link_list.append(f"C{ring_index}")

        return glycosidic_link_list
    
    @staticmethod
    def amide_check(conn_mat, rd):
        if len(list(rd.values())) == 7:
            C2 = rd['C2']
        elif len(list(rd.values())) == 8:
            C2 = rd['C3']
        else:
            C2 == None
        
        if C2 is not None:
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
        
        else:
            amide = None
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
                
            if len(rd2.keys()) == 7 and atom1_index is not None:
                edge_check_list.append([rd2['C1'], rd1[f"C{atom2_index}"]])
            elif len(rd2.keys()) == 6 and atom1_index is not None:
                edge_check_list.append([rd2['C2'], rd1[f"C{atom2_index}"]])

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
        
        if ring_graph.number_of_edges() == 0:
            ring_graph.add_node('Ring 0')

        return ring_graph

    @staticmethod
    def sort_rings(rd_list, conn_mat):
        c1_list = [rd['C1'] if len(list(rd.values())) == 7 else rd['C2'] for rd in rd_list]
        red_end = ConformerTest.find_red_end(c1_list=c1_list, rd_list=rd_list, conn_mat=conn_mat)
        ring_graph = ConformerTest.ring_graph_maker(rd_list=rd_list, conn_mat=conn_mat)
        
        tree = nx.dfs_tree(ring_graph, red_end)

        dfs_ring_list = list(nx.dfs_preorder_nodes(ring_graph, red_end))
        for dfs_index,node in enumerate(dfs_ring_list):
            if 'Amide' in node:
                dfs_ring_list.remove(node)
            else:
                rd_list_index = int(list(node.split())[-1])
                dfs_ring_list[dfs_index] = rd_list[rd_list_index]
                
        if dfs_ring_list == []:
            dfs_ring_list = rd_list
            tree = ring_graph

        branch_end_list = []
        new_dfs_ring_list = []

        for rd in dfs_ring_list:
            node_index = rd_list.index(rd)
            node = f"Ring {node_index}"
            neighbor_list = [rn for rn in ring_graph.neighbors(node)]
            if len(neighbor_list) == 1 and node != red_end:
                branch_end_list.append(rd)

        branch_end_list.reverse()
        
        for branch_end in branch_end_list:
            branch_node = f"Ring {rd_list.index(branch_end)}"
            branch = nx.shortest_path(ring_graph,red_end,branch_node)
            for node in branch:
                ring_dict_index = int(list(node.split())[-1])
                rd = rd_list[ring_dict_index]

                if rd not in new_dfs_ring_list:
                    new_dfs_ring_list.append(rd)

        glyco_list = [ConformerTest.glycosidic_link_check(conn_mat=conn_mat, rd=rd, c1_list=c1_list) for rd in new_dfs_ring_list]

        return tree, glyco_list, new_dfs_ring_list

    @staticmethod
    def ring_stereo_compiler(xyz_array, dfs_list, conn_mat):

        sugar_type_list = []
        glyco_type_list = []

        for rd in dfs_list:
            sugar_type = ConformerTest.sugar_stero(rd, xyz_array)
            link_type = ConformerTest.glycosidic_link_type(rd, sugar_type, xyz_array, conn_mat)
            sugar_type_list.append(sugar_type)
            glyco_type_list.append(link_type)
        
        return sugar_type_list,glyco_type_list

if __name__ == "__main__":
    #This section is just for testing (Will not be in final code)
    xyz_file = input()
    if '.xyz' in xyz_file:
        conn_mat = ConformerTest.xyztograph(xyz_file)
        xyz_array = ConformerTest.xyztoarray(xyz_file)
        cycles_in_graph = nx.cycle_basis(conn_mat)

    
        ring_dict_list = ConformerTest.sort_ring_atoms(cycles_in_conn_mat=cycles_in_graph, conn_mat=conn_mat)

        ring_tree, glyco_array, dfs_ring_list = ConformerTest.sort_rings(rd_list=ring_dict_list, conn_mat=conn_mat)
        sugar_type_list, link_type_list = ConformerTest.ring_stereo_compiler(xyz_array=xyz_array, dfs_list=dfs_ring_list, conn_mat=conn_mat)
        
        print(glyco_array)
        print()
        print(sugar_type_list)
        print(link_type_list)
        print()
        print(dfs_ring_list)
        
        pos=nx.drawing.nx_agraph.graphviz_layout(ring_tree, prog="dot")
        flipped_pos = {node: (-x,-y) for (node, (x,y)) in pos.items()}
        nx.draw(ring_tree, with_labels=True, pos=flipped_pos)
        plt.show()
 