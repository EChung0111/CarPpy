import networkx as nx

class ConformerTest:

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
                    elif (
                        ConformerTest.count_n(conn_mat, adj_at, 'H') == 0 and
                        ConformerTest.count_n(conn_mat, adj_at, 'O') == 2
                    ):
                        rd['C5'] = atom
                    elif ConformerTest.count_n(conn_mat, adj_at, 'H') == 3:
                        rd['C5'] = atom

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
                    elif (
                        ConformerTest.count_n(conn_mat, adj_at, 'H') == 0 and
                        ConformerTest.count_n(conn_mat, adj_at, 'O') == 2
                    ):
                        rd['C5'] = atom
                    elif ConformerTest.count_n(conn_mat, adj_at, 'H') == 3:
                        rd['C5'] = atom

        if sugar_basis.index(rd['C4']) == 0:
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
                sugar_basis = list(nx.cycle_basis(conn_mat, oxygen_atom_list[0])[0])
                rd['O'] = oxygen_atom_list[0]

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

        rd_list = [rd for rd in rd_list if rd != {}]
        return rd_list
    
    def glycosidic_link_check(conn_mat, rd, C1_list):
        glycosidic_link_list = []

        for ring_index in range(1, 7):
            if ring_index == 5:
                continue

            atom = rd[f"C{ring_index}"]

            for HetAt in ConformerTest.adjacent_atoms(conn_mat, atom):
                if 'C' not in HetAt and 'H' not in HetAt:
                    adj_atom_list = ConformerTest.adjacent_atoms(conn_mat, HetAt)

                    if ConformerTest.count_n(conn_mat, HetAt, 'C') == 2 and rd['C5'] not in adj_atom_list:
                        C1_count = 0

                        for adj_at in adj_atom_list:
                            if adj_at in C1_list:
                                C1_count += 1
                        
                        if C1_count >1:
                            glycosidic_link_list.append(f"C{ring_index}")

        return glycosidic_link_list

    def ring_dict_finder(atom, rd_list):
       
        for rd in rd_list:
            if atom in rd.values():
                return rd
    
    def sort_rings(rd_list, conn_mat):
        sorted_list = []
        C1_list = []
        for rd in rd_list:
            if len(rd) == 6:
                C1_list.append(rd['C1'])
            elif len(rd) == 5:
                C1_list.append(rd['C2'])
        
        for C1 in C1_list:

            ring_dict = ConformerTest.ring_dict_finder(C1, rd_list)
            
            for atom in ConformerTest.adjacent_atoms(conn_mat=conn_mat, node=C1):

                if 'O' in atom:
                    if ConformerTest.count_n(conn_mat=conn_mat, node=atom, filter='H') == 1:
                        sorted_list.append(rd)
                    else:
                        C1_count = 0
                        for adj_atom in ConformerTest.adjacent_atoms(conn_mat=conn_mat, node=atom):
                            if adj_atom in C1_list:
                                C1_count += 1
                        
                        if C1_count == 2 and len(ConformerTest.glycosidic_link_check(conn_mat, ring_dict, C1_list)) > 1:
                            pass
if __name__ == "__main__":
    #This section is just for testing (Will not be in final code)
    Furanose_conn_mat = nx.Graph()
    Furanose_conn_mat.add_edge('C1', 'C2')
    Furanose_conn_mat.add_edge('O12', 'C1')
    Furanose_conn_mat.add_edge('O9', 'C5')
    Furanose_conn_mat.add_edge('O10', 'C4')
    Furanose_conn_mat.add_edge('O8', 'C6')
    Furanose_conn_mat.add_edge('H24', 'O12')
    Furanose_conn_mat.add_edge('H13', 'C1')
    Furanose_conn_mat.add_edge('H15', 'C2')
    Furanose_conn_mat.add_edge('H17', 'C4')
    Furanose_conn_mat.add_edge('H18', 'C5')
    Furanose_conn_mat.add_edge('H19', 'C6')
    fcycles_in_conn_mat = nx.cycle_basis(Furanose_conn_mat)

    Pyranose_conn_mat = nx.Graph()
    nx.add_cycle(Pyranose_conn_mat, ['C2', 'C3', 'C4', 'C5', 'C6', 'O7'])
    Pyranose_conn_mat.add_edge('C1', 'C2')
    Pyranose_conn_mat.add_edge('O12', 'C1')
    Pyranose_conn_mat.add_edge('O11', 'C3')
    Pyranose_conn_mat.add_edge('O9', 'C5')
    Pyranose_conn_mat.add_edge('O10', 'C4')    

    Pyranose_conn_mat.add_edge('O8', 'C6')
    Pyranose_conn_mat.add_edge('H24', 'O12')
    Pyranose_conn_mat.add_edge('H13', 'C1')
    Pyranose_conn_mat.add_edge('H14', 'C1')
    Pyranose_conn_mat.add_edge('H15', 'C2')
    Pyranose_conn_mat.add_edge('H16', 'C3')
    Pyranose_conn_mat.add_edge('H17', 'C4')
    Pyranose_conn_mat.add_edge('H18', 'C5')
    Pyranose_conn_mat.add_edge('H19', 'C6')
    pcycles_in_conn_mat = nx.cycle_basis(Pyranose_conn_mat)

    Benzene_conn_mat = nx.conn_mat()
    nx.add_cycle(Benzene_conn_mat, ['C1', 'C2', 'C3', 'C4', 'C5', 'C6'])
    bcycles_in_conn_mat = nx.cycle_basis(Benzene_conn_mat)

    Fused_conn_mat = nx.conn_mat()
    nx.add_cycle(Fused_conn_mat, ['C2', 'C3', 'C4', 'C5', 'C6', 'O7'])
    nx.add_cycle(Fused_conn_mat, ['O7', 'C6', 'O32', 'C31', 'O9', 'C3', 'C2'])
    nx.add_cycle(Fused_conn_mat, ['C6', 'C5', 'C4', 'C3', 'O9', 'C31', 'O32'])
    Fused_conn_mat.add_edge('C1', 'C2')
    Fused_conn_mat.add_edge('O12', 'C1')
    Fused_conn_mat.add_edge('O18', 'C5')
    Fused_conn_mat.add_edge('O10', 'C4')
    Fused_conn_mat.add_edge('H24', 'O12')
    Fused_conn_mat.add_edge('H13', 'C1')
    Fused_conn_mat.add_edge('H14', 'C1')
    Fused_conn_mat.add_edge('H15', 'C2')
    Fused_conn_mat.add_edge('H16', 'C3')
    Fused_conn_mat.add_edge('H17', 'C4')
    Fused_conn_mat.add_edge('H18', 'C5')
    Fused_conn_mat.add_edge('H19', 'C6')
    fzcycles_in_conn_mat = nx.cycle_basis(Fused_conn_mat)
    
    ring_dict_list = ConformerTest.sort_ring_atoms(cycles_in_conn_mat=fzcycles_in_conn_mat, conn_mat=Fused_conn_mat)
    print(ring_dict_list)