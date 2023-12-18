import math
import rmsd
import numpy as np
import networkx
import sys, copy
from scipy import interpolate
from scipy.linalg import expm
from optparse import OptionParser
from datetime import datetime

def error(msg):

    """ write error message and quit
    :param msg: (string) the message to be outputted
    """

    sys.stderr.write(msg + "\n")
    sys.exit(3)

def dtime():

    """ finds the current date and time
    :return: the current date and time
    """

    now = datetime.now()
    return now.strftime("%m/%d/%Y %H:%M:%S")

def get_distance(at1, at2):

    """ Finds the distance between two atoms
    :param at1: (list) a list of xyz coordinates of atom1
    :param at2: (list) a list of xyz coordinates of atom2
    :return: (float) the distance between 2 atoms
    """

    return math.sqrt((at1[0]-at2[0])**2+(at1[1]-at2[1])**2+(at1[2]-at2[2])**2)

def norm(a):

    """Norm of a vector
    :param a: a vector in a form of a numpy array
    :return: (float) the norm of the vector
    """

    return math.sqrt(np.sum(a*a))

def clashcheck(conf, cutoff=1.2):

    """Checks if there is a clash between atoms
    :param conf: passes a conformer object. Conn_mat attributre of the conf is used to exclude 'bound' atoms. 
    :param cutoff: (float) the maximum value that would not be considered a clashing distance, default to 1.2
    :return: (bool) True for a clash or False for no clash
    """

    Inv_cm = np.ones( (conf.NAtoms, conf.NAtoms) ) + 10*conf.conn_mat
    dist = np.zeros((conf.NAtoms, conf.NAtoms))

    for at1 in range(conf.NAtoms):
        for at2 in range(conf.NAtoms):
            if at1 != at2 : 
                dist[at1,at2] = get_distance(conf.xyz[at1], conf.xyz[at2])
            else: dist[at1,at2] = 10.0

    Dist = dist*Inv_cm

    if np.amin(Dist) > cutoff:
        return False
    else:
        return True

def adjacent_atoms(conn_manumpyt, at):

    """returns all adjacent atoms to a specific atom in a conformation
    :param conn_mat: the connectivity matrix
    :param at: index of the selected atom
    :return: indices of all atoms adjacent to the selected atom
    """

    return np.nonzero(conn_mat[at,:])[0]

def connect_atoms(conf, at1, at2):

    """Connects two atoms in the conn_mat
    :param conf: conf objenumpyct that has associated conn_mat
    :return: None
    """

    conf.conn_mat[at1, at2] = 1
    conf.conn_mat[at2, at1] = 1

def disconnect_atoms(conf, at1, at2):

    """Disconnects two atoms in the conn_mat
    :param conf: conf object that has associated conn_mat
    :return: None                                            
    """

    conf.conn_mat[at1, at2] = 0
    conf.conn_mat[at2, at1] = 0

def draw_random():

    """Draw a random float between 0 and 1
    :return: (float) random sample
    """

    return np.random.random_sample()

def draw_random_int(top=1):

    """ draw a random int between 0-top
    :param top: (int) the upperbound of the random int generator
    :return: (int) returns a random integer
    """

    return np.random.randint(top)

def element_symbol(A):

  """ A dictionary for atomic number and atomic symbol
  :param A: either atomic number or atomic symbol for Hydrogen, Carbon, Nitrogen, Oxygen, Fluorine and Silicon
  :return: the corresponding atomic symbol or atomic number
  """

  periodic_table = { '1' : 'H', '2': 'He', 
          '3':'Li', '4':'Be', '5': 'B', '6': 'C', '7': 'N', '8': 'O', '9': 'F','10':'Ne', 
         '11':'Na','12':'Mg','13':'Al','14':'Si','15': 'P','16': 'S','17':'Cl','18':'Ar',
         '19': 'K','20':'Ca','35':'Br'
         }

  return periodic_table[A]

def element_number(A):

    periodic_table = { 'H' : 1, 'He': 2, 
            'Li': 3,'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9,'Ne':10,
            'Na':11,'Mg':12,'Al':13,'Si':14, 'P':15, 'S':16,'Cl':17,'Ar':18,
             'K':19,'Ca':20,'Br':35}

    return periodic_table[A]

def protecting_group_name(A):

    protecting_groups = { 'H1O1' : 'OH', 'H3C1': 'Me', 'H3C1O1': 'OMe', 'H4C2O1': 'OEt', 'H5C6O1': 'OPh', 'H7C10O1': 'ONh',
            'H7C7O1': 'OBn', 'H9C11O1': 'ONp', 
            'H3C2O2': 'OAc', 'H5C7O2': 'OBz', 'H9C4O2': 'OPiv', 
            'H4C2N1O1': 'NHCO-Me','H5C2N1O1': 'NHCOH-Me',
            'N3': 'Az', 'H2N1': 'NH2'}

    if A in protecting_groups.keys(): 
        return protecting_groups[A]
    else: 
        return A

def protecting_group_dihedrals(conf, atom, pg_type, PG_atoms):

    #PG_name = protecting_group_name(PG_sum)
    dih1 = [None, None, atom, None]

    for at1 in adjacent_atoms(conf.conn_mat, atom):
        if at1 not in PG_atoms: 
            for at2 in adjacent_atoms(conf.conn_mat, at1):
                if pg_type == 'C6' and conf.atoms[at2] == 'C': #C5
                    
                    dih1[0] = copy.copy(at2) ; dih1[1] = copy.copy(at1)

                elif pg_type != 'C6' and  conf.atoms[at2] == 'H': 
                    dih1[0] = copy.copy(at2) ; dih1[1] = copy.copy(at1)

        elif at1 in PG_atoms: 
            dih1[3] = copy.copy(at1)

    #print(dih1)

    if conf.atoms[atom] == 'O': 

        if conf.atoms[dih1[3]] == 'H':
            return [dih1]

        elif conf.atoms[dih1[3]] == 'C':
            adj_atoms = adjacent_atoms(conf.conn_mat, dih1[3])
            if len(adj_atoms) == 3: #sp2 carbon in carbonyl of aromatic

                adj_atoms_names = []
                for at in adj_atoms: adj_atoms_names.append(conf.atoms[at])

                if adj_atoms_names.count('O') == 2:  #carbonyl: OAc, OBz, OPiv, return only two dihedrals
                    for at in adj_atoms: 
                        if at not in dih1 and conf.atoms[at] == 'O': 
                            dih2 = [ dih1[1], dih1[2], dih1[3], at]
                            return [dih1, dih2]

                elif adj_atoms_names.count('C') == 2: #Aromatic group: OPh, ONh, return any angle. 
                    for at in adj_atoms: 
                        if at not in dih1 and conf.atoms[at] == 'C': 
                            dih2 = [ dih1[1], dih1[2], dih1[3], at]                      
                            return [dih1, dih2]

            if len(adj_atoms) == 4: #sp3 carbon - methyl/ene group. 

                adj_atoms_names = []
                for at in adj_atoms: adj_atoms_names.append(conf.atoms[at])

                if adj_atoms_names.count('H') == 3: #OMe
                    return [dih1]

                elif adj_atoms_names.count('C') == 1: #Another carbon attached
                    for at in adj_atoms: 
                        if at not in dih1 and conf.atoms[at] == 'C': 
                            dih2 = [ dih1[1], dih1[2], dih1[3], at]                      
                            return [dih1, dih2]

    if conf.atoms[atom] == 'N':

        if   len(PG_atoms) == 3 and conf.atoms[dih1[3]] == 'H': #NH2 group
            return [dih1]
        elif len(PG_atoms) == 3 and conf.atoms[dih1[3]] == 'N': #N3 group
            return [dih1] 

        adj_atoms =  adjacent_atoms(conf.conn_mat, atom) 
        adj_atoms_names = []
        for at in adj_atoms: adj_atoms_names.append(conf.atoms[at])
        
        for at, atn in zip(adj_atoms, adj_atoms_names):
            if atn == 'C' and at not in dih1: 

                dih3 = None
                adj_atoms2 = adjacent_atoms(conf.conn_mat, at) 
                adj_atoms2_names = []
                for at2 in adj_atoms2: adj_atoms2_names.append(conf.atoms[at2])

                if len(adj_atoms2) == 3 and 'O' in adj_atoms2_names: #NHCO-CH3 or NHCOH-CHH3 

                    for at2, atn2 in zip(adj_atoms2, adj_atoms_names2):

                        if atn2 == 'C' and at not in dih1: 
                           dih2 = [dih1[1], dih1[2], dih1[3], at]

                        if atn2 == 'O': 
                            adj_atoms3 = adjacent_atoms(conf.conf_conn, at2)
                            if len(and_atoms3) == 2:
                                if   adj_atoms3[0] == 'H': 
                                    dih3 = [ dih1[2], dih1[3], at, adj_atoms3[0]]
                                elif adj_atoms3[1] == 'H': 
                                    dih3 = [ dih1[2], dih1[3], at, adj_atoms3[1]]

                if dih3 is not None: return [dih1, dih2, dih3]
                return [dih1, dih2]


def determine_carried_atoms(conf, at1, at2):

    """Find all atoms necessary to be carried over during rotation
    of an atom 2

    :param at1: (list) the xyz coordinates of an atom
    :param at2: (list) the xyz coordinates of another atom
    :param conn_matt: the connectivity matrix of a conformer
    """
    #   1. Zero the connections in connectivity matrix
    #tmp_conn = np.copy(conf.conn_mat)

    broke_bond = False
    if conf.conn_mat[at1, at2] == 1 or conf.conn_mat[at2, at1] == 1: 
        disconnect_atoms(conf, at1, at2)
        broke_bond = True

    cm = networkx.graph.Graph(conf.conn_mat)

    if networkx.is_connected(cm): 
        print("Matrix still connected")
        return

    #   2. Determine the connected atoms:
    for subgraph in networkx.connected_components(cm):
        if at2 in subgraph: 
            #print('subgraph:',subgraph)
            carried_atoms = [ at for at in subgraph ]
    if broke_bond : connect_atoms(conf, at1, at2)

    #disconnect_atoms(conf, at1, at2)

    #carried_atoms = [at2]
    #start = True
    #while start:
    #    start = False
    #    #   Always iterate over entire list because of branching
    #    for at in carried_atoms:
    #        #   List of indexes of connected atoms:
    #        conn_atoms = np.where(conf.conn_mat[at] != 0)[0]
    #        conn_atoms.tolist
    #        conn_atoms_names = [ conf.atoms[x] for x in conn_atoms ] 
    #        conn_atoms.sort()
    #        conn_atoms = [x for _, x in sorted(zip(conn_atoms_names, conn_atoms))]
    #        for x in conn_atoms:
    #            if x not in carried_atoms:
    #                carried_atoms.append(x)
    #                start = True
    #connect_atoms(conf, at1, at2)

    return carried_atoms


def get_distance(at1, at2):

    """ Finds the distance between two atoms
    :param at1: (list) a list of xyz coordinates of atom1
    :param at2: (list) a list of xyz coordinates of atom2
    :return: (float) the distance between 2 atoms
    """

    return math.sqrt((at1[0]-at2[0])**2+(at1[1]-at2[1])**2+(at1[2]-at2[2])**2)

def measure_distance(conf, list_of_atoms):
 
    at1 = conf.xyz[list_of_atoms[0]]
    at2 = conf.xyz[list_of_atoms[1]]

    return math.sqrt((at1[0]-at2[0])**2+(at1[1]-at2[1])**2+(at1[2]-at2[2])**2)

def norm(a):

    """Norm of a vector
    :param a: a vector in a form of a numpy array
    :return: (float) the norm of the vector
    """

    return math.sqrt(np.sum(a*a))

def calculate_normal_vector(conf, list_of_atoms):

    """Calculate the normal vector of a plane by
    cross product of two vectors belonging to it.
    :param list_of_atoms: (list) of 3 atoms
    :param conf: a conformer object
    :return cross_product: cross product of two matricies
    """

    r0 = conf.xyz[list_of_atoms[1], :] - conf.xyz[list_of_atoms[0], :]
    r1 = conf.xyz[list_of_atoms[2], :] - conf.xyz[list_of_atoms[1], :]
    cross_product = np.cross(r1, r0)

    return cross_product

def measure_angle(conf, list_of_atoms):

    """Calculate an angle between three atoms:
    angle = acos(dot(X,Y)/(norm(X)*norm(Y)))
    :param list_of_atoms: (list) list of 3 atoms
    :param conf: a conformer object
    :return angle, axor: returns angle in degrees and axis of rotation of the angle
    """

    r0 = conf.xyz[list_of_atoms[0], :] - conf.xyz[list_of_atoms[1], :]
    r1 = conf.xyz[list_of_atoms[2], :] - conf.xyz[list_of_atoms[1], :]

    norm_r0 = np.sqrt(np.sum(r0**2))
    norm_r1 = np.sqrt(np.sum(r1**2))
    norm = norm_r0*norm_r1

    dot_product = np.dot(r0, r1)/norm
    angle = np.arccos(dot_product)

    #    Calculate the axis of rotation (axor):
    axor = np.cross(r0, r1)

    return angle*180.0/np.pi, axor

def set_angle(conf, list_of_atoms, new_ang):

    """Set a new angle between three atoms
    :param list_of_atoms: (list) list of three atoms
    :param new_ang: value of dihedral angle (in degrees) to be set
    """

    from scipy.linalg import expm

    if len(list_of_atoms) != 3:
        raise ValueError("The position needs to be defined by 4 integers")

    at1 = list_of_atoms[0]
    at2 = list_of_atoms[1] #midpoint
    at3 = list_of_atoms[2]
    #xyz = copy.copy(conf.xyz)
    xyz = conf.xyz

    #   Determine the axis of rotation:

    old_ang, axor = measure_angle(conf, [at1, at2, at3])
    norm_axor = np.sqrt(np.sum(axor**2))
    normalized_axor = axor/norm_axor

    #   Each carried_atom is rotated by euler-rodrigues formula:
    #   Also, I move the midpoint of the bond to the mid atom
    #   the rotation step and then move the atom back.

    rot_angle = np.pi*(new_ang - old_ang)/180.
    translation = xyz[at2, :]

    #apply rotations to at3.
    rot = expm(np.cross(np.eye(3), normalized_axor*(rot_angle)))
    xyz[at3, :] = np.dot(rot, xyz[at3, :]-translation)
    xyz[at3, :] = xyz[at3, :]+translation

    #return xyz

def measure_dihedral(conf, list_of_atoms):

    """Calculate a dihedral angle between two planes defined by
    a list of four atoms. It returns the angle and the rotation axis
    required to set a new dihedral.
    :param conf: conformer object which associated xyz matrix
    :param list_of_atoms: (list) list of 4 atoms
     """

    if len(list_of_atoms) == 4:
      plane1 = calculate_normal_vector(conf, list_of_atoms[:3])
      plane2 = calculate_normal_vector(conf, list_of_atoms[1:])

    else:
      plane1 = calculate_normal_vector(conf, list_of_atoms[:3])
      print(list_of_atoms[:3])
      plane2 = calculate_normal_vector(conf, list_of_atoms[3:])
      print(list_of_atoms[3:])

    #   Calculate the axis of rotation (axor)
    axor = np.cross(plane1, plane2)

    #   Calculate a norm of normal vectors:
    norm_plane1 = np.sqrt(np.sum(plane1**2))
    norm_plane2 = np.sqrt(np.sum(plane2**2))
    norm = norm_plane1 * norm_plane2

    #   Measure the angle between two planes:
    dot_product = np.clip(np.dot(plane1, plane2)/norm, -0.99999999, 0.99999999)
    alpha = np.arccos(dot_product)

    #   The cosine function is symetric thus, to distinguish between
    #   negative and positive angles, one has to calculate if the fourth
    #   point is above or below the plane defined by first 3 points:

    ppoint = - np.dot(plane1, conf.xyz[list_of_atoms[0], :])
    dpoint = (np.dot(plane1, conf.xyz[list_of_atoms[3], :])+ppoint)/norm_plane1

    if dpoint >= 0:
        return -(alpha*180.0)/np.pi, axor
    else:
        return (alpha*180.0)/np.pi, axor

def set_dihedral(conf, list_of_atoms, new_dih, incr = False,  axis_pos = "bond", threshold = 0.1):

    """Set a new dihedral angle between two planes defined by
    atoms first and last three atoms of the supplied list.
    :param list_of_atoms: (list) list of four atoms
    :param new_dih: (float) value of dihedral angle (in degrees) to be set
    :incr: (bool) whether the new_dih should be treated as incremental value to current dih
    :axis_pos: whether to set the axor along the central bond, or the terminal bond in the list of 4 atoms
    :returns: xyz modified numpy array with new atoms positions
    """
    #print("atoms of the planes:", list_of_atoms) #It's very poetic
    at1 = list_of_atoms[0]
    at2 = list_of_atoms[1] #midpoint
    at3 = list_of_atoms[2]
    at4 = list_of_atoms[3]
    #xyz = copy.copy(conf.xyz)
    xyz = conf.xyz

    #   Determine the axis of rotation:
    old_dih, axor = measure_dihedral(conf, [at1, at2, at3, at4])

    #if incr == True:
    #    print("current dihedral:",old_dih,"increm dihedral:",new_dih)
    #else:
    #    print("current dihedral:",old_dih,"target dihedral:",new_dih)
    try:
        norm_axor = np.sqrt(np.sum(axor**2))
        normalized_axor = axor/norm_axor

    except RuntimeWarning:
        print(axor, norm_axor, normalized_axor)
        print(at1, at2, at3, at4)
        print(axis_pos, incr)

    if incr == True: new_dih = old_dih + new_dih

    #(get it between -180. - 180.0
    if   new_dih >=  180.0 : new_dih -= 360.0
    elif new_dih <= -180.0 : new_dih += 360.0

    if abs(new_dih - old_dih) < threshold: return

    if old_dih >= 0.0 :
        if ( 180.0 - threshold) < new_dih:  new_dih = 180.0 - threshold
        rot_angle = new_dih - old_dih
        rot_angle = np.pi*(rot_angle)/180.
    else:
        if (-180.0 + threshold) > new_dih:  new_dih= -180.0 + threshold
        rot_angle = new_dih - old_dih
        rot_angle = -np.pi*(rot_angle)/180.

    #   Each carried_atom is rotated by Euler-Rodrigues formula:
    #   Reverse if the angle is less than zero, so it rotates in
    #   right direction.
    #   Also, I move the midpoint of the bond to the center for
    #   the rotation step and then move the atom back.

    rot = expm(np.cross(np.eye(3), normalized_axor*rot_angle))

    if axis_pos == "bond":
        translation = (xyz[list_of_atoms[1], :]+xyz[list_of_atoms[2], :])/2
        carried_atoms = determine_carried_atoms(conf, at2, at3)

    elif axis_pos == "term":
        translation = np.array([x for x in xyz[list_of_atoms[3], :]])
        #Determine which atoms should be dragged along with the bond:
        carried_atoms = determine_carried_atoms(conf, at3, at4)
        carried_atoms.remove(at4)

    for at in carried_atoms:
      #print("atom #:", at)
      #print("original xyz:", xyz[at, :])
      xyz[at, :] = np.dot(rot, xyz[at, :]-translation)
      xyz[at, :] = xyz[at, :]+translation
      #print("new xyz:", xyz[at, :])

    #dih, axor = measure_dihedral(conf, [at1, at2, at3, at4])
    #print("new dihedral:", dih, "\n")

def ring_pucker_dict(pucker):

    """ Returns a value of theta angles associated with a ring pucker """
    topol_table = {
    '1C4' : [ -35.26, -35.26, -35.26],  '4C1': [  35.26,  35.26,  35.26],
    '1,4B': [ -35.26,  74.20, -35.26], 'B1,4': [  35.26, -74.20,  35.26],
    '2,5B': [  74.20, -35.26, -35.26], 'B2,5': [ -74.20,  35.26,  35.26],
    '3,6B': [ -35.26, -35.26,  74.20], 'B3,6': [  35.26,  35.26, -74.20],
    '1H2' : [ -42.16,   9.07, -17.83],  '2H1': [  42.16,  -9.07,  17.83],
    '2H3' : [  42.16,  17.83,  -9.06],  '3H2': [ -42.16, -17.83,   9.06],
    '3H4' : [ -17.83, -42.16,   9.07],  '4H3': [  17.83,  42.16,  -9.07],
    '4H5' : [  -9.07,  42.16,  17.83],  '5H4': [   9.07, -42.16, -17.83],
    '5H6' : [   9.07, -17.83, -42.16],  '6H5': [  -9.07,  17.83,  42.16],
    '6H1' : [  17.83,  -9.07,  42.16],  '1H6': [ -17.83,   9.07,  -42.16],
    '1S3' : [   0.00,  50.84, -50.84],  '3S1': [   0.00, -50.84,  50.84],
    '5S1' : [  50.84, -50.84,   0.00],  '1S5': [ -50.84,  50.84,   0.00],
    '6S2' : [ -50.84,   0.00,  50.84],  '2S6': [  50.84,   0.00, -50.84],
    '1E'  : [ -35.26,  17.37, -35.26],  'E1' : [  35.26, -17.37,  35.26],
    '2E'  : [  46.86,   0.00,   0.00],  'E2' : [ -46.86,   0.00,   0.00],
    '3E'  : [ -35.26, -35.26,  17.37],  'E3' : [  35.26,  35.26, -17.37],
    '4E'  : [   0.00,  46.86,   0.00],  'E4' : [   0.00, -46.86,   0.00],
    '5E'  : [  17.37, -35.26, -35.26],  'E5' : [ -17.37,  35.26,  35.26],
    '6E'  : [   0.00,   0.00,  46.86],  'E6' : [   0.00,   0.00, -46.86]
    }

    if pucker in topol_table:
        return topol_table[pucker]
    else:
       error("the provided topology is not in the table\n")

def ring_dihedrals(conf, ring_number):

    """ This function will return the 3 theta angles that define a ring pucker
    :param conf: conformer object with associated rings: 
    :param ring_number: which ring will be measured
    :return: theta dihedrals associated with the ring
    """

    ra = conf.graph.nodes[ring_number]['ring_atoms']
    dih_atoms = [
         [ra['C5'],ra['C3'],ra['C1'],ra['C2']],
         [ra['C1'],ra['C5'],ra['C3'],ra['C4']],
         [ra['C3'],ra['C1'],ra['C5'],ra['O' ]]]

    theta = []
    for n in range(3):
        t = measure_dihedral( conf, dih_atoms[n])[0]
        #print (old)
        if   t < 180.0 and t > 0.0      : t =  180.0 - t
        elif t > 180.0                  : t = -180.0 + t
        elif t < 0.0   and t > -180.0   : t = -180.0 - t
        theta.append(t)

    return(theta)

def set_ring_pucker(conf, ring_number, pucker):

    """ Edits the ring pucker by assigning a new angle to the C2, C4 and O angles. This is based on the ring puckering model proposed in Puckering Coordinates of Monocyclic Rings by Triangular Decomposition Anthony D. Hill and Peter J. Reilly
    :param conf: a conformer object
    :param ring_number: (int) selects which ring of the conformer, an index to select the graph node
    :param ring_pucker: (list) or (string) this is the ring puckering angles. Either a list with 3 numbers or a string that defines the intended topology which is looked up in the topol_dict
    """

    #ring_pucker is either a string or a list of 3 numbers
    #check if it's a string

    if type(pucker) is str: new_theta  = ring_pucker_dict(pucker)

    #checks if it is a list with 3 numbers

    elif (type(pucker) is list or type(pucker) is np.ndarray) and len(pucker) == 3:
        new_theta  = pucker

    else:
        error("neither existing topology nor list of 3 dihedral angles are provided")

    ra = conf.graph.nodes[ring_number]['ring_atoms']

    #xyz_backup = copy.copy(conf.xyz)

    #print("ring atoms:",ra)

    #break the ring bonds
    #Flip, Twist, Tilt:
    #1. Move the even carbons according to the theta angles
    #2. Twist the odd carbons to be perpendical to the plane formed by even atoms
    #3. Tilt the odd atoms to form the ring.

    dih_atoms = [
         [ra['C5'],ra['C3'],ra['C1'],ra['C2']],
         [ra['C1'],ra['C5'],ra['C3'],ra['C4']],
         [ra['C3'],ra['C1'],ra['C5'],ra['O' ]]]
    dih_atoms2= [
         [ra['C4'],ra['C2'],ra['O' ],ra['C1']],
         [ra['O' ],ra['C4'],ra['C2'],ra['C3']],
         [ra['C2'],ra['O' ],ra['C4'],ra['C5']]]

    #at  => any atom
    #rat => atom in a ring

    #1. Flap:
    #print("Step 1: Flap")

    adj_atoms = [] ; old_theta = []
    for rat in ['C1', 'C3', 'C5']:
        adj_atoms.append(adjacent_atoms(conf.conn_mat, ra[rat]))
        #print(adj_atoms[-1])
        for at in adj_atoms[-1]:
            disconnect_atoms(conf, ra[rat], at)

    for rat1, rat2 in zip(['C1', 'C3', 'C5'], ['C2', 'C4', 'O']):
        connect_atoms(conf, ra[rat1], ra[rat2])

    for n in range(3):
        old  = measure_dihedral( conf, dih_atoms[n])[0]
        if   old < 180.0 and old > 0.0      : old =  180.0 - old
        elif old > 180.0                    : old = -180.0 + old
        elif old < 0.0   and old > -180.0   : old = -180.0 - old
        old_theta.append(old)
        set_dihedral(conf, dih_atoms[n], 180.0-new_theta[n])

    for n, rat in enumerate(['C1', 'C3', 'C5']):
        for at in adj_atoms[n]:
            connect_atoms(conf, ra[rat], at)

    #print("Step 2: Twist")
    ring_order = ['C1', 'C2', 'C3', 'C4', 'C5', 'O', 'C1']
    for i in range(len(ring_order)-1):
        disconnect_atoms(conf, ra[ring_order[i]],  ra[ring_order[i+1]])

    plane_even_vec = calculate_normal_vector(conf, [ra['O'], ra['C2'], ra['C4']])
    norm_ep = plane_even_vec /  np.sqrt(np.sum(plane_even_vec**2))
   
    for n, oa in enumerate(['C1', 'C3', 'C5']):

        op_atoms_adj = adjacent_atoms(conf.conn_mat, ra[oa]) ;

        if oa == 'C1' and conf.anomer == 'carbocation': #sp2 C1
           #print(conf.atoms[op_atoms_adj[0]])
            anomeric_h = copy.copy(op_atoms_adj[0])
            set_dihedral(conf, [ra['C2'], ra['O'], ra['C1'], anomeric_h],   179.)
            #set_dihedral(conf, [ra['C2'], ra['O'], ra['C1'], anomeric_h],   179.) #No idea why I need to do it twice
            continue

        if conf.atoms[op_atoms_adj[0]] == 'H':
            op_atoms = [op_atoms_adj[0], ra[oa], op_atoms_adj[1]]
        else: 
            op_atoms = [op_atoms_adj[1], ra[oa], op_atoms_adj[0]]
        if oa == 'C1':
            if   conf.anomer == 'beta'  and conf.graph.nodes[ring_number]['absconf'] == 'D':
               op_atoms.reverse()
            elif conf.anomer == 'alpha' and conf.graph.nodes[ring_number]['absconf'] == 'L':
               op_atoms.reverse()
        else:
  
            if conf.graph.nodes[ring_number]['absconf'] == 'D':
               op_atoms.reverse()

        for step in range(3):
            #Calculate the deviation from pi/2:
            plane_odd_vec = calculate_normal_vector(conf, op_atoms)
            norm_op = plane_odd_vec / np.sqrt(np.sum(plane_odd_vec**2))
            dot_product = np.dot(norm_ep, norm_op) ; rot_angle =  np.pi/2 - np.arccos(dot_product)
            #print(rot_angle)
            if rot_angle < 0.005: rot_angle = 0.0
            axor = np.cross(norm_ep, norm_op)
            rot = expm(np.cross(np.eye(3), axor*rot_angle)) ; translation = conf.xyz[op_atoms[1], :]

            #Determine which atoms should be dragged along with the bond:
            carried_atoms = determine_carried_atoms(conf, ra['O'], op_atoms[1]) #Bond with 'O' is zeroed anyway.
            carried_atoms.remove(op_atoms[1])
            #rotate the atoms:
            for at in carried_atoms:
                conf.xyz[at, :] = np.dot(rot, conf.xyz[at, :]-translation)
                conf.xyz[at, :] = conf.xyz[at, :]+translation

      #1. Get the axis between the farthest atom in the even plane (C2, C4, O)  and the group that is being adjusted
      #2. Use this axis to rotate the normal of the plane by 90.0 degrees.
        axor = conf.xyz[op_atoms[1],:] - conf.xyz[dih_atoms2[n][0],:]
        naxor = axor / np.sqrt(np.sum(axor**2))
        rot_mat = expm(np.cross(np.eye(3), naxor*np.pi/2))
        norm_ep_perp = np.dot(rot_mat, norm_ep)

      # Now, calculate the angle between the rotate even plane and the group and calculate the matrix to adjust it to 180.0
        dot_product = np.dot(norm_ep_perp, norm_op) ; rot_angle =  np.pi - np.arccos(dot_product)
        axor = np.cross(norm_ep_perp, norm_op)
        rot = expm(np.cross(np.eye(3), axor*rot_angle)) ; translation = conf.xyz[op_atoms[1],:]
      #Do the rotation on all atoms
        for at in carried_atoms:
            conf.xyz[at, :] = np.dot(rot, conf.xyz[at, :]-translation)
            conf.xyz[at, :] = conf.xyz[at, :]+translation

    #3. Tilt:
    #print("Step 3: Tilt")

    for n, rat in zip([0,1,2],['C1', 'C3', 'C5']):

        if rat == 'C1' and conf.anomer == 'carbocation': #sp2 C1
            set_angle(conf, [ra['C2'], ra['C1'], anomeric_h], 120.)
            continue

        N1 = (n+3)%3
        N2 = (n+2)%3
        differ =  (new_theta[N1] - old_theta[N1])/2 + (new_theta[N2] - old_theta[N2])/2
        if abs(differ) < 1.0 or abs(differ) > 359.0 : continue
        set_dihedral(conf, dih_atoms2[n], differ, incr = True, axis_pos="term")

    # Reconnect:
    for i in range(len(ring_order)-1):
        connect_atoms(conf, ra[ring_order[i]],  ra[ring_order[i+1]])

def ring_canon(theta, norm = 'l2'):

    canon_list = ('1C4',  '4C1', '1,4B', 'B1,4', '2,5B', 'B2,5','3,6B', 'B3,6', '1H2',  '2H1',
    '2H3',  '3H2','3H4',  '4H3','4H5',  '5H4','5H6',  '6H5','6H1' ,  '1H6','1S3' ,  '3S1','5S1' ,  '1S5',
    '6S2' ,  '2S6','1E'  ,  'E1' ,'2E'  ,  'E2' ,'3E'  ,  'E3' ,'4E'  ,  'E4' ,'5E'  ,  'E5' , '6E'  ,  'E6' )

  #find the closest coordinate to theta
    distances=[]
    for i in canon_list:
        co=ring_pucker_dict(i)
        if norm == 'l2': 
            d=get_distance(theta,co)
        elif norm == 'l1': 
            d=abs(theta[0] - co[0]) + abs(theta[1] - co[1]) + abs(theta[2] - co[2])

        entry = (d,i)
        distances.append(entry)

    distances.sort(key= lambda i: i[0])
    canon=distances[0][1]

    return canon

def select_freeze(freeze, ra):

    """Returns the string parameter for gaussian to freeze either the dihedral angles or atom postitions that form the dihedral angles
    :param freeze: (str) a string either "dih", "atoms" or "fhiaims"; this will return the string parameter to freeze the dihedral angles or freezing the atoms that form the dihedrals
    :param ra: (dict) ra (ring atoms) is a dictionary with the atom name and the atom number in the list of molecules
  """

    dih_atoms = [
         [ra['C5'],ra['C3'],ra['C1'],ra['C2']],
         [ra['C1'],ra['C5'],ra['C3'],ra['C4']],
         [ra['C3'],ra['C1'],ra['C5'],ra['O' ]]]

    if freeze == "dih":
        freeze_dih=''
        for dih in dih_atoms:
            for at in dih:
                freeze_dih=freeze_dih+str(at+1)+' '
            freeze_dih=freeze_dih+'F\n'
        return(freeze_dih)

    elif freeze == "atoms":
        freeze_atoms= 'notatoms='+','.join([str(ra[x] + 1) for x in ['O', 'C1', 'C2', 'C3', 'C4', 'C5']])
        return(freeze_atoms)

    elif freeze == "fhiaims":
        return [ra[x] for x in ['O', 'C1', 'C2', 'C3', 'C4', 'C5']]
    
def order_layer(layer):

    ord_layer=[]
    ord_layer.append(layer[0][1])

    del layer[:1]

    for i,k in zip(layer[0::2], layer[1::2]):
        temp = [i[1],k[1]]
        temp.sort(key=lambda n:n[0])
        midpoint = len(ord_layer)//2+1
        ord_layer.insert(midpoint, temp[1])
        ord_layer.insert(midpoint, temp[0])
    midpoint = len(ord_layer)//2+1
    ord_layer.insert(midpoint,layer[-1][1])
    return ord_layer

def pucker_scan(detail):

    canon = ('1C4',  '4C1', '1,4B', 'B1,4', '2,5B', 'B2,5','3,6B', 'B3,6', '1H2',  '2H1',
    '2H3',  '3H2','3H4',  '4H3','4H5',  '5H4','5H6',  '6H5','6H1' ,  '1H6','1S3' ,  '3S1','5S1' ,  '1S5',
    '6S2' ,  '2S6','1E'  ,  'E1' ,'2E'  ,  'E2' ,'3E'  ,  'E3' ,'4E'  ,  'E4' ,'5E'  ,  'E5' , '6E'  ,  'E6' )

    top = ring_pucker_dict('1C4') #top represent the actual cooridnates of the positive pole
    bot = ring_pucker_dict('4C1') #bot (bottom) represents the coordinates of the negative pole

    #order in order of distance from the positive pole
    positive = []
    #skip the first two entries because they are poles
    for i in canon[2:]:
        xyz=ring_pucker_dict(i)
        d=get_distance(top,xyz)
        entry = (d,xyz)
        positive.append(entry)
    positive.sort(key= lambda i: i[0])

    L1 = positive[:12] #L1 is a copy, not a pointer
    del positive[:12]

    negative = []

    for i in positive:
        d=get_distance(bot,i[1])
        entry = (d,i[1])
        negative.append(entry)
    negative.sort(key= lambda i: i[0])

    L3 = negative[:12]
    del negative[:12]

    L2 = negative

    #sort the levels in terms of the z axis
    L1.sort(key=lambda i: i[1][2])
    L2.sort(key=lambda i: i[1][2])
    L3.sort(key=lambda i: i[1][2])

    #this orders the levels a counter clockwise direction

    L1=order_layer(L1)
    L2=order_layer(L2)
    L3=order_layer(L3)

    #Slice scan
    detail_dict = {'high':[4,5],'medium':[3,4],'low':[2,2]}
    polar = detail_dict[detail][0] ; tropical = detail_dict[detail][1]

    slist=[]

    for i in range(len(L1)):

        #positive pole to L1, we don't want to include the point on L1 to avoid duplicates with the next segment
        p_L1 =np.linspace(start=top  , stop=L1[i], num=polar,   endpoint=False)
        #Layer 1 to Layer 2
        L1_L2=np.linspace(start=L1[i], stop=L2[i], num=tropical,endpoint=False)
        #Layer 2 to Layer 3
        L2_L3=np.linspace(start=L2[i], stop=L3[i], num=tropical,endpoint=False)
        #L3 to negative pole, we do want to include the endpoint to get the pole
        L3_p =np.linspace(start=L3[i], stop=bot  , num=(polar+1)) #MM: Why no endpoint = False? 

        temp=np.concatenate((p_L1,L1_L2,L2_L3,L3_p))
        slist.append(temp)

    #Layer scan
    pt_dict = {'high':[1,1,2,3,4,5,6,7,8], 'medium':[1,1,2,2,3,3,4], 'low':[1,2,2,3]}
    layer_pts = pt_dict[detail]

    #postive pole to layer1
    L1_index = polar
    L2_index = polar+tropical
    L3_index = L2_index+tropical
    temp=[]

    for i, j  in zip(slist[::], slist[1::]): #MM: Can I delete ":"?

        for pt in range(len(i[1:L2_index])):
            top_half=np.linspace(start=i[1:L2_index][pt], stop=j[1:L2_index][pt], num=layer_pts[pt], endpoint=False)
        for array in top_half:
            temp.append(array)

        #these 2 for loops need to be separated because sometimes the top range [1:L2] and 
        #the bottom range [L2:-1] are != related to the number of slice points generated

        for pt in range(len(i[L2_index:-1])):
            bot_half=np.linspace(start=i[L2_index:-1][pt], stop=j[L2_index:-1][pt], num=layer_pts[-(pt+1)], endpoint=False)
        for array in bot_half:
            temp.append(array)

    last  = slist[-1]
    first = slist[0]

    for pt in range(len(last[1:L2_index])): #MM: I don't understand this two loops. Check indendation level.  
        top_half=np.linspace(start=last[1:L2_index][pt], stop=first[1:L2_index][pt], num=layer_pts[pt], endpoint=False)

        for array in top_half:
            temp.append(array)

    for pt in range(len(last[L2_index:-1])):
        bot_half=np.linspace(start=last[L2_index:-1][pt], stop=first[L2_index:-1][pt], num=layer_pts[-(pt+1)], endpoint=False)

        for array in bot_half:
            temp.append(array)

    temp.append(np.array(bot))
    temp.append(np.array(top))

    return temp

def calculate_rmsd(conf1, conf2, atoms=None):

    """calculate the rmsd of two conformers; how similar the positions of the atoms are to each other

    :param conf1: a conformer object
    :param conf2: another conformer object
    :atoms: (string) passing the atomic symbol the function will omit those atoms when calculating the rmsd. Most commonly used is 'H' to remove hydrogen
    """

    xyz1 = []
    xyz2 = []

    #exclude the specified atom

    if atoms != None:

        if type(atoms) == set:
            for n, i in enumerate(conf1.atoms):
                if i in atoms: 
                    xyz1.append(conf1.xyz[n])
                    xyz2.append(conf2.xyz[n])

        elif type(atoms) == list: 
            for n in atoms:
                xyz1.append(conf1.xyz[n])
                xyz2.append(conf2.xyz[n])
    else:
        xyz1 = conf1.xyz ; xyz2 = conf2.xyz

    return rmsd.rmsd_qcp(xyz1, xyz2)

def deriv(spec,h):
  """ calculate first derivative of function 'spec'
       using the central finite difference method up to 6th order,
       and for the first 3 and last 3 grid points the
       forward/backward finite difference method up to 2nd order.
       ...as used in f77-program and suggested by Zanazzi-Jona...

       :param spec: (list) an IR spectrum
       :param h: (float) the delta x between each data point, must be less than 1 for accurate integral results
       :return: the first derivative of the parameter spec
  """
  der_spec =[[i[0],0] for i in spec]

  length=len(spec)
  for i in range(3,length-3):
    der_spec[i][1]=(-1*spec[i-3][1]+9*spec[i-2][1]-45*spec[i-1][1]+45*spec[i+1][1]-9*spec[i+2][1]+1*spec[i+3][1])/(60*h)
  for i in range(0,3):
    der_spec[i][1]=(-11*spec[i][1]+18*spec[i+1][1]-9*spec[i+2][1]+2*spec[i+3][1])/(6*h)
  for i in range(length-3,length):
    der_spec[i][1]=(11*spec[i][1]-18*spec[i-1][1]+9*spec[i-2][1]-2*spec[i-3][1])/(6*h)

  return der_spec


def get_range(tspec,espec,w_incr,shift,start,stop):
  """determine wavenumber range within the comparison between theoretical
  and experimental spectrum is performed (depends on the shift)

  :param tspec: (list) theoretical spectrum
  :param espec: (list) experimental spectrum
  :param w_incr: (float) grid interval of the spectra -- should be 1 or smaller!
  :param shift: the shift on the spectrum
  :param start: (int) the starting point on the spectrum
  :param stop: (int) the ending point on the spectrum
  :return: (tnstart, tnstop, enstart, enstop) the start and stop for thee range of the theoretical and experimental spectra
  """
  de1=start+shift-espec[0][0]
  if (de1 >= 0 ):
    de1=int((start+shift-espec[0][0])/w_incr+0.00001)
    enstart=de1
    tnstart=int((start-tspec[0][0])/w_incr+0.00001)
  else:
    de1=int((start+shift-espec[0][0])/w_incr-0.00001)
    enstart=0
    tnstart=int((start-tspec[0][0])/w_incr-de1+0.00001)
  de2=stop+shift-espec[-1][0]
  if (de2 <= 0 ):
    de2=int((stop+shift-espec[-1][0])/w_incr-0.00001)
    enstop=len(espec)+de2
    tnstop=len(tspec)+int((stop-tspec[-1][0])/w_incr-0.00001)
  else:
    de2=int((stop+shift-espec[-1][0])/w_incr+0.00001)
    enstop=len(espec)
    tnstop=len(tspec)+int((stop-tspec[-1][0])/w_incr-de2-0.00001)
  return tnstart, tnstop, enstart, enstop


def integrate(integrand,delta):
  """ integrate using the trapezoid method as Zanazzi-Jona suggested and was used in the f77-program...

  :param integrand: (list) A spectrum that is to be integrated
  :param delta: (float) the delta x in the trapezoidal method, the length of the base of each trapezoid. In implementation it is the increment of the data point of the spectrum, must be less than 1 for accurate integral results
  :return: (float) returns a product of the calculated integral and the delta
  """
  integral = 0.5*(integrand[0][1]+integrand[-1][1])
  for i in range(1,len(integrand)-1):
    integral += integrand[i][1]
  return integral*delta

def ypendry(spec,d1_spec,VI):
  """ calculate the Pendry Y-function: y= l^-1/(l^-2+VI^2) with l=I'/I (logarithmic derivative),
      J.B. Pendry, J. Phys. C: Solid St. Phys. 13 (1980) 937-44

  :param spec: (list) a conformer IR spectrum
  :param d1_spec: (list) the first derivative of the IR spectrum
  :param VI: (int) approximate half-width of the peaks
  :return: (2D list) returns the calculated pendry y-function
  """
  y=[[i[0],0] for i in spec]

  for i in range(len(spec)):
    if (abs(spec[i][1]) <= 1.E-7):
      if (abs(d1_spec[i][1]) <= 1.E-7):
        y[i][1] = 0
      else:
        y[i][1] = (spec[i][1]/d1_spec[i][1])/((spec[i][1]/d1_spec[i][1])**2+VI**2)
    else:
      y[i][1] = (d1_spec[i][1]/spec[i][1])/(1+(d1_spec[i][1]/spec[i][1])**2*(VI**2))
  return y


def rfac(espec, tspec, start=1000, stop=1800, w_incr=1.0, shift_min=-10, shift_max=+10, shift_incr=1, r="pendry", VI=10):

  """ %prog [options] r-fac.in
        Reads two spectra and calculates various R-factors -- FS 2011
        Attention: both spectra have to be given on the same, equidistant grid!
        NOTE: in the f77-program R1 is scaled by 0.75 and R2 is scaled by 0.5; this is not done here
        Please provide a file r-fac.in with the following specifications (without the comment lines!!!)
        (the numbers are just examples, choose them according to your particular case)

  :param espec: (list) the experimental spectrum
  :param tspec: (list) the theoretical spectrum, the conformer spectrum that the experimental spectrum is being compared to
  :param start: (int) where to start the comparison, default to 1000
  :param stop: (int) where to stop the comparison, default to 1800
  :param w_incr: (float) grid interval of the spectra -- should be 1 or smaller! (otherwise integrations/derivatives are not accurate) Default to 0.5
  :param shift_min: (int) minimal shift of the theoretical spectrum, default to -10
  :param shift_max: (int) maximal shift of the experimental spectrum, default to +10
  :param shift_incr: (int) shift interval, default to 1
  :param r: (string) specify which r-factor should be calculated options: pendry, ZJ, R1, R2 (give a list of the requested r-factors separated by comma). The default is "pendry"
  :param VI: (int) approximate half-width of the peaks (needed for pendry r-factor). Default to 10
  """
   #for shift in numpy.arange(shift_min,shift_max+shift_incr,shift_incr):# get the interval within the two spectra are compared
                #tnstart,tnstop,enstart,enstop = get_range(tspec,espec,w_incr,shift,start,stop)


# perform some checks of the input data...
  if (int(shift_incr/w_incr+0.00001) == 0):
    error("Error: shift_incr cannot be smaller than w_incr!")
  if (start-espec[0][0] < 0) or (espec[-1][0]-stop < 0):
    error("check experimental spectrum!!")
  if (start-tspec[0][0] < 0) or (tspec[-1][0]-stop < 0):
    error("check theoretical spectrum!!")
  if (int((espec[-1][0]-espec[0][0])/w_incr+0.0001) != len(espec)-1 ) or (int((tspec[-1][0]-tspec[0][0])/w_incr+0.0001) != len(tspec)-1 ):
    error("check w_incr!!")


# cut out data points that are not needed in order to save time...
  if (espec[0][0]-(start+shift_min-w_incr*25) < 0):
    espec=espec[-1*int((espec[0][0]-(start+shift_min-w_incr*25))/w_incr-0.00001):][:]
  if (espec[-1][0]-(stop+shift_max+w_incr*25) > 0):
    espec=espec[:-1*(int((espec[-1][0]-(stop+shift_max+w_incr*25))/w_incr+0.00001)+1)][:]
  if (tspec[0][0]-(start-w_incr*25) < 0):
    tspec=tspec[-1*int((tspec[0][0]-(start-w_incr*25))/w_incr-0.00001):][:]
  if (tspec[-1][0]-(stop+w_incr*25) > 0):
    tspec=tspec[:-1*(int((tspec[-1][0]-(stop+w_incr*25))/w_incr+0.00001)+1)][:]


# set negative intensity values to zero
  for i in range(0,len(espec)):
    if (espec[i][1]<0):
      espec[i][1]=0
  for i in range(0,len(tspec)):
    if (tspec[i][1]<0):
      tspec[i][1]=0

# start calculating derivatives...
  d1_espec = deriv(espec,w_incr)
  d1_tspec = deriv(tspec,w_incr)
# calculate the second derivatives if the Zanazzi-Jona R-factor is requested
  if "ZJ" in r:
    d2_tspec = deriv(d1_tspec,w_incr)
    d2_espec = deriv(d1_espec,w_incr)
# calculate Pendry Y-function if Pendry R-factor is requested
  if "pendry" in r:
    ye = ypendry(espec,d1_espec,VI)
    yt = ypendry(tspec,d1_tspec,VI)



  min_pendry = [1.E100,0]
  min_r1     = [1.E100,0]
  min_r2     = [1.E100,0]
  min_zj     = [1.E100,0]
# start with loop over x-axis shifts
  for shift in np.arange(shift_min,shift_max+shift_incr,shift_incr):
    # get the interval within the two spectra are compared
    tnstart,tnstop,enstart,enstop = get_range(tspec,espec,w_incr,shift,start,stop)
    #sys.stdout.write("\nshift: %9.3f, theory-start: %5d, theory-end: %5d, exp-start: %5d, exp-end: %5d\n" % (shift,tspec[tnstart][0],tspec[tnstop-1][0],espec[enstart][0],espec[enstop-1][0]))
    s_espec = np.array(espec[enstart:enstop]) # cut out the interval within which the comparison takes place
    s_tspec = np.array(tspec[tnstart:tnstop])
    s_d1_espec = np.array(d1_espec[enstart:enstop])
    s_d1_tspec = np.array(d1_tspec[tnstart:tnstop])
    c_scale=integrate(s_espec,w_incr)/integrate(s_tspec,w_incr)
    if "pendry" in r:
      # see J.B. Pendry, J. Phys. C: Solid St. Phys. 13 (1980) 937-44
      s_yt = np.array(yt[tnstart:tnstop]) # cut out the interval within which the comparison takes place
      s_ye = np.array(ye[enstart:enstop])
      te2 = integrate((s_yt-s_ye)**2,w_incr) # integrate (yt-ye)^2
      t2e2 = integrate(s_yt**2+s_ye**2,w_incr) # integrate yt^2+ye^2
      r_pend = te2/t2e2
      #sys.stdout.write("Pendry R-factor : %f, shift: %f\n" % (r_pend,shift))
      if (r_pend < min_pendry[0] ):
         min_pendry=[r_pend,shift]
    if "R1" in r:
      # see  M.A. van Hove, S.Y. Tong, and M.H. Elconin, Surfac Science 64 (1977) 85-95
      r1 = integrate(abs(s_espec-c_scale*s_tspec),w_incr)/integrate(abs(s_espec),w_incr)
      sys.stdout.write("R1 R-factor     : %f, shift: %f\n" % (r1,shift))
      if (r1 < min_r1[0]):
         min_r1=[r1,shift]
    if "R2" in r:
      # see  M.A. van Hove, S.Y. Tong, and M.H. Elconin, Surfac Science 64 (1977) 85-95
      r2 = integrate((s_espec-c_scale*s_tspec)**2,w_incr)/integrate(s_espec**2,w_incr)
      sys.stdout.write("R2 R-factor     : %f, shift: %f\n" % (r2,shift))
      if (r2 < min_r2[0]):
         min_r2=[r2,shift]
    if "ZJ" in r:
      # E. Zanazzi, F. Jona, Surface Science 62 (1977), 61-88
      s_d2_tspec = np.array(d2_tspec[tnstart:tnstop])
      s_d2_espec = np.array(d2_espec[enstart:enstop])

      epsilon = 0
      for i in s_d1_espec:
        if abs(i[1]) > epsilon:
          epsilon = abs(i[1])

      integrand = abs(c_scale*s_d2_tspec-s_d2_espec)*abs(c_scale*s_d1_tspec-s_d1_espec)/(abs(s_d1_espec)+epsilon)
      # interpolate integrand onto 10 times denser grid, see publication by Zanazzi & Jona
      incr = 0.1*w_incr
      grid_old = np.arange(0,len(integrand))*w_incr
      grid_new = np.arange(grid_old[0],grid_old[-1]+incr,incr)
      spl = interpolate.splrep(grid_old,integrand.T[1],k=3,s=0)
      integrand_dense = interpolate.splev(grid_new,spl,der=0)
      integrand_dense = np.vstack((grid_new,integrand_dense)).T
      # calculate reduced Zanazzi-Jona R-factor r=r/0.027
      r_zj = integrate(integrand_dense,incr)/(0.027*integrate(abs(s_espec),w_incr))
      sys.stdout.write("red. ZJ R-factor: %f, shift %f\n" % (r_zj,shift))
      if (r_zj < min_zj[0]):
        min_zj=[r_zj,shift]


# find minimal r-factor and write it out
   #sys.stdout.write("\nMinimal r-factors:\n")
  if "pendry" in r:
     #sys.stdout.write("minimal r-factor: Delta = %8.5f, Pendry R-factor = %7.5f \n" % ( min_pendry[1], min_pendry[0]))
       #print  (min_pendry[1], min_pendry[0])
       #I'm adding a return statement
    return min_pendry[0]
  if "R1" in r:
    sys.stdout.write("minimal r-factor: Delta = %8.5f, R1 R-factor = %7.5f \n" % ( min_r1[1], min_r1[0]))
  if "R2" in r:
    sys.stdout.write("minimal r-factor: Delta = %8.5f, R2 R-factor = %7.5f \n" % ( min_r2[1], min_r2[0]))
  if "ZJ" in r:
    sys.stdout.write("minimal r-factor: Delta = %8.5f, ZJ R-factor = %7.5f \n" % ( min_zj[1], min_zj[0]))
