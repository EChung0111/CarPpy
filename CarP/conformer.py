
#  10/2018 - CP and MM / HC-CUNY
#  A class that creates an instance of a molecule defined as conformer.
#  It parses gaussian output file (optimization + freq at this moment to
#  set proper flags, to be fixed for only freq calcs) and creates an object with
#  following attibutes:
#  - self.Geom -  np.array with xyz of all atoms
#  - self.atoms - list of atomic numbers/int sorted as xyz
#  - self.[EHG/Ezpe] - float, respective energy function value
#  - self.Freq + self.Ints - np.array consisting either freqs or ints
#  - self.Vibs - 3D np.array with normal modes of all vibrations
#  - self.NAtoms - int with number of atoms
#  - self._ir    - identification/directory name
#  - self.conn_mat - a NxN matrix, N = num of atoms, containing 0 or 1 indicating if there is a bond present or not
#  - self.graph  - each node on this graph is a ring of the conformer and contains ring related info


import numpy as np
import re, os
from subprocess import Popen, PIPE

from nltk.sem.chat80 import continent

from .utilities import *
import networkx as nx
from operator import itemgetter, attrgetter
import matplotlib.pyplot as plt
import py3Dmol as p3D
from itertools import zip_longest
import PIL

class Conformer():

    """
    A class that creates an instance of a molecule defined as conformer.
    It parses gaussian output file (optimization + freq at this moment to
    set proper flags, to be fixed for only freq calcs) and creates an object
    """

    def __init__(self, path):

        """Construct a conformer object
        :param path: (string) a path to the directory that holds the log-file or will be used to create the input file
        """

        self.topol = None 
        self.path  = path
        self._id   = path.split('/')[-1]

        #self.status = False

    def create_input(self, theory, output=None,  software = 'g16'):

        """ Creates the parameters to run simulation in Gaussian
        :param theory: (dict) a dictionary with the simulation parameters
        :param output: (string) this is the name of the output directory to be created
        :param software: (string) g16 or fhiaims
        """
        if output == None:
            self.outdir = self.path
        else: self.outdir = output

        try:
            os.makedirs(self.outdir)
        except:
            for ifiles in os.walk(self.outdir):
                for filename in ifiles[2]:
                    os.remove('/'.join([self.outdir,filename])) 

        if software == 'g16':

            if theory['disp'] in [True, 'EmpiricalDispersion=GD3BJ', 'GD3BJ']:
                theory['disp'] = 'EmpiricalDispersion=GD3BJ'
            elif theory['disp'] in ['EmpiricalDispersion=GD3', 'GD3']:
                theory['disp'] = 'EmpiricalDispersion=GD3'
            else: 
                theory['disp'] = ' '

            input_file = self.outdir + '/input.com'
            f = open(input_file, 'w')
            f.write('%nproc=' + str(theory['nprocs'])+'\n')
            f.write('%mem='+theory['mem']+'\n')
            f.write(' '.join(['#P', theory['method'], theory['basis_set'],  theory['jobtype'], theory['other_options'], theory['disp'], '\n']))
            f.write('\n')
            f.write(self._id + '\n')
            f.write('\n ')
            f.write(str(theory['charge']) + ' ' + str(theory['multiplicity']) + '\n')
            for at, xyz in zip(self.atoms, self.xyz):
                line = '{0:5s} {1:10.3f} {2:10.3f} {3:10.3f}\n'.format(at, xyz[0], xyz[1], xyz[2])
                f.write(line)
            f.write(' ')

            if theory['extra'] == None: f.close()
            else:
               f.write('\n')
               f.write(theory['extra'] + '\n')
               f.write(' ') 
            f.close()

        elif software == 'fhiaims':

            control_file = self.outdir + '/control.in'
            geom_file    = self.outdir + '/geometry.in'

            c = open(control_file, 'w')
            c.write('xc ' + str(theory['xc']) + '\n')
            c.write(theory['disp'] + '\n')
            c.write('charge ' + str(theory['charge']) + '\n')
            c.write(theory['jobtype']+'\n')
            c.write(theory['convergence_options'] + '\n')
            c.write('density_update_method ' + theory['density_update_method'] + '\n')
            c.write('check_cpu_consistency ' + theory['check_cpu_consistency'] + '\n')
            diff_atoms = set(self.atoms)
            
            for at in diff_atoms:
                EN="{0:02d}".format(element_number(at))
                with open('/exports/apps/fhi-aims.210226/species_defaults/'+theory['basis_set'] + '/' + EN + '_' +at+'_default','r') as light: 
                    for line in light.readlines():
                        c.write(line)
            c.close()

            g = open(geom_file, 'w')
            for n, at, xyz in zip(range(self.NAtoms), self.atoms, self.xyz):
                if n in theory['extra']: freeze = 'constrain_relaxation .true.'
                else: freeze = '' 
                line = 'atom      {0:10.3f} {1:10.3f} {2:10.3f} {3:3s}{4:s}\n'.format( xyz[0], xyz[1], xyz[2], at, freeze)
                g.write(line)
            g.close()

    def run_qm(self, theory, software='g16'):

        """ Opens and runs a simulation in the Gaussian application. To run this function GausView must already be intalled on the device
        :param mpi: (bool) message passing interface, set true to use parallel programming. experimental.
        """

        try: hasattr(self, 'outdir')
        except:
            print("Create input first")
            sys.exit(1)
            
        cwd=os.getcwd(); os.chdir(self.outdir)

        if software == 'g16':
            with open('input.log', 'w') as out:
                gauss_job = Popen("g16 input.com ", shell=True, stdout=out, stderr=out)
                gauss_job.wait()
            os.chdir(cwd)
            return gauss_job.returncode 
            #could pose an error with the puckscan script, inverted return 

        elif software == 'fhiaims':
            with open('aims.log', 'w') as out: 
                fhi_job = Popen("mpiexec -np " + str(theory['nprocs']) + '  ' + str(theory['exec']), shell=True, stdout=out, stderr=out)
                fhi_job.wait()

            os.chdir(cwd)
            return fhi_job.returncode

    def calculate_ccs(self, params = None, method = 'pa', accuracy = 1):


        """ Calls program sigma to calculate collision cross section, the sigma must be in the PATH variable. Need to change hardcoded paths otherwise it won't work

        :param temp_dir: (string) name of a directory that will be generated to hold onto some files generated during the calculations
        :param methond: (string) pa or ehss, different methods of calculation
        :param accuracy: dont change the default, return a value converged within 1%
        """ 

        #make a temp dir to store dat files
        #need to make a specialized .xyz file for sigma

        if not params: params = "/home/matma/bin/sigma-parameters.dat"

        if hasattr(self, 'ccs'):
            print("{0:20s} already has ccs".format(self._id))
            return None

        #if not hasattr(self, 'outdir'):
        #    outdir = '/'.join([output, self._id])
        #    self.outdir = outdir

        for ifiles in os.walk(self.path):
            if "pa.dat" in ifiles[2]:
                for line in open('/'.join([self.path, "pa.dat"])).readlines():
#                     if re.search('Average PA', line.decode('utf-8')):
#                         self.ccs  = float(line.decode('utf-8').split()[4])
                     if re.search('Average PA', line):
                         self.ccs  = float(line.split()[4])
                         return None 

        with open( '/'.join([self.path, 'sig.xyz']),'w') as ccs:

            ccs.write("{0:3d}\n".format(self.NAtoms))
            for at, xyz in zip(self.atoms, self.xyz):
                ccs.write("{0:3s}{1:10.3f}{2:10.3f}{3:10.3f}\n".format(at, xyz[0], xyz[1], xyz[2] ))
            ccs.close()

        if method == 'pa':
            #requires a parameter file and sigma code available, needs to be a path variable !!!

            with open('/'.join([self.path, 'pa.dat']), 'w') as out:
                ccs_job = Popen("sigma -f xyz -i " + '/'.join([self.path,'sig.xyz']) +' -n ' +  str(accuracy) + " -p " + params, shell=True, stdout=out, stderr=out)
                ccs_job.wait()
                #out, err = ccs_job.communicate()
            for line in open('/'.join([self.path, 'pa.dat']), 'r').readlines():
                #if re.search('Average PA', line.decode('utf-8')): 
                #    self.ccs  = float(line.decode('utf-8').split()[4])
                if re.search('Average PA', line): 
                    self.ccs  = float(line.split()[4])

    def load_log(self, software="g16"):

        """ Creates a conformer object using infromation from the self.path attribute
        """

        # why did this try function get commented out?
        #try:
        #    logfile = open(file_path, 'r')
        #except IOError: 
        #    print("%30s not accessible", file_path)
        #    return 1 


        if software == "g16" :

            flags = { 'freq_flag': False, 'nmr_flag': False, 'opt_flag': False, 'jcoup_flag': False, 'normal_mode': False, 'read_geom': False}
            job_type  = None 

            #temprorary variables to hold the data
            freq = [] ; ints = [] ; vibs = [] ; geom = [] ; atoms = [] ; nmr = [] 
            self.NAtoms = None
    
            for line in open("/".join([self.path, "input.log"]), 'r').readlines():
    
                    if not job_type and re.search('^ #', line):

                        if "opt" in line:
                            if "freq" in line: 
                                job_type = 'optfreq'
                            else: 
                                job_type = 'opt'
                        elif "freq" in line: 
                            if "opt" in line: 
                                job_type = 'optfreq'
                            else: 
                                job_type = 'freq'
                                flags["freq_flag"] = True
                        elif "nmr" in line:
                            job_type = 'nmr' 
                        else: 
                            job_type = 'sp'
    
                    if self.NAtoms is None and re.search('^ NAtoms=', line): 
                        self.NAtoms = int(line.split()[1])
    
                    if job_type == 'optfreq' or job_type == "freq":
    
                        if flags['freq_flag'] == False and re.search('Normal termination', line): flags['freq_flag'] = True 
                        #We skip the opt part of optfreq job, all info is in the freq part

                        if flags['freq_flag'] == True: 
    
                            if re.search('SCF Done',   line): self.E = float(line.split()[4]) 
                            elif re.search('Sum of electronic and zero-point Energies',   line): self.Ezpe = float(line.split()[6])
                            elif re.search('Sum of electronic and thermal Enthalpies' ,   line): self.H    = float(line.split()[6])                    
                            elif re.search('Sum of electronic and thermal Free Energies', line): self.F    = float(line.split()[7])
     
                            elif re.search('Coordinates', line) and  len(geom) == 0: flags['read_geom'] = True

                            elif flags['read_geom'] == True and re.search('^\s*.\d', line):
                                geom.append([float(x) for x in line.split()[3:6]])
                                atoms.append(element_symbol(line.split()[1]))
                                if int(line.split()[0]) == self.NAtoms:
                                   flags['read_geom'] = False

                            elif  re.search('Deg. of freedom', line):
                                self.NVibs  = int(line.split()[3])

                            elif re.search('^ Frequencies', line):
                                freq_line = line.strip()
                                for f in freq_line.split()[2:5]: freq.append(float(f))
                                flags['normal_mode'] = False

                            elif re.search('^ IR Inten', line):
                                ir_line = line.strip()
                                for i in ir_line.split()[3:6]: ints.append(float(i))
    
                            elif re.search('^  Atom  AN', line):
                                flags['normal_mode'] = True          #locating normal modes of a frequency
                                mode_1 = []; mode_2 = []; mode_3 = [];
                                #continue

                            elif flags['normal_mode'] == True and re.search('^\s*\d*\s*.\d*', line) and len(line.split()) > 3:
                                mode_1.append([float(x) for x in line.split()[2:5]])
                                mode_2.append([float(x) for x in line.split()[5:8]])
                                mode_3.append([float(x) for x in line.split()[8:11]])
    
                            elif flags['normal_mode'] == True:
                                flags['normal_mode'] = False
                                for m in [mode_1, mode_2, mode_3]: vibs.append(np.array(m))

                    elif job_type == 'opt': 
    
                        if re.search('SCF Done',   line): E = float(line.split()[4])
                        if re.search('Optimization completed.', line): 
                            self.E = E ; 
                            flags['opt_flag'] = True  
                        if flags['opt_flag'] == True:
                            if re.search('Standard orientation:', line) : 
                                flags['read_geom'] = True

                            elif flags['read_geom'] == True and re.search('^\s*.\d', line):
                                geom.append([float(x) for x in line.split()[3:6]])
                                atoms.append(element_symbol(line.split()[1]))
                                if int(line.split()[0]) == self.NAtoms:
                                   flags['read_geom'] = False

                    elif job_type == 'nmr':

                        if re.search('SCF Done',   line): 
                            self.E = float(line.split()[4])
                        elif re.search('Coordinates', line) and len(geom) == 0: 
                            flags['read_geom'] = True

                        elif flags['read_geom'] == True and re.search('^\s*.\d', line):
                            geom.append([float(x) for x in line.split()[3:6]])
                            atoms.append(element_symbol(line.split()[1]))
                            if int(line.split()[0]) == self.NAtoms:
                                flags['read_geom'] = False

                        elif re.search('Total nuclear spin-spin coupling J', line):
                            spin = [ [] for i in range(self.NAtoms)]
                            flags['jcoup_flag'] = True

                        elif flags['jcoup_flag'] == True and re.search('-?\d\.\d+[Dd][+\-]\d\d?', line):
                            for x in line.split()[1:]:
                                spin[int(line.split()[0])-1].append(float(x.replace('D','E')))

                        elif flags['jcoup_flag'] == True and re.search('End of Minotr F.D. properties file', line):
                            flags['jcoup_flag'] = False

                    elif job_type == 'sp': 

                        if re.search('SCF Done',   line): self.E = float(line.split()[4])
                        elif re.search('Standard orientation:', line) : 
                            flags['read_geom'] = True
                        elif flags['read_geom'] == True and re.search('^\s*.\d', line):
                            geom.append([float(x) for x in line.split()[3:6]])
                            atoms.append(element_symbol(line.split()[1]))
                            if int(line.split()[0]) == self.NAtoms:
                               flags['read_geom'] = False

            #postprocessing: 
            if job_type == 'freq' or job_type == 'optfreq': 
                self.Freq = np.array( freq ) 
                self.Ints = np.array( ints )
                self.Vibs=np.zeros((self.NVibs, self.NAtoms, 3))
                for i in range(self.NVibs): self.Vibs[i,:,:] = vibs[i]

            if job_type == 'nmr':
                for at in spin:
                    while len(at) < self.NAtoms: at.append(0)
                self.nmr = np.tril(spin) 

            self.xyz = np.array(geom)
            self.atoms = atoms

        elif software == "fhiaims" :

            geom = [] ; atoms = []
    
            read_geom = False
            self.NAtoms = None
            self._id    = self.path.split('/')[-1]
    
            for line in open("/".join([self.path, "aims.log"]) , 'r').readlines():
    
                if  "Number of atoms" in  line: 
                    self.NAtoms = int(line.split()[5])
    
                #Final energy:
                if " | Total energy of the DFT" in line: 
                    self.E = float(line.split()[11])/27.211384500 #eV to Ha
                #Reading final geom:
                elif " Final atomic structure:" in line: read_geom = True
                elif read_geom == True and "atom " in line:
                    geom.append([ float(x) for x in line.split()[1:4] ])
                    atoms.append(line.split()[-1])
                elif read_geom == True and "--------" in line: read_geom = False

            self.xyz = np.array(geom)
            self.atoms = atoms

        elif software == "xyz" : 
#
            self.NAtoms = None
            self._id    = self.path.split('/')[-1]
            self.topol = self._id
            geom = [] ; atoms = []

            for n, line in enumerate(open('/'.join([self.path, "geometry.xyz"]), 'r').readlines()): #this should be anything .xyz

                if n == 0 and self.NAtoms == None: self.NAtoms = int(line)
                if n == 1:
                   try: 
                       self.E = float(line)
                   except: 
                       pass 
                if n > 1:
                    if len(line.split()) == 0: break 
                    geom.append([float(x) for x in line.split()[1:4]])
                    if line.split()[0].isalpha(): atoms.append(line.split()[0])
                    else: atoms.append(element_symbol(line.split()[0]))

            self.xyz = np.array(geom)
            self.atoms = atoms

        return True

    def __str__(self): 

        """Prints a some molecular properties"""

        print ("%20s%20s   NAtoms=%5d" %(self._id, self.topol, self.NAtoms))

        if hasattr(self, 'F'):  print ("E={0:12.6f} H={1:12.6f} F={2:12.6f}".format(self.E, self.H, self.F))
        else: print("E={0:12.6f}".format(self.E))

        if hasattr(self, 'graph'):
            for n  in self.graph.nodes:
                ring = self.graph.nodes[n]
                print ("Ring    {0:3d}:  {1:6s} {2:6.1f} {3:6.1f} {4:6.1f}".format(n, ring['pucker'], *ring['theta']), end='')
                if 'c6_atoms' in ring:
                    print("{0:10.1f}".format(ring['c6_dih']), end = '\n')
                else:
                    print('')

            for e in self.graph.edges:
                edge = self.graph.edges[e]

                if len(edge['dihedral']) == 2: 
                    print ("Link {0}:  {1:6s} {2:6.1f} {3:6.1f}".format(e, edge['linker_type'], edge['dihedral'][0], edge['dihedral'][1]), end='\n' )

                elif len(edge['dihedral']) == 3: 
                    print ("Link {0}:  {1:6s} {2:6.1f} {3:6.1f} {4:6.1f}".format(e, edge['linker_type'], edge['dihedral'][0], edge['dihedral'][1], edge['dihedral'][2]), end='\n')

                elif len(edge['dihedral']) == 4: 
                    print ("Link {0}:  {1:6s} {2:6.1f} {3:6.1f} {4:6.1f} {5:6.1f}".format(e, edge['linker_type'], edge['dihedral'][0], edge['dihedral'][1], edge['dihedral'][2], edge['dihedral'][3]), end='\n')

        return ' '

    def gaussian_broadening(self, broaden, resolution=1):
 
        """ Performs gaussian broadening on IR spectrum
        generates attribute self.IR - np.array with dimmension 4000/resolution consisting gaussian-boraden spectrum
        
        :param broaden: (float) gaussian broadening in wn-1
        :param resolution: (float) resolution of the spectrum (number of points for 1 wn) defaults is 1, needs to be fixed in plotting
        """

        IR = np.zeros((int(4000/resolution) + 1,))
        X = np.linspace(0,4000, int(4000/resolution)+1)
        for f, i in zip(self.Freq, self.Ints):  IR += i*np.exp(-0.5*((X-f)/int(broaden))**2)
        self.IR=np.vstack((X, IR)).T #tspec

    def connectivity_matrix(self, distXX, distXH):

        """ Creates a connectivity matrix of the molecule. A connectivity matrix holds the information of which atoms are bonded and to what. 

        :param distXX: The max distance between two atoms (not hydrogen) to be considered a bond
        :param distXH: The max distance between any atom and a hydrogen atom to be considered a bond
        """

        Nat = self.NAtoms
        self.conn_mat = np.zeros((Nat, Nat))

        for at1 in range(Nat):
            for at2 in range(Nat):
                
                dist = get_distance(self.xyz[at1], self.xyz[at2])
                if at1 == at2: pass
                elif (self.atoms[at1] == 'H' or self.atoms[at2] == 'H') and dist < distXH: 
                    self.conn_mat[at1,at2] = 1; self.conn_mat[at2,at1] = 1 
                elif (self.atoms[at1] != 'H' and self.atoms[at2] != 'H') and dist < distXX:
                    self.conn_mat[at1,at2] = 1; self.conn_mat[at2,at1] = 1   

        #Remove bifurcated Hs:
        for at1 in range(Nat):
            if self.atoms[at1] == 'H' and np.sum(self.conn_mat[at1,:]) > 1:

                    at2list = np.where(self.conn_mat[at1,:] == 1) 
                    at2dist = [ round(get_distance(self.xyz[at1], self.xyz[at2x]), 3) for at2x in at2list[0]]
                    at2 = at2list[0][at2dist.index(min(at2dist))]
                    for at2x in at2list[0]: 
                        if at2x != at2: 
                            print('remove', self._id, at2x, at1, at2)
                            self.conn_mat[at1, at2x] = 0 ; self.conn_mat[at2x, at1] = 0

        cm = nx.graph.Graph(self.conn_mat)
        self.Nmols = nx.number_connected_components(cm)

    def pyranose_basis(self, rd, sugar_basis):
        adj_atom_O = adjacent_atoms(self.conn_mat, rd['O'])

        for atom in adj_atom_O:
            if self.atoms[atom].count('H') == 2 or [self.atoms[adj_at] for adj_at in adjacent_atoms(atom)].count(
                    'H') == 2:
                rd['C5'] = atom
            elif (
                    [self.atoms[adj_at] for adj_at in adjacent_atoms(atom)].count('H') > 1 and
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
        sugar_basis_no_rep = []
        for at in sugar_basis:
            if at not in sugar_basis_no_rep:
                sugar_basis_no_rep.append(at)

        sugar_basis = sugar_basis_no_rep

        Carb_Oxygen = [atom for atom in sugar_basis if 'O' in atom][0]
        sugar_basis.remove(Carb_Oxygen)

        for atom_index, atom in enumerate(sugar_basis):
            rd[f"C{atom_index + 1}"] = atom

        if (
                [self.atoms[rd['C6']]].count('H') >= 1 and
                [self.atoms[rd['C1']]].count('C') > 1):

            rd_index = 6
            while rd_index > 0:
                rd[f"C{rd_index + 1}"] = rd[f"C{rd_index}"]
                rd_index -= 1

            for C2_adjaceent in adjacent_atoms(self.atoms[rd['C2']]):
                if C2_adjaceent not in sugar_basis and 'C' in C2_adjaceent:
                    rd['C1'] = C2_adjaceent

        return rd

    def furanose_basis(self, rd, sugar_basis):
        adj_atom_O = adjacent_atoms(self.conn_mat, rd['O'])

        for atom in adj_atom_O:
            if self.atoms[atom].count('H') == 2 or [self.atoms[adj_at] for adj_at in adjacent_atoms(atom)].count(
                    'H') == 2:
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

        sugar_basis_no_rep = []
        for at in sugar_basis:
            if at not in sugar_basis_no_rep:
                sugar_basis_no_rep.append(at)

        sugar_basis = sugar_basis_no_rep

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

                sugar_basis_list = list(nx.cycle_basis(conn_mat, oxygen_atom_list[0]))
                rd['O'] = oxygen_atom_list[0]

                for sugar_basis in sugar_basis_list:
                    if len(sugar_basis) < 5:
                        continue

                    if type(rd) is not dict:
                        continue

                    if 'O' not in rd.keys():
                        continue

                    if len(sugar_basis) == len(ring) and rd['O'] in sugar_basis:

                        if len(ring) == 6:
                            rd = Conformer.pyranose_basis(conn_mat, rd['O'], sugar_basis, rd)

                        elif len(ring) == 5:
                            rd = Conformer.furanose_basis(conn_mat, rd['O'], sugar_basis, rd)

                    elif len(sugar_basis) != len(ring) and rd['O'] in sugar_basis:
                        cycle = ring

                        new_cycle = []

                        oxygen_atom = rd['O']
                        if oxygen_atom not in cycle:
                            continue

                        oxygen_index = cycle.index(oxygen_atom)
                        new_cycle.append(oxygen_atom)

                        index = oxygen_index + 1
                        cycle_len = len(cycle)

                        if 'O' not in cycle[0]:
                            while index != oxygen_index:

                                if index != cycle_len:
                                    new_cycle.append(cycle[index])
                                    index += 1
                                else:
                                    index = 0
                                    new_cycle.append(cycle[index])
                                    index += 1

                        cycle = new_cycle
                        sugar_basis = cycle

                        if len(cycle) == 6:
                            rd = Conformer.pyranose_basis(conn_mat, rd['O'], sugar_basis, rd)

                        elif len(cycle) == 5:
                            rd = Conformer.furanose_basis(conn_mat, rd['O'], sugar_basis, rd)

            if oxygen_atoms > 1 and len(ring) >= 7:
                for oxygen_atom in oxygen_atom_list:
                    test_basis = nx.minimum_cycle_basis(conn_mat, oxygen_atom)

                    for cycle in test_basis:
                        if len(cycle) == 6:
                            oxygen_atom_counter = sum(1 for cycle_atom in cycle if self.atoms[cycle_atom] == 'O')
                            oxygen_atom_cycle_list = [cycle_atom for cycle_atom in cycle if
                                                      self.atoms[cycle_atom] == 'O']

                            if oxygen_atom_counter == 1:
                                rd['O'] = oxygen_atom_cycle_list[0]
                                sugar_basis = list(cycle)
                                rd = self.pyranose_basis(rd, sugar_basis)

                        elif len(cycle) == 5:
                            oxygen_atom_cycle_list = [cycle_atom for cycle_atom in cycle if
                                                      self.atoms[cycle_atom] == 'O']
                            oxygen_atom_counter = len(oxygen_atom_cycle_list)

                            if oxygen_atom_counter == 1:
                                rd['O'] = oxygen_atom_cycle_list[0]
                                sugar_basis = list(cycle)
                                rd = self.furanose_basis(rd, sugar_basis)

            rd_list.append(rd)

        rd_list = [rd for rd in rd_list if rd is not None]
        rd_list = [rd for rd in rd_list if len(rd.keys()) >= 5]
        return rd_list

    def glycosidic_link_check(self, conn_mat, edge, c1_list, rd_list):
        glycosidic_link_list = []

        node_1 = edge[0]
        node_2 = edge[1]

        rd1_index = int(node_1.split(' ')[-1])
        rd2_index = int(node_2.split(' ')[-1])

        rd1 = rd_list[rd1_index]
        rd2 = rd_list[rd2_index]

        for ring_index in range(1, len(list(rd1.values()))):
            if ring_index == 5:
                continue

            atom = rd1[f"C{ring_index}"]

            for het_at in adjacent_atoms(atom):
                if 'C' not in het_at and 'H' not in het_at and het_at not in rd1.values():
                    adj_atom_list = adjacent_atoms(het_at)

                    if het_at.count('C') == 2 and rd1['C5'] not in adj_atom_list:
                        c1_count = sum(1 for adj_at in adj_atom_list if adj_at in c1_list)

                        for c1_atom in adjacent_atoms(het_at):
                            if c1_atom in c1_list and c1_atom in rd2.values():

                                if c1_count > 0:
                                    return f"C{ring_index}"

        return glycosidic_link_list

    def ring_dict_finder(atom, rd_list):
        for rd in rd_list:
            if atom in rd.values():
                return rd

    def find_red_end(self, c1_list, rd_list, conn_mat):
        for c1 in c1_list:
            ring_dict = Conformer.ring_dict_finder(c1, rd_list)

            for atom in adjacent_atoms(self.atoms[c1]):
                if 'C' not in atom and 'H' not in atom:
                    if atom in ring_dict.values():
                        continue

                    if [self.atoms[atom]].count('H') >= 1:
                        return rd_list.index(ring_dict)
                    else:
                        c1_count = sum(1 for adj_atom in adjacent_atoms(self.atoms(atom))
                                       if adj_atom in c1_list)

                        if c1_count == 2 and len(
                                Conformer.glycosidic_link_check(rd=ring_dict, c1_list=c1_list)) > 1:
                            return rd_list.index(ring_dict)

    def ring_connectivity_checker(self, rd1, rd2, conn_mat):
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
            HC2_count = self.atoms[C2].count('H')
            NC2_count = self.atoms[C2].count('N')

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

    def amine_check(self,conn_mat, rd):

        if len(list(rd.values())) == 7:
            C2 = rd['C2']
        elif len(list(rd.values())) == 8:
            C2 = rd['C3']
        else:
            C2 = None

        if C2 is not None:
            HC2_count = self.atoms[C2].count('H')
            NC2_count = self.atoms[C2].count('N')

            for C2_adj_at in adjacent_atoms(self.atoms[C2]):
                if 'N' in C2_adj_at:
                    HN_count = self.atoms[C2_adj_at].count('H')
                    CN_count = self.atoms[C2_adj_at].count('C')

                    if HN_count >= 1 and CN_count == 1:
                        amine = True

                    else:
                        amine = False

            if 'amide' not in locals():
                amine = False

            return amine

    def ring_graph_maker(self, rd_list, conn_mat):
        ring_graph = nx.Graph()

        for rd1 in rd_list:
            for rd2 in rd_list:
                if rd1 != rd2 and Conformer.ring_connectivity_checker(rd1=rd1, rd2=rd2, conn_mat=conn_mat) \
                        and not ring_graph.has_edge(rd_list.index(rd1), rd_list.index(rd2)) \
                        and not ring_graph.has_edge(rd_list.index(rd2), rd_list.index(rd1)):
                    ring_graph.add_edge(rd_list.index(rd1), rd_list.index(rd2))

        if Conformer.amide_check(conn_mat=conn_mat, rd=rd1) == True:
            ring_graph.add_edge(f"Amide {rd_list.index(rd1)}", f"Ring {rd_list.index(rd1)}", weight=2)

        if ring_graph.number_of_edges() == 0:
            ring_graph.add_node('Ring 0')

        return ring_graph

    def sort_rings(self, rd_list, conn_mat):
        c1_list = [rd['C1'] if 'C1' in rd else rd['C2'] for rd in rd_list]
        red_end = Conformer.find_red_end(c1_list=c1_list, rd_list=rd_list, conn_mat=conn_mat)
        ring_graph = Conformer.ring_graph_maker(rd_list=rd_list, conn_mat=conn_mat)

        dfs_ring_list = list(nx.dfs_preorder_nodes(ring_graph, red_end))
        for dfs_index, node in enumerate(dfs_ring_list):
            if 'Amide' in node:
                dfs_ring_list.remove(node)
            else:
                rd_list_index = int(list(node.split())[-1])
                dfs_ring_list[dfs_index] = rd_list[rd_list_index]

        if dfs_ring_list == []:
            dfs_ring_list = rd_list

        tree =  nx.dfs_tree(ring_graph, red_end)

        glyco_array = [Conformer.glycosidic_link_check(conn_mat,edge,c1_list) for edge in list(nx.dfs_edges(tree, red_end))]
        return dfs_ring_list, tree, glyco_array

    def dihedral_angle(self, atom1, atom2, atom3, atom4, conf):

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

    def sugar_stero(self, rd, conf):
        if len(rd.values()) == 7:
            dihedral_angle = Conformer.dihedral_angle(rd['O'], rd['C5'], rd['C4'], rd['C6'], conf)
        elif len(rd.values()) > 7:
            dihedral_angle = Conformer.dihedral_angle(rd['O'], rd['C6'], rd['C5'], rd['C7'], conf)
        else:
            dihedral_angle = None

        if dihedral_angle is not None and 0 > dihedral_angle:
            sugar_type = 'D'
        elif dihedral_angle is not None and dihedral_angle > 0:
            sugar_type = "L"
        else:
            sugar_type = 'None'

        return sugar_type

    def glycosidic_link_type(self, rd, sugar_type, conf, conn_mat):

        if len(rd.values()) == 7:

            enumeric_H = [adj_at for adj_at in Conformer.adjacent_atoms(conn_mat, rd['C1']) if
                          'H' in adj_at and adj_at not in rd][0]
            dihedral_angle = Conformer.dihedral_angle(rd['O'], rd['C1'], rd['C2'], enumeric_H, conf)

        elif len(rd.values()) > 7:

            enumeric_H = [adj_at for adj_at in Conformer.adjacent_atoms(conn_mat, rd['C2']) if
                          'H' in adj_at and adj_at not in rd][0]
            dihedral_angle = Conformer.dihedral_angle(rd['O'], rd['C2'], rd['C3'], enumeric_H, conf)

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

    def find_pg(self, dfs_list, conn_mat):
        pg_list = []

        for ring in dfs_list:
            pg_dict = {}

            if len(ring.values()) == 7:
                pg_root = ring['C6']
                root_adj = adjacent_atoms(at=pg_root,conn_mat=conn_mat)

                for adj_at in root_adj:
                    if 'O' in adj_at:
                        pg_O_adj = [at for at in adjacent_atoms(at=adj_at,conn_mat=conn_mat) if at  != ring['C6']]
                        OC_count = adj_at.count('C')
                        OH_count = pg_O_adj.count('H')

                        pg_dict['O'] = adj_at
                        if OC_count == 2:
                            pg_dict['Ac'] = pg_O_adj[0]


            elif len(ring.values()) > 7:
                pg_root = ring['C7']
                root_adj = adjacent_atoms(at=pg_root, conn_mat=conn_mat)

                for adj_at in root_adj:
                    if 'O' in adj_at:
                        pg_O_adj = [at for at in adjacent_atoms(at=adj_at, conn_mat=conn_mat) if at != ring['C7']]
                        OC_count = adj_at.count('C')
                        OH_count = pg_O_adj.count('H')

                        pg_dict['O'] = adj_at
                        if OC_count == 2:
                            pg_dict['Ac'] = pg_O_adj[0]

            pg_list.append(pg_dict)
        return pg_list

    def ring_stereo_compiler(self, conf, dfs_list, conn_mat):

        sugar_type_list = []
        glyco_type_list = []

        for rd in dfs_list:
            sugar_type = Conformer.sugar_stero(rd, conf)
            link_type = Conformer.glycosidic_link_type(rd, sugar_type, conf, conn_mat)
            sugar_type_list.append(sugar_type)
            glyco_type_list.append(link_type)

        return sugar_type_list, glyco_type_list

    def sugar_type_checker(rd,xyz_array,conn_mat):

        # Bit Order: Ring Size, C6, O2, O3, O4, Amide, O6, Amine
        sugar_dict = {'Tal':11111010, 'TalNac':11111110, 'TalA':11111020, 'TalN':11111011,'6dTal':11111000, '6dTalNac':11111100,
                      'Man':11110010, 'ManNac':11110110, 'ManA':11110020, 'ManN':11110011, 'Rha':11110000, 'RhaNac':11110100,
                      'Ido':11101010, 'IdoNac':11101110, 'IdoA':11101020, 'IdoN':11101011,
                      'Alt':11100010, 'AltNac':11100110, 'AltA':11100020, 'AltN':11100011, '6dAlt':11011011, '6dAltNac':11011100,
                      'Gul':11001010, 'GulNac':11001110, 'GulA':11001020, 'GulN':11001011, '6dGul':11001000,
                      'All':11000010, 'AllNac':11000110, 'AllA':11000020, 'AllN':11000011,
                      'Gal':11011010, 'GalNac':11011110, 'GalA':11011020, 'GalN':11011011, 'Fuc':11100000, 'FucNac':11100100,
                      'Glc':11010010, 'GlcNac':11010110, 'GlcA':11010020, 'GlcN':11010011, 'Qui':11010000, 'QuiNac':11010100, 'Xyl':11010030}

        if len(rd.keys()) >=7:

            O2 = [atom for atom in adjacent_atoms(rd['C2']) if 'H' not in atom and atom not in rd.values()][0]
            O3 = [atom for atom in adjacent_atoms(rd['C3']) if 'H' not in atom and atom not in rd.values()][0]
            O4 = [atom for atom in adjacent_atoms(rd['C4']) if 'H' not in atom and atom not in rd.values()][0]

            O2_Dihedral = Conformer.dihedral_angle(rd['C1'],rd['C2'],rd['C3'],O2,xyz_array)
            O3_Dihedral = Conformer.dihedral_angle(rd['C2'],rd['C3'],rd['C4'],O3,xyz_array)
            O4_Dihedral = Conformer.dihedral_angle(rd['C3'],rd['C4'],rd['C5'],O4,xyz_array)

            amide_check = Conformer.amide_check(conn_mat, rd)
            amine_check = Conformer.amine_check(conn_mat, rd)

            C5H = rd['C5'].count('H')

            if C5H == 1:
                O6_num = rd['C6'].count('O')
            else:
                O6_num = None

            sugar_bit = '11'
            for value in [O2_Dihedral, O3_Dihedral, O4_Dihedral, amide_check, O6_num, amine_check]:
                if type(value) == float:
                    if value < 0:
                        sugar_bit += '0'
                    elif value > 0:
                        sugar_bit += '1'

                elif type(value) == bool:
                    if value is True:
                        sugar_bit += '1'
                    else:
                        sugar_bit += '0'

                elif type(value) == int:
                    if value == 1:
                        sugar_bit += '1'
                    elif value == 2:
                        sugar_bit += '2'
                    else:
                        sugar_bit += '0'

                elif type(value) == None:
                    sugar_bit += '3'

            sugar_bits = int(sugar_bit)

            for key,value in zip(sugar_dict.keys(),sugar_dict.values()):
                if value == sugar_bits:
                    return key

    def snfg(tree,dfs_list,glyco_list,stero_list,sugar_list, rd_list, node_size:float = 3, edge_length:float = 5):

        snfg_graph = nx.Graph()

        sugar_dict = {'Tal': 11111010, 'TalNac': 11111110, 'TalA': 11111020, 'TalN': 11111011, '6dTal': 11111000,
                      '6dTalNac': 11111100,
                      'Man': 11110010, 'ManNac': 11110110, 'ManA': 11110020, 'ManN': 11110011, 'Rha': 11110000,
                      'RhaNac': 11110100,
                      'Ido': 11101010, 'IdoNac': 11101110, 'IdoA': 11101020, 'IdoN': 11101011,
                      'Alt': 11100010, 'AltNac': 11100110, 'AltA': 11100020, 'AltN': 11100011, '6dAlt': 11011011,
                      '6dAltNac': 11011100,
                      'Gul': 11001010, 'GulNac': 11001110, 'GulA': 11001020, 'GulN': 11001011, '6dGul': 11001000,
                      'All': 11000010, 'AllNac': 11000110, 'AllA': 11000020, 'AllN': 11000011,
                      'Gal': 11011010, 'GalNac': 11011110, 'GalA': 11011020, 'GalN': 11011011, 'Fuc': 11100000,
                      'FucNac': 11100100,
                      'Glc': 11010010, 'GlcNac': 11010110, 'GlcA': 11010020, 'GlcN': 11010011, 'Qui': 11010000,
                      'QuiNac': 11010100, 'Xyl': 11010030}

        current_dir = os.path.abspath('')
        string_list = current_dir.split('CarPpy')
        image_dir = os.path.join(string_list[0], 'CarPpy', 'CarP', 'snfg')

        image_dict = {}
        for key in sugar_dict:
            image = os.path.join(image_dir,f"{key}.png")
            image_dict[key] = image

        images = {k: PIL.Image.open(fname) for k, fname in image_dict.items()}

        covered_nodes = []
        dfs_edges = list(nx.dfs_edges(tree))

        for index, node in enumerate(sugar_list):
            if node in images.keys():
                snfg_graph.add_node(f"{node} {index}", image=images[node])
            else:
                snfg_graph.add_node(f"{node} {index}")

        pos = {}

        root_node = dfs_edges[0][0]
        pos[f"{sugar_list[0]} 0"] = [0,0]
        covered_nodes.append(root_node)

        for link,edge in zip(glyco_list,dfs_edges):
            node = edge[-1]
            prev_node = edge[0]
            if node not in covered_nodes:
                covered_nodes.append(node)

                rd2_index = int(node.split(' ')[-1])
                rd2 = rd_list[rd2_index]

                rd1_index = int(prev_node.split(' ')[-1])
                rd1 = rd_list[rd1_index]

                dfs2_index = dfs_list.index(rd2)
                dfs1_index = dfs_list.index(rd1)

                snfg_graph.add_edge(f"{sugar_list[dfs1_index]} {dfs1_index}", f"{sugar_list[dfs2_index]} {dfs2_index}")
                prev_coords = pos[f"{sugar_list[dfs1_index]} {dfs1_index}"]
                if len(sugar_list) <= 3:
                    if link == 'C4':
                        pos[f"{sugar_list[dfs2_index]} {dfs2_index}"] = [prev_coords[0] - edge_length,prev_coords[1]]
                    elif link == 'C6':
                        pos[f"{sugar_list[dfs2_index]} {dfs2_index}"] = [prev_coords[0] - edge_length/2, prev_coords[1] + math.sqrt(3)*edge_length/2]
                    elif link == 'C3':
                        pos[f"{sugar_list[dfs2_index]} {dfs2_index}"] = [prev_coords[0] - edge_length/2, prev_coords[1] - math.sqrt(3)*edge_length/2]
                    elif link == 'C2':
                        pos[f"{sugar_list[dfs2_index]} {dfs2_index}"] = [prev_coords[0], prev_coords[1] - edge_length]
                else:
                    if link == 'C4' or link == 'C2':
                        pos[f"{sugar_list[dfs2_index]} {dfs2_index}"] = [prev_coords[0] - edge_length,prev_coords[1]]
                    elif link == 'C6':
                        pos[f"{sugar_list[dfs2_index]} {dfs2_index}"] = [prev_coords[0] - edge_length/2, prev_coords[1] + math.sqrt(3)*edge_length/2]
                    elif link == 'C3':
                        pos[f"{sugar_list[dfs2_index]} {dfs2_index}"] = [prev_coords[0] - edge_length/2, prev_coords[1] - math.sqrt(3)*edge_length/2]

        fig, ax = plt.subplots(1,1)
        nx.draw(snfg_graph, pos=pos, ax=ax, with_labels=False, width=50*node_size/(len(sugar_list)*edge_length**1.5), node_size=0)

        tr_figure = ax.transData.transform
        tr_axes = fig.transFigure.inverted().transform

        icon_size = (ax.get_xlim()[1] - ax.get_xlim()[0]) * (0.4*node_size/(len(sugar_list)*edge_length**1.5))
        icon_center = icon_size / 2.0

        for n in snfg_graph.nodes:
            xf, yf = tr_figure(pos[n])
            xa, ya = tr_axes((xf, yf))
            # get overlapped axes and plot icon
            a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
            print(snfg_graph.nodes[n]['image'])

            if 'image' in snfg_graph.nodes[n].keys():
                a.imshow(snfg_graph.nodes[n]["image"])

            a.axis("off")

        return fig,ax

    def measure_c6(self): 

        """Dihedral angle between carbon 5 and carbon 6. Sugars with 1,6 glycosidic bond does not have c6 atoms. This angle would just the angle on the glycocidic bond
        """

        for n in self.graph.nodes:
            if 'c6_atoms' in self.graph.nodes[n]:
                self.graph.nodes[n]['c6_dih'] = measure_dihedral(self, self.graph.nodes[n]['c6_atoms'])[0]

    def set_c6(self, ring, dih):

        """Sets a new dihedral angle between carbon 5 and carbon 6
        :param ring: index to indicate which ring is being considered in the molecule
        :param dih: the new dihedral angle 
        """

        if 'c6_atoms' in self.graph.nodes[ring]:
            atoms = self.graph.nodes[ring]['c6_atoms']
            set_dihedral(self, atoms, dih)
            self.measure_c6()

    def measure_pg(self, ring, PG = 'all'):
        
        if PG == 'all': 
            for pg in ['C2', 'C3', 'C4', 'C6']: 
                if pg in self.graph.nodes[ring].keys():
                    for n, dih in enumerate(self.graph.nodes[ring][pg]['dih_atoms']):
                        self.graph.nodes[ring][pg]['PG_dihs'][n] = measure_dihedral(self, dih)[0]
        else: 
            for n, dih in enumerate(self.graph.nodes[ring][PG]['dih_atoms']):
                self.graph.nodes[ring][PG]['PG_dihs'][n] = measure_dihedral(self, dih)[0]

    def set_pg(self, ring, PG, dihs):
        
        for n, dih in enumerate(self.graph.nodes[ring][PG]['dih_atoms']):
            set_dihedral(self, dih, dihs[n])
        self.measure_pg(ring, PG)

    def measure_glycosidic(self):

        """ Measures the dihedral angle of the glycosidic bond
        """

        for e in self.graph.edges:

            atoms = self.graph.edges[e]['linker_atoms']
            phi, ax = measure_dihedral(self, atoms[:4])
            psi, ax = measure_dihedral(self, atoms[1:5])

            if len(atoms) == 6: #1-6 linkage
                omega, ax = measure_dihedral(self, atoms[2:6])
                self.graph.edges[e]['dihedral'] = [phi, psi, omega]

            elif len(atoms) == 7: # linkage at NAc
                omega, ax = measure_dihedral(self, atoms[2:6])
                gamma, ax = measure_dihedral(self, atoms[3:7])
                self.graph.edges[e]['dihedral'] = [phi, psi, omega, gamma]

            else: self.graph.edges[e]['dihedral'] = [phi, psi]

    def set_glycosidic(self, bond, phi, psi, omega=None, gamma=None):

        """ Changes the dihedral angle of the glycosidic bond

        :param bond: (int) index of which glycosidic bond to alter
        :param phi: (float) phi angle
        :param psi: (float) psi angle 
        """

        #atoms = sort_linkage_atoms(self.dih_atoms[bond])
        atoms = self.graph.edges[bond]['linker_atoms']
        set_dihedral(self, atoms[:4], phi)
        set_dihedral(self, atoms[1:5], psi)

        if omega != None: 
            set_dihedral(self, atoms[2:6], omega)
        if gamma != None: 
            set_dihedral(self, atoms[3:7], gamma)

        self.measure_glycosidic()

    def measure_ring(self):

        """ Assigns the theta angles and the canonical ring pucker 
        """

        for n in self.graph.nodes:

            #atoms = self.graph.nodes[n]['ring_atoms']
            self.graph.nodes[n]['theta']    = ring_dihedrals(self, n)
            self.graph.nodes[n]['pucker']   = ring_canon(self.graph.nodes[n]['theta']); 

    def set_ring(self, ring_number, pucker):

       set_ring_pucker(self, ring_number, pucker)

    def update_topol(self, models):

        """ Updates topology and checks for proton shifts
        """

        conf_links = [ self.graph.edges[e]['linker_type'] for e in self.graph.edges]
        self.topol = 'unknown'

        for m in models:

            m_links = [ m.graph.edges[e]['linker_type'] for e in m.graph.edges ]
            mat = self.conn_mat - m.conn_mat #difference in connectivity

            if not np.any(mat) and conf_links == m_links and self.anomer == m.anomer : 
                self.topol = m.topol
                return 0  

            elif conf_links == m_links and self.anomer == m.anomer:
                atc = 0 #atom counter
                acm = np.argwhere(np.abs(mat) == 1) #absolute connectivity matrix
                for at in acm:
                    if self.atoms[at[0]] == 'H' or self.atoms[at[1]] == 'H': atc += 1 
                if atc == len(acm):
                        self.topol = m.topol+'_Hs' #identify if there is only proton shifts 
                        return 0  

        return 0 

    def save_xyz(self):

        xyz_file='/'.join([self.path, "geometry.xyz"])
        print(xyz_file)
        f = open(xyz_file, 'w')
        f.write('{0:3d}\n'.format(self.NAtoms))
        f.write('xyz test file\n')
        for at, xyz in zip(self.atoms, self.xyz):
            line = '{0:5s} {1:10.3f} {2:10.3f} {3:10.3f}\n'.format(at, xyz[0], xyz[1], xyz[2])
            f.write(line)

    def show(self, width=600, height=600, print_xyz = False):

        """ Displays a 3D rendering of the conformer using Py3Dmol

        :param width: the width of the display window 
        :param height: the height of the display window
        """

        XYZ = "{0:3d}\n{1:s}\n".format(self.NAtoms, self._id)
        for at, xyz in zip(self.atoms, self.xyz):
            XYZ += "{0:3s}{1:10.3f}{2:10.3f}{3:10.3f}\n".format(at, xyz[0], xyz[1], xyz[2] )
        xyzview = p3D.view(width=width,height=height)
        xyzview.addModel(XYZ,'xyz')
        xyzview.setStyle({'stick':{}})
        xyzview.zoomTo()
        xyzview.show()
        if print_xyz == True: print(XYZ)

    def plot_ir(self,  xmin = 900, xmax = 1700, scaling_factor = 0.965,  plot_exp = False, exp_data = None, exp_int_split=False, normal_modes=False, save_fig=False, save_dat=False):

        """ Plots the IR spectrum in xmin -- xmax range,
        x-axis is multiplied by scaling factor, everything
        is normalized to 1. If exp_data is specified, 
        then the top panel is getting plotted too. 
        Need to add output directory. Default name is self._id
        """

        import matplotlib.pyplot as plt
        from matplotlib.ticker import NullFormatter

        fig, ax = plt.subplots(1, figsize=(10,3))

        #left, width = 0.02, 0.98 ; bottom, height = 0.15, 0.8
        #ax  = [left, bottom, width, height ]
        #ax  = plt.axes(ax)
        exten = 20

        ax.tick_params(axis='both', which='both', bottom=True, top=False, labelbottom=True, right=False, left=False, labelleft=False)
        ax.spines['top'].set_visible(False) ; ax.spines['right'].set_visible(False) ; ax.spines['left'].set_visible(False)
        ax.xaxis.set_tick_params(direction='out')
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.set_ylim(0,1.1)

        xticks = np.linspace(xmin,xmax,int((xmax-xmin+2*exten)/100)+1)
        ax.set_xticks(xticks)
        ax.set_xticklabels([int(x) for x in xticks], fontsize=10)
        ax.set_xlim(xmin-exten, xmax+exten+10)

        shift = 0.05 ;         incr = (self.IR[-1,0] - self.IR[0,0])/(len(self.IR)-1)
        scale_t  =  1/np.amax(self.IR[int(xmin/incr):int(xmax/incr)+100,1])

        if plot_exp == True:
            if exp_int_split == False:  
                scale_exp=  1/np.amax(exp_data[:,1])
                ax.plot(exp_data[:,0], exp_data[:,1]*scale_exp+shift, color='r', alpha=0.5, linewidth=2)
                ax.fill_between(exp_data[:,0], exp_data[:,1]*scale_exp+shift, np.linspace(shift,shift, len(exp_data[:,1])), color='r', alpha=0.5)

            else:
                print("split")
                scale_expL=  1/np.amax(exp_data[:,1])
                scale_expH= scale_t * np.amax(self.IR[int(1200/incr):int(xmax/incr)+100,1]) /(np.amax(np.where(exp_data[:,0] > 1200, 0, exp_data[:,1])))
                split_wn = np.where(exp_data[:,0] == 1200) ; split_wn = split_wn[0][0]
                ax.plot(exp_data[:split_wn,0], exp_data[:split_wn,1]*scale_expL+shift, color='r', alpha=0.75, linewidth=2)
                ax.fill_between(exp_data[:split_wn,0], exp_data[:split_wn,1]*scale_expL+shift, np.linspace(shift,shift, len(exp_data[:split_wn,1])), color='r', alpha=0.5)

                ax.plot(exp_data[split_wn:,0], exp_data[split_wn:,1]*scale_expH+shift, color='r', alpha=0.75, linewidth=2)
                ax.fill_between(exp_data[split_wn:,0], exp_data[split_wn:,1]*scale_expH+shift, np.linspace(shift,shift, len(exp_data[split_wn:,1])), color='r', alpha=0.5)

        Xsc = self.IR[:,0]* scaling_factor ; IRsc = self.IR[:,1]*scale_t
        ir_theo = ax.plot(Xsc, IRsc+shift, color='0.25', linewidth=2)
        ax.fill_between(Xsc, np.linspace(shift, shift, len(IRsc)), IRsc+shift, color='0.5', alpha=0.5)

        if normal_modes == True:
            for l in range(len(self.Freq)):
                 ax.plot([scaling_factor*self.Freq[l], scaling_factor*self.Freq[l]], [shift, self.Ints[l]*scale_t+shift], linewidth=2, color='0.25')        

        fig.tight_layout()
        current_path = os.getcwd()
        #output_path =  os.path.join(current_path, self.path, self._id+'.png')
        #print(output_path + self._id+'.png')
        if save_fig == True:  
            plt.savefig('/'.join([self.path,'ir_plot.pdf']) , dpi=300)

        if save_dat == True:
            with open('/'.join([self.path, 'ir_harm.dat']),'w') as out:
                for ir in self.IR:
                    out.write("{0:10.3f}{1:103f}\n".format(ir[0], ir[1]))

    def rotate_conf(self, conf_index, rotmat):

        """Stores the following in each conformer obj the name of the conformer it is being rotated to match, the index of that conformer and the rotation matrix.
        The rotation matrix is then multiplied to the existing xyz matrix and the vibrations matrix. Those rotated matrices are also saved.


        :param conf_name: (string) name of the conformer this conformer has been rotated to
        :param conf_index: (int) index of the conformer rotated to, in the list of conformers of the conformer space
        :param rotmat: (3x3 numpy array) the rotation matrix
        """
        self.rot_mat = rotmat
        self.rot_index = conf_index
        self.xyz = np.matmul(self.rot_mat, self.xyz.T).T

        for vib in range(3*self.NAtoms-6):
            self.Vibs[vib,:,:] = np.matmul(self.rot_mat,self.Vibs[vib,:,:].T).T

