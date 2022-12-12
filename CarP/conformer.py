
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
from .utilities import *
import networkx as nx
from operator import itemgetter, attrgetter
import matplotlib.pyplot as plt
import py3Dmol as p3D

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

    def assign_atoms(self, sort_atoms = False, deter_PGs = False):

        """ Labels each atom in the graph with its atomic symbol
        """

        def get_C1s(self, cycles_in_graph): 

            C1s = []
            for r in cycles_in_graph:
                for atom in r:
                    adj_atoms = [ self.atoms[x] for x in adjacent_atoms(self.conn_mat, atom)]
                    if adj_atoms.count('O') == 2:
                        C1s.append(atom)

                    elif len(adj_atoms) == 3 and adj_atoms.count('O') == 1: #carbocation
                        C1s.append(atom)
            return C1s

        def order_rings(cm, C1s):

            C1pos = []
            for C1 in C1s:
                NRed=0 #NRed = reducing end NNon = non reducing end
                for C12 in C1s:
                    path = nx.shortest_path(cm, C1, C12)
                    if len(path) == 1: continue
                    elif len(path) == 3:  #1-1 glycosidic bond, oxygen will belong to the Reducing Carb, 
                        NRed += 0.5
                        break

                    nred = 0
                    for r in cycles_in_graph:
                       if path[1] not in r: nred += 1

                    if nred == len(cycles_in_graph): NRed += 1

                if int(NRed) != NRed:
                    if NRed == 0.5: pass
                    else: NRed+=1 #It's will be placed last
                C1pos.append(NRed)

            ring_atoms = sort_ring_atoms(self, cycles_in_graph)
            C1s = [ i[0] for i in sorted(zip(C1s, C1pos), key=itemgetter(1)) ]
            ring_atoms = [ i[0] for i in sorted(zip(ring_atoms, C1pos), key=itemgetter(1)) ]
            C1pos.sort()
            #print(self._id, C1s, C1pos)

            for i in range(len(C1pos)):
                for j in range(i+1, len(C1pos)):
                    if C1pos[i] == C1pos[j]: 
                        path = nx.shortest_path(cm, C1s[i], C1s[j])
                        link = []
                        #print(path)
                        for at in path[1:-1]: 
                            for r in ring_atoms: 
                                if at in r.values(): 
                                    link.append(list(r.keys())[list(r.values()).index(at)])

                        #print(link)
                        if int(link[0][-1:]) > int(link[-1][-1:]):
                            C1pos[i] += 0.1
                        elif int(link[0][-1:]) < int(link[-1][-1:]):
                            C1pos[j] += 0.1 

            C1s = [ i[0] for i in sorted(zip(C1s, C1pos), key=itemgetter(1)) ]
            ring_atoms = [ i[0] for i in sorted(zip(ring_atoms, C1pos), key=itemgetter(1)) ]
            C1pos.sort()

            return C1s, C1pos, ring_atoms

        def sort_ring_atoms(self, cycles_in_graph): 

            ring_atoms = []

            for r in cycles_in_graph:
                if len(r) != 6: continue #Non six-membered rings not implemented

                ring_atoms.append({}) #dictionary, probably atom desc
                # C5 and O
                rd = ring_atoms[-1] # rd = ring dicitionary
                for at in r:
                    if self.atoms[at] == 'O':
                        rd['O'] = at #Ring Oxygen, we start counting from here

                adj_atoms_O  = adjacent_atoms(self.conn_mat, rd['O'])
                for at in adj_atoms_O:
                    adj_atoms = adjacent_atoms(self.conn_mat, at)
                    if [self.atoms[atom] for atom in adj_atoms].count('H') == 2: #no C6 carbon:
                        rd['C5'] = at
                    else:
                        for at2 in adj_atoms:
                            if self.atoms[at2] == 'C' and at2 not in r: #C5 atom with C6 carbon
                                rd['C5'] = at
                                rd['C6'] = at2
                                for at3 in adjacent_atoms(self.conn_mat, rd['C6']):
                                    if self.atoms[at3] == 'O': rd['O6'] = at3

                for at in [rd['O'], rd['C5']]: r.remove(at)

                for at in r:
                    if self.conn_mat[at][rd['O']] == 1: rd['C1'] = at
                    elif self.conn_mat[at][rd['C5']] == 1: rd['C4'] = at
                for at in [rd['C4'], rd['C1']]:  r.remove(at)

                for at in r:
                    if self.conn_mat[at][rd['C1']] == 1: rd['C2'] = at
                    elif self.conn_mat[at][rd['C4']] == 1: rd['C3'] = at
                for at in [rd['C3'], rd['C2']]:  r.remove(at)

            return ring_atoms


        def determine_PGs(self, ring_number):

            ra = self.graph.nodes[ring_number]['ring_atoms']
            self.graph.nodes[ring_number]['PGs_dih'] = []
            linkages = []
            for e in self.graph.edges:
                for at in self.graph.edges[e]['linker_atoms']: linkages.append(at)
            #print(linkages)

            for at in ['C2', 'C3', 'C4', 'C6']:
                if at in ra.keys():
                    adj_atoms = adjacent_atoms(self.conn_mat, ra[at])
                    for at2 in adj_atoms:
                        if self.atoms[at2] in ['C', 'N', 'O', 'F', 'S'] and at2 not in linkages:
                            if at2 not in ra.values() or ('O6' in ra.keys() and ra['O6'] == at2):

                                disconnect_atoms(self, ra[at], at2)
                                PG_atoms = determine_carried_atoms(self, ra[at], at2)
                                connect_atoms(self, ra[at], at2)

                                self.graph.nodes[ring_number][at] = {}
                                #self.graph.nodes[ring_number][at]['PG_atoms'] = PG_atoms

                                PG_sum = ""
                                PG_atoms_names = [ self.atoms[pgat] for pgat in PG_atoms ]
                                for atom in [ 'H', 'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br']: 
                                    natom = PG_atoms_names.count(atom) 
                                    if  natom > 0: 
                                        PG_sum += atom + str(natom)

                                self.graph.nodes[ring_number][at]['PG_name']   = protecting_group_name(PG_sum)
                                self.graph.nodes[ring_number][at]['dih_atoms'] = protecting_group_dihedrals(self, at2, at, PG_atoms)
                                self.graph.nodes[ring_number][at]['PG_dihs'] = []
                                for d in self.graph.nodes[ring_number][at]['dih_atoms']:
                                    self.graph.nodes[ring_number][at]['PG_dihs'].append(measure_dihedral(self, d)[0])


                                #if at == 'C6': 
                                #    self.graph.nodes[ring_number][at]['dihedrals'][0] = copy.copy(self.graph.nodes[ring_number]['c6_atoms'])

        def resort_atoms(self):

            order = []
            C1s = [ x['C1'] for x in ring_atoms] #get ring atoms in order

            for n, C1 in enumerate(C1s):
                for C in ["C1", "C2", "C3", "C4", "C5", "C6", "O"]:
                    C = ring_atoms[n][C]
                    order.append(C)
                    for at in [ x for x in adjacent_atoms(self.conn_mat, C)]:
                        if at not in ring_atoms[n].values() and self.atoms[at] != 'H':
                            carried_atoms =  determine_carried_atoms(self, C, at)
                            for a, c in enumerate(carried_atoms):
                                if c in order or c in C1s: #Check whether last three atoms of the linkage is branched (amide for instance)
                                    for prev_atoms in range(1,a):
                                        #print(a, prev_atoms, carried_atoms[a], carried_atoms[a-prev_atoms])
                                        for adj in adjacent_atoms(self.conn_mat, carried_atoms[a-prev_atoms]):
                                            if adj not in C1s and adj not in order: order.append(adj)
                                    break
                                else: order.append(c)

                        elif "O6" in ring_atoms[n].keys() and at == ring_atoms[n]["O6"]: #O6 is encoded in a part of the ring
                            carried_atoms =  determine_carried_atoms(self, C, at)
                            for c in carried_atoms:
                                if c in order or c in C1s: break
                                else: order.append(c)



                    for at in [ x for x in adjacent_atoms(self.conn_mat, C)]:
                        if at not in ring_atoms[n].values() and self.atoms[at] == 'H': order.append(at)

            #print(order, len(order))

            self.atoms =  [ self.atoms[x]  for x in order]
            self.xyz   =  [ self.xyz[x]    for x in order]
            self.xyz = np.array(self.xyz)

        self.graph = nx.DiGraph()
        cm = nx.graph.Graph(self.conn_mat)
        cycles_in_graph = nx.cycle_basis(cm) #a cycle in the conn_mat would be a ring

        #print(cycles_in_graph)

        #Remove Bz/Np/etx, i.e. rings without 'O'
        #Remove bond in dioxolenium/oxonium/fused rings:

        atoms_in_cycles = []
        for cycle in cycles_in_graph:
            atoms_in_cycles += cycle
        unique_atoms_in_cycles = set(atoms_in_cycles)

        for unique in unique_atoms_in_cycles: 
            if atoms_in_cycles.count(unique) >= 2: #Atom in multiple rings

                #print('multiple', unique)
                adj_atoms = []
                for adj in adjacent_atoms(self.conn_mat, unique): #check how many Os are adj
                    adj_atoms.append(self.atoms[adj])

                if adj_atoms.count('O') == 2:
                    #print('remove', unique)

                    #dioxolenium ion => break the C1-O bond, we will check 
                    #(1) how many bonds around O (oxonium), 
                    #(2) whether next C is carbonyl or not. 

                    for adj in adjacent_atoms(self.conn_mat, unique): 
                        if self.atoms[adj] == 'O':
                            #print(adj)
                            if len(adjacent_atoms(self.conn_mat, adj)) == 3:
                                disconnect_atoms(self, unique, adj)
                            else:
                                for adj2 in adjacent_atoms(self.conn_mat, adj):
                                    if adj2 != unique and len(adjacent_atoms(self.conn_mat, adj2)) == 3:
                                        disconnect_atoms(self, unique, adj)

        #redo cycles in new matrix: 
        cm = nx.graph.Graph(self.conn_mat)
        cycles_in_graph = nx.cycle_basis(cm) 
        #print(cycles_in_graph)

        #Remove Bz/Np/etx, i.e. rings without 'O'
        cycles_to_keep = []
        for cycle in cycles_in_graph: 
            for at in cycle: 
                if self.atoms[at] == 'O': cycles_to_keep.append(cycle)
        cycles_in_graph = cycles_to_keep
        #print(cycles_in_graph)
        C1s = get_C1s(self, cycles_in_graph)
        C1s, C1pos, ring_atoms = order_rings(cm, C1s)

        #ring_atoms = sort_ring_atoms(self, cycles_in_graph)
        #C1s = [ i[0] for i in sorted(zip(C1s, C1pos), key=itemgetter(1)) ]
        #ring_atoms = [ i[0] for i in sorted(zip(ring_atoms, C1pos), key=itemgetter(1)) ]

        #print(self.atoms)
        #print(C1s, ring_atoms)

        if sort_atoms == True:

            resort_atoms(self)
            self.connectivity_matrix(distXX=self.distXX, distXH=self.distXH)
            self.graph = nx.DiGraph()
            cm = nx.graph.Graph(self.conn_mat)

            cycles_in_graph = nx.cycle_basis(cm)
            C1s = get_C1s(self, cycles_in_graph) 
            C1s, C1pos, ring_atoms = order_rings(cm, C1s)

        for n, i in enumerate(ring_atoms): 
            self.graph.add_node(n, ring_atoms = i)

        for n in self.graph.nodes:
            if 'O6' not in self.graph.nodes[n]['ring_atoms'].keys(): pass 
            else: 
               atoms = [] 
               for at in ['O', 'C5', 'C6', 'O6']:
                    atoms.append(self.graph.nodes[n]['ring_atoms'][at])
               self.graph.nodes[n]['c6_atoms'] = atoms
               self.graph.nodes[n]['c6_dih'] = measure_dihedral(self, atoms)[0]

        C1s = [ x['C1'] for x in ring_atoms] #Sorted list of C1s, first C1 is reducing end. 
        cycles_in_graph = nx.cycle_basis(cm) #a cycle in the conn_mat would be a ring

        for r1 in range(self.graph.number_of_nodes()):
            for r2 in range(self.graph.number_of_nodes()):
                linker_atoms = [] ; linked = False
                if r1 >= r2 : pass
                else:
                    path = nx.shortest_path(cm, self.graph.nodes[r1]['ring_atoms']['C1'], self.graph.nodes[r2]['ring_atoms']['C1'])
                    n = 1 ; term = False
                    while n <= len(path):
                        at = path[-n]
                        #Check wheter path[n] is inside a cycle
                        c = 0 
                        for cycle in cycles_in_graph:
                            if at in cycle: 
                                if at in self.graph.nodes[r2]['ring_atoms'].values():
                                    linker_atoms.append(self.graph.nodes[r2]['ring_atoms']['O'])
                                    linker_atoms.append(at)
                                    n += 1 ; break 
                                elif at in self.graph.nodes[r1]['ring_atoms'].values():
                                    linker_atoms.append(at)
                                    linked = True ; term = True 
                                    linker_type = (list(self.graph.nodes[r1]['ring_atoms'].keys())[list(self.graph.nodes[r1]['ring_atoms'].values()).index(at)])[-1]
                                    if len(path) == 3:  C_psi='O' 
                                    else:               C_psi = 'C'+str(int(linker_type)-1)
                                    linker_atoms.append(self.graph.nodes[r1]['ring_atoms'][C_psi])
                                    break
                                else:
                                    term = True ; break
                            else: c += 1 

                        if term == True: 
                            break
                        if c == len(cycles_in_graph):
                          linker_atoms.append(at) 
                          n += 1

                    #print(linker_type, linker_atoms)

                    if linked == True:
                        adj = adjacent_atoms(self.conn_mat, linker_atoms[1])
                        for at in adj:
                            if self.atoms[at] == 'H':
                                list_of_atoms = linker_atoms[:3] + [at]
                        #print(list_of_atoms)
                        idih = measure_dihedral( self, list_of_atoms )[0]
                        if linker_type == '5': linker_type = '6'
                        if self.atoms[linker_atoms[4]] == 'N':
                            linker_type += 'N'
                        #print(idih)
                        if idih < 0.0:
                            if 'O6' in self.graph.nodes[r2]['ring_atoms'].keys(): linkage = 'b1'+linker_type 
                            #whether it's a Fucose or not
                            else: linkage = 'a1'+linker_type
                        elif idih >= 0.0: 
                            if 'O6' in self.graph.nodes[r2]['ring_atoms'].keys(): linkage = 'a1'+linker_type
                            else: linkage = 'b1'+linker_type
                        self.graph.add_edge(r1, r2, linker_atoms = linker_atoms, linker_type = linkage ) 

        #Delete C6 bond if 16-linkage if present:
        for n in self.graph.nodes:
            node = self.graph.nodes[n]
            edge = self.graph.out_edges(n)
            if len(edge) == 0: break
            #print(edge)
            for e in edge:
                if self.graph.edges[e]['linker_type'][-2:] == '16': 
                    del node['c6_atoms'] ; del node['c6_dih']

        #determine whether L- (Fuc) or D-sugar:
        for n in self.graph.nodes:
            node = self.graph.nodes[n]
            adj = adjacent_atoms(self.conn_mat, node['ring_atoms']['C5'])
            for at in adj: 
                if self.atoms[at] == 'H': H5 = at
            list_of_atoms = [ node['ring_atoms']['C4'], node['ring_atoms']['C5'], node['ring_atoms']['C6'], H5]
            idih = measure_dihedral( self, list_of_atoms )[0]
            if idih < 0.0 : node['absconf'] = 'D'
            else: node['absconf'] = 'L'

        #determine anomaricity of the redicing end: 

        adj = adjacent_atoms(self.conn_mat, self.graph.nodes[0]['ring_atoms']['C1'])
        for at in adj:
            if self.atoms[at] == 'H': Ha = at
            elif self.atoms[at] in ['N', 'O', 'F', 'S']  and at not in self.graph.nodes[0]['ring_atoms'].values(): O = at
            else: self.anomer = 'carbocation' 
        if not self.anomer:
            list_of_atoms = [ self.graph.nodes[0]['ring_atoms']['O'], self.graph.nodes[0]['ring_atoms']['C1'], O, Ha] 
            idih = measure_dihedral( self, list_of_atoms )[0]

            if   idih <  0.0 and self.graph.nodes[0]['absconf'] == 'D': self.anomer = 'beta'
            elif idih <  0.0 and self.graph.nodes[0]['absconf'] == 'L': self.anomer = 'alpha'
            elif idih >= 0.0 and self.graph.nodes[0]['absconf'] == 'D': self.anomer = 'alpha'
            elif idih >= 0.0 and self.graph.nodes[0]['absconf'] == 'L': self.anomer = 'beta'

        if deter_PGs == True: 
            for n in self.graph.nodes: determine_PGs(self, n)

        #print (self.dih_atoms, self.dih, self.anomer)

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

