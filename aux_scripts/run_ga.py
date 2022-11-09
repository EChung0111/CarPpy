import CarP
import copy, sys

GAsettings = {
            "initial_pool"  : 8,
            "alive_pool"    : 8,
            "generations"   : 60,
            "prob_dih_mut"  : 0.65,
            "prob_c6_mut"   : 0.65,
            "prob_ring_mut" : 0.33,
            "pucker_P_model": [0.2, 0.35, 0.35, 0.05, 0.05],
            "rmsd_cutoff"   : 0.1,
            "models"        : "models_ga",
            "output"        : "GAout.log",
            "output_dir"    : "GArun",
            "software"      : "g16",
            "Fmaps"         : None
            }

dtime = CarP.utilities.dtime

def  spawn_initial(GArun, n):

    m = CarP.utilities.draw_random_int(len(GArun.models))
    GArun.append(copy.deepcopy(GArun.models[m]))
    GArun[-1]._id = "initial-{0:02d}".format(n)
    GArun[-1].path= '/'.join([GArun.path, GArun[-1]._id])
    print("Generate initial-{0:02d} form {1:20s}  Date: {2:30s}".format(n, GArun.models[m]._id, dtime()))

def spawn_offspring(GArun, n, IP=GAsettings['initial_pool']):

    m = CarP.utilities.draw_random_int(GAsettings['alive_pool'])
    GArun.append(copy.deepcopy(GArun[m]))
    GArun[-1]._id = "offspring-{0:02d}".format(n-IP)
    GArun[-1].path= '/'.join([GArun.path, GArun[-1]._id])
    print("Generate offspring-{0:02d} from {1:20s} Date: {2:30s}".format(n-IP, GArun[m]._id, dtime()))

def remove_duplicates(GArun):

    for i in range(len(GArun)-1):
        rmsd = CarP.utilities.calculate_rmsd(GArun[-1], GArun[i])
        if rmsd < GAsettings['rmsd_cutoff']:
            print("RMSD: {0:6.3f} already exist as {1:20s}".format(rmsd, GArun[i]._id))
            return True 
    return False

def run_ga():

    """A genetic algorithm script that implements functions of the CarP package.
    """

    output = GAsettings["output"]
    with open(output, 'w') as out: 

        sys.stdout = out 
        sys.stderr = out
        print("Initialize:", dtime())

        #Generation
        #Creates a conformer space and fills it with inital guesses, which are  generated from models from GAsettings['models']
        GArun = CarP.Space(GAsettings["output_dir"], software=GAsettings["software"])
        GArun.load_models(GAsettings["models"])
        if GAsettings["Fmaps"]: GArun.load_Fmaps(GAsettings["Fmaps"])

        GArun.set_theory(nprocs=8, mem='16GB', 
                         method='PM3', basis_set='', charge=0, multiplicity=1, 
                         jobtype='sp', disp=False)

        #GArun.set_theory(nprocs=24, mem='64GB', charge=1, method='PBE1PBE' , basis_set='6-31G(d)', jobtype='opt=loose', disp=True)
        #GArun.set_theory(software='fhiaims', charge='1.0', basis_set='light', nprocs=24) 

        #First, chec whether there are any models in the output_dir, if yes then we sort and print them
        n = len(GArun)
        if n > 0:
            GArun.sort_energy(energy_function='E')
            GArun.reference_to_zero(energy_function='E')
            GArun.print_relative(alive=GAsettings['alive_pool'])
        out.flush()

        phase = "populate" ; IP = GAsettings['initial_pool']

        while True:

            if n >= IP: phase = "evolve"
            if n >= (GAsettings['generations'] + IP) : break

            succ_job = True
            while succ_job: 

                if    phase == "populate": 
                    spawn_initial(GArun, n)
                elif  phase == "evolve": 
                    spawn_offspring(GArun,n)

                offspring = GArun[-1]

                clash = True ; new_parent = False ; attempt = 0 
                xyz_backup = copy.copy(offspring.xyz)

                while clash:

                    #print("Attempt {0:3d}".format(attempt), end='\n')
                    if attempt > 100: 
                         print("More attempts than threshold, trying another parent.")
                         del offspring ; del GArun[-1] ; 
                         new_parent = True ; break

                    if attempt > 0:
                        offspring.xyz = copy.copy(xyz_backup)

                    if phase == "populate":

                        for e in offspring.graph.edges:
                            CarP.ga_operations.modify_glyc(offspring, e)

                        for r in offspring.graph.nodes:
                            CarP.ga_operations.modify_c6(offspring, r)
                            #CarP.ga_operations.modify_ring(offspring, r) #Don't modify rings in initial

                    elif phase == "evolve": 

                        for e in offspring.graph.edges: 
 
                            if CarP.utilities.draw_random() < GAsettings['prob_dih_mut']:
                                if GAsettings["Fmaps"]:
                                    CarP.ga_operations.modify_glyc(offspring, e, model = "Fmaps", Fmap = GArun.linkages)
                                else:
                                    CarP.ga_operations.modify_glyc(offspring, e)

                        for r in offspring.graph.nodes:

                            if CarP.utilities.draw_random() < GAsettings['prob_c6_mut']:
                                CarP.ga_operations.modify_c6(offspring, r)

                            if CarP.utilities.draw_random() < GAsettings['prob_ring_mut']:
                                CarP.ga_operations.modify_ring(offspring, r, GAsettings['pucker_P_model'])

                    attempt += 1 #records the number of attempts before a modification without clashes occurs
                    clash = CarP.utilities.clashcheck(offspring)

                if new_parent == True: continue 

                #print("Attempt {0:3d} successful".format(attempt), end='\n')
                offspring.measure_c6(); offspring.measure_glycosidic() ; offspring.measure_ring()

                offspring.create_input(GArun.theory, software=GAsettings["software"])
                succ_job = offspring.run_qm(GArun.theory, software=GAsettings["software"]) #with a proper execution of gaussian

            offspring.load_log(software=GAsettings["software"])
            offspring.measure_c6() ; offspring.measure_glycosidic() ; offspring.measure_ring()
            offspring.update_topol(GArun.models)

            #sometimes in calculation a mol will break into 2 molecules. If so it is removed.

            if offspring.Nmols > 1: 
                print("{0:3d} molecules present, remove".format(offspring.Nmols))
                del offspring ; del GArun[-1] ; continue 

            #checks for copies
            if n > 0: 
                if remove_duplicates(GArun): 
                    del GArun[-1] ; continue 

            if  phase == "populate": print("Finished initial-{0:02d}{1:27s} Date: {2:30s}".format(n, '', dtime()))
            else:                    print("Finished offspring-{0:02d}{1:27s} Date: {2:30s}".format(n-IP, '', dtime()))
            print(offspring)

            GArun.sort_energy(energy_function='E')
            GArun.reference_to_zero(energy_function='E')
            GArun.print_relative(alive=GAsettings['alive_pool'], pucker='pucker')
            out.flush()
            n += 1

if __name__ == '__main__':

    run_ga()
