#Scanning 3D surface of the ring puckering angles

import CarP
import copy, sys, os, shutil, argparse


def puck_scan(**kwargs): #take a command line argument for the ring number

    scan_settings = {
        "in_dir"   : "models",
        "out_dir"  : "output-atoms",
        "output"   : "scan-atoms.log",
        "details"  : "low",
        "ring"     : 0,
        "sample_c6": True,
        "freeze"   : "atoms",
        "software" : "g16"
        }

    for key in kwargs:
        if kwargs[key] != None: scan_settings[key] = kwargs[key]

    output = scan_settings["output"]

    with open(output, 'w') as out: 

        sys.stdout = out
        #sys.stderr = out
        print("Scan settings:\n", scan_settings)

        #creates a space
        ring_scan = CarP.Space(scan_settings["out_dir"], software=scan_settings["software"])
        finished_puckers = [conf._id for conf in ring_scan]
        print("Complete puckers:\n", finished_puckers)

        #loads a model from in_dir that will be used to generate puckers
        ring_scan.load_models(scan_settings["in_dir"])
        reference_conf = ring_scan.models[0]

        #atoms in the ring: 
        ra = reference_conf.graph.nodes[scan_settings["ring"]]['ring_atoms']
        freeze = CarP.utilities.select_freeze(scan_settings["freeze"],ra)   
        jobtype = {"dih" : "opt=modredundant", "atoms": "opt=readopt"}

        ring_scan.set_theory(software='g16', method='PM3', basis_set= '', disp=False,  nprocs=12, mem='24GB', jobtype=jobtype[scan_settings["freeze"]], extra=freeze)
        #ring_scan.set_theory(software='fhiaims', nprocs=12, charge=0.0, extra=freeze)

        out.flush()

        pucker_conformations = CarP.utilities.pucker_scan(scan_settings["details"])
        for n, thetas in enumerate(pucker_conformations):

            _id = '-'.join([reference_conf._id,'{0:03d}'.format(n)])
            if _id in finished_puckers: continue

            ring_scan.append(copy.deepcopy(reference_conf))
            new_conf=ring_scan[-1]
            new_conf._id = '-'.join([new_conf._id,'{0:03d}'.format(n)])
            new_conf.path= '/'.join([ring_scan.path, new_conf._id])

            if os.path.exists(new_conf.path): 
                print("delete", new_conf._id)
                shutil.rmtree(new_conf.path)

            CarP.utilities.set_ring_pucker(new_conf, scan_settings["ring"], thetas)
            new_conf.create_input(ring_scan.theory, ring_scan.path, software=scan_settings["software"])
            succ_job = new_conf.run_qm(ring_scan.theory, software=scan_settings["software"])

            if not succ_job:
                new_conf.load_log(software=scan_settings["software"])
                new_conf.measure_ring() ; new_conf.measure_c6() 
                print(new_conf)
            else:
                print(new_conf._id + " failed. Moving to the next pucker.")
                del ring_scan[-1]
            out.flush()

        ring_scan.reference_to_zero()
        ring_scan.sort_energy()
        ring_scan.print_relative(pucker=True)


def main():

	#parses the command line arguments, every argument is required
	#python3 puck_scan.py --in_dir glucose -r 0 -d low --out_dir g
	#for more specifics: python3 puck_scan.py -h 

    parser = argparse.ArgumentParser(description='Selects the ring number, level of detail of the scan and the input/output directories')

    parser.add_argument('--in_dir', required=False, help='the name of the input directory')
    parser.add_argument('--out_dir', required=False, help='the name of the output directory')
    parser.add_argument('--output',  required=False, help='output log-file')
    parser.add_argument('--ring', '-r', type=int, required=False, help='the index of which ring to select on a molecule')
    parser.add_argument('--details', '-d', choices=['low','medium','high'], required=False, help='the level of detail of the pucker scan, more detail means more conformations')
    args = parser.parse_args()

    puck_scan(**vars(args))

if __name__ == '__main__':
    main()



