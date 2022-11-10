# Auto sampling-submisions-analysis tool

import CarP
import copy, sys, os, shutil, argparse


dtime = CarP.utilities.dtime

def run_benchmark(**kwargs): 

    Bsettings = {
        "model"    : "glc",
        "job_type" : "sp scf=tight",
        "outdir"   : "benchmark",
        "output"   : "benchmark.log",
        "method"   : "PM3",
        "basis_set": " ",
        "disp"     : False,
        "software" : "g16"
        }

    for key in kwargs:
        if kwargs[key] != None: Bsettings[key] = kwargs[key]

    for p in sys.path: 
        if "CarP" in p: carp_path = p

    if not os.path.exists(Bsettings["outdir"]): os.mkdir(Bsettings["outdir"])

    output = '/'.join([Bsettings["outdir"], Bsettings["output"]])

    with open(output, 'w') as out:

        sys.stdout = out
        sys.stderr = out

        print("benchmark settings:\n", Bsettings)
        reference = CarP.Space('/'.join([carp_path, 'Benchmark_sets', Bsettings["model"]]), software='xyz')
        reference.sort_energy()

        benchmark = CarP.Space(Bsettings["outdir"])
        fin_bench = [ conf._id for conf in benchmark ] 
        if len(fin_bench) > 0: print("Completed benchmarks:\n", fin_bench)

        benchmark.set_theory(software = Bsettings["software"], method = Bsettings["method"], basis_set = Bsettings["basis_set"], disp=Bsettings["disp"],  
            nprocs=8, mem='8GB', charge=0, multiplicity='1',
            jobtype=Bsettings["job_type"])

        for n, conf in enumerate(reference):
            
            benchmark.append(copy.deepcopy(conf))
            bench = benchmark[-1]

            if bench._id in fin_bench: continue
            bench.path= '/'.join([benchmark.path, bench._id]) 
            if os.path.exists(bench.path):
                print("delete", bench._id)
                shutil.rmtree(bench.path)

            bench.create_input(benchmark.theory, bench.path, software=Bsettings["software"])
            print('{0:3d} Running {1:>12s} Date: {2:30s}'.format(n, bench._id, dtime()))
            succ_job = bench.run_qm(benchmark.theory)

            if not succ_job:
                bench.load_log(software = Bsettings["software"])
            else:
                print(bench._id + " failed.")
                bench.E = 0.0

            out.flush()

        f = open('/'.join([Bsettings["outdir"], "Benchmark.dat"]), 'w')
        f.write("{0:>3s}{1:>12s}{2:>16s}{3:>16s}\n".format('n', 'id', 'Ref [Ha]', 'Bench'))
        for n, r, b in zip(range(len(benchmark)), reference, benchmark):
            f.write("{0:3d}{1:>12s}{2:16.8f}{3:16.8f}\n".format(n, r._id, r.E, b.E))
        f.close()


def main():

        #parses the command line arguments, every argument is required
        #python3 puck_scan.py --in_dir glucose -r 0 -d low --out_dir g
        #for more specifics: python3 puck_scan.py -h 

    parser = argparse.ArgumentParser()

    parser.add_argument('--model',     choices = ['glc', 'glc_oc', 'glcnac', 'nana', 'a14a', 'a16a'] , required=False, help='benchmark set')
    parser.add_argument('--outdir',    required=False, help='outdir that will output files files')
    parser.add_argument('--output',    required=False, help='output file with energies')
    parser.add_argument('--method',    required=False, help='selected benchmark method')
    parser.add_argument('--basis_set', required=False, help='selected benchmark basis set')
    parser.add_argument('--software' , choices = ['g16', 'fhiaims'] , required=False, help='selected software')
    args = parser.parse_args()

    run_benchmark(**vars(args))

if __name__ == '__main__':

    main()

