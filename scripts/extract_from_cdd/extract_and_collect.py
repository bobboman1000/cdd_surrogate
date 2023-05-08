import sys, os
from joblib import delayed, Parallel
from tqdm import tqdm

N_JOBS=50
def main(args):
    path_to_exp_set = args[0]
    target_path = args[1]
    def process_exp(path_to_exp):
        os.system(f"python extract.py {path_to_exp}")
        # Compute time series
        os.system(f"./pre_plot_stress.sh {path_to_exp}")
        os.system(f"./pre_plot_distribution.sh {path_to_exp}")
        os.system(f"./pre_plot_dislocation_density.sh {path_to_exp}")
        
    paths = [os.path.join(path_to_exp_set, filename) for filename in os.listdir(path_to_exp_set)]
    pool = Parallel(n_jobs=N_JOBS)
    pool((delayed(process_exp)(path) for path in tqdm(paths)))
    
    os.system(f"python collect.py {path_to_exp_set} {target_path}")

if __name__ == "__main__":
    main(sys.argv[1:])
