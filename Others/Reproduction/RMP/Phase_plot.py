from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d, spin_basis_general
import numpy as np
from Models import J_matrix
import matplotlib.pyplot as plt
import os
from joblib import Parallel, delayed
import resource

# parameters and basis
N, J = 10, 1
s = np.arange(N)
Z = -(s+1)
basis = spin_basis_general(N,zblock=(Z,0))

def run_simulation(h,W,seed):
    ## define operators using site-coupling lists
    rng = np.random.default_rng(seed)
    J_ij = J_matrix(N,-J/2,J/2,rng)
    D_i = rng.uniform(-W,W,N)
    J_zz = [[J_ij[i,j],i,j] for i in range(N-1) for j in range(i+1,N)]
    h_x = [[0.5*(h+D_i[i]),i] for i in range(N)]
    ## Static and dynamic lists
    static = [["zz",J_zz],["x",h_x]]
    dynamic = []
    ## build hamiltonian
    H = hamiltonian(static, dynamic, basis=basis, dtype=np.float64, check_herm=False, check_symm=False)
    E = H.eigvalsh()
    E = np.sort(E)
    delta = np.diff(E)
    MIN = np.minimum(delta[:-1],delta[1:])
    MAX = np.maximum(delta[:-1],delta[1:])
    r = MIN/MAX
    return np.mean(r)

if __name__ == "__main__":
    n_realizations=100
    seeds = range(100)
    values = np.logspace(1e-2,1e2,50)
    n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK',1))

    results_flat = Parallel(n_jobs=n_cpus)(delayed(run_simulation)(h,W,seed) for h in values for W in values for seed in seeds)
    results = np.array(results_flat).reshape(len(values),len(values),n_realizations)
    results_mean = np.mean(results,axis=2)
    np.savez_compressed('Phase_plot.npz',r_mean=results_mean, h_values=values, W_values=values,  N_spins=N, J=J, model='Fully Connected TFIM')

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    
    print(f"--- Resource Usage Report ---")
    print(f"Peak Memory Usage: {usage / 1024:.2f} MB")