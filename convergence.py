## Code to check the convergence property for different values of W, h with average over 600 realizations for each set of parameter values

import resource
import os
from joblib import Parallel, delayed
import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import LinearRegression
# Assuming these are your custom modules
from Models import get_Pauli_X, get_Pauli_Y, get_Pauli_Z, get_ZZ, Ising 
from Density_matrix import trace_1, mixed_density_matrix

# -- 1. Global Data Generation --
rng_global = np.random.default_rng(seed=54)
s = rng_global.uniform(200)
n_spins,J_val,tau_val = 10,1,10


# -- 2. Initial States --
def norm(A,B):
    C = A-B
    return np.sqrt(np.real(np.trace(C.conj().T @ C)))

def generate_states():
      N=10
      J = 1
      h = 0.05
      tau = 10
      rng = np.random.default_rng(seed=54)
      s = rng.uniform(low=0,high=1,size=24)
      rho_a = mixed_density_matrix(10,2,N,rng)
      rho_b = mixed_density_matrix(10,2,N,rng)
      x_ops = get_Pauli_X(N)
      z_ops = get_Pauli_Z(N)
      Hamiltonian, _ = Ising(N,J,h,rng,x_ops,z_ops)
      E, U = eigh(Hamiltonian)
      U_dag = U.conj().T
      phase_mat = np.exp(-1j * (E[:, np.newaxis] - E[np.newaxis, :]) * tau)
      def evolve(rho_in, phase_mat):
            rho_energy = U_dag @ rho_in @ U
            return U @ (rho_energy * phase_mat) @ U_dag

      def input_map(rho_in, s, N):
            """Map input s to the first spin and trace out the rest using the module's reshape trick."""
            psi_s = np.array([np.sqrt(1-s), np.sqrt(s)], dtype=complex)
            rho_s = np.outer(psi_s, psi_s.conj())
            # Use optimized partial trace from Density_matrix
            rho_rest = trace_1(rho_in, N)
            return np.kron(rho_s, rho_rest)

      print(f'initial norm = {norm(rho_a,rho_b)}')

      for s_in in s:
            rho_a = evolve(input_map(rho_a, s_in, N),phase_mat)

      for s_in in s:
            rho_b = evolve(input_map(rho_b, s_in, N),phase_mat)

      print(f'final norm = {norm(rho_a,rho_b)}')
      return rho_a, rho_b

rho_a, rho_b = generate_states()


# -- 3. defining the simulation function --
def run_simulation(W,h,seed,N=n_spins,J=J_val,tau=tau_val):    
    rng_local = np.uniform.default_rng(seed)
    Hamiltonian, _ = Ising(N,J,h=h,rng=rng_local,x_ops=x_ops,z_ops=z_ops,disorder=True,D=W)
    x_ops = get_Pauli_X(N)
    z_ops = get_Pauli_Z(N)
    E, U = eigh(Hamiltonian)
    U_dag = U.conj().T
    phase_mat = np.exp(-1j * (E[:, np.newaxis] - E[np.newaxis, :]) * tau)
    def evolve(rho_in, phase_mat):
        rho_energy = U_dag @ rho_in @ U
        return U @ (rho_energy * phase_mat) @ U_dag

    def input_map(rho_in, s, N):
        """Map input s to the first spin and trace out the rest using the module's reshape trick."""
        psi_s = np.array([np.sqrt(1-s), np.sqrt(s)], dtype=complex)
        rho_s = np.outer(psi_s, psi_s.conj())
        # Use optimized partial trace from Density_matrix
        rho_rest = trace_1(rho_in, N)
        return np.kron(rho_s, rho_rest)
    
    for s_in in s:
        rho_a = evolve(input_map(rho_a,s_in,N),phase_mat)
    
    for s_in in s:
        rho_b = evolve(input_map(rho_b,s_in,N),phase_mat)

    return norm(rho_a,rho_b)

# -- 4. Running the code in parellel: scan over W and h with 600 realizations for each parameter value set
n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK',1))
n_realizations = 600
seeds = range(n_realizations)
W_values = np.logspace(-2,2,50)*0.5
h_values = np.logspace(-2,2,50)*0.5

results = Parallel(n_jobs=n_cpus)(delayed(run_simulation)(W,h,seed) for W in W_values for h in h_values for seed in seeds)
results = np.array(results).reshape(len(W_values),len(h_values),n_realizations)
norm_mean = np.mean(results,axis=2)
norm_var = np.var(results,axis=2)

np.savez_compressed('Convergence_plot.npz',
                    raw_data=results, 
                    mean = norm_mean, 
                    variance = norm_var, 
                    model = 'Fully Connected Transverse Field Ising Model', 
                    n_spins = n_spins, 
                    J_val = J_val, 
                    tau_val = tau_val)