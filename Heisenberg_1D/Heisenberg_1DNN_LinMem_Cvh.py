# NARMA task with RMP.
# C v/s h

import resource
import os
from joblib import Parallel, delayed
import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import LinearRegression
from Models import get_Pauli_X, get_Pauli_Y, get_Pauli_Z, get_ZZ, Heisenberg_1DNN 
from Density_matrix import trace_1

# ---- 1. Global data generation (linear memory task) ---
n = 10  # delay
washout, train, test = 1000, 2000, 2000
total_steps = washout + train + test + n
rng_data = np.random.default_rng(seed=42)
s = rng_data.uniform(0, 1, total_steps)
y = np.zeros(total_steps)

for i in range(n, total_steps):
    y[i] = s[i-n]

s_washout = s[:washout]
s_train = s[washout:washout+train]
s_test = s[washout+train:washout+train+test]
y_train = y[washout:washout+train]
y_test = y[washout+train:washout+train+test]

# Create the operators amd the initial state once
N, J, tau = 10, 1, 10
x_ops = get_Pauli_X(N)
y_ops = get_Pauli_Y(N)
z_ops = get_Pauli_Z(N)
zz_ops = get_ZZ(N,z_ops)
rho = (1/2**N)*np.ones([2**N,2**N]) # maximally coherent initial state

# --- 2. THE SIMULATION FUNCTION ---
def run_simulation(h_val, seed, N=N,J=J,tau=tau,rho=rho):
    # Create a local RNG for this task
    local_rng = np.random.default_rng(seed)
    
    # --- 2.1. MODEL SETUP ---   
    Hamiltonian, _ = Heisenberg_1DNN(N,J,h_val,local_rng)
    Hamiltonian = Hamiltonian.toarray()
    E, U = eigh(Hamiltonian)
    U_dag = U.conj().T
    phase_mat = np.exp(-1j * (E[:, np.newaxis] - E[np.newaxis, :]) * tau)

    # --- 2.2. VECTORIZE OBSERVABLES ---
    # We flatten observables into (n_obs, dim**2) to use dot products instead of Tr(rho @ O)
    raw_obs = x_ops + y_ops + z_ops + zz_ops
    obs_matrix = np.array([o.flatten() for o in raw_obs]) 

    def get_features(rho_matrix):
        # Tr(A @ B) is the dot product of A.flatten() and B.T.flatten()
        # Since observables are often Hermitian, we just use the flattened obs_matrix
        return np.real(obs_matrix @ rho_matrix.flatten())

    def time_evolve(rho_0, phase_mat):
        rho_energy_t = (U_dag @ rho_0 @ U) * phase_mat
        return U @ rho_energy_t @ U_dag

    def inpt_map(rho_in, s_val, N_spins):
        # Pre-calculated basis states for speed
        psi_s = np.array([np.sqrt(s_val), np.sqrt(1-s_val)]) # Simplified basis logic
        rho_s = np.outer(psi_s, psi_s)
        return np.kron(rho_s, trace_1(rho_in, N_spins))

    # --- 2.3. EXECUTION LOOPS ---

    # Washout (No data storage)
    for val in s_washout:
        rho = time_evolve(inpt_map(rho, val, N),phase_mat)

    # Training (Vectorized feature extraction)
    X_train = np.zeros((train, len(raw_obs)))
    for k in range(train):
        rho = time_evolve(inpt_map(rho, s_train[k], N),phase_mat)
        X_train[k, :] = get_features(rho)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Testing (Batch prediction)
    X_test = np.zeros((test, len(raw_obs)))
    for k in range(test):
        rho = time_evolve(inpt_map(rho, s_test[k], N),phase_mat)
        X_test[k, :] = get_features(rho)

    y_pred = model.predict(X_test)

    # --- 2.4. RESULTS ---
    cov = np.cov(y_test, y_pred)
    return (cov[0, 1]**2) / (cov[0, 0] * cov[1, 1])

# --- 3. PARAMETER SCAN SETUP ---
h_values = np.logspace(-2, 2, 60)
n_realizations = 100 
# Create a flat list of (h, seed) tuples
seed_values = range(n_realizations)

# --- 4. PARALLEL EXECUTION ---
n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
print(f"Running in parallel with {n_cpus} CPUs")

results_flat = Parallel(n_jobs=n_cpus)(
        delayed(run_simulation)(h, seed) 
        for h in h_values 
        for seed in seed_values
    )

results_matrix = np.array(results_flat).reshape(len(h_values), n_realizations)
    
c_mean = np.mean(results_matrix, axis=1)
c_std = np.std(results_matrix,axis=1)
c_se = np.std(results_matrix, axis=1, ddof=1) / np.sqrt(n_realizations)

np.savez_compressed(''
    'results/Heisenberg_1DNN_LinMem_Cvh.npz',
    h_values = h_values,
    c_raw=results_matrix,
    c_mean=c_mean,
    c_std=c_std,
    c_se=c_se,
    n_spins=N,
    J_val=J,
    tau_val=tau,
    n_realizations=n_realizations,
    model="Heisenberg 1-dimensional nearest neighbour"
)
print("Simulation complete. Final data saved.")

# Get peak memory usage in kilobytes
usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

# Convert to Megabytes or Gigabytes
print(f"--- Resource Usage Report ---")
print(f"Peak Memory Usage: {usage / 1024:.2f} MB")

