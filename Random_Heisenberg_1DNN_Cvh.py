# Linear memory and NARMA task with RMP.
# C v/s h
# Random anti-ferromagnetic heisenberg spin chain (1DNN)

import resource
import os
from joblib import Parallel, delayed
import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import LinearRegression
from Models import get_Pauli_X, get_Pauli_Y, get_Pauli_Z, get_ZZ, spin_basis_1D, Heisenberg_1DNN 
from Density_matrix import trace_1

# ---- 1. Global data generation (linear memory task and NARMA) ---
n = 10  # delay
washout, train, test = 1000, 2000, 2000
total_steps = washout + train + test + n + 100
rng_data = np.random.default_rng(seed=42)
s_raw = rng_data.uniform(0.0, 0.2, total_steps)
y_raw = np.zeros(total_steps)

for i in range(n, total_steps):
    y_raw[i] = 0.1 + 1.5 * s_raw[i-n] * s_raw[i-1] + 0.05 * y_raw[i-1] * np.sum(y_raw[i-n:i]) + 0.3 * y_raw[i-1]

s = s_raw[100:] / 0.2 
y_NARMA = y_raw[100:]

y_LinMem = np.zeros(total_steps)
for i in range(n, total_steps):
    y_LinMem[i] = s[i-n]

s_washout = s[:washout]
s_train = s[washout:washout+train]
s_test = s[washout+train:washout+train+test]
y_NARMA_train = y_NARMA[washout:washout+train]
y_NARMA_test = y_NARMA[washout+train:washout+train+test]
y_LinMem_train = y_LinMem[washout:washout+train]
y_LinMem_test = y_LinMem[washout+train:washout+train+test]

# --- 2. Parameters, readout operators, initial state, and the spin 1D basis ---
N, J, tau = 10, 1, 10
x_ops = get_Pauli_X(N)
y_ops = get_Pauli_Y(N)
z_ops = get_Pauli_Z(N)
zz_ops = get_ZZ(N,z_ops)
rho = (1/2**N)*np.ones([2**N,2**N]) # maximally coherent initial state
basis = spin_basis_1D(N)

# --- 3. THE SIMULATION FUNCTION ---
def run_simulation(h_val, seed, N=N,J=J,tau=tau,rho=rho,basis=basis):
    # Create a local RNG for this task
    local_rng = np.random.default_rng(seed)
    
    # --- 2.1. MODEL SETUP ---   
    Hamiltonian, _ = Heisenberg_1DNN(N,J,h_val,basis,local_rng)
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

    model_LinMem = LinearRegression()
    model_LinMem.fit(X_train, y_LinMem_train)
    model_NARMA = LinearRegression()
    model_NARMA.fit(X_train, y_NARMA_train)

    # Testing (Batch prediction)
    X_test = np.zeros((test, len(raw_obs)))
    for k in range(test):
        rho = time_evolve(inpt_map(rho, s_test[k], N),phase_mat)
        X_test[k, :] = get_features(rho)

    y_LinMem_pred = model_LinMem.predict(X_test)
    y_NARMA_pred = model_NARMA.predict(X_test)

    # --- 2.4. RESULTS ---
    cov_LinMem = np.cov(y_LinMem_test, y_LinMem_pred)
    cov_NARMA = np.cov(y_NARMA_test, y_NARMA_pred)
    return (cov_LinMem[0, 1]**2) / (cov_LinMem[0, 0] * cov_LinMem[1, 1]), (cov_NARMA[0, 1]**2) / (cov_NARMA[0, 0] * cov_NARMA[1, 1])

# --- 4. PARAMETER SCAN SETUP ---
h_values = np.logspace(-2, 2, 60)
n_realizations = 100 
# Create a flat list of (h, seed) tuples
seed_values = range(n_realizations)

# --- 5. PARALLEL EXECUTION ---
n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
print(f"Running in parallel with {n_cpus} CPUs")

results_flat = Parallel(n_jobs=n_cpus)(
        delayed(run_simulation)(h, seed) 
        for h in h_values 
        for seed in seed_values
    )

# Reshape into (len(h_values), n_realizations, 2) because each run returns 2 outputs
results_matrix = np.array(results_flat).reshape(len(h_values), n_realizations, 2)

# Separate the tasks along the last axis
matrix_LinMem = results_matrix[:, :, 0]
matrix_NARMA  = results_matrix[:, :, 1]

# Compute statistics for Linear Memory
c_mean_LinMem = np.mean(matrix_LinMem, axis=1)
c_std_LinMem  = np.std(matrix_LinMem, axis=1)
c_se_LinMem   = np.std(matrix_LinMem,axis=1,ddof=1) / np.sqrt(n_realizations)

# Compute statistics for NARMA
c_mean_NARMA = np.mean(matrix_NARMA, axis=1)
c_std_NARMA  = np.std(matrix_NARMA, axis=1)
c_se_NARMA   = np.std(matrix_NARMA,axis=1,ddof=1) / np.sqrt(n_realizations)

# Save everything comprehensively
np.savez_compressed(
    'results/Heisenberg_1DNN_Cvh.npz',
    h_values=h_values,
    # Raw matrix outputs
    c_raw_LinMem=matrix_LinMem,
    c_raw_NARMA=matrix_NARMA,
    # LinMem Metrics
    c_mean_LinMem=c_mean_LinMem,
    c_std_LinMem=c_std_LinMem,
    c_se_LinMem=c_se_LinMem,
    # NARMA Metrics
    c_mean_NARMA=c_mean_NARMA,
    c_std_NARMA=c_std_NARMA,
    c_se_NARMA=c_se_NARMA,
    # Metadata
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

