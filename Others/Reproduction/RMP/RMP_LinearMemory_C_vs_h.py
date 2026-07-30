# Linear memory task with RMP.
# Optmized
# C v/s h

import resource
import os
from joblib import Parallel, delayed
import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import LinearRegression
# Assuming these are your custom modules
from Models import get_Pauli_X, get_Pauli_Y, get_Pauli_Z, get_ZZ, Ising 
from Density_matrix import trace_1, mixed_density_matrix

# ---- 1. Global data generation ---
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

# Create the operators once
N, J, h_val, tau = 10, 1, 10*0.5, 10
x_ops = get_Pauli_X(N)
y_ops = get_Pauli_Y(N)
z_ops = get_Pauli_Z(N)
zz_ops = get_ZZ(N,z_ops)

# --- 2. THE SIMULATION FUNCTION ---
def run_simulation(W_val, seed):
    # Create a local RNG for this task
    local_rng = np.random.default_rng(seed)
    # --- 2.1. MODEL SETUP ---   
    Hamiltonian, _ = Ising(N, J, h_val, local_rng, x_ops=x_ops, z_ops=z_ops, disorder=True, D=W_val)
    rho = mixed_density_matrix(10, 2, N, local_rng, complex_ensemble=True)
    E, U = eigh(Hamiltonian)
    U_dag = U.conj().T
    phase_factors = np.exp(-1j * (E[:, np.newaxis] - E[np.newaxis, :]) * tau)

    # --- 2.2. VECTORIZE OBSERVABLES ---
    # We flatten observables into (n_obs, dim**2) to use dot products instead of Tr(rho @ O)
    raw_obs = x_ops + y_ops + z_ops + zz_ops
    obs_matrix = np.array([o.flatten() for o in raw_obs]) 

    def get_features(rho_matrix):
        # Tr(A @ B) is the dot product of A.flatten() and B.T.flatten()
        # Since observables are often Hermitian, we just use the flattened obs_matrix
        return np.real(obs_matrix @ rho_matrix.flatten())

    def time_evolve_fast(rho_0):
        rho_energy_t = (U_dag @ rho_0 @ U) * phase_factors
        return U @ rho_energy_t @ U_dag

    def inpt_fast(rho_in, s_val, N_spins):
        # Pre-calculated basis states for speed
        psi_s = np.array([np.sqrt(s_val), np.sqrt(1-s_val)]) # Simplified basis logic
        rho_s = np.outer(psi_s, psi_s)
        return np.kron(rho_s, trace_1(rho_in, N_spins))

    # --- 2.3. EXECUTION LOOPS ---

    # Washout (No data storage)
    for val in s_washout:
        rho = time_evolve_fast(inpt_fast(rho, val, N))

    # Training (Vectorized feature extraction)
    X_train = np.zeros((train, len(raw_obs)))
    for k in range(train):
        rho = time_evolve_fast(inpt_fast(rho, s_train[k], N))
        X_train[k, :] = get_features(rho)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Testing (Batch prediction)
    X_test = np.zeros((test, len(raw_obs)))
    for k in range(test):
        rho = time_evolve_fast(inpt_fast(rho, s_test[k], N))
        X_test[k, :] = get_features(rho)

    y_pred = model.predict(X_test)

    # --- 2.4. RESULTS ---
    cov = np.cov(y_test, y_pred)
    return (cov[0, 1]**2) / (cov[0, 0] * cov[1, 1])

# --- 3. PARAMETER SCAN SETUP ---
W_values = np.logspace(-2, 2, 50)*0.5
n_realizations = 100 
# Create a flat list of (h, seed) tuples
seed_values = range(n_realizations)

# --- 4. PARALLEL EXECUTION ---
n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
print(f"Running in parallel with {n_cpus} CPUs")

results_flat = []
chunk_size = 5    # Process 5 h-values at a time (5 * 100 = 500 simulations per checkpoint)
total_W = len(W_values)

for i in range(0, total_W, chunk_size):
    current_chunk = W_values[i : i + chunk_size]

    print(f"--- Starting Batch: h-indices {i} to {i + len(current_chunk) - 1} ---")

    # This nested list comprehension is correct, it produces:
    # [ (h0,s0), (h0,s1)...(h0,s99), (h1,s0)... ]
    chunk_results = Parallel(n_jobs=n_cpus)(
        delayed(run_simulation)(W, seed) 
        for W in current_chunk 
        for seed in seed_values
    )
    
    results_flat.extend(chunk_results)

    # Use the unique name so you don't overwrite every time
    checkpoint_name = f"LinearMemory_checkpoint_W_index_{i}.npz"
    np.savez_compressed(checkpoint_name, data=results_flat)
    print(f"Successfully saved checkpoint: {checkpoint_name}")

# --- 5. RESHAPE AND SAVE ---
if len(results_flat) == len(W_values) * n_realizations:
    results_matrix = np.array(results_flat).reshape(len(W_values), n_realizations)
    
    c_mean = np.mean(results_matrix, axis=1)
    c_std = np.std(results_matrix, axis=1)

    np.savez_compressed(
        'LinearMemory_Cvsh.npz',
        W = np.logspace(-2,+2,50),
        c_raw=results_matrix,
        c_mean=c_mean,
        c_std=c_std,
        N_spins=N,
        J=J,
        tau=tau,
        realizations=n_realizations,
        model="Fully Connected Transverse Field Ising Model"
    )
    print("Simulation complete. Final data saved.")
else:
    print(f"Warning: Simulation ended early. Collected {len(results_flat)}/{total_W*n_realizations} results.")
    # Optional: save whatever we managed to get
    np.savez_compressed('PARTIAL_Final_Results.npz', data=results_flat)


# Get peak memory usage in kilobytes
usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

# Convert to Megabytes or Gigabytes
print(f"--- Resource Usage Report ---")
print(f"Peak Memory Usage: {usage / 1024:.2f} MB")
