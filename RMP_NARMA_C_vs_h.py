# NARMA task with delay 10, with RMP.
# Optmized.
# C v/s h

import os
from joblib import Parallel, delayed
import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import LinearRegression
# Assuming these are your custom modules
from Models import get_Pauli_X, get_Pauli_Y, get_Pauli_Z, get_ZZ, Ising 
from Density_matrix import trace_1, mixed_density_matrix

# --- 1. GLOBAL DATA GENERATION (NARMA) ---
n = 10
washout, train, test = 1000, 2000, 2000
total_steps = washout + train + test + n + 100
rng_data = np.random.default_rng(seed=42)
s_raw = rng_data.uniform(0.0, 0.2, total_steps)
y_raw = np.zeros(total_steps)

for i in range(n, total_steps):
    y_raw[i] = 0.1 + 1.5 * s_raw[i-n] * s_raw[i-1] + 0.05 * y_raw[i-1] * np.sum(y_raw[i-n:i]) + 0.3 * y_raw[i-1]

s = s_raw[100:] / 0.2 
y = y_raw[100:]

s_washout = s[:washout]
s_train = s[washout:washout+train]
s_test = s[washout+train:washout+train+test]
y_train = y[washout:washout+train]
y_test = y[washout+train:washout+train+test]

# Create the operators once
N, J, tau = 10, 1, 10
x_ops = get_Pauli_X(N)
y_ops = get_Pauli_Y(N)
z_ops = get_Pauli_Z(N)
zz_ops = get_ZZ(N,z_ops)

# --- 2. THE SIMULATION FUNCTION ---
def run_simulation(h_val, seed):
    # Create a local RNG for this task
    local_rng = np.random.default_rng(seed)
    # --- 2.1. MODEL SETUP ---   
    Hamiltonian, _ = Ising(N, J, h_val, local_rng, x_ops=x_ops, z_ops=z_ops)
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
h_values = np.logspace(-2, 2, 20)
n_realizations = 100 
# Create a flat list of (h, seed) tuples
tasks = [(h, seed) for h in h_values for seed in range(n_realizations)]

# --- 4. PARALLEL EXECUTION ---
# results will be a flat list of length (20 * 100 = 2000)
n_cpus = int(os.environ.get('PBS_NP', 1))
results_flat = Parallel(n_jobs=n_cpus)(
    delayed(run_simulation)(h, seed) for h, seed in tasks
)

# --- 5. RESHAPE AND SAVE ---
# Reshape to (number_of_h, number_of_realizations)
results_matrix = np.array(results_flat).reshape(len(h_values), n_realizations)

# Calculate stats for plotting
c_mean = np.mean(results_matrix, axis=1)
c_std = np.std(results_matrix, axis=1)

np.savez_compressed(
    'quantum_sim_averaged.npz',
    h=h_values,
    c_raw=results_matrix,
    c_mean=c_mean,
    c_std=c_std,
    N_spins=N,
    J=J,
    tau=tau,
    realizations=n_realizations,
    model="Ising"
)

print("Simulation complete. Data saved.")