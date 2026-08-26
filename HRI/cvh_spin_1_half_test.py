# This file performs a scan over some parameter evaluating the performance metric C (capacity) for Linear Memory 10 and NARMA 10 tasks.
# You can put the model Hamiltonian of your choice.
# Changes in the test file:(0) globally RHO_INIT is defined and the function copies that rho = RHO_INIT.copy();
# (1) adding extract features; (2) taking obs_matrix construction outside of the simulation funciton and correcting o.flatten() to o.conj().flatten(); 
# (3) adding xx_obs and yy_obs; (4) adding if __name__ == "__main__": ; and (5) output dir and output file etc. 
# Also changed the Hamiltonian to FullyConnected_TFIM

import resource
import os
import time
from joblib import Parallel, delayed
import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import LinearRegression
from Models import get_Pauli_X, get_Pauli_Y, get_Pauli_Z, get_XX, get_YY, get_ZZ, J_matrix, FullyConnected_TFIM 
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
total_steps = total_steps - 100
y_LinMem = np.zeros(total_steps)
for i in range(n, total_steps):
    y_LinMem[i] = s[i-n]

s_washout = s[:washout]
s_train = s[washout:washout+train]
s_test = s[washout+train:washout+train+test]
y_train_NARMA = y_NARMA[washout:washout+train]
y_test_NARMA = y_NARMA[washout+train:washout+train+test]
y_trian_LinMem = y_LinMem[washout:washout+train]
y_test_LinMem = y_LinMem[washout+train:washout+train+test]

# --- 2. Parameters, readout operators, initial state, and the spin 1D basis ---
N, J, tau = 10, 1, 10
dims = 2**N
x_ops = get_Pauli_X(N)
y_ops = get_Pauli_Y(N)
z_ops = get_Pauli_Z(N)
xx_ops = get_XX(N, x_ops)
yy_ops = get_YY(N, y_ops)
zz_ops = get_ZZ(N,z_ops)
# We flatten observables into (n_obs, dim**2) to use dot products instead of Tr(rho @ O)
raw_obs = x_ops + y_ops + z_ops + xx_ops + yy_ops + zz_ops
obs_matrix = np.array([o.conj().flatten() for o in raw_obs])

# maximally coherent initial state
RHO_INIT = np.full((dims,dims), 1/dims, dtype=complex)  

# --- 3. THE SIMULATION FUNCTION ---
def run_simulation(h_val, seed, N=N,J=J,tau=tau):
    # Create a local RNG for this task
    local_rng = np.random.default_rng(seed)
    
    # --- 2.1. MODEL SETUP ---
    J_ij = J_matrix(N,-J/2,J/2,local_rng)   
    Hamiltonian = FullyConnected_TFIM(N,J_ij,h_val)   #PUT YOUR MODEL HAMILTONIAN HERE!
    Hamiltonian = Hamiltonian.toarray()
    E, U = eigh(Hamiltonian)
    U_dag = U.conj().T
    phase_mat = np.exp(-1j * (E[:, np.newaxis] - E[np.newaxis, :]) * tau)

    # --- 2.2. define functions ---
    def time_evolve(rho_0, phase_mat):
        rho_energy_t = (U_dag @ rho_0 @ U) * phase_mat
        return U @ rho_energy_t @ U_dag

    def inpt_map(rho_in, s_val, N_spins):
        # Pre-calculated basis states for speed
        psi_s = np.array([np.sqrt(s_val), np.sqrt(1-s_val)]) # Simplified basis logic
        rho_s = np.outer(psi_s, psi_s)
        return np.kron(rho_s, trace_1(rho_in, N_spins))

    def extract_features(rho, s_in):
        X = np.zeros((len(s_in),len(obs_matrix)))

        for idx, s_val in enumerate(s_in):
            # input and evolve rho
            rho = time_evolve(inpt_map(rho, s_val, N),phase_mat)
            # measure
            X[idx,:] = np.real(obs_matrix @ rho.flatten())
            
        return X, rho

    # --- 2.3. EXECUTION LOOPS ---

    # copy global initial state for worker specific dynamic evolution
    rho = RHO_INIT.copy()

    # Washout (No data storage)
    for val in s_washout:
        rho = time_evolve(inpt_map(rho, val, N),phase_mat)

    # Training
    X_train, rho = extract_features(rho, s_train)

    model_LinMem = LinearRegression()
    model_LinMem.fit(X_train, y_trian_LinMem)
    model_NARMA = LinearRegression()
    model_NARMA.fit(X_train, y_train_NARMA)

    # Testing
    X_test, _ = extract_features(rho, s_test)

    y_pred_LinMem = model_LinMem.predict(X_test)
    y_pred_NARMA = model_NARMA.predict(X_test)

    # --- 2.4. RESULTS ---
    cov_LinMem = np.cov(y_test_LinMem, y_pred_LinMem)
    cov_NARMA = np.cov(y_test_NARMA, y_pred_NARMA)
    return (cov_LinMem[0, 1]**2) / (cov_LinMem[0, 0] * cov_LinMem[1, 1]), (cov_NARMA[0, 1]**2) / (cov_NARMA[0, 0] * cov_NARMA[1, 1])


# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    start_time = time.time()

    #h_values = np.logspace(-2, 2, 60)*0.5
    h_values = [0.5e-1]
    n_realizations = 64 #100
    seed_values = range(n_realizations)

    #n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    n_cpus = os.environ.get("JOBLIB_CPU_COUNT", os.cpu_count())
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
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
  
    output_file = os.path.join(output_dir, "fully_connected_tfim_cvh.npz")
    np.savez_compressed(
        output_file,
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
        model="fully connected transverse field ising model; H = sum_ij J_ij X_i X_j + h sum_i Z_i; J_ij in U(-J_val/2,J_val/2)."
    )
    print("Simulation complete. Final data saved.")
    print(f"\nGrid Search Finished in {time.time() - start_time:.2f} seconds.")
    # Get peak memory usage in kilobytes
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Convert to Megabytes or Gigabytes
    print(f"--- Resource Usage Report ---")
    print(f"Peak Memory Usage: {usage / 1024:.2f} MB")

