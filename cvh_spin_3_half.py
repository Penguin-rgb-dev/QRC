# This file performs a scan over some parameter evaluating the performance metric C (capacity) for Linear Memory 10 and NARMA 10 tasks.
# You can put the spin 3/2 hamiltonian of your choice.


import resource
import os
import time
from joblib import Parallel, delayed
import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import LinearRegression
from Models import get_spin_operators, get_spin_xx, get_spin_yy, get_spin_zz, heisenberg_spin_3_half
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
N, J, tau = 5, 1, 10
dims = 4**N
x_ops = get_spin_operators(N, op_type="x", d=4, sparse=False)
y_ops = get_spin_operators(N, op_type="y", d=4, sparse=False)
z_ops = get_spin_operators(N, op_type="z", d=4, sparse=False)
xx_ops = get_spin_xx(N, d=4, sparse=False)
yy_ops = get_spin_yy(N, d=4, sparse=False)
zz_ops = get_spin_zz(N, d=4, sparse=False)
# We flatten observables into (n_obs, dim**2) to use dot products instead of Tr(rho @ O)
raw_obs = x_ops + y_ops + z_ops + xx_ops + yy_ops + zz_ops
obs_matrix = np.array([o.conj().flatten() for o in raw_obs])

def trace_leftmost_site(rho, d=4):
    """Traces out the leftmost spin (Site N-1, MSB) from a chain with local dimension d."""
    # Convert sparse matrix to dense array if necessary
    if hasattr(rho, "toarray"):
        rho = rho.toarray()

    D = rho.shape[0]
    dim_rest = D // d

    # Reshape: (site_N-1_row, rest_row, site_N-1_col, rest_col)
    reshaped_rho = rho.reshape(d, dim_rest, d, dim_rest)

    # Trace out axes 0 and 2 (leftmost site N-1)
    return np.trace(reshaped_rho, axis1=0, axis2=2)


def inject_leftmost_state(s, rho, d=4):
    """Replaces the state at the leftmost site (Site N-1) with state psi(s)."""
    if not (0 <= s <= 1):
        raise ValueError(f"Parameter s must be in range [0, 1], got {s}")

    if 0 <= s < 1 / 3:
        psi = np.array([0, 0, np.sqrt(3 * s), np.sqrt(1 - 3 * s)])
    elif 1 / 3 <= s <= 2 / 3:
        psi = np.array([0, np.sqrt(3 * s - 1), np.sqrt(2 - 3 * s), 0])
    else:
        psi = np.array([np.sqrt(3 * s - 2), np.sqrt(3 - 3 * s), 0, 0])

    psi_dm = np.outer(psi, psi)
    rho_reduced = trace_leftmost_site(rho, d=d)

    # Places psi_dm at the leftmost position: psi_dm ⊗ rho_rest
    return np.kron(psi_dm, rho_reduced)

# maximally coherent initial state
RHO_INIT = np.full((dims,dims),1/dims,dtype=complex) 

# --- 3. THE SIMULATION FUNCTION ---
def run_simulation(h_val, seed, N=N,J=J,tau=tau):
    # Create a local RNG for this task
    local_rng = np.random.default_rng(seed)
    
    # --- 2.1. MODEL SETUP ---
    J_array = np.full(N, -1)
    h_array = local_rng.uniform(low=-h_val,high=h_val,size=N)
    Hamiltonian = heisenberg_spin_3_half(N, J_array, h_array,periodic=True)   #PUT YOUR MODEL HAMILTONIAN HERE!
    Hamiltonian = Hamiltonian.toarray()
    E, U = eigh(Hamiltonian)
    U_dag = U.conj().T
    phase_mat = np.exp(-1j * (E[:, np.newaxis] - E[np.newaxis, :]) * tau)

    # --- 2.2. define functions ---
    def evolve(rho_in, phase_mat):
            rho_energy = U_dag @ rho_in @ U
            return U @ (rho_energy * phase_mat) @ U_dag

    def extract_features(rho, s_in):
        X = np.zeros((len(s_in),len(obs_matrix)))

        for idx, s_val in enumerate(s_in):
            # input and evolve rho
            rho = evolve(inject_leftmost_state(s_val,rho),phase_mat)
            # measure
            X[idx,:] = np.real(obs_matrix @ rho.flatten())
            
        return X, rho

    # --- 2.3. EXECUTION LOOPS ---

    # copy global initial state for worker specific dynamic evolution
    rho = RHO_INIT.copy()

    # Washout (No data storage)
    for val in s_washout:
        rho = evolve(inject_leftmost_state(rho, val, N),phase_mat)

    # Training (Vectorized feature extraction)
    X_train, rho = extract_features(rho, s_train)

    model_LinMem = LinearRegression()
    model_LinMem.fit(X_train, y_trian_LinMem)
    model_NARMA = LinearRegression()
    model_NARMA.fit(X_train, y_train_NARMA)

    # Testing (Batch prediction)
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

    h_values = np.logspace(-2, 2, 60)
    n_realizations = 100
    seed_values = range(n_realizations)

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
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
  
    output_file = os.path.join(output_dir, "heisenberg_spin_3_half_cvh.npz")
    np.savez_compressed(
        output_file,
        h_values=h_values,
        n_realizations=n_realizations,
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
        j_val=J,
        tau_val=tau,
        model="fully connected transverse field ising model; H = sum_ij J_ij X_i X_j + h sum_i Z_i; J_ij in U(-j_val/2,j_val/2)."
    )
    print("Simulation complete. Final data saved.")
    print(f"\nGrid Search Finished in {time.time() - start_time:.2f} seconds.")
    # Get peak memory usage in kilobytes
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Convert to Megabytes or Gigabytes
    print(f"--- Resource Usage Report ---")
    print(f"Peak Memory Usage: {usage / 1024:.2f} MB")

