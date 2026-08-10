import os
import time
import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import Ridge
from joblib import Parallel, delayed

# Import external modules
from Models import get_Pauli_X, get_Pauli_Z, J_matrix, FullyConnected_TFIM
from Density_matrix import trace_1

# ---- 0. DATA GENERATION (NARMA) ----
n_values = [2, 5, 10, 15, 20]
washout, train, test = 1000, 3000, 1000
num_samples = washout + train + test

max_n = max(n_values)
total_steps = max_n + 100 + num_samples

# Dataset generation (fixed seed for input generation across quantum runs)
rng_data = np.random.default_rng(seed=42)
s_raw = rng_data.uniform(0.0, 0.2, total_steps)

Y = np.zeros((num_samples, len(n_values)))
for idx, n in enumerate(n_values):
    y_raw = np.zeros(total_steps)
    for i in range(n, total_steps):
        if n == 2:
            y_raw[i] = (
                0.4 * y_raw[i - 1]
                + 0.4 * y_raw[i - 1] * y_raw[i - 2]
                + 0.6 * (s_raw[i - 1] ** 3)
                + 0.1
            )
        else:
            y_raw[i] = (
                0.1
                + 1.5 * s_raw[i - n] * s_raw[i - 1]
                + 0.05 * y_raw[i - 1] * np.sum(y_raw[i - n : i])
                + 0.3 * y_raw[i - 1]
            )
    Y[:, idx] = y_raw[-num_samples:]

S = (s_raw[-num_samples:] / 0.2).flatten()

S_washout = S[0:washout]
S_train, Y_train = S[washout : washout + train], Y[washout : washout + train]
S_test, Y_test = S[washout + train : num_samples], Y[washout + train : num_samples]


# ---- 1. CORE SIMULATION FUNCTION ----
N = 5
J_val = 1.0
h_val = 0.5

z_ops = get_Pauli_Z(N)
Z_matrix = np.array([np.asarray(op.diagonal()).ravel() for op in z_ops])

# Pre-calculate initial density matrix globally
RHO_INIT = np.full((2**N, 2**N), 1.0 / (2**N), dtype=complex)

def evaluate_narma_nmse(tau, V, seed):
    """
    Simulates the Quantum Reservoir for a given tau, V, and seed realization.
    Returns:
        NMSE (np.ndarray): 1D array of shape (len(n_values),) containing NMSE for each NARMA task.
    """
    
    # Quantum Reservoir Setup (Hamiltonian depends on variable seed)
    rng_qr = np.random.default_rng(seed)
      
    J_ij = J_matrix(N, -J_val / 2, J_val / 2, rng_qr)
    Hamiltonian, _ = FullyConnected_TFIM(N, J_ij, h_val)
    Hamiltonian = Hamiltonian.toarray()

    # Exact Diagonalization
    E, U = eigh(Hamiltonian)
    U_dag = U.conj().T

    dt = tau / V
    Phase_dt = np.exp(-1j * (E[:, np.newaxis] - E[np.newaxis, :]) * dt)
    Phase_tau = np.exp(-1j * (E[:, np.newaxis] - E[np.newaxis, :]) * tau)


    def evolve(rho_in, phase_mat):
        rho_energy = U_dag @ rho_in @ U
        return U @ (rho_energy * phase_mat) @ U_dag

    def input_map(rho_in, s):
        psi_s = np.array([np.sqrt(s), np.sqrt(1.0 - s)], dtype=complex)
        rho_s = np.outer(psi_s, psi_s.conj())
        rho_rest = trace_1(rho_in, N)
        return np.kron(rho_s, rho_rest)

    def extract_features(s_sequence, rho_start):
        n_steps = len(s_sequence)
        features = np.zeros((n_steps, N * V))
        rho_curr = rho_start

        for k in range(n_steps):
            rho_curr = input_map(rho_curr, s_sequence[k])
            for v in range(V):
                rho_curr = evolve(rho_curr, Phase_dt)
                features[k, v * N : (v + 1) * N] = np.real(Z_matrix @ rho_curr.diagonal())

        return features, rho_curr

    # Copy global initial state for worker-specific dynamic evolution
    rho = RHO_INIT.copy()

    # Washout, Train, Test
    for k in range(washout):
        rho = evolve(input_map(rho, S_washout[k]), Phase_tau)

    X_train, rho = extract_features(S_train, rho)
    X_train = (X_train + 1.0) / 2.0

    model = Ridge(alpha=1e-4).fit(X_train, Y_train)

    X_test, _ = extract_features(S_test, rho)
    X_test = (X_test + 1.0) / 2.0

    Y_pred = model.predict(X_test)

    # Compute NMSE
    nmse = np.sum((Y_pred - Y_test) ** 2, axis=0) / np.sum(Y_test ** 2, axis=0)
    return nmse


# ---- 2. MAIN PARALLEL EXECUTION ----

if __name__ == "__main__":
    start_time = time.time()

    # Define Grid Parameters
    tau_list = np.logspace(0,7,8,base=2)         # 8 values of tau
    V_list = [1,2,5,10,25,50]                     # 6 values of V
    seeds = np.arange(100, 120)                  # 20 distinct Hamiltonian realization seeds

    n_taus = len(tau_list)
    n_Vs = len(V_list)
    n_seeds = len(seeds)
    n_tasks = 5  # NARMA-2, 5, 10, 15, 20

    # Check if running under SLURM; if not, auto-detect workstation cores.
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    
    if slurm_cpus is not None:
        n_jobs = int(slurm_cpus)
    else:
        # Leaves 1-2 cores free for system responsiveness (e.g., UI, web browsing)
        total_cores = os.cpu_count() or 4
        n_jobs = max(1, total_cores - 2) 

    print(f"Workstation detected ({os.cpu_count()} CPU threads available).")
    print(f"Running grid search using {n_jobs} parallel workers...")

    # Parallel Execution via Joblib
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(evaluate_narma_nmse)(tau, V, seed)
        for tau in tau_list
        for V in V_list
        for seed in seeds
    )

    # 1. Reshape raw results into 4D matrix: (n_taus, n_Vs, 20_realizations, 5_tasks)
    nmse_4d = np.array(results).reshape(n_taus, n_Vs, n_seeds, n_tasks)

    # 2. Compute Mean and Standard Deviation across the 20 realizations (axis=2)
    nmse_mean_3d = np.mean(nmse_4d, axis=2)  # Shape: (len(tau), len(V), 5_tasks)
    nmse_std_3d  = np.std(nmse_4d, axis=2)   # Shape: (len(tau), len(V), 5_tasks)

    # Save outputs
    output_dir = "Data/Reproduction_1"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "nmse_grid_results.npz")
    np.savez_compressed(
        output_file,
        NMSE_4D=nmse_4d,             # Shape: (len(tau), len(V), 20, 5)
        NMSE_MEAN=nmse_mean_3d,       # Shape: (len(tau), len(V), 5)
        NMSE_STD=nmse_std_3d,         # Shape: (len(tau), len(V), 5)
        tau_list=tau_list,
        V_list=V_list,
        seeds=seeds,
        n_values=[2, 5, 10, 15, 20],
        n_spins = N,
        J_val = J_val,
        h_val = h_val
    )

    print(f"\nGrid Search Finished in {time.time() - start_time:.2f} seconds.")
    print("--- SAVED MATRIX SHAPES ---")
    print(f"Full 4D Matrix (tau, V, seeds, tasks): {nmse_4d.shape}")
    print(f"Mean 3D Matrix (tau, V, tasks):        {nmse_mean_3d.shape}")
    print(f"Std 3D Matrix  (tau, V, tasks):        {nmse_std_3d.shape}")