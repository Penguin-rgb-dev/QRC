import os
import time
import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import Ridge, LinearRegression
from joblib import Parallel, delayed

# External quantum system imports
from Models import get_Pauli_X, get_Pauli_Z, J_matrix, FullyConnected_TFIM
from Density_matrix import trace_1

# ---- Global Task & Data Parameters ----
n_values = [2, 5, 10, 15, 20]
washout, train, test = 1000, 3000, 1000
num_samples = washout + train + test  # 5000 total samples
max_n = max(n_values)
total_steps = max_n + 100 + num_samples

# ---- 1. SUPERIMPOSED SINE WAVE INPUT GENERATION ----
# Equation: s_k = 0.1 * [ sin(2*pi*alpha*k/T) * sin(2*pi*beta*k/T) * sin(2*pi*gamma*k/T) + 1 ]
alpha, beta, gamma = 2.11, 3.73, 4.11
T = 100.0
k_indices = np.arange(total_steps)

s_raw = 0.1 * (
    np.sin(2.0 * np.pi * alpha * k_indices / T)
    * np.sin(2.0 * np.pi * beta * k_indices / T)
    * np.sin(2.0 * np.pi * gamma * k_indices / T)
    + 1.0
)

# ---- Generate NARMA Targets Y ----
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

# Rescale input s_k to [0, 1] for state injection
S = (s_raw[-num_samples:] / 0.2).flatten()

# Dataset Splits
S_washout = S[0:washout]
S_train, Y_train = S[washout : washout + train], Y[washout : washout + train]
S_test, Y_test = S[washout + train : num_samples], Y[washout + train : num_samples]


# ==============================================================================
# BLOCK A: BASELINE LINEAR REGRESSION (WITHOUT RESERVOIR)
# ==============================================================================
print("--- Running Baseline Linear Regression (No Reservoir) ---")
# Direct input mapping: map S_train directly to Y_train
linear_baseline_model = LinearRegression().fit(S_train.reshape(-1, 1), Y_train)
Y_pred_linear_baseline = linear_baseline_model.predict(S_test.reshape(-1, 1))

# Compute Baseline NMSE
nmse_baseline = np.sum((Y_pred_linear_baseline - Y_test) ** 2, axis=0) / np.sum(Y_test ** 2, axis=0)
print("Baseline Linear Regression Y_pred shape:", Y_pred_linear_baseline.shape)  # (1000, 5)
for idx, n in enumerate(n_values):
    print(f"  NARMA-{n:02d} Baseline NMSE: {nmse_baseline[idx]:.6e}")
print("-" * 55 + "\n")


# ==============================================================================
# BLOCK B: QUANTUM RESERVOIR SIMULATION FUNCTION
# ==============================================================================

N = 6            # Fixed number of spins across all runs
J_val = 1.0      # Coupling strength
h_val = 0.5     # Transverse field
tau_val = 1.0    # Default tau

# Pre-calculate invariant Pauli Z diagonals globally
z_ops = get_Pauli_Z(N)
Z_matrix = np.array([np.asarray(op.diagonal()).ravel() for op in z_ops])  # Shape: (N, 2^N)

# Pre-calculate initial density matrix globally
RHO_INIT = np.full((2**N, 2**N), 1.0 / (2**N), dtype=complex)

def run_narma_for_V(V, tau=tau_val, seed=42):
    """
    Runs Quantum Reservoir evaluation for a specified V value.
    Returns:
        Y_pred (np.ndarray): 2D array of shape (test_len, 5) containing predictions.
    """

    rng_qr = np.random.default_rng(seed)

    # Hamiltonian & Operator construction
    z_ops = get_Pauli_Z(N)
    J_ij = J_matrix(N, -J_val / 2, J_val / 2, rng_qr)
    Hamiltonian, _ = FullyConnected_TFIM(N, J_ij, h_val)
    Hamiltonian = Hamiltonian.toarray()

    # Diagonalization
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

    # Washout Phase
    for k in range(washout):
        rho = evolve(input_map(rho, S_washout[k]), Phase_tau)

    # Training Feature Extraction
    X_train, rho = extract_features(S_train, rho)
    X_train = (X_train + 1.0) / 2.0

    #model = Ridge(alpha=1e-4).fit(X_train, Y_train)
    model = LinearRegression().fit(X_train, Y_train)

    # Testing Feature Extraction
    X_test, _ = extract_features(S_test, rho)
    X_test = (X_test + 1.0) / 2.0

    # Output predictions for all 5 NARMA tasks: shape (1000, 5)
    Y_pred = model.predict(X_test)
    return Y_pred


# ==============================================================================
# BLOCK C: PARALLEL EXECUTION & 3D MATRIX STACKING
# ==============================================================================

if __name__ == "__main__":
    start_time = time.time()
    V_list = [1, 2, 5, 10]

    # Core allocation strategy
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    n_jobs = int(slurm_cpus) if slurm_cpus is not None else max(1, (os.cpu_count() or 4) - 2)

    print(f"Launching parallel Quantum Reservoir simulations for V = {V_list} using {n_jobs} cores...")

    # Parallel Execution returning list of 2D arrays: [ (1000, 5), (1000, 5), (1000, 5), (1000, 5) ]
    Y_pred_list = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(run_narma_for_V)(V) for V in V_list
    )

    # Stack 2D matrices into a 3D matrix along axis 0
    # Axis 0: V values [V=1, V=2, V=5, V=10]
    # Axis 1: Time steps (1000 samples)
    # Axis 2: NARMA tasks (5 columns: n=2, 5, 10, 15, 20)
    Y_pred_3D = np.stack(Y_pred_list, axis=0)

    print("\n--- SIMULATION COMPLETE ---")
    print("Stacked 3D Matrix Shape (len(V), test_len, n_tasks):", Y_pred_3D.shape)

    # Calculate and display nmse
    NMSE_matrix = np.sum((Y_pred_3D - Y_test) ** 2, axis=1) / np.sum(Y_test ** 2, axis=0)

    print("\n--- RESULTS ---")
    print("Stacked 3D Matrix Shape:", Y_pred_3D.shape)

    # 2. Print results cleanly from the precomputed NMSE matrix
    #for idx_V, V_val in enumerate(V_list):
    #    print(f"\nNMSE for V = {V_val:02d}:")
    #    for idx_n, n in enumerate(n_values):
    #        print(f"  NARMA-{n:02d}: {NMSE_matrix[idx_V, idx_n]:.6e}")

    for idx_n, n in enumerate(n_values):
        print(f'\nNMSE for n = {n:02d}:')
        for idx_V, V_val in enumerate(V_list):    
            print(f'  QR-(V={V_val:02d}): {NMSE_matrix[idx_V,idx_n]:.6e}')

    # Saving outputs
    output_dir = "Data/Reproduction_1"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "nmse_table1_results.npz")
    np.savez_compressed(
        output_file,
        y_pred = Y_pred_3D,
        y_target = Y_test,
        nmse = NMSE_matrix,
        nmse_lr = nmse_baseline,
        n_spins = N,
        j_val = J_val,
        h_val = h_val,
        tau_val = tau_val
    )

    print(f"\nTotal Execution Time: {time.time() - start_time:.2f} seconds.")