# This code evaluates nmse scanning over h_values, for the mg task with temporal multiplexing.

# ---- 0. IMPORTS ----
import resource
import os
import time
from joblib import Parallel, delayed
import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import Ridge
from Models import get_Pauli_Z, J_matrix, FullyConnected_TFIM 
from Density_matrix import trace_1


# ---- 1. DATA GENERATION ----
rng_data = np.random.default_rng(seed=42)
discard, washout_len, train_len, test_len = 1000, 1000, 10000, 2000
sigma, tau_MG, total_mg_steps = 0.1, 17, (discard + washout_len + train_len + test_len)*10
A = np.zeros(total_mg_steps)
A[0] = 1.2
delay_idx = int(tau_MG / sigma)

for i in range(total_mg_steps - 1):
    delayed_val = A[i - delay_idx] if i >= delay_idx else 1.2
    A[i + 1] = A[i] + sigma * ((0.2 * delayed_val) / (1.0 + delayed_val**10) - 0.1 * A[i])

# Subsample (every 10th point) and discard initial transient
A = A[10000:]
y = A[::10]

# Min-Max Normalization to [0, 1]
y = (y - np.min(y)) / (np.max(y) - np.min(y))
print(f"Dataset normalized. y_min = {np.min(y):.4f}, y_max = {np.max(y):.4f}")

# Input/Target partitioning
s_washout = y[:washout_len -1]
s_train = y[washout_len - 1 : washout_len + train_len - 1]
y_train = y[washout_len : washout_len + train_len]
y_target = y[washout_len + train_len : washout_len + train_len + test_len]

# --- 2. PARAMETERS, OBSERVABLES, GLOBAL INITIAL STATE, AND REGULARIZATION NOISE ---
N, J, tau, V = 10, 1, 10, 10
dims = 2**N
# We flatten(diagonal elements only) observables into (n_obs, dim) to use dot products instead of Tr(rho @ O)
z_ops = get_Pauli_Z(N)
obs_matrix = np.array([o.diagonal() for o in z_ops])

# Globally setting the regularization noise
sigma_noise = 1e-5
NOISE = rng_data.uniform(-sigma_noise, sigma_noise, [len(s_train), N*V])

# maximally coherent initial state
RHO_INIT = np.full((dims,dims), 1/dims, dtype=complex)


# --- 3. THE SIMULATION FUNCTION ---
def run_simulation(h_val, seed):
    # Create a local RNG for this task
    local_rng = np.random.default_rng(seed)

    # --- 3.1. MODEL SETUP ---
    J_ij = J_matrix(N,-J/2,J/2,local_rng)   
    Hamiltonian = FullyConnected_TFIM(N,J_ij,h_val)   #PUT YOUR MODEL HAMILTONIAN HERE!
    Hamiltonian = Hamiltonian.toarray()
    E, U = eigh(Hamiltonian)
    U_dag = U.conj().T
    dt = tau / V
    Phase_dt = np.exp(-1j * (E[:, np.newaxis] - E[np.newaxis, :]) * dt)
    Phase_tau = np.exp(-1j * (E[:, np.newaxis] - E[np.newaxis, :]) * tau)

    def evolve(rho_in, phase_mat):
        rho_energy = U_dag @ rho_in @ U
        return U @ (rho_energy * phase_mat) @ U_dag

    def input_map(rho_in, s, N):
        """Map input s to the first spin and trace out the rest using the module's reshape trick."""
        psi_s = np.array([np.sqrt(s), np.sqrt(1-s)], dtype=complex)
        rho_s = np.outer(psi_s, psi_s.conj())
        # Use optimized partial trace from Density_matrix
        rho_rest = trace_1(rho_in, N)
        return np.kron(rho_s, rho_rest)

    def extract_features(rho, s_in):
        X = np.zeros((len(s_in),len(obs_matrix)*V))

        for idx, s_val in enumerate(s_in):
            rho = input_map(rho, s_val, N)
            for v in range(V):
                rho = evolve(rho, Phase_dt)
                X[idx, v*N : (v+1)*N] = np.real(obs_matrix @ rho.diagonal())

        return X, rho

    def teacher_force(rho, s_in):
        for val in s_in:
            rho = evolve(input_map(rho, val, N), Phase_tau)
        return rho

    # --- 3.2 RUN ---
    rho = RHO_INIT.copy()

    # washout
    rho = teacher_force(rho, s_washout)

    # training
    X_train, rho = extract_features(rho, s_train)

    # normalize to [0,1] and add regularization noise U[-sigma_noise, sigma_noise]
    X_train = (X_train + 1) / 2
    X_train += local_rng.uniform(-sigma_noise, sigma_noise, X_train.shape)

    # Ridge model prevents ill-conditioned weights and feedback explosion
    model = Ridge(alpha=1e-4).fit(X_train, y_train)

    # testing
    y_pred = np.zeros(test_len)

    # Warm start first prediction step with the last training sample
    input_signal = y_train[-1]

    for i in range(test_len):
        rho = input_map(rho, input_signal, N)
        X_features = np.zeros(N * V)
        
        for v in range(V):
            rho = evolve(rho, Phase_dt)
            X_features[v*N : (v+1)*N] = np.real(obs_matrix @ rho.diagonal())
                
        feat = ((X_features + 1.0) / 2.0).reshape(1, -1)
        pred_val = model.predict(feat)[0]
        
        # Clip predictions to prevent numerical divergence in feedback loop
        pred_val = np.clip(pred_val, 0.0, 1.0)
        y_pred[i] = pred_val
        
        # Feedback loop: set current prediction as next step's input signal
        input_signal = pred_val

    # ---- 3.3 METRIC (nmse) ----
    nmse = np.sum((y_target - y_pred)**2) / np.sum(y_target**2)

    return nmse

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    start_time = time.time()

    #h_values = np.logspace(-2, 2, 60)*0.5
    h_values = [0.5e-1]
    n_realizations = 64 # usually 100
    seed_values = range(n_realizations)

    n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    print(f"Running in parallel with {n_cpus} CPUs")

    results_flat = Parallel(n_jobs=n_cpus)(
            delayed(run_simulation)(h, seed) 
            for h in h_values 
            for seed in seed_values
        )

    # Reshape into (len(h_values), n_realizations)
    results_matrix = np.array(results_flat).reshape(len(h_values), n_realizations)

    mean = np.mean(results_matrix, axis=1)
    std = np.std(results_matrix, axis=1)
    se = np.std(results_matrix, axis=1, ddof=1) / np.sqrt(n_realizations)

    # Save everything comprehensively
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
  
    output_file = os.path.join(output_dir, "scan_tm_mg_nmse_fully_connected_tfim.npz")
    np.savez_compressed(
        output_file,
        h_values = h_values,
        raw_data = results_matrix,
        mean = mean,
        std = std,
        se = se,
        # Metadata
        n_spins = N,
        J_val = J,
        tau_val = tau,
        temporal_multiplexing = V,
        n_realizations=n_realizations,
        model="fully connected transverse field ising model; H = sum_ij J_ij X_i X_j + h sum_i Z_i; J_ij in U(-J_val/2,J_val/2)."
    )
    print("Simulation complete. Final data saved.")
    
    # Get peak memory usage in kilobytes
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Convert to Megabytes or Gigabytes
    print(f"--- Resource Usage Report ---")
    print(f"Peak Memory Usage: {usage / 1024:.2f} MB")
    print(f"Grid Search Finished in {time.time() - start_time:.2f} seconds.")