# This code evaluates nmse for a range of h_values.

# ---- 0. IMPORTS ----
import resource
import os
import time
from joblib import Parallel, delayed
import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import Ridge
from Models import get_Pauli_X, get_Pauli_Y, get_Pauli_Z, get_XX, get_YY, get_ZZ, J_matrix, FullyConnected_TFIM 
from Density_matrix import trace_1

# ---- 1. DATASET GENERATION ----
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

# Globally setting the regularization noise
sigma_noise = 1e-5
NOISE = rng_data.uniform(-sigma_noise, sigma_noise, [len(s_train),len(obs_matrix)])

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
    phase_mat = np.exp(-1j * (E[:, np.newaxis] - E[np.newaxis, :]) * tau)

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
        X = np.zeros((len(s_in),len(obs_matrix)))

        for idx, s_val in enumerate(s_in):
            # input and evolve rho
            rho = evolve(input_map(rho, s_val, N), phase_mat)
            # measure
            X[idx,:] = np.real(obs_matrix @ rho.flatten())
            
        return X, rho

    def teacher_force(rho, s_in):
        for val in s_in:
            rho = evolve(input_map(rho, val, N), phase_mat)
        return rho

    # --- 3.2 RUN ---
    rho = RHO_INIT.copy()

    # washout
    rho = teacher_force(rho, s_washout)

    # training
    X_train, rho = extract_features(rho, s_train)

    # normalize to [0,1] and add regularization noise U[-sigma_noise, sigma_noise]
    X_train = (X_train + 1) / 2
    X_train += NOISE

    # Ridge model prevents ill-conditioned weights and feedback explosion
    model = Ridge(alpha=1e-4).fit(X_train, y_train)

    # test
    y_pred = np.zeros(test_len)
    input_signal = y_train[-1]

    for i in range(test_len):
        rho = evolve(input_map(rho, input_signal, N), phase_mat)
        x_features = (np.real(obs_matrix @ rho.flatten()) + 1) / 2
        pred_val = model.predict(x_features.reshape(1,-1))[0]

        # Clip predictions to prevent numerical divergence in feedback loop
        pred_val = np.clip(pred_val, 0, 1)
        y_pred[i] = pred_val

        # Feedback loop: set current prediction as next step's input signal
        input_signal = pred_val


    # --- 3.3 EVALUATE NMSE (nmse = sum_i (y_target_i - y_pred_i)^2 / sum_i y_target_i^2 ) ----
    nmse = np.mean((y_target - y_pred)**2) / np.mean(y_target**2)

    return nmse


# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    start_time = time.time()

    #h_values = np.logspace(-2, 2, 60)*0.5
    h_values = [0.5e-1]
    n_realizations = 32 # usually 100
    seed_values = range(n_realizations)

    # Read the CPU limit exported by your .sh script
    n_cpus = int(os.environ.get('JOBLIB_CPU_COUNT') or os.environ.get('PBS_NUM_PPN') or os.cpu_count() or 1)
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
  
    output_file = os.path.join(output_dir, "scan_mg_nmse_fully_connected_tfim.npz")
    np.savez_compressed(
        output_file,
        h_values=h_values,
        raw_data = results_matrix,
        mean = mean,
        std = std,
        se = se,
        # Metadata
        n_spins=N,
        J_val=J,
        tau_val=tau,
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