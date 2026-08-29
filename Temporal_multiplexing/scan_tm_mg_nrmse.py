# This code evaluates nrmse over a range of h_values for the mg task with temporal multilexing.

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

# ---- 1. GENERATE MG SERIES DATA ----
rng_data = np.random.default_rng(seed=42)
discard_len, washout_len, train_len, teacher_force_len, test_len, num_test_seq = 1000, 1000, 2000, 1000, 84, 20
sigma, tau_MG, total_mg_steps = 0.1, 17, (washout_len + train_len + num_test_seq * (teacher_force_len + test_len) + discard_len)*10
A = np.zeros(total_mg_steps)
A[0] = 1.2
delay_idx = int(tau_MG / sigma)

for i in range(total_mg_steps - 1):
    delayed_val = A[i - delay_idx] if i >= delay_idx else 1.2
    A[i + 1] = A[i] + sigma * ((0.2 * delayed_val) / (1.0 + delayed_val**10) - 0.1 * A[i])

# discard initial transient and subsample (every 10th point).
A = A[discard_len*10:]
s_raw = A[::10]

# variance
var = np.var(s_raw)

# normalizing to [0,1]
minimum, maximum = np.min(s_raw), np.max(s_raw)
s_raw = (s_raw - minimum) / (maximum - minimum)
print(f's_min = {np.min(s_raw):.4f}, s_max = {np.max(s_raw):.4f}')

# defining washout, train, test data
s_washout = s_raw[:washout_len-1]
s_train = s_raw[washout_len-1 : washout_len + train_len-1]
y_train = s_raw[washout_len : washout_len + train_len]
s_test = s_raw[washout_len + train_len:].reshape(num_test_seq, teacher_force_len+test_len)

# collecting the 84th step target values in the original coordinates
y_target_84 = (s_test[:,-1] * (maximum - minimum)) + minimum


# --- 2. PARAMETERS, READOUT OPERATORS, INITIAL STATE, AND NOISE ---
N, J, tau, V = 7, 1, 10, 10
dims = 2**N
# We flatten observables into (n_obs, dim**2) to use dot products instead of Tr(rho @ O)
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

    # --- 3.2 TRAINING ---
    rho = RHO_INIT.copy()

    # washout
    rho = teacher_force(rho, s_washout)

    # train
    X_train, rho = extract_features(rho, s_train)

    # normalise and add regularization noise
    X_train = (X_train + 1) / 2
    X_train += NOISE

    # Ridge model prevents ill-conditioned weights and feedback explosion
    model = Ridge(alpha=1e-4).fit(X_train, y_train)

    # ---- 3.3 TESTING ----
    diff = []
    for i in range(num_test_seq):
        rho = RHO_INIT.copy()
        rho = teacher_force(rho, s_test[i,:teacher_force_len-1])
        input_signal = s_test[i, teacher_force_len-1]

        pred_val = 0.0  # Initialize variable to ensure it's always bound

        for j in range(test_len):
            rho = input_map(rho, input_signal, N)
            X_features = np.zeros(N*V)
            for v in range(V):
                rho = evolve(rho, Phase_dt)
                X_features[v*N : (v+1)*N] = np.real(obs_matrix @ rho.diagonal())

            feat = ((X_features + 1.0) / 2.0).reshape(1, -1)
            pred_val = model.predict(feat)[0]

            # Clip predictions to prevent numerical divergence in feedback loop
            pred_val = np.clip(pred_val, 0, 1)
            input_signal = pred_val

        pred_val = (pred_val * (maximum - minimum)) + minimum   # converting back to the original coordinates
        diff.append(y_target_84[i]-pred_val)

    # ---- 5. NRMSE_84----
    diff = np.array(diff)
    nrmse = np.sqrt(np.mean(diff**2)/var)

    return nrmse

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    start_time = time.time()

    #h_values = np.logspace(-2, 2, 60)*0.5
    h_values = [0.5e-1]
    n_realizations = 64 # usually 100
    seed_values = range(n_realizations)

    n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK') or os.cpu_count() or 1)
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
  
    output_file = os.path.join(output_dir, "scan_tm_mg_nrmse_84_fully_connected_tfim.npz")
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
        n_realizations = n_realizations,
        model = "fully connected transverse field ising model; H = sum_ij J_ij X_i X_j + h sum_i Z_i; J_ij in U(-J_val/2,J_val/2)."
    )
    print("Simulation complete. Final data saved.")
    
    # Get peak memory usage in kilobytes
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Convert to Megabytes or Gigabytes
    print(f"--- Resource Usage Report ---")
    print(f"Peak Memory Usage: {usage / 1024:.2f} MB")
    print(f"Grid Search Finished in {time.time() - start_time:.2f} seconds.")