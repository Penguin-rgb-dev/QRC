# This code produces data to plot mse vs h plot for the mg task using direct k_step prediction.

import resource
import os
import time
from joblib import Parallel, delayed
import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import Ridge
from Models import get_Pauli_X, get_Pauli_Y, get_Pauli_Z, get_XX, get_YY, get_ZZ, J_matrix, FullyConnected_TFIM 
from Density_matrix import trace_1

# ---- 1. DATA GENERATION ----
# l is the length of input sequences and k is the prediction step size.
discard_len, l, k, train_len, test_len = 1000, 16, 1, 800, 200
sigma, tau_MG, total_mg_steps = 0.1, 17, (discard_len + train_len + test_len + l + k -1)*10
A = np.zeros(total_mg_steps)
A[0] = 1.2
delay_idx = int(tau_MG / sigma)

for i in range(total_mg_steps - 1):
    delayed_val = A[i - delay_idx] if i >= delay_idx else 1.2
    A[i + 1] = A[i] + sigma * ((0.2 * delayed_val) / (1.0 + delayed_val**10) - 0.1 * A[i])

# discard initial transient and subsample (every 10th point).
A = A[discard_len*10:]
s_raw = A[::10]

# normalizing to [0,1]
min, max = np.min(s_raw), np.max(s_raw)
s_raw = (s_raw - min) / (max - min)
print(f's_min = {np.min(s_raw):.4f}, s_max = {np.max(s_raw):.4f}')

# generating S and Y
S = np.zeros([train_len+test_len, l])
y = np.zeros(train_len+test_len)
for i in range(train_len+test_len):
    S[i,:] = s_raw[i:i+l]
    y[i] = s_raw[i+l+k-1]

#train/test split
S_train, y_train = S[:train_len,:], y[:train_len]
S_test, y_test = S[train_len:], y[train_len:]


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

    def extract_features(S_in):
        num_rows = S_in.shape[0]
        X_features = np.zeros([num_rows,len(obs_matrix)])
        for idx in range(num_rows):
            rho = RHO_INIT.copy()
            for val in S_in[idx,:]:
                rho = evolve(input_map(rho, val, N), phase_mat)
            X_features[idx,:] = np.real(obs_matrix @ rho.flatten())
        return X_features

    # ---- 3.2 training ----
    X_train = extract_features(S_train)

    # Ridge model prevents ill-conditioned weights and feedback explosion
    model = Ridge(alpha=1e-4).fit(X_train, y_train)

    # ---- 3.3 testing ----
    X_test = extract_features(S_test)
    y_pred = model.predict(X_test)


    # ---- 3.4 evaluation metric ----
    # mean squared error; mse = mean( sum_i [ y_test_i - y_pred_i ]^2 )
    mse = np.mean((y_test - y_pred)**2)

    # mean absolute error; mae = mean( sum_i | y_test_i - y_pred_i | )
    mae = np.mean(abs(y_test - y_pred))

    # pearson correlation coefficient; pcc = sum_i (y_test_i - mean(y_test_i))(y_pred_i - mean(y_pred_i)) / [sqrt(sum_i (y_test_i - mean(y_test_i))^2 * sqrt(sum_i (y_pred_i - mean(y_pred_i))^2)]
    cov = np.cov(y_test, y_pred)
    pcc = np.sqrt(cov[0,1]**2 / (cov[0,0] * cov[1,1]))

    return mse, mae, pcc


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

    # Reshape into (len(h_values), n_realizations, 3) as there as 3 outputs for each parameter set
    results_matrix = np.array(results_flat).reshape(len(h_values), n_realizations, 3)

    # Separate the metrics along the last axis
    matrix_mse = results_matrix[:,:,0]
    matrix_mae = results_matrix[:,:,1]
    matrix_pcc = results_matrix[:,:,2]

    # Compute statistics for mse
    mse_mean = np.mean(matrix_mse,axis=1)
    mse_std = np.std(matrix_mse, axis=1)
    mse_se = np.std(matrix_mse, axis=1, ddof=1) / np.sqrt(n_realizations)

    # Compute statistics for mae
    mae_mean = np.mean(matrix_mae,axis=1)
    mae_std = np.std(matrix_mae, axis=1)
    mae_se = np.std(matrix_mae, axis=1, ddof=1) / np.sqrt(n_realizations)

    # Compute statistics for pcc
    pcc_mean = np.mean(matrix_pcc,axis=1)
    pcc_std = np.std(matrix_pcc, axis=1)
    pcc_se = np.std(matrix_pcc, axis=1, ddof=1) / np.sqrt(n_realizations)

    # Save everything comprehensively
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "scan_mg_k_step_pred_fully_connected_tfim.npz")
    np.savez_compressed(
        output_file,
        h_values=h_values,
        # Raw matrix outputs
        raw_data_mse = matrix_mse,
        raw_data_mae = matrix_mae,
        raw_data_pcc = matrix_pcc,
        # mse
        mse_mean = mse_mean,
        mse_std = mse_std,
        mse_se = mse_se,
        # mae
        mae_mean = mae_mean,
        mae_std = mae_std,
        mae_se = mae_se,
        # pcc
        pcc_mean = pcc_mean,
        pcc_std = pcc_std,
        pcc_se = pcc_se,        
        # Metadata
        k = k,
        l = l,
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