import os
import time
import numpy as np
import torch

from Models import J_matrix_alpha, FullyConnected_TFIM, get_Pauli_X, get_Pauli_Y, get_Pauli_Z, get_XX, get_YY, get_ZZ

# ---- 0. Set Device & Precision ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
torch.set_default_dtype(torch.float64)

# ---- 1. Global Data Generation ---
n = 10  # delay
washout, train, test = 1000, 2000, 2000
total_steps_raw = washout + train + test + n + 100
rng_data = np.random.default_rng(seed=42)
s_raw = rng_data.uniform(0.0, 0.2, total_steps_raw)
y_raw = np.zeros(total_steps_raw)

for i in range(n, total_steps_raw):
    y_raw[i] = 0.1 + 1.5 * s_raw[i-n] * s_raw[i-1] + 0.05 * y_raw[i-1] * np.sum(y_raw[i-n:i]) + 0.3 * y_raw[i-1]

s = s_raw[100:] / 0.2 
y_NARMA = y_raw[100:]
total_steps = len(s)

y_LinMem = np.zeros(total_steps)
for i in range(n, total_steps):
    y_LinMem[i] = s[i-n]

s_washout = torch.tensor(s[:washout], device=device, dtype=torch.float64)
s_train   = torch.tensor(s[washout:washout+train], device=device, dtype=torch.float64)
s_test    = torch.tensor(s[washout+train:washout+train+test], device=device, dtype=torch.float64)

y_train_NARMA  = torch.tensor(y_NARMA[washout:washout+train], device=device, dtype=torch.float64)
y_test_NARMA   = torch.tensor(y_NARMA[washout+train:washout+train+test], device=device, dtype=torch.float64)
y_train_LinMem = torch.tensor(y_LinMem[washout:washout+train], device=device, dtype=torch.float64)
y_test_LinMem  = torch.tensor(y_LinMem[washout+train:washout+train+test], device=device, dtype=torch.float64)

# --- 2. Parameters & Readout Operators ---
N, J, h_val, tau = 10, 1, 1e-1, 10
dims = 2**N
sub_dim = 2**(N - 1)

x_ops = get_Pauli_X(N)
y_ops = get_Pauli_Y(N)
z_ops = get_Pauli_Z(N)
xx_ops = get_XX(N, x_ops)
yy_ops = get_YY(N, y_ops)
zz_ops = get_ZZ(N, z_ops)

raw_obs = x_ops + y_ops + z_ops + xx_ops + yy_ops + zz_ops
obs_matrix_np = np.array([o.conj().flatten() for o in raw_obs], dtype=np.complex128)
obs_matrix_gpu = torch.tensor(obs_matrix_np, device=device, dtype=torch.complex128)

# --- Batched Helper Functions ---

def partial_trace_spin1_batched(rho_batch):
    """
    Batched partial trace over qubit 0 for shape (B, 1024, 1024).
    Reshapes (B, 2, 512, 2, 512) and sums over traced qubit indices.
    """
    B = rho_batch.shape[0]
    rho_tensor = rho_batch.view(B, 2, sub_dim, 2, sub_dim)
    return rho_tensor[:, 0, :, 0, :] + rho_tensor[:, 1, :, 1, :]

def fast_ridge_predict_batched(X_tr_batch, Y_tr, X_te_batch, alpha=1e-6):
    """
    Batched Linear/Ridge Regression fit and prediction.
    X_tr_batch: (B, T, M), Y_tr: (T,)
    Solves B independent linear systems in parallel on GPU.
    """
    B, T, M = X_tr_batch.shape
    
    # Compute X^T X for each realization in batch: (B, M, M)
    XtX = torch.bmm(X_tr_batch.transpose(1, 2), X_tr_batch)
    
    if alpha > 0:
        idx = torch.arange(M, device=device)
        XtX[:, idx, idx] += alpha
        
    # Compute X^T Y for each realization: (B, M, 1)
    Y_tr_expanded = Y_tr.unsqueeze(0).unsqueeze(2).expand(B, T, 1)
    XtY = torch.bmm(X_tr_batch.transpose(1, 2), Y_tr_expanded)
    
    # Batched linear solve: W has shape (B, M, 1)
    W = torch.linalg.solve(XtX, XtY)
    
    # Predict X_test @ W for each realization: (B, T_test)
    Y_pred = torch.bmm(X_te_batch, W).squeeze(2)
    return Y_pred

def R2_score_batched(Y_true, Y_pred_batch):
    """Computes Capacity Metric C for each batch trajectory in parallel."""
    B = Y_pred_batch.shape[0]
    Y_true_centered = Y_true - Y_true.mean()
    Y_pred_centered = Y_pred_batch - Y_pred_batch.mean(dim=1, keepdim=True)
    
    cov = torch.sum(Y_pred_centered * Y_true_centered.unsqueeze(0), dim=1)
    var_true = torch.sum(Y_true_centered**2)
    var_pred = torch.sum(Y_pred_centered**2, dim=1)
    
    capacity = (cov**2) / (var_true * var_pred + 1e-12)
    return capacity

# --- 3. BATCHED GPU SIMULATION FUNCTION ---
def run_simulation_batched_gpu(alpha_val, seeds):
    B = len(seeds)
    
    # 3.1. Build Hamiltonians & Diagonalize B realizations in parallel
    U_list = []
    phase_mat_list = []
    
    for seed in seeds:
        local_rng = np.random.default_rng(seed)
        J_ij = J_matrix_alpha(N, -J/2, J/2, alpha_val, local_rng, use_kac=False)   
        Hamiltonian = FullyConnected_TFIM(N, J_ij, h_val).toarray()
        
        E_cpu, U_cpu = np.linalg.eigh(Hamiltonian)
        
        U_gpu = torch.tensor(U_cpu, device=device, dtype=torch.complex128)
        E_gpu = torch.tensor(E_cpu, device=device, dtype=torch.float64)
        phase_gpu = torch.exp(-1j * (E_gpu.unsqueeze(1) - E_gpu.unsqueeze(0)) * tau)
        
        U_list.append(U_gpu)
        phase_mat_list.append(phase_gpu)
        
    # Stack into 3D Tensors: (B, 1024, 1024)
    U_batch = torch.stack(U_list)
    U_dag_batch = U_batch.mH  # Batched conjugate transpose
    phase_mat_batch = torch.stack(phase_mat_list)

    def step_forward_batched(rho_batch, s_val):
        # 1. Batched partial trace over qubit 0: Shape (B, 512, 512)
        rho_reduced = partial_trace_spin1_batched(rho_batch)
        
        c00 = s_val
        c11 = 1.0 - s_val
        c01 = torch.sqrt(s_val * c11)
        
        # 2. Block-assembly for input state: Shape (B, 1024, 1024)
        rho_in = torch.empty((B, dims, dims), device=device, dtype=torch.complex128)
        rho_in[:, :sub_dim, :sub_dim] = c00 * rho_reduced
        rho_in[:, :sub_dim, sub_dim:] = c01 * rho_reduced
        rho_in[:, sub_dim:, :sub_dim] = c01 * rho_reduced
        rho_in[:, sub_dim:, sub_dim:] = c11 * rho_reduced
        
        # 3. Batched Matrix Evolution: U @ ((U_dag @ rho_in @ U) * phase_mat) @ U_dag
        # Uses torch.bmm for 3D tensor multiplication
        tmp1 = torch.bmm(U_dag_batch, rho_in)
        tmp2 = torch.bmm(tmp1, U_batch)
        rho_tilde = tmp2 * phase_mat_batch
        
        tmp3 = torch.bmm(U_batch, rho_tilde)
        return torch.bmm(tmp3, U_dag_batch)

    # 3.2. Initial state batch: (B, 1024, 1024)
    rho_batch = torch.full((B, dims, dims), 1.0 / dims, device=device, dtype=torch.complex128)

    # Washout phase
    for idx in range(len(s_washout)):
        rho_batch = step_forward_batched(rho_batch, s_washout[idx])

    # Feature extraction loop fully batched: Shape (B, T, M)
    def extract_features_batched(s_sequence, rho_start):
        T_len = len(s_sequence)
        M_obs = obs_matrix_gpu.shape[0]
        X_batch = torch.empty((B, T_len, M_obs), device=device, dtype=torch.float64)
        curr_rho = rho_start
        
        for idx in range(T_len):
            curr_rho = step_forward_batched(curr_rho, s_sequence[idx])
            # Batched vector expectation computation
            # reshaped_rho: (B, dims^2) -> (B, dims^2) @ obs_matrix.T (M, dims^2).T -> (B, M)
            rho_flat = curr_rho.reshape(B, -1)
            X_batch[:, idx, :] = torch.real(rho_flat @ obs_matrix_gpu.T)
            
        return X_batch, curr_rho

    # Train phase
    X_train_batch, rho_batch = extract_features_batched(s_train, rho_batch)

    # Test phase
    X_test_batch, _ = extract_features_batched(s_test, rho_batch)

    # Batched Ridge Regression solves all B realizations concurrently
    y_pred_LinMem_batch = fast_ridge_predict_batched(X_train_batch, y_train_LinMem, X_test_batch)
    y_pred_NARMA_batch  = fast_ridge_predict_batched(X_train_batch, y_train_NARMA, X_test_batch)

    # Evaluate Capacities for all B realizations in parallel
    c_LinMem_batch = R2_score_batched(y_test_LinMem, y_pred_LinMem_batch)
    c_NARMA_batch  = R2_score_batched(y_test_NARMA, y_pred_NARMA_batch)

    return c_LinMem_batch.cpu().numpy(), c_NARMA_batch.cpu().numpy()


# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    start_time = time.time()

    alpha_values = np.linspace(0, 8, 41)
    n_realizations = 100
    seeds = list(range(n_realizations))

    results_matrix = np.zeros((len(alpha_values), n_realizations, 2))

    print(f"Starting fully batched GPU execution for {len(alpha_values)} alpha parameters across {n_realizations} parallel seeds...")

    for a_idx, alpha in enumerate(alpha_values):
        # All 100 realizations for this alpha execute in parallel on GPU inside one batch call
        c_lin_batch, c_narma_batch = run_simulation_batched_gpu(alpha, seeds)
        
        results_matrix[a_idx, :, 0] = c_lin_batch
        results_matrix[a_idx, :, 1] = c_narma_batch

    matrix_LinMem = results_matrix[:, :, 0]
    matrix_NARMA  = results_matrix[:, :, 1]

    # Compute statistics
    c_mean_LinMem = np.mean(matrix_LinMem, axis=1)
    c_std_LinMem  = np.std(matrix_LinMem, axis=1)
    c_se_LinMem   = np.std(matrix_LinMem, axis=1, ddof=1) / np.sqrt(n_realizations)

    c_mean_NARMA  = np.mean(matrix_NARMA, axis=1)
    c_std_NARMA   = np.std(matrix_NARMA, axis=1)
    c_se_NARMA    = np.std(matrix_NARMA, axis=1, ddof=1) / np.sqrt(n_realizations)

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "power_law_fully_connected_tfim_c_vs_alpha_gpu.npz")

    np.savez_compressed(
        output_file,
        n_realizations=n_realizations,
        alpha_values=alpha_values,
        c_raw_LinMem=matrix_LinMem,
        c_raw_NARMA=matrix_NARMA,
        c_mean_LinMem=c_mean_LinMem,
        c_std_LinMem=c_std_LinMem,
        c_se_LinMem=c_se_LinMem,
        c_mean_NARMA=c_mean_NARMA,
        c_std_NARMA=c_std_NARMA,
        c_se_NARMA=c_se_NARMA,
        n_spins=N,
        J_val=J,
        h_val=h_val,
        tau_val=tau,
        model="long range transverse field ising model; H = sum_ij J_ij X_i X_j + h sum_i Z_i; J_ij = j_ij_0 / (N * min(|i-j| , N-|i-j|)^alpha); J_ij_0 in U(-J_val/2,J_val/2)."
    )
    
    print(f"Simulation complete. Final data saved to {output_file}.")
    print(f"Grid Search Finished in {time.time() - start_time:.2f} seconds.")