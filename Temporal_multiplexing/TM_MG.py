import time
import tracemalloc
import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import Ridge

# Import custom modules if available, otherwise fallback provided
from Models import get_Pauli_X, get_Pauli_Z, J_matrix, FullyConnected_TFIM

tracemalloc.start()
start_time = time.perf_counter()

# --- 1. Parameters ---
N = 7            # Number of spins (2^7 = 128 dimensional Hilbert space)
J_val = 1.0      # Coupling strength
h_val = 0.05     # Transverse field
tau = 10.0       # Total evolution time per input step
V = 10           # Internal time steps (sub-sampling)
sigma_noise = 1e-5
seed = 43
rng = np.random.default_rng(seed)

# --- 2. Mackey-Glass Data Generation ---
sigma, tau_MG, total_mg_steps = 0.1, 17, 140000
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

washout_len, train_len, test_len = 1000, 10000, 2000

# Input/Target partitioning
s_washout = y[:washout_len]
s_train = y[washout_len : washout_len + train_len]
y_train = y[washout_len + 1 : washout_len + train_len + 1]
y_test_target = y[washout_len + train_len : washout_len + train_len + test_len]

# --- 3. Hamiltonian & Operator Setup ---
x_ops = get_Pauli_X(N)
z_ops = get_Pauli_Z(N)

J_ij = J_matrix(N, -J_val / 2.0, J_val / 2.0, rng)
Hamiltonian, _ = FullyConnected_TFIM(N, J_ij, h_val)
Hamiltonian = Hamiltonian.toarray()

# --- 4. Exact Diagonalization & Propagation Setup ---
E, U = eigh(Hamiltonian)
U_dag = U.conj().T

dt = tau / V
Phase_dt = np.exp(-1j * (E[:, np.newaxis] - E[np.newaxis, :]) * dt)
Phase_tau = np.exp(-1j * (E[:, np.newaxis] - E[np.newaxis, :]) * tau)

# Pre-calculate diagonal operators in eigenbasis for fast expectation calculation
z_diags = [op.diagonal() for op in z_ops]

def evolve(rho_in, phase_mat):
    """Evolves density matrix in unit time step via precomputed phase matrix."""
    rho_energy = U_dag @ rho_in @ U
    return U @ (rho_energy * phase_mat) @ U_dag

def partial_trace_first_spin(rho, N):
    """Traces out the 1st spin from an N-spin density matrix (returns 2^(N-1) x 2^(N-1))."""
    dim_rest = 2**(N - 1)
    reshaped = rho.reshape(2, dim_rest, 2, dim_rest)
    return reshaped[0, :, 0, :] + reshaped[1, :, 1, :]

def input_map(rho_in, s, N):
    """Injects single scalar input s into spin 0 while preserving rest of the reservoir state."""
    # State mapping for single qubit: |psi> = sqrt(s)|0> + sqrt(1-s)|1>
    psi_s = np.array([np.sqrt(s), np.sqrt(1.0 - s)], dtype=complex)
    rho_s = np.outer(psi_s, psi_s.conj())
    rho_rest = partial_trace_first_spin(rho_in, N)
    return np.kron(rho_s, rho_rest)

# --- 5. Washout Phase ---
print("Starting Washout Phase...")
rho = np.full((2**N, 2**N), 1.0 / (2**N), dtype=complex)  # Maximally mixed state initialization

for k in range(washout_len):
    rho = evolve(input_map(rho, s_washout[k], N), Phase_tau)

# --- 6. Training Phase ---
print("Starting Training Phase...")
X_features = np.zeros((train_len, N * V))

for k in range(train_len):
    rho = input_map(rho, s_train[k], N)
    for v in range(V):
        rho = evolve(rho, Phase_dt)
        rho_diag = rho.diagonal()
        for n in range(N):
            X_features[k, v * N + n] = np.real(np.sum(rho_diag * z_diags[n]))

# Normalization & Regularization noise
X_features = (X_features + 1.0) / 2.0
X_features += rng.uniform(-sigma_noise, sigma_noise, X_features.shape)

# Ridge model prevents ill-conditioned weights and feedback explosion
model = Ridge(alpha=1e-4).fit(X_features, y_train)
print("Training Complete.")

# --- 7. Evaluation Phase (Autonomous Feedback / One-Step Ahead) ---
print("Starting Evaluation Phase...")
y_pred = np.zeros(test_len)

# Warm start first prediction step with the last training sample
input_signal = s_train[-1]

for i in range(test_len):
    rho = input_map(rho, input_signal, N)
    X_eval = np.zeros(N * V)
    
    for v in range(V):
        rho = evolve(rho, Phase_dt)
        rd = rho.diagonal()
        for n in range(N):
            X_eval[v * N + n] = np.real(np.sum(rd * z_diags[n]))
            
    feat = ((X_eval + 1.0) / 2.0).reshape(1, -1)
    pred_val = model.predict(feat)[0]
    
    # Clip predictions to prevent numerical divergence in feedback loop
    pred_val = np.clip(pred_val, 0.0, 1.0)
    y_pred[i] = pred_val
    
    # Feedback loop: set current prediction as next step's input signal
    input_signal = pred_val

# --- 8. Metrics & Output ---
# Correctly dimensioned NMSE evaluation
NMSE_val = np.sum((y_pred - y_test_target)**2) / np.sum(y_test_target**2)

print("\n--- RESULTS ---")
print(f"NMSE: {NMSE_val:.6e}")
print(f"Prediction Range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")

#np.savez_compressed(
#   'mg_results.npz',
#    pred=y_pred,
#    target=y_test_target,
#    time_steps=np.arange(test_len)
#)

end_time = time.perf_counter()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"Total Execution Time: {end_time - start_time:.2f} s")
print(f"Peak Memory Usage: {peak / 1e6:.2f} MB")