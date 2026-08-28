# This code does the mg task as done in the 2017 paper by Fujii and Nakajima. It trains on the first 1000 steps (after a washout of 1000 steps added by me)
# and then autonomously predicts the last 2000 steps
# all of these 2000 steps are compared with those of the original series to calculate the nmse.
# Here we have added the use of ridge regression and output clipping to deal with the unreported anomalies.

# ---- 0. IMPORTS ----
import os
import time
import tracemalloc
import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import Ridge
from Models import get_Pauli_X, get_Pauli_Y, get_Pauli_Z, get_XX, get_YY ,get_ZZ, J_matrix, FullyConnected_TFIM
from Density_matrix import trace_1

tracemalloc.start()
start_time = time.perf_counter()
seed = 42
rng = np.random.default_rng(seed)

# ---- 1. DATA GENERATION ----
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

# ---- 2. PARAMETERS, OBSERVABLES, HAMILTONIAN, FUNCTIONS, INITIAL STATE ----
N, J, h_val, tau = 7, 1, 0.5e-1, 10
dims = 2**N
J_ij = J_matrix(N,-J/2,J/2,rng)
Hamiltonian = FullyConnected_TFIM(N,J_ij,h_val)  # PUT YOU HAMILTONIAN HERE!
Hamiltonian = Hamiltonian.toarray()

# maximally coherent initial state
RHO_INIT = np.full((dims,dims), 1/dims, dtype=complex)  

# diagonalizing the hamiltonian
E, U = eigh(Hamiltonian)
U_dag = U.conj().T
phase_mat = np.exp(-1j * (E[:, np.newaxis] - E[np.newaxis, :]) * tau)

# We flatten observables into (n_obs, dim**2) to use dot products instead of Tr(rho @ O)
x_ops = get_Pauli_X(N)
y_ops = get_Pauli_Y(N)
z_ops = get_Pauli_Z(N)
xx_ops = get_XX(N,x_ops)
yy_ops = get_YY(N,y_ops)
zz_ops = get_ZZ(N,z_ops)
raw_obs = x_ops + y_ops + z_ops + xx_ops + yy_ops + zz_ops
obs_matrix = np.array([o.conj().flatten() for o in raw_obs])

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

# ---- 3. RUN ----
sigma_noise = 1e-5
rho = RHO_INIT.copy()

# washout
rho = teacher_force(rho, s_washout)

# training
X_train, rho = extract_features(rho, s_train)

# normalize to [0,1] and add regularization noise U[-sigma_noise, sigma_noise]
X_train = (X_train + 1) / 2
X_train += rng.uniform(-sigma_noise, sigma_noise, X_train.shape)

# Ridge model prevents ill-conditioned weights and feedback explosion
model = Ridge(alpha=1e-4).fit(X_train, y_train)
print("Training Complete.")

# test
y_pred = np.zeros(test_len)
input_signal = y_pred[-1]

for i in range(test_len - 1):
    rho = evolve(input_map(rho, input_signal, N), phase_mat)
    x_features = (np.real(obs_matrix @ rho.flatten()) + 1) / 2
    pred_val = model.predict(x_features.reshape(1,-1))[0]

    # Clip predictions to prevent numerical divergence in feedback loop
    pred_val = np.clip(pred_val, 0, 1)
    y_pred[i] = pred_val

    # Feedback loop: set current prediction as next step's input signal
    input_signal = pred_val


# ---- 4. EVALUATE NMSE ----
nmse = np.mean((y_target - y_pred)**2) / np.mean(y_target**2)

print("\n--- RESULTS ---")
print(f"nmse: {nmse:.6e}")
print(f"Prediction Range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")

# Save everything comprehensively
output_dir = "data/mg"
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, "mg_nmse.npz")
np.savez_compressed(
    output_file,
    nmse = nmse,
    pred = y_pred,
    target = y_target,
    # metadata
    n_spins = N,
    j_val = J,
    h_val = h_val,
    tau_val = tau,
    model="fully connected transverse field ising model; H = sum_ij J_ij X_i X_j + h sum_i Z_i; J_ij in U(-J_val/2,J_val/2)."
)

end_time = time.perf_counter()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print("Simulation completed!")
print(f"Total time: {end_time-start_time:.4f}s")
print(f"Peak RAM: {peak/10**6:.2f} MB")
print(f"Current RAM: {current/10**6:.2f} MB")