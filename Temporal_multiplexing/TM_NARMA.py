import os
import time
import tracemalloc
import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import Ridge

# External model imports
from Models import get_Pauli_X, get_Pauli_Z, J_matrix, FullyConnected_TFIM
from Density_matrix import trace_1

tracemalloc.start()
start_time = time.perf_counter()

# ---- 1. DATA GENERATION (NARMA TASKS) ----

n_values = [2, 5, 10, 15, 20]
washout, train, test = 1000, 3000, 1000
num_samples = washout + train + test  # Fixed total length (5000 samples)

# Max n determines padding needed for perfect alignment across all outputs
max_n = max(n_values)
total_steps = max_n + 100 + num_samples

# Shared raw input sequence
rng_data = np.random.default_rng(seed=42)
s_raw = rng_data.uniform(0.0, 0.2, total_steps)

# Output matrix: 5000 rows (samples) x 5 columns (for n = 2, 5, 10, 15, 20)
Y = np.zeros((num_samples, len(n_values)))

for idx, n in enumerate(n_values):
    y_raw = np.zeros(total_steps)
    for i in range(n, total_steps):
        if n == 2:
            # NARMA-2 exact formula
            y_raw[i] = (
                0.4 * y_raw[i - 1]
                + 0.4 * y_raw[i - 1] * y_raw[i - 2]
                + 0.6 * (s_raw[i - 1] ** 3)
                + 0.1
            )
        else:
            # NARMA-n exact formula
            y_raw[i] = (
                0.1
                + 1.5 * s_raw[i - n] * s_raw[i - 1]
                + 0.05 * y_raw[i - 1] * np.sum(y_raw[i - n : i])
                + 0.3 * y_raw[i - 1]
            )

    # Slice aligned tail portion
    Y[:, idx] = y_raw[-num_samples:]

# Input matrix rescaled to [0, 1]
S = (s_raw[-num_samples:] / 0.2).flatten()

# Dataset Partitioning
S_washout = S[0:washout]
S_train, Y_train = S[washout : washout + train], Y[washout : washout + train]
S_test, Y_test = S[washout + train : num_samples], Y[washout + train : num_samples]


# ---- 2. QUANTUM RESERVOIR SYSTEM SETUP ----

seed = 42
rng = np.random.default_rng(seed)
N = 5            # Number of spins
J_val = 1.0      # Coupling strength
h_val = 0.05e-1     # Transverse field
tau = 1.0        # Total time per step
V = 10           # Internal time steps (sub-sampling)

# Construct Hamiltonian
x_ops = get_Pauli_X(N)
z_ops = get_Pauli_Z(N)

J_ij = J_matrix(N, -J_val / 2, J_val / 2, rng)
Hamiltonian, _ = FullyConnected_TFIM(N, J_ij, h_val)
Hamiltonian = Hamiltonian.toarray()

# Diagonalization for Exact Time Evolution
E, U = eigh(Hamiltonian)
U_dag = U.conj().T

# Pre-calculate propagator phase matrices
dt = tau / V
Phase_dt = np.exp(-1j * (E[:, np.newaxis] - E[np.newaxis, :]) * dt)
Phase_tau = np.exp(-1j * (E[:, np.newaxis] - E[np.newaxis, :]) * tau)

# Pre-stack Pauli Z diagonals for vectorized fast expectation calculation: shape (N, 2^N)
Z_matrix = np.array([np.asarray(op.diagonal()).ravel() for op in z_ops])

def evolve(rho_in, phase_mat):
    """Evolves state via precomputed phase matrix."""
    rho_energy = U_dag @ rho_in @ U
    return U @ (rho_energy * phase_mat) @ U_dag

def input_map(rho_in, s, N):
    """Inject scalar s into spin 0 while preserving rest of reservoir state."""
    psi_s = np.array([np.sqrt(s), np.sqrt(1.0 - s)], dtype=complex)
    rho_s = np.outer(psi_s, psi_s.conj())
    rho_rest = trace_1(rho_in, N)
    return np.kron(rho_s, rho_rest)

def extract_features(s_sequence, rho_start):
    """Encodes input stream into quantum feature matrix using vectorized measurement."""
    n_steps = len(s_sequence)
    features = np.zeros((n_steps, N * V))
    rho_curr = rho_start

    for k in range(n_steps):
        rho_curr = input_map(rho_curr, s_sequence[k], N)
        for v in range(V):
            rho_curr = evolve(rho_curr, Phase_dt)
            # Vectorized expectation across all spins at once: Z_matrix @ diag(rho)
            features[k, v * N : (v + 1) * N] = np.real(Z_matrix @ rho_curr.diagonal())

    return features, rho_curr


# ---- 3. WASHOUT, TRAINING, AND TESTING ----

# Maximally coherent density matrix initialization
rho = np.full((2**N, 2**N), 1.0 / (2**N), dtype=complex)

print("Starting Washout Phase...")
for k in range(washout):
    rho = evolve(input_map(rho, S_washout[k], N), Phase_tau)

print("Starting Training Feature Extraction...")
X_features_train, rho = extract_features(S_train, rho)
X_features_train = (X_features_train + 1.0) / 2.0  # Rescale features to [0, 1]

# Ridge Regression model for robust multi-output training
model = Ridge(alpha=1e-4).fit(X_features_train, Y_train)

print("Starting Testing Feature Extraction...")
X_features_test, _ = extract_features(S_test, rho)
X_features_test = (X_features_test + 1.0) / 2.0

Y_pred = model.predict(X_features_test)


# ---- 4. PERFORMANCE EVALUATION ----

# Corrected NMSE (squared denominator): evaluated individually per column/delay order
NMSE = np.sum((Y_pred - Y_test) ** 2, axis=0) / np.sum(Y_test ** 2, axis=0)

print("\n--- PERFORMANCE RESULT ---")
for idx, n in enumerate(n_values):
    print(f"NARMA-{n:02d} | NMSE: {NMSE[idx]:.6e}")


# ---- 5. SAVING RESULTS ----

output_dir = 'Data/Reproduction_1'
os.makedirs(output_dir, exist_ok=True)

np.savez_compressed(
    os.path.join(output_dir, 'NARMA.npz'),
    NMSE=NMSE,
    Y_pred=Y_pred,
    Y_target=Y_test,
    n_values=n_values,
    model='fully connected TFIM',
    n_spins=N,
    J_val=J_val,
    h_val=h_val,
    J_ij=J_ij,
    V=V,
    tau=tau
)

end_time = time.perf_counter()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"\nExecution Time: {end_time - start_time:.2f} s")
print(f"Peak RAM Usage: {peak / 1e6:.2f} MB")