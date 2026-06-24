# Linear memory task using Ising 1D NN system of 10 spins

import time
import tracemalloc
import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import LinearRegression
from Models import get_Pauli_X, get_Pauli_Y, get_Pauli_Z, get_ZZ, Heisenberg_1DNN
from Density_matrix import trace_1, mixed_density_matrix


tracemalloc.start()
start_time = time.perf_counter()
rng = np.random.default_rng(seed=42)

# --- 1. DATA GENERATION (NARMA) ---
n = 10
washout, train, test = 1000, 2000, 2000
total_steps = washout + train + test + n + 100
s_raw = rng.uniform(0.0, 0.2, total_steps)
y_raw = np.zeros(total_steps)

for i in range(n, total_steps):
    y_raw[i] = 0.1 + 1.5 * s_raw[i-n] * s_raw[i-1] + 0.05 * y_raw[i-1] * np.sum(y_raw[i-n:i]) + 0.3 * y_raw[i-1]

s = s_raw[100:] / 0.2 
y = y_raw[100:]

s_washout = s[:washout]
s_train = s[washout:washout+train]
s_test = s[washout+train:washout+train+test]
y_train = y[washout:washout+train]
y_test = y[washout+train:washout+train+test]

# --- 2. MODEL SETUP ---
N, J, h_val, tau = 10, 1, 0.5, 10
Hamiltonian, _ = Heisenberg_1DNN(N,h_val,J,rng)
#rho = mixed_density_matrix(10, 2, N, rng, complex_ensemble=True)
rho = (1/2**N)*np.ones([2**N,2**N]) # maximally coherent initial state


E, U = Hamiltonian.eigh()
U_dag = U.conj().T
phase_mat = np.exp(-1j * (E[:, np.newaxis] - E[np.newaxis, :]) * tau)

# --- 3. VECTORIZE OBSERVABLES ---
# We flatten observables into (n_obs, dim**2) to use dot products instead of Tr(rho @ O)
x_ops = get_Pauli_X(N)
y_ops = get_Pauli_Y(N)
z_ops = get_Pauli_Z(N)
zz_ops = get_ZZ(N,z_ops)
raw_obs = x_ops + y_ops + z_ops + zz_ops
obs_matrix = np.array([o.flatten() for o in raw_obs]) 

def get_features(rho_matrix):
    # Tr(A @ B) is the dot product of A.flatten() and B.T.flatten()
    # Since observables are often Hermitian, we just use the flattened obs_matrix
    return np.real(obs_matrix @ rho_matrix.flatten())

def evolve(rho_in, phase_mat):
            rho_energy = U_dag @ rho_in @ U
            return U @ (rho_energy * phase_mat) @ U_dag

def input_map(rho_in, s, N):
    """Map input s to the first spin and trace out the rest using the module's reshape trick."""
    psi_s = np.array([np.sqrt(1-s), np.sqrt(s)], dtype=complex)
    rho_s = np.outer(psi_s, psi_s.conj())
    # Use optimized partial trace from Density_matrix
    rho_rest = trace_1(rho_in, N)
    return np.kron(rho_s, rho_rest)

# --- 4. EXECUTION LOOPS ---

# Washout (No data storage)
for val in s_washout:
    rho = evolve(input_map(rho, val, N),phase_mat)

# Training (Vectorized feature extraction)
X_train = np.zeros((train, len(raw_obs)))
for k in range(train):
    rho = evolve(input_map(rho, s_train[k], N),phase_mat)
    X_train[k, :] = get_features(rho)

model = LinearRegression()
model.fit(X_train, y_train)

# Testing (Batch prediction)
X_test = np.zeros((test, len(raw_obs)))
for k in range(test):
    rho = evolve(input_map(rho, s_test[k], N),phase_mat)
    X_test[k, :] = get_features(rho)

y_pred = model.predict(X_test)

# --- 5. RESULTS ---
cov = np.cov(y_test, y_pred)
C = (cov[0, 1]**2) / (cov[0, 0] * cov[1, 1])

print(f"C={C:.2f}")
end_time = time.perf_counter()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"Total time: {end_time-start_time:.4f}s")
print(f"Peak RAM: {peak/10**6:.2f} MB")
print(f"Current RAM: {current/10**6:.2f} MB")
