# Linear Memory Task for delay=10 with RMP.
# Optimized
# Calculating C.

import time
import tracemalloc
import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import LinearRegression
from Models import get_Pauli_X, get_Pauli_Y, get_Pauli_Z, get_ZZ, Ising 
from Density_matrix import trace_1, mixed_density_matrix

tracemalloc.start()
start_time = time.perf_counter()

rng = np.random.default_rng(seed=42)

# --- 1. DATA GENERATION (Linear Memory) ---
n = 10  # delay
washout, train, test = 1000, 2000, 2000
total_steps = washout + train + test + n
s = rng.uniform(0, 1, total_steps)
y = np.zeros(total_steps)

for i in range(n, total_steps):
    y[i] = s[i-n]

s_washout = s[:washout]
s_train = s[washout:washout+train]
s_test = s[washout+train:washout+train+test]
y_train = y[washout:washout+train]
y_test = y[washout+train:washout+train+test]

# --- 2. MODEL SETUP ---
N, J, h_val, tau = 10, 1, 0.1, 10
x_ops = get_Pauli_X(N)
y_ops = get_Pauli_Y(N)
z_ops = get_Pauli_Z(N)
zz_ops = get_ZZ(N,z_ops)
Hamiltonian, _ = Ising(N, J, h_val, rng, x_ops=x_ops, z_ops=z_ops)
rho = mixed_density_matrix(10, 2, N, rng, complex_ensemble=True)

E, U = eigh(Hamiltonian)
U_dag = U.conj().T
phase_factors = np.exp(-1j * (E[:, np.newaxis] - E[np.newaxis, :]) * tau)

# --- 3. VECTORIZE OBSERVABLES ---
# We flatten observables into (n_obs, dim**2) to use dot products instead of Tr(rho @ O)
raw_obs = x_ops + y_ops + z_ops + zz_ops
obs_matrix = np.array([o.flatten() for o in raw_obs]) 

def get_features(rho_matrix):
    # Tr(A @ B) is the dot product of A.flatten() and B.T.flatten()
    # Since observables are often Hermitian, we just use the flattened obs_matrix
    return np.real(obs_matrix @ rho_matrix.flatten())

def time_evolve_fast(rho_0):
    rho_energy_t = (U_dag @ rho_0 @ U) * phase_factors
    return U @ rho_energy_t @ U_dag

def inpt_fast(rho_in, s_val, N_spins):
    # Pre-calculated basis states for speed
    psi_s = np.array([np.sqrt(s_val), np.sqrt(1-s_val)]) # Simplified basis logic
    rho_s = np.outer(psi_s, psi_s)
    return np.kron(rho_s, trace_1(rho_in, N_spins))

# --- 4. EXECUTION LOOPS ---

# Washout (No data storage)
for val in s_washout:
    rho = time_evolve_fast(inpt_fast(rho, val, N))

# Training (Vectorized feature extraction)
X_train = np.zeros((train, len(raw_obs)))
for k in range(train):
    rho = time_evolve_fast(inpt_fast(rho, s_train[k], N))
    X_train[k, :] = get_features(rho)

model = LinearRegression()
model.fit(X_train, y_train)

# Testing (Batch prediction)
X_test = np.zeros((test, len(raw_obs)))
for k in range(test):
    rho = time_evolve_fast(inpt_fast(rho, s_test[k], N))
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

