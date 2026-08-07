# This code evaluates performace on linear memory and NARMA10 tasks for the Hamiltonian of your choice.

import time
import tracemalloc
import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import LinearRegression
from Models import get_Pauli_X, get_Pauli_Y, get_Pauli_Z, get_XX, get_YY ,get_ZZ, J_matrix, FullyConnected_TFIM
from Density_matrix import trace_1, mixed_density_matrix


tracemalloc.start()
start_time = time.perf_counter()
rng = np.random.default_rng(seed=42)

# ---- 1. Global data generation (linear memory task and NARMA) ---
n = 10  # delay
washout, train, test = 1000, 2000, 2000
total_steps = washout + train + test + n + 100
rng_data = np.random.default_rng(seed=42)
s_raw = rng_data.uniform(0.0, 0.2, total_steps)
y_raw = np.zeros(total_steps)

for i in range(n, total_steps):
    y_raw[i] = 0.1 + 1.5 * s_raw[i-n] * s_raw[i-1] + 0.05 * y_raw[i-1] * np.sum(y_raw[i-n:i]) + 0.3 * y_raw[i-1]

s = s_raw[100:] / 0.2   #rescaling the input to [0,1]
y_NARMA = y_raw[100:]
total_steps = total_steps - 100
y_LinMem = np.zeros(total_steps)
for i in range(n, total_steps):
    y_LinMem[i] = s[i-n]

s_washout = s[:washout]
s_train = s[washout:washout+train]
s_test = s[washout+train:washout+train+test]
y_train_NARMA = y_NARMA[washout:washout+train]
y_test_NARMA = y_NARMA[washout+train:washout+train+test]
y_trian_LinMem = y_LinMem[washout:washout+train]
y_test_LinMem = y_LinMem[washout+train:washout+train+test]

# --- 2. MODEL SETUP ---
N, J, h_val, tau = 10, 1, 1e2, 10
J_ij = J_matrix(N,-J/2,J/2,rng)
Hamiltonian, _ = FullyConnected_TFIM(N,J_ij,h_val)  # PUT YOU HAMILTONIAN HERE!
Hamiltonian = Hamiltonian.toarray()
#rho = mixed_density_matrix(10, 2, N, rng, complex_ensemble=True)
rho = (1/2**N)*np.ones([2**N,2**N]) # maximally coherent initial state


E, U = eigh(Hamiltonian)
U_dag = U.conj().T
phase_mat = np.exp(-1j * (E[:, np.newaxis] - E[np.newaxis, :]) * tau)

# --- 3. VECTORIZE OBSERVABLES ---
# We flatten observables into (n_obs, dim**2) to use dot products instead of Tr(rho @ O)
x_ops = get_Pauli_X(N)
y_ops = get_Pauli_Y(N)
z_ops = get_Pauli_Z(N)
xx_ops = get_XX(N,x_ops)
yy_ops = get_YY(N,y_ops)
zz_ops = get_ZZ(N,z_ops)
raw_obs = x_ops + y_ops + z_ops + xx_ops + yy_ops + zz_ops
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
    psi_s = np.array([np.sqrt(s), np.sqrt(1-s)], dtype=complex)
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

model_LinMem = LinearRegression()
model_LinMem.fit(X_train, y_trian_LinMem)
model_NARMA = LinearRegression()
model_NARMA.fit(X_train, y_train_NARMA)

# Testing (Batch prediction)
X_test = np.zeros((test, len(raw_obs)))
for k in range(test):
    rho = evolve(input_map(rho, s_test[k], N),phase_mat)
    X_test[k, :] = get_features(rho)

y_pred_LinMem = model_LinMem.predict(X_test)
y_pred_NARMA = model_NARMA.predict(X_test)

# --- 5. RESULTS ---
cov_LinMem = np.cov(y_test_LinMem, y_pred_LinMem)
cov_NARMA = np.cov(y_test_NARMA, y_pred_NARMA)
C_LinMem, C_NARMA = (cov_LinMem[0, 1]**2) / (cov_LinMem[0, 0] * cov_LinMem[1, 1]), (cov_NARMA[0, 1]**2) / (cov_NARMA[0, 0] * cov_NARMA[1, 1])

#np.savez_compressed('Data/Prediction/Fully_connected_TFIM_LinMem_NARMA_IV.npz',
#                    pred_linmem = y_pred_LinMem,
#                    target_linmem = y_test_LinMem,
#                    c_linmem = C_LinMem,
#                    pred_narma = y_pred_NARMA,
#                    target_narma = y_test_NARMA,
#                    c_narma = C_NARMA,
#                    model = 'Fully connected TFIM with uniform random coupling',
#                    n_spins = N,
#                    J = J,
#                    J_ij = J_ij,
#                    h = h_val,
#                    tau = tau,
#                    delay = n)

print(f"C_LinMem={C_LinMem:.2f}, C_NARMA={C_NARMA:.2f}")
end_time = time.perf_counter()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"Total time: {end_time-start_time:.4f}s")
print(f"Peak RAM: {peak/10**6:.2f} MB")
print(f"Current RAM: {current/10**6:.2f} MB")
