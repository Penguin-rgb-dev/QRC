# This code evaluates performace on the NARMA 10 task for the model Hamiltonian of your choice.

import time
import tracemalloc
import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import LinearRegression
from Models import get_spin_operators, get_spin_xx, get_spin_yy, get_spin_zz, heisenberg_spin_3_half
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
N, J_val, h_val, tau = 5, 1, 1, 10
dims = 4**N
J_i = np.full(N,-1)
h_i = rng.uniform(-h_val,h_val,size=N)
Hamiltonian = heisenberg_spin_3_half(N, J_i, h_i, periodic=True)  # PUT YOUR HAMILTONIAN HERE!
Hamiltonian = Hamiltonian.toarray()
rho = np.full((dims,dims), 1/dims, dtype=complex) # maximally coherent initial state
E, U = eigh(Hamiltonian)
U_dag = U.conj().T
phase_mat = np.exp(-1j * (E[:, np.newaxis] - E[np.newaxis, :]) * tau)

# --- 3. VECTORIZE OBSERVABLES ---
# We flatten observables into (n_obs, dim**2) to use dot products instead of Tr(rho @ O)
x_ops = get_spin_operators(N,op_type="x",d=4,sparse=False)
y_ops = get_spin_operators(N,op_type="y",d=4,sparse=False)
z_ops = get_spin_operators(N,op_type="z",d=4,sparse=False)
xx_ops = get_spin_xx(N, d=4, sparse=False)
yy_ops = get_spin_yy(N, d=4, sparse=False)
zz_ops = get_spin_zz(N, d=4, sparse=False)
raw_obs = x_ops + y_ops + z_ops + xx_ops + yy_ops + zz_ops
obs_matrix = np.array([o.conj().flatten() for o in raw_obs])


def evolve(rho_in, phase_mat):
            rho_energy = U_dag @ rho_in @ U
            return U @ (rho_energy * phase_mat) @ U_dag

def trace_leftmost_site(rho, d=4):
    """Traces out the leftmost spin (Site N-1, MSB) from a chain with local dimension d."""
    # Convert sparse matrix to dense array if necessary
    if hasattr(rho, "toarray"):
        rho = rho.toarray()

    D = rho.shape[0]
    dim_rest = D // d

    # Reshape: (site_N-1_row, rest_row, site_N-1_col, rest_col)
    reshaped_rho = rho.reshape(d, dim_rest, d, dim_rest)

    # Trace out axes 0 and 2 (leftmost site N-1)
    return np.trace(reshaped_rho, axis1=0, axis2=2)


def inject_leftmost_state(s, rho, d=4):
    """Replaces the state at the leftmost site (Site N-1) with state psi(s)."""
    if not (0 <= s <= 1):
        raise ValueError(f"Parameter s must be in range [0, 1], got {s}")

    if 0 <= s < 1 / 3:
        psi = np.array([0, 0, np.sqrt(3 * s), np.sqrt(1 - 3 * s)])
    elif 1 / 3 <= s <= 2 / 3:
        psi = np.array([0, np.sqrt(3 * s - 1), np.sqrt(2 - 3 * s), 0])
    else:
        psi = np.array([np.sqrt(3 * s - 2), np.sqrt(3 - 3 * s), 0, 0])

    psi_dm = np.outer(psi, psi)
    rho_reduced = trace_leftmost_site(rho, d=d)

    # Places psi_dm at the leftmost position: psi_dm ⊗ rho_rest
    return np.kron(psi_dm, rho_reduced)

def extract_features(rho, s_in):
    X = np.zeros((len(s_in),len(obs_matrix)))

    for idx, s_val in enumerate(s_in):
          # input and evolve rho
          rho = evolve(inject_leftmost_state(s_val,rho),phase_mat)
          # measure
          X[idx,:] = np.real(obs_matrix @ rho.flatten())
        
    return X, rho


# --- 4. EXECUTION LOOPS ---

# Washout (No data storage)
for val in s_washout:
    rho = evolve(inject_leftmost_state(val, rho),phase_mat)

# Training (Vectorized feature extraction)
X_train, rho = extract_features(rho, s_train)

model_LinMem = LinearRegression()
model_LinMem.fit(X_train, y_trian_LinMem)
model_NARMA = LinearRegression()
model_NARMA.fit(X_train, y_train_NARMA)

# Testing (Batch prediction)
X_test, _ = extract_features(rho,s_test)

y_pred_LinMem = model_LinMem.predict(X_test)
y_pred_NARMA = model_NARMA.predict(X_test)

# --- 5. RESULTS ---
cov_LinMem = np.cov(y_test_LinMem, y_pred_LinMem)
cov_NARMA = np.cov(y_test_NARMA, y_pred_NARMA)
C_LinMem, C_NARMA = (cov_LinMem[0, 1]**2) / (cov_LinMem[0, 0] * cov_LinMem[1, 1]), (cov_NARMA[0, 1]**2) / (cov_NARMA[0, 0] * cov_NARMA[1, 1])

#np.savez_compressed('Data/Prediction/Heisenberg_1DNN_Ji_LinMem_NARMA_II.npz',
#                   pred_linmem = y_pred_LinMem,
#                    target_linmem = y_test_LinMem,
#                    c_linmem = C_LinMem,
#                    pred_narma = y_pred_NARMA,
#                    target_narma = y_test_NARMA,
#                    c_narma = C_NARMA,
#                    model = 'Heisenberg 1DNN with random J_i',
#                    n_spins = N,
#                    J_val = J_val,
#                    h_val = h_val,
#                    Ji = Ji,
#                    tau = tau,
#                    delay = n)

print(f"C_LinMem={C_LinMem:.2f}, C_NARMA={C_NARMA:.2f}")
end_time = time.perf_counter()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"Total time: {end_time-start_time:.4f}s")
print(f"Peak RAM: {peak/10**6:.2f} MB")
print(f"Current RAM: {current/10**6:.2f} MB")
