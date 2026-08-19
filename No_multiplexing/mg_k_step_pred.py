# This code evaluates performance on the mg time series task for l step input sequence with k step direct prediction same as that done in the masters thesis by felner.

# imports
import time
import tracemalloc
import os
import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import Ridge
from Models import get_Pauli_X, get_Pauli_Y, get_Pauli_Z, get_XX, get_YY ,get_ZZ, J_matrix, FullyConnected_TFIM
from Density_matrix import trace_1

tracemalloc.start()
start_time = time.perf_counter()
seed = 42
rng = np.random.default_rng(seed)

# ---- 1. data generation ----
# l is the length of input sequences and k is the prediction step size.
discard_len, l, k, train_len, test_len = 1000, 16, 100, 800, 200
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


# ---- 2. parameters, observables, hamiltonian, functions, initial state ----
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

def extract_features(S_in):
    num_rows = S_in.shape[0]
    X_features = np.zeros([num_rows,len(obs_matrix)])
    for idx in range(num_rows):
        rho = RHO_INIT.copy()
        for val in S_in[idx,:]:
            rho = evolve(input_map(rho, val, N), phase_mat)
        X_features[idx,:] = np.real(obs_matrix @ rho.flatten())
    return X_features

# ---- 3. training ----
X_train = extract_features(S_train)

# Ridge model prevents ill-conditioned weights and feedback explosion
model = Ridge(alpha=1e-4).fit(X_train, y_train)
print("Training Complete.")

# ---- 4. testing ----
X_test = extract_features(S_test)
y_pred = model.predict(X_test)


# ---- 5. evaluation metric ----
# mean squared error; mse = mean( sum_i [ y_test_i - y_pred_i ]^2 )
mse = np.mean((y_test - y_pred)**2)

# mean absolute error; mae = mean( sum_i | y_test_i - y_pred_i | )
mae = np.mean(abs(y_test - y_pred))

# pearson correlation coefficient; pcc = sum_i (y_test_i - mean(y_test_i))(y_pred_i - mean(y_pred_i)) / [sqrt(sum_i (y_test_i - mean(y_test_i))^2 * sqrt(sum_i (y_pred_i - mean(y_pred_i))^2)]
cov = np.cov(y_test, y_pred)
pcc = np.sqrt(cov[0,1]**2 / (cov[0,0] * cov[1,1]))

print(f" mse = {mse:.4f}")
print(f" mae = {mae:.4f}")
print(f"pcc = {pcc:.4f}")

end_time = time.perf_counter()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"Total time: {end_time-start_time:.4f}s")
print(f"Peak RAM: {peak/10**6:.2f} MB")
print(f"Current RAM: {current/10**6:.2f} MB")

# ---- 6. save the results ----
output_dir = "Data/mg/k_step_pred"
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, "k_step_pred_{seed}.npz")
np.savez_compressed(
    output_file,
    mse = mse,
    mae = mae,
    pcc = pcc,
    model = 'fully connected tfim',
    n_spins = N,
    J_ij = J_ij,
    h_val = h_val,
    tau = tau
)