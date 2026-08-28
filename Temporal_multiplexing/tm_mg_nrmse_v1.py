# this code trains for the mg time series prediction task and tests using 20 sequences of length 1084 
# to evaluate the normalised root mean squared error of the 84th step prediction (nrmse84).
# version 1: uses 1. linear rescale to [0,1]; 2. input map s in [0,1]; 3. normalization of x_features to [0,1]; 4. cliping pred value to [0,1].

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

# ---- 1. GENERATE MG SERIES DATA ----
discard_len, washout_len, train_len, teacher_force_len, test_len, num_test_seq = 1000, 1000, 2000, 1000, 84, 20
sigma, tau_MG, total_mg_steps = 0.1, 17, (washout_len + train_len + num_test_seq * (teacher_force_len + test_len) + discard_len)*10
A = np.zeros(total_mg_steps)
A[0] = 1.2
delay_idx = int(tau_MG / sigma)

for i in range(total_mg_steps - 1):
    delayed_val = A[i - delay_idx] if i >= delay_idx else 1.2
    A[i + 1] = A[i] + sigma * ((0.2 * delayed_val) / (1.0 + delayed_val**10) - 0.1 * A[i])

# discard initial transient and subsample (every 10th point).
A = A[discard_len*10:]
s_raw = A[::10]

# variance
var = np.var(s_raw)

# normalizing to [0,1]
min, max = np.min(s_raw), np.max(s_raw)
s_raw = (s_raw - min) / (max - min)
print(f's_min = {np.min(s_raw):.4f}, s_max = {np.max(s_raw):.4f}')

# defining washout, train, test data
s_washout = s_raw[:washout_len-1]
s_train = s_raw[washout_len-1 : washout_len + train_len-1]
y_train = s_raw[washout_len : washout_len + train_len]
s_test = s_raw[washout_len + train_len:].reshape(num_test_seq, teacher_force_len+test_len)

# collecting the 84th step target values in the original coordinates
y_target_84 = (s_test[:,-1] * (max - min)) + min


# ---- 2. parameters, observables, hamiltonian, functions, initial state ----
N, J, h_val, tau, V = 7, 1, 0.5e-1, 10, 10
dims = 2**N
J_ij = J_matrix(N,-J/2,J/2,rng)
Hamiltonian = FullyConnected_TFIM(N,J_ij,h_val)  # PUT YOU HAMILTONIAN HERE!
Hamiltonian = Hamiltonian.toarray()

# maximally coherent initial state
RHO_INIT = np.full((dims,dims), 1/dims, dtype=complex)  

# diagonalizing the hamiltonian
E, U = eigh(Hamiltonian)
U_dag = U.conj().T
dt = tau / V
Phase_dt = np.exp(-1j * (E[:, np.newaxis] - E[np.newaxis, :]) * dt)
Phase_tau = np.exp(-1j * (E[:, np.newaxis] - E[np.newaxis, :]) * tau)

# We flatten observables into (n_obs, dim**2) to use dot products instead of Tr(rho @ O)
z_ops = get_Pauli_Z(N)
obs_matrix = np.array([o.diagonal() for o in z_ops])

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
    X = np.zeros((len(s_in),len(obs_matrix)*V))

    for idx, s_val in enumerate(s_in):
        rho = input_map(rho, s_val, N)
        for v in range(V):
            rho = evolve(rho, Phase_dt)
            X[idx, v*N : (v+1)*N] = np.real(obs_matrix @ rho.diagonal())

    return X, rho

def teacher_force(rho, s_in):
    for val in s_in:
        rho = evolve(input_map(rho, val, N), Phase_tau)
    return rho


# ---- 3. training ----
sigma_noise = 1e-5
rho = RHO_INIT.copy()

# washout
rho = teacher_force(rho, s_washout)

# train
X_train, rho = extract_features(rho, s_train)

# normalise and add regularization noise
X_train = (X_train + 1) / 2
X_train += rng.uniform(-sigma_noise, sigma_noise, X_train.shape)

# Ridge model prevents ill-conditioned weights and feedback explosion
model = Ridge(alpha=1e-4).fit(X_train, y_train)
print("Training Complete.")

# ---- 4. testing ----
diff = []
for i in range(num_test_seq):
    rho = RHO_INIT.copy()
    rho = teacher_force(rho, s_test[i,:teacher_force_len-1])
    input_signal = s_test[i, teacher_force_len-1]

    pred_val = 0.0  # Initialize variable to ensure it's always bound

    for j in range(test_len):
        rho = input_map(rho, input_signal, N)
        X_features = np.zeros(N*V)
        for v in range(V):
            rho = evolve(rho, Phase_dt)
            X_features[v*N : (v+1)*N] = np.real(obs_matrix @ rho.diagonal())

        feat = ((X_features + 1.0) / 2.0).reshape(1, -1)
        pred_val = model.predict(feat)[0]

        # Clip predictions to prevent numerical divergence in feedback loop
        pred_val = np.clip(pred_val, 0, 1)
        input_signal = pred_val

    pred_val = (pred_val * (max - min)) + min   # converting back to the original coordinates
    diff.append(y_target_84[i]-pred_val)


# ---- 5. NRMSE_84 and memory-time----
diff = np.array(diff)
nrmse = np.sqrt(np.mean(diff**2)/var)

print("\n--- RESULTS ---")
print(f"nrmse: {nrmse:.6e}")

end_time = time.perf_counter()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"Total time: {end_time-start_time:.4f}s")
print(f"Peak RAM: {peak/10**6:.2f} MB")
print(f"Current RAM: {current/10**6:.2f} MB")

# ---- 6. save the results ----
output_dir = "Data/mg"
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, f"tm_mg_nrmse_84_v1_{seed}.npz")
np.savez_compressed(
    output_file,
    nrmse = nrmse,
    model = 'fully connected tfim',
    n_spins = N,
    J_ij = J_ij,
    h_val = h_val,
    tau = tau,
    temporal_multiplexing = V,
)
