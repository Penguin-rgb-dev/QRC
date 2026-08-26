# This code produces data to plot mse vs h plot for the mg task using direct k_step prediction.

import resource
import os
import time
from joblib import Parallel, delayed
import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import Ridge
from Models import get_Pauli_X, get_Pauli_Y, get_Pauli_Z, get_XX, get_YY, get_ZZ, J_matrix, FullyConnected_TFIM 
from Density_matrix import trace_1

# ---- 1. DATA GENERATION ----
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


# --- 2. Parameters, readout operators, initial state, and the spin 1D basis ---
N, J, tau = 10, 1, 10
dims = 2**N
x_ops = get_Pauli_X(N)
y_ops = get_Pauli_Y(N)
z_ops = get_Pauli_Z(N)
xx_ops = get_XX(N, x_ops)
yy_ops = get_YY(N, y_ops)
zz_ops = get_ZZ(N,z_ops)
# We flatten observables into (n_obs, dim**2) to use dot products instead of Tr(rho @ O)
raw_obs = x_ops + y_ops + z_ops + xx_ops + yy_ops + zz_ops
obs_matrix = np.array([o.conj().flatten() for o in raw_obs])

# maximally coherent initial state
RHO_INIT = np.full((dims,dims), 1/dims, dtype=complex)
