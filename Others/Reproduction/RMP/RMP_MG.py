## Mackey Glass using RMP and ED for time evolution.

##Imports
import numpy as np
import matplotlib.pyplot as plt
from Density_matrix import trace_1, mixed_density_matrix
from scipy.linalg import eigh
from Models import X, Y, Z, ZZ, Ising, Mixed_Heisenberg, Ferromagnetic_Heisenberg, Anti_Ferromagnetic_Heisenberg
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(seed=42)

# --- 1. DATA GENERATION (Mackey Glass) ---
washout, train, test = 1000, 10000, 2000
total_steps = washout+train+test
sigma = 1/10
tau_MG = 17
s_raw = np.zeros(140000)
s_raw[0] = 1.2
for i in range(0,len(s_raw)-1):
    s_raw[i+1] = s_raw[i] + sigma*((0.2*s_raw[i-int(tau_MG/sigma)])/(1 + s_raw[i - int(tau_MG/sigma)]**10) - 0.1*s_raw[i])

## Subsample after washout
s_raw = s_raw[10000:]
y = np.zeros(total_steps)
for i in range(len(y)):
    y[i] = s_raw[i*10]
y = (y - min(y))/(max(y) - min(y))
print(max(y), min(y))

# Training data
s_washout = y[:washout]
s_train = y[washout:washout+train]
y_train = y[washout+1:washout+train+1]


# --- 2. MODEL SETUP ---
N, J, h, tau = 5, 1, 0.1, 10
Hamiltonian, _ = Ising(N, J, h,rng)
rho = mixed_density_matrix(10, 2, N)

E, U = eigh(Hamiltonian)
U_dag = U.conj().T
phase_factors = np.exp(-1j * (E[:, np.newaxis] - E[np.newaxis, :]) * tau)

# --- 3. VECTORIZE OBSERVABLES ---
# We flatten observables into (n_obs, dim**2) to use dot products instead of Tr(rho @ O)
raw_obs = list(X(N)) + list(Y(N)) + list(Z(N)) + ZZ(N)
obs_matrix = np.array([o.flatten() for o in raw_obs]) 

def get_features(rho_matrix):
    # Tr(y_raw @ B) is the dot product of y_raw.flatten() and B.T.flatten()
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
print("training done!")

# Testing (Batch prediction)
y_pred = np.zeros(test)
y_pred[0] = model.predict(X_train[-1].reshape(1,-1))[0]
for k in range(1,test):
    if y_pred[k-1]<0 or y_pred[k-1]>1:
        print(f"Prediction exceeded the range at {k-1}.")
        break
    rho = time_evolve_fast(inpt_fast(rho, y_pred[k-1], N))
    X_test = get_features(rho)
    y_pred[k] = model.predict(X_test.reshape(1,-1))[0]
    
## NMSE
NMSE_val = np.sum((y_pred-y[washout+train:washout+train+test])**2)/np.sum(y[washout+train:washout+train+test:]**2)
print('NMSE_val =', NMSE_val)