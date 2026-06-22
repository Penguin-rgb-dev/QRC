# Evaluating NRMSE over 84 step prediction horizon.
import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import LinearRegression, Ridge
from Models import get_Pauli_X, get_Pauli_Z, Ising
from Density_matrix import trace_1, mixed_density_matrix
import joblib
import sys
import torch
import torch.nn as nn
seed=42

# ---- 1. Data Generation ----
## Generate 50x1084 step MG series
n_data_realisations = 50
sigma, tau_MG, discard_len, teacher_force_len, test_len = 0.1, 17, 12000, 1000, 84
steps = discard_len + n_data_realisations*(teacher_force_len+test_len)
A = np.zeros(steps*10)
A[0] = 1.2
delay_idx = int(tau_MG / sigma)

for i in range(steps*10 - 1):
    # Using 1.2 as history to avoid the fixed-point at 0
    delayed_val = A[i - delay_idx] if i >= delay_idx else 1.2
    A[i+1] = A[i] + sigma * ((0.2 * delayed_val) / (1 + delayed_val**10) - 0.1 * A[i])

## Subsample and shift data for the task
A = A[discard_len*10:] #discarding initial 120000 steps which were previously seen by the model and also the initial transient before that.
y_original = A[::10]    # subsample every 10th step to get unit time step
Delta = max(y_original) - min(y_original)
y = (y_original - min(y_original))/(max(y_original)-min(y_original))
y = 0.1 + 0.8*y 
print(f"y_max = {max(y)}; y_min = {min(y)}")
delta = max(y) - min(y)



# ---- 2. Load model weights, Hamiltonian. ----
N, J_val, h_val, tau, V = 7, 1, 0.5, 4, 10
#--------------- MLP at the readout layer  -----------------------------------
class QuantumReadoutMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(QuantumReadoutMLP, self).__init__()
        # Small MLP Architecture
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # Input: Number of observables
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # Optional second layer
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # Output: Task dependent
        )
        
    def forward(self, x):
        return self.network(x)

model = QuantumReadoutMLP(input_dim=N*V, hidden_dim=32, output_dim=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def predict_single(sample_array, model):
    """
    sample_array: a single reservoir state (e.g., a list or 1D numpy array)
    """
    model.eval()
    with torch.no_grad():
        # 1. Convert to tensor
        sample_tensor = torch.tensor(sample_array, dtype=torch.float32)
        
        # 2. Add batch dimension: (input_dim) -> (1, input_dim)
        sample_tensor = sample_tensor.unsqueeze(0)
        
        # 3. Predict
        prediction = model(sample_tensor)
        
        # 4. Return as a scalar or list
        return prediction.item()

model.load_state_dict(torch.load('Data/mlp_weights.pth'))
model.eval()

x_ops = get_Pauli_X(N)
z_ops = get_Pauli_Z(N)
z_diags = [op.diagonal() for op in z_ops]
rng = np.random.default_rng(seed)
Hamiltonian, _ = Ising(N, J_val, h_val, rng, x_ops, z_ops)
rho_0 = mixed_density_matrix(10, 2, N, rng)

# Exact Diagonalization (ED) for Time Evolution
E, U = eigh(Hamiltonian)
U_dag = U.conj().T

# Pre-calculate evolution phases
dt = tau / V
Phase_dt = np.exp(-1j * (E[:, np.newaxis] - E[np.newaxis, :]) * dt)
Phase_tau = np.exp(-1j * (E[:, np.newaxis] - E[np.newaxis, :]) * tau)


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


# ---- 3. Testing ----
# loop over n_data_realisations sets
y_84_error = np.zeros(n_data_realisations)
for i in range(n_data_realisations):
    z = y[(teacher_force_len+test_len)*i:(teacher_force_len+test_len)*(i+1)]
    y_tf = z[:teacher_force_len]
    y_test = z[teacher_force_len:teacher_force_len+test_len]
    rho = rho_0   ### Loading the same initial state
    # teacher forcing
    for k in range(teacher_force_len-1):
        rho = evolve(input_map(rho, y_tf[k], N),Phase_tau)
    features = np.zeros(N*V)
    rho = input_map(rho,y_tf[-1],N)
    for v in range(V):
        rho = evolve(rho,Phase_dt)
        rho_diag = rho.diagonal()
        for n in range(N):
            features[v*N+n] = np.real(np.sum(rho_diag * z_diags[n]))
    features = (features+1)/2   #normalizing the features
    # prediction
    y_pred = np.zeros(test_len)
    y_pred[0] = predict_single(features,model)
    for k in range(1,test_len):
        rho = input_map(rho,y_pred[k-1],N)
        for v in range(V):
            rho = evolve(rho,Phase_dt)
            rho_diag = rho.diagonal()
            for n in range(N):
                features[v*N+n] = np.real(np.sum(rho_diag * z_diags[n]))
        features = (features+1)/2   #normalizing the features
        y_pred[k] = predict_single(features,model)
        if not (0 <= y_pred[k] <= 1):
            print(f"Warning: Divergence detected at run {i}, step {k}")
            sys.exit()
    y_84_error[i] = (y_pred[-1] - y_test[-1])

# ----3. Estimate NRMSE ----
y_84_error = y_84_error * (Delta/delta) # Rescaling to original coordinates
NRMSE = np.sqrt(np.sum(y_84_error**2)/(n_data_realisations*np.var(y_test)))
print(f'NRMSE of the 84th step prediction averaged over 50 runs is {NRMSE}')

