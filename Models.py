#-------------------------------------------------------------------------------
# Name:        Various Spin model hamiltonians
# Purpose:
#
# Author:      Divesh Mathur
#
# Created:     10/02/2025
# Copyright:   (c) Divesh Mathur 2025
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
from scipy.linalg import expm
from quspin.operators import hamiltonian

# --- Helper for Identity ---
def I(i): 
    return np.identity(2**i)

# --- Updated RNG-based J Matrix ---
def J_matrix(N, K_min, K_max, rng):    
    # Vectorized generation is much faster than Python nested loops
    # rng should be an instance of np.random.default_rng()
    j_raw = rng.uniform(K_min, K_max, (N, N))
    # Make it symmetric and set diagonal to zero
    j_sym = (j_raw + j_raw.T) / 2
    np.fill_diagonal(j_sym, 0)
    return j_sym

# --- Optimized Operator Generation ---
# These are now "Static" – you should generate them ONCE and pass them 
# to your Hamiltonian functions to avoid redundant math.

def get_Pauli_X(N):
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    return [np.kron(np.kron(I(i), x), I(N-i-1)) for i in range(N)]

def get_Pauli_Y(N):
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    return [np.kron(np.kron(I(i), y), I(N-i-1)) for i in range(N)]

def get_Pauli_Z(N):
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    return [np.kron(np.kron(I(i), z), I(N-i-1)) for i in range(N)]

def get_ZZ(N, z_ops):
    # Instead of re-calculating, we use the pre-calculated Z operators
    zz = []
    for i in range(N):
        for j in range(i + 1, N):
            zz.append(z_ops[i] @ z_ops[j])
    return zz

# --- Optimized Hamiltonian Functions ---

import numpy as np

def Ising(N, K, h, rng, x_ops=None, z_ops=None, disorder=False, D=0):
    """
    Constructs a Hermitian Transverse Field Ising Hamiltonian.
    """
    if x_ops is None: x_ops = get_Pauli_X(N)
    if z_ops is None: z_ops = get_Pauli_Z(N)
    
    # Weights matrix J is real and symmetric
    W = J_matrix(N, -K/2, K/2, rng)

    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)
    
    # ---------------------------------------------------------
    # 1. INTERACTION TERMS (Optimized from O(N^2) matmuls to N)
    # ---------------------------------------------------------
    for i in range(N):
        # Accumulate the scalar-matrix additions first (computationally cheap)
        V_i = np.zeros((dim, dim), dtype=complex)
        for j in range(N):
            if i != j:
                V_i += W[i, j] * x_ops[j]
        
        # Perform the expensive matrix multiplication only ONCE per site
        H += x_ops[i] @ V_i

    # Divide by 2 because the logic above double-counts pairs (i,j) and (j,i)
    H /= 2

    # ---------------------------------------------------------
    # 2. EXTERNAL FIELD TERMS
    # ---------------------------------------------------------
    # Pre-calculate the field strengths for all sites cleanly
    fields = h + rng.uniform(-D, D, N) if disorder else [h] * N
        
    for i in range(N):
        H += fields[i] * z_ops[i]
            
    # ---------------------------------------------------------
    # 3. NUMERICAL SAFETY
    # ---------------------------------------------------------
    # Force Hermiticity to cancel out tiny rounding errors
    H = (H + H.conj().T) / 2
        
    return H, W

def Ising_1DNN(N, K, h, rng):
    # 1. Generate random coupling weights
    W = rng.uniform(low=-K/2, high=K/2, size=N)
    
    # 2. Define the interaction terms (X_i X_{i+1}) with Periodic Boundary Conditions
    # Format: [[weight, site_i, site_j], ...]
    x_interactions = [[W[i], i, (i + 1) % N] for i in range(N)]
    
    # 3. Define the transverse field terms (Z_i)
    # Format: [[weight, site_i], ...]
    z_fields = [[h, i] for i in range(N)]
    
    # 4. Construct operator lists for QuSpin
    # 'xx' means product of two X operators, 'z' means single Z operator
    static_list = [
        ["xx", x_interactions],
        ["z", z_fields]
    ]
    
    # 5. Build the Hamiltonian as a sparse matrix
    # check_herm=False and check_pcon=False speed up initialization
    H = hamiltonian(static_list, [], N=N, dtype=np.float64, 
                    check_herm=False, check_pcon=False)
    
    # Returns a QuSpin object. You can get the sparse matrix via H.tocsr() 
    # or get its eigenvalues directly using H.eigsh()
    return H, W

def Heisenberg(N, K, h, rng, x_ops=None, y_ops=None, z_ops=None):
    """
    Constructs a Hermitian Heisenberg (XXX) Hamiltonian.
    """
    if x_ops is None: x_ops = get_Pauli_X(N)
    if y_ops is None: y_ops = get_Pauli_Y(N)
    if z_ops is None: z_ops = get_Pauli_Z(N)
    
    W = J_matrix(N, -K/2, K/2, rng)
    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)
    
    for i in range(N):
        for j in range(i + 1, N):
            # Heisenberg interaction: XiXj + YiYj + ZiZj
            interaction = (x_ops[i] @ x_ops[j]) + \
                          (y_ops[i] @ y_ops[j]) + \
                          (z_ops[i] @ z_ops[j])
            H += W[i, j] * interaction
            
    for i in range(N):
        H -= h * z_ops[i]
        
    # Final Hermitian enforcement
    H = (H + H.T) / 2
    return H, W

#---------------------------------xxxxxxxxxxxxxxxxxxxxxxxxxxxxx-------------------------------------------------------------
# This part needs to be updated.


def Ferromagnetic_Heisenberg(N,K,h):   #Weights are in the range (0,K); mag. field = h
    x = X(N)
    y = Y(N)
    z = Z(N)
    W = J(N,0,K)

    H = np.zeros([2**N,2**N])
    for i in range(N):
        for j in range (N):
            if j > i:
                H = H - W[i][j]*(x[i]@x[j] + y[i]@y[j] + z[i]@z[j])
            else:
                continue

    for i in range(N):
        H = H - h*z[i]

    return H, W

def Anti_Ferromagnetic_Heisenberg(N,K,h):   #Weights are in the range (0,K); mag. field = h
    x = X(N)
    y = Y(N)
    z = Z(N)
    W = J(N,0,K)

    H = np.zeros([2**N,2**N])
    for i in range(N):
        for j in range (N):
            if j > i:
                H = H + W[i][j]*(x[i]@x[j] + y[i]@y[j] + z[i]@z[j])
            else:
                continue

    for i in range(N):
        H = H - h*z[i]

    return H, W

def Mixed_Heisenberg(N,K,h):   #Weights are in the range (0,K); mag. field = h
    x = X(N)
    y = Y(N)
    z = Z(N)
    W = J(N,-K/2,K/2)

    H = np.zeros([2**N,2**N])
    for i in range(N):
        for j in range (N):
            if j > i:
                H = H - W[i][j]*(x[i]@x[j] + y[i]@y[j] + z[i]@z[j])
            else:
                continue

    for i in range(N):
        H = H - h*z[i]

    return H, W





def Time_evolution_operator(H, time_step):
    return expm(-1j * H * time_step)

def time_evolution(rho,H,T):
    U = Time_evolution_operator(H, 1/10)
    P = (rho,)
    for i in range(10*T):
        P = P + (U @ P[i] @ U.T.conj(),)

    return P

def getstate():
    state = rng.getstate()
    return state

def setstate(x):
    rng.setstate(x)

def ran():
    for i in range(5):
        print(rng.random())




