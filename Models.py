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
import scipy.sparse as sp

#from quspin.operators import hamiltonian
#from quspin.basis import spin_basis_1d

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

def get_XX(N, x_ops):
    # Instead of re-calculating, we use the pre-calculated Z operators
    xx = []
    for i in range(N):
        for j in range(i + 1, N):
            xx.append(x_ops[i] @ x_ops[j])
    return xx

def get_YY(N, y_ops):
    # Instead of re-calculating, we use the pre-calculated Z operators
    yy = []
    for i in range(N):
        for j in range(i + 1, N):
            yy.append(y_ops[i] @ y_ops[j])
    return yy

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


#def Heisenberg_1DNN(N,h,J,rng):
#    h_i = rng.uniform(low=-h,high=h,size=N)
#    x_interactions=[[J,i,(i+1) % N] for i in range(N)]
#    y_interactions=[[J,i,(i+1) % N] for i in range(N)]
#    z_interactions=[[J,i,(i+1) % N] for i in range(N)]
#    z_fields = [[h_i[i],i] for i in range(N)]
#    static = [
#        ['xx',x_interactions],
#        ['yy',y_interactions],
#        ['zz',z_interactions],
#        ['z',z_fields]
#    ]
#
#    basis = spin_basis_1d(L=N)
#    H = hamiltonian(static,[],basis=basis,check_herm=False,check_pcon=False)
#    return H, h_i




#---------------------------------xxxxxxxxxxxxxxxxxxxxxxxxxxxxx-------------------------------------------------------------
 # ------ Various Hamiltonians using a different method. --------
## Anti-ferromagnetic Heisenberg spin chain with site-disorder using magnetic field. (1DNN, SPARSE)
def Heisenberg_1DNN(N, J, h, rng):
    """
    Generates the 1DNN Antiferromagnetic Heisenberg Hamiltonian with on-site
    disorder and with periodic boundary conditions 
    and random transverse/longitudinal fields using sparse COO/CSR matrices.
    
    Returns:
        H (csr_matrix): Sparse Hamiltonian of shape (2^N, 2^N)
        h_values (ndarray): Random field values drawn for each site
    """
    dims = 1 << N  # Equivalent to 2**N
    states = np.arange(dims, dtype=np.int32)
    h_values = rng.uniform(-h, h, N)

    # 1. DIAGONAL ELEMENTS (S_z S_z interaction + On-site Field)
    # -----------------------------------------------------------
    # Calculate S_z S_z term across all bonds using bitwise XOR:
    # Bitwise XOR (state ^ (state shifted)) returns 1 where adjacent spins differ, 0 where same.
    # Scale from {0, 1} to {-1, +1} mapping: S_i^z S_{i+1}^z = 1 - 2 * bit_diff
    
    sz_interaction = np.zeros(dims, dtype=np.float64)
    for pos in range(N):
        next_pos = (pos + 1) % N
        bit_diff = ((states >> pos) ^ (states >> next_pos)) & 1
        sz_interaction += 0.25*(1 - 2 * bit_diff)
    
    # Calculate single-site S_z term: spin state 1 -> +0.5, 0 -> -0.5
    # Vectorised sum across sites:
    sz_site_sum = np.zeros(dims, dtype=np.float64)
    for pos in range(N):
        spin_dir = 0.5 - ((states >> pos) & 1)   # Map 0 -> -1, 1 -> +1
        sz_site_sum += spin_dir * h_values[pos]

    # Combine diagonal values (Assuming standard S=1/2 notation: J * S_i * S_j)
    # Note: Adjust constant factors if using Pauli matrices vs Spin-1/2 operators
    diag_values = J * sz_interaction + sz_site_sum

    # 2. OFF-DIAGONAL ELEMENTS (Flip-flop S_i^+ S_j^- + S_i^- S_j^+)
    # -----------------------------------------------------------
    # Flips occur only when adjacent spins are opposite.
    rows_list = []
    cols_list = []

    for pos in range(N):
        next_pos = (pos + 1) % N
        bond_mask = (1 << pos) | (1 << next_pos)
        
        # Check where spins differ on this bond
        opposite_spins = (((states >> pos) ^ (states >> next_pos)) & 1).astype(bool)
        
        # Connected state index after flipping both bits
        rows = states[opposite_spins]
        cols = states[opposite_spins] ^ bond_mask
        
        rows_list.append(rows)
        cols_list.append(cols)

    all_rows = np.concatenate(rows_list)
    all_cols = np.concatenate(cols_list)
    
    # Each flip contributes 2 * J (or 0.5 * J depending on standard Pauli vs Spin-1/2 convention)
    off_diag_data = np.full(len(all_rows), 0.5 * J, dtype=np.float64)

    # 3. BUILD SPARSE MATRIX
    # ----------------------
    row_indices = np.concatenate([states, all_rows])
    col_indices = np.concatenate([states, all_cols])
    data = np.concatenate([diag_values, off_diag_data])

    H = sp.csr_matrix((data, (row_indices, col_indices)), shape=(dims, dims))
    
    return H, h_values

# Random anti-ferromagnetic Heisenberg spin chain (1-dimensional nearest neighbour) SPARSE
def random_antiferro_Heisenberg_1DNN_sparse(N, J, h, rng):
    dims = 2**N
    J_i = rng.uniform(0, J, N)
    
    # 1. Generate the state basis indices directly
    state_indices = np.arange(dims)
    
    # 2. Compute Diagonal Terms (Sz * Sz interactions and External Field)
    # Extract all bit values for all states at once: shape (dims, N)
    # bit_matrix[i, p] is the value (0 or 1) of the p-th bit of state i
    bit_matrix = (state_indices[:, None] >> np.arange(N)) & 1
    sz_matrix = -1*(2 * bit_matrix - 1)  # Map 0 -> -1, 1 -> +1 (Pauli Z representation) #### ABC multiply by -1
    
    # Longitudinal field term: sum_p (h * sz_p) / 2  (assuming spin-1/2)
    # Remove the division by 2 if h is acting on raw Pauli matrices
    diag_h = np.sum(sz_matrix * (h / 2.0), axis=1)
    
    # Ising J_z term: sum_p J_i * sz_p * sz_{p+1}
    # Roll the columns to get the NN neighbor (periodic boundary conditions)
    sz_neighbors = np.roll(sz_matrix, -1, axis=1)
    diag_J = np.sum(sz_matrix * sz_neighbors * (J_i / 4.0), axis=1)
    
    total_diag = diag_h + diag_J
    
    # 3. Compute Off-Diagonal Terms (Sx*Sx + Sy*Sy -> Spin Flips)
    row_indices = []
    col_indices = []
    data = []
    
    for pos in range(N):
        next_pos = (pos + 1) % N
        bond_mask = (1 << pos) | (1 << next_pos)
        
        # Check where spins are opposite
        bit1 = (state_indices >> pos) & 1
        bit2 = (state_indices >> next_pos) & 1
        opposite_spins = bit1 != bit2
        
        # Where spins are opposite, the flip operator acts
        cols = state_indices[opposite_spins]
        rows = state_indices[opposite_spins] ^ bond_mask
        
        row_indices.append(rows)
        col_indices.append(cols)
        # For Pauli matrices, off-diagonal is J/2. For spin-1/2 operators, it's J/2 as well if properly scaled.
        data.append(np.full(len(cols), J_i[pos] / 2.0))
        
    # Concatenate all sparse coordinates
    flat_rows = np.concatenate(row_indices)
    flat_cols = np.concatenate(col_indices)
    flat_data = np.concatenate(data)
    
    # Add the diagonal elements to the sparse data arrays
    flat_rows = np.concatenate([flat_rows, state_indices])
    flat_cols = np.concatenate([flat_cols, state_indices])
    flat_data = np.concatenate([flat_data, total_diag])
    
    # Build CSR sparse matrix
    H = sp.csr_matrix((flat_data, (flat_rows, flat_cols)), shape=(dims, dims))
    
    return H, J_i



