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

## ---- 0. Some useful functions and the Fully connected transverse field Ising spin network evaluated by explicit matrix multiplications ----

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
def get_Pauli_X_L_R(N):
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    return [np.kron(np.kron(np.eye(2**i), x), np.eye(2**(N-i-1))) for i in range(N)]

def get_Pauli_Y_L_R(N):
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    return [np.kron(np.kron(np.eye(2**i), y), np.eye(2**(N-i-1))) for i in range(N)]

def get_Pauli_Z_L_R(N):
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    return [np.kron(np.kron(np.eye(2**i), z), np.eye(2**(N-i-1))) for i in range(N)]

def get_Pauli_X(N):
    """
    Constructs all single site Pauli x operators
    with spin ordering from right to left.
    """
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    return [np.kron(np.kron(np.eye(2**(N-i-1)), x), np.eye(2**(i))) for i in range(N)]

def get_Pauli_Y(N):
    """
    Constructs all single site Pauli y operators 
    with spin ordering from right to left.
    """
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    return [np.kron(np.kron(np.eye(2**(N-i-1)), y), np.eye(2**(i))) for i in range(N)]

def get_Pauli_Z(N):
    """
    Constructs all single site Pauli z operators 
    with spin ordering from right to left.
    """
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    return [np.kron(np.kron(np.eye(2**(N-i-1)), z), np.eye(2**(i))) for i in range(N)] 

def get_XX(N, x_ops):
    # Instead of re-calculating, we use the pre-calculated X operators
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

## ---- 1. Fully connected transverse field Ising spins ----
def FullyConnected_TFIM(N, J, h):
    """
    Constructs the Fully Connected (All-to-All) Transverse Field Ising Model 
    Hamiltonian using sparse matrices.
    
    H = sum_{i < j} J_{ij} * X_i * X_j + h * sum_i Z_i
    
    Parameters:
        N (int): Number of spin-1/2 sites
        J (float or 2D ndarray): Interaction coupling. 
            - If float: Uniform coupling J applied to all pairs (i, j).
            - If 2D ndarray of shape (N, N): Site-dependent matrix J[i, j].
        h (float): Uniform transverse field strength along the z-axis.
        
    Returns:
        H (csr_matrix): Sparse Hamiltonian of shape (2^N, 2^N)
    """
    dims = 1 << N  # 2^N states
    states = np.arange(dims, dtype=np.int32)

    # Standardise J into an (N, N) matrix if given as a scalar
    if np.isscalar(J):
        J_matrix = np.full((N, N), J, dtype=np.float64)
    else:
        J_matrix = np.asarray(J, dtype=np.float64)

    # 1. DIAGONAL ELEMENTS (Transverse Field: h * sum_i Z_i)
    # -----------------------------------------------------------
    # Mapping: Bit 0 -> 1, Bit 1 -> -1
    sz_sum = np.zeros(dims, dtype=np.float64)
    for pos in range(N):
        spin_dir = 1 - 2*((states >> pos) & 1)
        sz_sum += spin_dir

    diag_values = h * sz_sum

    # 2. OFF-DIAGONAL ELEMENTS (All-to-All Interaction: sum_{i < j} J_{ij} X_i X_j)
    # ---------------------------------------------------------------------------------
    # Every pair (i, j) with i < j flips bits at site i and site j.
    # Matrix element amplitude is J_{ij}
    rows_list = []
    cols_list = []
    data_list = []

    for i in range(N):
        for j in range(i + 1, N):
            bond_mask = (1 << i) | (1 << j)
            
            # Every state flips bits at positions i and j
            rows = states
            cols = states ^ bond_mask
            
            rows_list.append(rows)
            cols_list.append(cols)
            data_list.append(np.full(dims, J_matrix[i, j], dtype=np.float64))

    all_rows = np.concatenate(rows_list)
    all_cols = np.concatenate(cols_list)
    off_diag_data = np.concatenate(data_list)

    # 3. CONSTRUCT SPARSE CSR MATRIX
    # ------------------------------
    row_indices = np.concatenate([states, all_rows])
    col_indices = np.concatenate([states, all_cols])
    data = np.concatenate([diag_values, off_diag_data])

    H = sp.csr_matrix((data, (row_indices, col_indices)), shape=(dims, dims))
    
    return H, J_matrix

## ---- 2. One-dimensional nearest neighbor transverse field Ising spin chain ---
def TFIM_1DNN(N, J, h):
    """
    Constructs the 1D Transverse Field Ising Model (TFIM) Hamiltonian 
    with Periodic Boundary Conditions using sparse matrices.
    
    H = sum_i ( J_i * X_i * X_{i+1} ) + h * sum_i ( X_i )
    
    Parameters:
        N (int): Number of spin-1/2 sites
        J (float or ndarray): Coupling constant(s). If float, uniform J is used.
                             If ndarray, length must be N (site-dependent bonds).
        h (float): Uniform transverse field strength along z-axis.
        
    Returns:
        H (csr_matrix): Sparse Hamiltonian of shape (2^N, 2^N)
    """
    dims = 1 << N  # 2^N states
    states = np.arange(dims, dtype=np.int32)

    # Standardise J to an array of length N for each bond (i, (i+1)%N)
    if np.isscalar(J):
        J_bonds = np.full(N, J, dtype=np.float64)
    else:
        J_bonds = np.asarray(J, dtype=np.float64)

    # 1. DIAGONAL ELEMENTS (Transverse Field: h * sum_i S_i^z)
    # -----------------------------------------------------------
    # Mapping: Bit 0 -> +1, Bit 1 -> -1
    sz_sum = np.zeros(dims, dtype=np.float64)
    for pos in range(N):
        spin_dir = 1 - 2*((states >> pos) & 1)
        sz_sum += spin_dir

    diag_values = h * sz_sum

    # 2. OFF-DIAGONAL ELEMENTS (Interaction: sum_i J_i X_i X_{i+1})
    # -----------------------------------------------------------
    # S_i^x S_{i+1}^x flips both spins at pos and next_pos.
    # Contribution to the matrix element is J_i.
    rows_list = []
    cols_list = []
    data_list = []

    for pos in range(N):
        next_pos = (pos + 1) % N
        bond_mask = (1 << pos) | (1 << next_pos)
        
        # Every state flips both bits at this bond
        rows = states
        cols = states ^ bond_mask
        
        rows_list.append(rows)
        cols_list.append(cols)
        data_list.append(np.full(dims, J_bonds[pos], dtype=np.float64))

    all_rows = np.concatenate(rows_list)
    all_cols = np.concatenate(cols_list)
    off_diag_data = np.concatenate(data_list)

    # 3. CONSTRUCT SPARSE CSR MATRIX
    # ------------------------------
    row_indices = np.concatenate([states, all_rows])
    col_indices = np.concatenate([states, all_cols])
    data = np.concatenate([diag_values, off_diag_data])

    H = sp.csr_matrix((data, (row_indices, col_indices)), shape=(dims, dims))
    
    return H, J_bonds

## ---- 3. One-dimensional nearest neighbor Heisenberg spin chain  --- (NEEDS TO BE UNDERSTOOD)
def Heisenberg_1DNN_general(N, J, h, rng=None):
    """
    Constructs the 1D Nearest-Neighbor Heisenberg Hamiltonian with periodic boundary 
    conditions, supporting uniform or random/site-dependent J_i and h_i values.
    
    H = sum_i J_i (sigma_i * sigma_{i+1}) + sum_i h_i Z_i
    
    Parameters:
        N (int): Number of spin-1/2 sites.
        J (float, tuple, or ndarray): 
            - float: Uniform coupling J across all bonds.
            - tuple (low, high): Uniform random J_i ~ U(low, high) drawn per bond.
            - ndarray: Exact site-dependent coupling array of length N.
        h (float, tuple, or ndarray): 
            - float: Uniform field h across all sites.
            - tuple (low, high): Uniform random h_i ~ U(low, high) drawn per site.
            - ndarray: Exact site-dependent field array of length N.
        rng (np.random.Generator, optional): Random number generator instance.
        
    Returns:
        H (csr_matrix): Sparse Hamiltonian of shape (2^N, 2^N)
        J_bonds (ndarray): Coupling values used for each bond
        h_sites (ndarray): Field values used for each site
    """
    if rng is None:
        rng = np.random.default_rng()

    dims = 1 << N  # 2^N states
    states = np.arange(dims, dtype=np.int32)

    # --- Parse J_i couplings ---
    if isinstance(J, tuple):
        J_bonds = rng.uniform(J[0], J[1], N)
    elif np.isscalar(J):
        J_bonds = np.full(N, J, dtype=np.float64)
    else:
        J_bonds = np.asarray(J, dtype=np.float64)

    # --- Parse h_i fields ---
    if isinstance(h, tuple):
        h_sites = rng.uniform(h[0], h[1], N)
    elif np.isscalar(h):
        h_sites = np.full(N, h, dtype=np.float64)
    else:
        h_sites = np.asarray(h, dtype=np.float64)

    # 1. DIAGONAL ELEMENTS (Z_i Z_{i+1} + On-site h_i Z_i)
    # -----------------------------------------------------------
    # Coupling Z_i Z_{i+1}: +1 (parallel) or -1 (antiparallel)
    sz_sz_interaction = np.zeros(dims, dtype=np.float64)
    for pos in range(N):
        next_pos = (pos + 1) % N
        bit_diff = ((states >> pos) ^ (states >> next_pos)) & 1
        # bit_diff == 0 (same) -> 1 | bit_diff == 1 (opposite) -> -1
        sz_sz_interaction += J_bonds[pos] * (1 - 2 * bit_diff)

    # Field h_i S_i^z: bit 0 -> +0.5, bit 1 -> -0.5
    sz_site_sum = np.zeros(dims, dtype=np.float64)
    for pos in range(N):
        spin_dir = 1 - 2*((states >> pos) & 1)
        sz_site_sum += spin_dir * h_sites[pos]

    diag_values = sz_sz_interaction + sz_site_sum

    # 2. OFF-DIAGONAL ELEMENTS (Flip-flop: 2 * J_i * (sigma_i^+ sigma_{i+1}^- + sigma_i^- sigma_{i+1}^+))
    # -----------------------------------------------------------------------------------
    # Spins flip ONLY when adjacent bits are opposite (bit_diff == 1)
    rows_list = []
    cols_list = []
    data_list = []

    for pos in range(N):
        next_pos = (pos + 1) % N
        bond_mask = (1 << pos) | (1 << next_pos)
        
        # Select states with opposite spins on this bond
        opposite_spins = (((states >> pos) ^ (states >> next_pos)) & 1).astype(bool)
        
        rows = states[opposite_spins]
        cols = states[opposite_spins] ^ bond_mask
        
        rows_list.append(rows)
        cols_list.append(cols)
        data_list.append(np.full(len(rows), 2 * J_bonds[pos], dtype=np.float64))

    all_rows = np.concatenate(rows_list)
    all_cols = np.concatenate(cols_list)
    off_diag_data = np.concatenate(data_list)

    # 3. CONSTRUCT SPARSE CSR MATRIX
    # ------------------------------
    row_indices = np.concatenate([states, all_rows])
    col_indices = np.concatenate([states, all_cols])
    data = np.concatenate([diag_values, off_diag_data])

    H = sp.csr_matrix((data, (row_indices, col_indices)), shape=(dims, dims))
    
    return H, J_bonds, h_sites
