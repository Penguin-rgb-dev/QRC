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

# --- power law dependence with kac factor and PBC ---
def J_matrix_alpha(N, K_min, K_max, alpha, rng, pbcs=True, use_kac=True):
    """
    Generates a symmetric all-to-all coupling matrix with power-law decay.
    Supports Open (OBC) and Periodic Boundary Conditions (PBC).
    """
    j_raw = rng.uniform(K_min, K_max, (N, N))
    j_sym = (j_raw + j_raw.T) / 2.0

    idx = np.arange(N)
    dist = np.abs(idx[:, None] - idx[None, :])

    # 1. Apply Periodic Boundary Conditions if requested
    if pbcs:
        dist = np.minimum(dist, N - dist)

    # 2. Compute Power-Law Decay
    with np.errstate(divide='ignore'):
        decay = np.where(dist > 0, 1.0 / (dist ** alpha), 0.0)

    # 3. Apply Kac Normalization
    if use_kac:
        # Sum of a single row gives the total coupling strength per site
        kac_factor = np.sum(decay[0]) 
        decay = decay / kac_factor

    j_sym *= decay
    return j_sym

# --- Optimized Operator Generation ---
# Spin 1/2
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

# Spin 3/2
from functools import reduce
def get_spin_3_half_operators(sparse=True):
    """Generates single-site spin-3/2 operators (Sx, Sy, Sz)."""
    Sz = np.diag([1.5, 0.5, -0.5, -1.5])

    Sp = np.zeros((4, 4))
    Sp[0, 1] = np.sqrt(3)
    Sp[1, 2] = 2.0
    Sp[2, 3] = np.sqrt(3)

    Sm = Sp.T

    Sx = 0.5 * (Sp + Sm)
    Sy = -0.5j * (Sp - Sm)

    if sparse:
        return sp.csr_matrix(Sx), sp.csr_matrix(Sy), sp.csr_matrix(Sz)
    return Sx, Sy, Sz


def embed_operators(num_sites, site_ops, d=4, sparse=True):
    """Helper to embed localized operators into the full N-site Hilbert space.

    Site ordering: Site 0 is the rightmost tensor factor.
    """
    if sparse:
        eye_op = sp.eye(d, format="csr")
        op_list = [
            site_ops.get(num_sites - 1 - idx, eye_op)
            for idx in range(num_sites)
        ]
        return reduce(lambda a, b: sp.kron(a, b).tocsr(), op_list)
    else:
        eye_op = np.eye(d)
        op_list = [
            site_ops.get(num_sites - 1 - idx, eye_op)
            for idx in range(num_sites)
        ]
        return reduce(np.kron, op_list)


def get_spin_operators(num_sites, op_type="x", d=4, sparse=True):
    """Generates single-site spin operators embedded in the N-site Hilbert space."""
    Sx, Sy, Sz = get_spin_3_half_operators(sparse=sparse)
    op_map = {"x": Sx, "y": Sy, "z": Sz}

    if op_type not in op_map:
        raise ValueError("op_type must be 'x', 'y', or 'z'.")

    op = op_map[op_type]
    return [
        embed_operators(num_sites, {site: op}, d=d, sparse=sparse)
        for site in range(num_sites)
    ]

def get_spin_xx(num_sites, d=4, sparse=True):
    """Generates two-site Sz_i * Sz_j spin operators for all site pairs (i, j)."""
    Sx, _, _ = get_spin_3_half_operators(sparse=sparse)
    operators = []

    for site_i in range(num_sites):
        for site_j in range(site_i+1,num_sites):
            site_dict = {site_i: Sx, site_j: Sx}       
            full_op = embed_operators(
                num_sites, site_dict, d=d, sparse=sparse
            )
            operators.append(full_op)

    return operators

def get_spin_yy(num_sites, d=4, sparse=True):
    """Generates two-site Sz_i * Sz_j spin operators for all site pairs (i, j)."""
    _, Sy, _ = get_spin_3_half_operators(sparse=sparse)
    operators = []

    for site_i in range(num_sites):
        for site_j in range(site_i+1,num_sites):
            site_dict = {site_i: Sy, site_j: Sy}       
            full_op = embed_operators(
                num_sites, site_dict, d=d, sparse=sparse
            )
            operators.append(full_op)

    return operators

def get_spin_zz(num_sites, d=4, sparse=True):
    """Generates two-site Sz_i * Sz_j spin operators for all site pairs (i, j)."""
    _, _, Sz = get_spin_3_half_operators(sparse=sparse)
    operators = []

    for site_i in range(num_sites):
        for site_j in range(site_i+1,num_sites):
            site_dict = {site_i: Sz, site_j: Sz}       
            full_op = embed_operators(
                num_sites, site_dict, d=d, sparse=sparse
            )
            operators.append(full_op)

    return operators

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

# ---- hamiltonians ----
## ---- 1. Fully connected transverse field Ising spins ----
import numpy as np
import scipy.sparse as sp

def FullyConnected_TFIM(N, J, h):
    """
    Constructs the Fully Connected (All-to-All) Transverse Field Ising Model 
    Hamiltonian using sparse matrices.
    
    H = sum_{i < j} J_{ij} * X_i * X_j + sum_i h_i * Z_i
    
    Parameters:
        N (int): Number of spin-1/2 sites
        J (float or 2D ndarray): Interaction coupling. 
            - If float: Uniform coupling J applied to all pairs (i, j).
            - If 2D ndarray of shape (N, N): Site-dependent matrix J[i, j].
        h (float or 1D ndarray): Transverse field strength along the z-axis.
            - If float: Uniform field h applied to all sites i.
            - If 1D ndarray of shape (N,): Site-dependent field h[i] for each site i.
        
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

    # Standardise h into an (N,) array if given as a scalar
    if np.isscalar(h):
        h_array = np.full(N, h, dtype=np.float64)
    else:
        h_array = np.asarray(h, dtype=np.float64)
        if h_array.shape != (N,):
            raise ValueError(f"Array h must have shape ({N},), but got {h_array.shape}")

    # 1. DIAGONAL ELEMENTS (Transverse Field: sum_i h_i * Z_i)
    # -----------------------------------------------------------
    # Mapping: Bit 0 -> +1, Bit 1 -> -1
    diag_values = np.zeros(dims, dtype=np.float64)
    for pos in range(N):
        spin_dir = 1 - 2 * ((states >> pos) & 1)
        diag_values += h_array[pos] * spin_dir

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
    
    return H

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

## ---- 3. One-dimensional nearest neighbor Heisenberg spin chain  ----
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

## ---- 4. Heisenberg_spin_3/2 nearest neighbor ----
def heisenberg_spin_3_half(N, J_array, h_array, periodic=False):
    """
    Builds a spin-3/2 Heisenberg Hamiltonian with site-dependent J and magnetic field h.
    H = - sum_i J_i (S_i . S_{i+1}) - sum_i (h_i S^z_i)
    
    Parameters:
    - N (int): Number of sites.
    - J_array (list or ndarray): Coupling strengths. 
                                 Length must be N-1 (open) or N (periodic).
    - h_array (list or ndarray): Magnetic field at each site. Length must be N.
    - periodic (bool): If True, applies periodic boundary conditions.
    """
    d = 4
    num_states = d**N
    
    # Convert inputs to numpy arrays for safety
    J = np.asarray(J_array)
    h = np.asarray(h_array)
    
    # Validate array lengths
    num_bonds = N if periodic else N - 1
    if len(J) != num_bonds:
        raise ValueError(f"J_array must have length {num_bonds} for N={N} (periodic={periodic})")
    if len(h) != N:
        raise ValueError(f"h_array must have length {N} to match the number of sites")
    
    # Map the base-4 digit q in {0, 1, 2, 3} to m_z quantum number
    q_to_m = np.array([1.5, 0.5, -0.5, -1.5])
    
    # Pre-calculate matrix element amplitudes for S+ and S-
    Sp_amp = np.array([0.0, np.sqrt(3), 2.0, np.sqrt(3)])
    Sm_amp = np.array([np.sqrt(3), 2.0, np.sqrt(3), 0.0])
    
    rows = []
    cols = []
    data = []
    
    # Loop over every possible many-body state integer
    for s in range(num_states):
        diag_val = 0.0
        
        # --- MODIFICATION 2: Site-dependent Magnetic Field (Zeeman Term) ---
        # H_field = - \sum_i h_i * S_i^z
        for i in range(N):
            qi = (s >> (2 * i)) & 3
            diag_val += -h[i] * q_to_m[qi]
            
        # --- MODIFICATION 1: Site-dependent J Coupling ---
        for i in range(num_bonds):
            j = (i + 1) % N
            J_local = J[i]  # Use the specific J for this bond
            
            qi = (s >> (2 * i)) & 3
            qj = (s >> (2 * j)) & 3
            
            # Diagonal Contribution (S_i^z S_j^z)
            diag_val += -J_local * q_to_m[qi] * q_to_m[qj]
            
            # Off-Diagonal Contribution 0.5 * (S_i^+ S_j^- + S_i^- S_j^+)
            if qi > 0 and qj < 3:
                s_new = s - (1 << (2 * i)) + (1 << (2 * j))
                val = -J_local * 0.5 * Sp_amp[qi] * Sm_amp[qj]
                
                rows.append(s)
                cols.append(s_new)
                data.append(val)
                
            if qi < 3 and qj > 0:
                s_new = s + (1 << (2 * i)) - (1 << (2 * j))
                val = -J_local * 0.5 * Sm_amp[qi] * Sp_amp[qj]
                
                rows.append(s)
                cols.append(s_new)
                data.append(val)
        
        # Store the accumulated diagonal value (Field + SzSz) for this state
        rows.append(s)
        cols.append(s)
        data.append(diag_val)
        
    H = sp.coo_matrix((data, (rows, cols)), shape=(num_states, num_states))
    return H.tocsr()