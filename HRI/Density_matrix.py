#-------------------------------------------------------------------------------
# Name:        Density matrix
# Purpose:
#
# Author:      Divesh Mathur
#
# Created:     08/02/2025
# Copyright:   (c) Divesh Mathur 2025
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np

# --- Vectorized Random Utilities ---

def vector(m, rng, complex_state=True): 
    """
    Generates a random normalized vector of dimension m.
    If complex_state is True, it generates a complex vector (standard for QM).
    """
    # Use standard_normal for both real and imaginary parts
    # This ensures a uniform distribution on the unit sphere after normalization
    real_part = rng.standard_normal(m)
    
    if complex_state:
        imag_part = rng.standard_normal(m)
        coeff = real_part + 1j * imag_part
    else:
        coeff = real_part

    # Normalize using the complex norm: sqrt(sum(|c_i|^2))
    return coeff / np.linalg.norm(coeff)

def weight_dist(n, rng):
    """Generates a random distribution of n weights that sum to 1."""
    weights = rng.random(n)
    return weights / np.sum(weights)

# --- Optimized State Construction ---

def composite_state(m, N, rng, complex_state=True):
    """Generates a complex composite vector state for N particles."""
    vectors = [vector(m, rng, complex_state) for _ in range(N)]
    
    state = vectors[0]
    for i in range(1, N):
        # np.kron automatically handles complex numbers
        state = np.kron(state, vectors[i])
    return state

def pure_density_matrix(m, N, rng, complex_state):
    """Generates a pure state density matrix."""
    psi = composite_state(m, N, rng, complex_state)
    return np.outer(psi, psi)


def mixed_density_matrix(n, m, N, rng, complex_ensemble=True): 
    dim = m**N
    # CRITICAL: Must specify dtype=complex to store imaginary components
    rho = np.zeros((dim, dim), dtype=complex)
    weights = weight_dist(n, rng)
    
    for i in range(n):
        # Generating complex pure states
        psi = composite_state(m, N, rng, complex_state=complex_ensemble)
        rho += weights[i] * np.outer(psi, psi.conj()) # conj() for complex outer product
        
    return rho

# --- Fast Partial Trace (The "Reshape" Trick) ---

def trace_1(rho, N):
    """
    Calculates the partial trace over the first subsystem (spin 1).
    This version avoids matrix multiplications and uses memory reshaping.
    """
    dim_rest = 2**(N-1)
    # Reshape rho to (dim_sub1, dim_rest, dim_sub1, dim_rest)
    # For a qubit, dim_sub1 is 2.
    reshaped_rho = rho.reshape(2, dim_rest, 2, dim_rest)
    
    # Trace over the first and third axes (the first spin)
    return np.trace(reshaped_rho, axis1=0, axis2=2)

def basis(N):
    """Returns the natural basis vectors using identity matrix rows (fast)."""
    dim = 2**N
    eye = np.eye(dim)
    return [eye[i] for i in range(dim)]

