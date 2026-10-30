# This code calculates the <r> ratio for the Heisenberg spin-3/2 chain in the M=0 sector using GPU acceleration. It dynamically determines the optimal sub-batch size based on available GPU VRAM, constructs the Hamiltonian matrix, and computes eigenvalues in parallel. The results are averaged over multiple realizations of random disorder fields.


import itertools
import torch
import csv
import os


def get_optimal_sub_batch_size(
    dim, num_realizations, device="cuda", dtype=torch.float64, safety_factor=0.70
):
    """Dynamically calculates the maximum safe sub_batch_size based on free GPU VRAM.

    Accounts for the dense matrix size and cuSOLVER workspace overhead during
    torch.linalg.eigvalsh.
    """
    if device == "cpu" or not torch.cuda.is_available():
        return min(10, num_realizations)

    # Query currently available free VRAM in bytes
    free_vram, total_vram = torch.cuda.mem_get_info(device)

    # Use a safety margin (e.g., 70% of free memory) to accommodate PyTorch context & caching
    usable_vram = free_vram * safety_factor

    bytes_per_elem = 8 if dtype == torch.float64 else 4
    matrix_bytes = dim * dim * bytes_per_elem

    # cuSOLVER (eigvalsh) requires ~2.5x to 3x matrix size in workspace memory per batch item
    bytes_per_realization = matrix_bytes * 3.0

    calc_batch_size = int(usable_vram // bytes_per_realization)

    # Clamp batch size between 1 and num_realizations
    optimal_batch = max(1, min(calc_batch_size, num_realizations))

    print(
        f"[GPU Memory Detection] Free VRAM: {free_vram / (1024**3):.2f} GB / {total_vram / (1024**3):.2f} GB"
    )
    print(
        f"[Auto Batching] Matrix Dim: {dim}x{dim} ({matrix_bytes / (1024**2):.1f} MB per matrix)"
    )
    print(f"[Auto Batching] Selected Sub-Batch Size: {optimal_batch}\n")

    return optimal_batch


def compute_r_ratio(evals_batch, keep_fraction=0.50):
    """Computes <r> from a batch of sorted eigenvalues on GPU.

    evals_batch shape: (num_realizations, dim)
    """
    num_realizations, dim = evals_batch.shape

    # Trim spectrum to retain central fraction
    discard = int((1.0 - keep_fraction) / 2.0 * dim)
    trimmed_evals = evals_batch[:, discard : dim - discard]

    # Adjacent level spacings s_n = E_{n+1} - E_n
    s = torch.diff(trimmed_evals, dim=1)

    # Ratio r_n = min(s_n, s_{n-1}) / max(s_n, s_{n-1})
    s_curr = s[:, 1:]
    s_prev = s[:, :-1]

    r_n = torch.minimum(s_curr, s_prev) / torch.maximum(s_curr, s_prev)

    # Average over spectrum levels within each realization
    r_per_realization = torch.mean(r_n, dim=1)

    # Global mean and standard error across realizations
    r_mean = torch.mean(r_per_realization).item()
    r_err = (torch.std(r_per_realization) / (num_realizations**0.5)).item()

    return r_mean, r_err


def gpu_batch_r_parameter_s32_m0(
    N,
    W_list,
    num_realizations,
    J=1.0,
    sub_batch_size="auto",
    keep_fraction=0.50,
    device="cuda",
):
    if N % 2 != 0:
        raise ValueError("N must be even for M=0 sector.")

    dtype = torch.float64

    # 1. Build basis on CPU
    target_digit_sum = int(1.5 * N)
    basis = [
        state
        for state in itertools.product(range(4), repeat=N)
        if sum(state) == target_digit_sum
    ]
    dim = len(basis)
    state_to_idx = {state: idx for idx, state in enumerate(basis)}

    basis_tensor = torch.tensor(basis, dtype=dtype, device=device)

    # 2. Determine optimal sub_batch_size if set to 'auto'
    if sub_batch_size == "auto" or sub_batch_size is None:
        sub_batch_size = get_optimal_sub_batch_size(
            dim=dim,
            num_realizations=num_realizations,
            device=device,
            dtype=dtype,
            safety_factor=0.70,
        )

    # 3. Construct off-diagonal exchange matrix on CPU and move to GPU
    H_offdiag = torch.zeros((dim, dim), dtype=dtype)
    Sp_coeffs = {
        1: torch.sqrt(torch.tensor(3.0)),
        2: 2.0,
        3: torch.sqrt(torch.tensor(3.0)),
    }
    Sm_coeffs = {
        0: torch.sqrt(torch.tensor(3.0)),
        1: 2.0,
        2: torch.sqrt(torch.tensor(3.0)),
    }

    for idx, state in enumerate(basis):
        for i in range(N):
            j = (i + 1) % N
            d_i, d_j = state[i], state[j]

            if d_i in Sp_coeffs and d_j in Sm_coeffs:
                new_state = list(state)
                new_state[i] -= 1
                new_state[j] += 1
                new_idx = state_to_idx[tuple(new_state)]
                H_offdiag[new_idx, idx] += (
                    0.5 * J * Sp_coeffs[d_i] * Sm_coeffs[d_j]
                )

            if d_i in Sm_coeffs and d_j in Sp_coeffs:
                new_state = list(state)
                new_state[i] += 1
                new_state[j] -= 1
                new_idx = state_to_idx[tuple(new_state)]
                H_offdiag[new_idx, idx] += (
                    0.5 * J * Sm_coeffs[d_i] * Sp_coeffs[d_j]
                )

    H_offdiag = H_offdiag.to(device)

    # Pre-calculate ZZ interaction energy for diagonal
    sz_all = 1.5 - basis_tensor
    E_zz = torch.zeros(dim, dtype=dtype, device=device)
    for i in range(N):
        j = (i + 1) % N
        E_zz += J * sz_all[:, i] * sz_all[:, j]

    results = {}
    diag_indices = torch.arange(dim, device=device)

    print(
        f"Running N={N} (M=0, Dim={dim}) | Total Realizations={num_realizations} (Sub-batch size={sub_batch_size})...\n"
    )

    for W in W_list:
        evals_list = []
        torch.manual_seed(42)  # For reproducibility of random fields

        # Sub-batch realizations using auto-computed batch size
        for start_idx in range(0, num_realizations, sub_batch_size):
            current_sub_size = min(sub_batch_size, num_realizations - start_idx)

            # Generate random fields in [-W, W] for current sub-batch
            h_i = (
                2
                * torch.rand(
                    (current_sub_size, N), dtype=dtype, device=device
                )
                - 1
            ) * W

            # Calculate diagonal disorder energy
            E_disorder = h_i @ sz_all.T

            # Build 3D batch tensor for current sub-batch
            H_batch = H_offdiag.unsqueeze(0).repeat(current_sub_size, 1, 1)
            H_batch[:, diag_indices, diag_indices] += (
                E_zz.unsqueeze(0) + E_disorder
            )

            # Parallel GPU Diagonalization
            evals_sub = torch.linalg.eigvalsh(H_batch)
            evals_list.append(evals_sub)

        # Concatenate all sub-batch results back into one realization tensor
        evals_all_realizations = torch.cat(evals_list, dim=0)

        # Calculate <r> ratio
        r_mean, r_err = compute_r_ratio(
            evals_all_realizations, keep_fraction=keep_fraction
        )
        results[W] = (r_mean, r_err)

    return results


# Example Execution
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    W_list = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0]

    # sub_batch_size="auto" dynamically sizes the batch based on available VRAM
    r_results = gpu_batch_r_parameter_s32_m0(
        N=6,
        W_list=W_list,
        num_realizations=50,
        sub_batch_size="auto",
        keep_fraction=0.50,
        device=device,
    )

    print(f"{'W':<8} | {'<r>':<10} | {'Std Error':<10} | {'Regime'}")
    print("-" * 45)
    for W, (r_mean, r_err) in r_results.items():
        regime = (
            "GOE (Ergodic)"
            if r_mean > 0.48
            else "Poisson (Localized)" if r_mean < 0.42 else "Crossover"
        )
        print(f"{W:<8.2f} | {r_mean:<10.4f} | {r_err:<10.4f} | {regime}")

    # Save to CSV
    output_dir = "Data/r_parameter/Heisenberg"
    os.makedirs(output_dir, exist_ok=True) 
    output_file = os.path.join(output_dir, "r_parameter_N6.csv")
    
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["W", "r_mean", "r_err"])  # Header
        for W, (r_mean, r_err) in r_results.items():
            writer.writerow([W, r_mean, r_err])

    print("Saved to r_parameter_N6.csv")