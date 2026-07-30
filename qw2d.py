import numpy as np
import time
from joblib import Parallel, delayed
import resource
import os

start_time = time.time()

n=1000
L=500
marked=[(6,7),(7,7)]
marked_str = "_".join(f"{x}-{y}" for x, y in marked)
tau=5
reset="Localised Superposition around Most Probable Position"
reset_type = "LocalSup_MPP"


# ---- 1. Simulation function ----
def qw2d(r, radius, n=n, L=L, marked=marked, tau=tau, reset=reset, coin=None):

    if reset is None:
        reset_type = "NA"

    if coin is None:
        coin = 0.5*np.array([
            [-1,  1,  1,  1],
            [ 1, -1,  1,  1],
            [ 1,  1, -1,  1],
            [ 1,  1,  1, -1]])

    center = L//2, L//2
    marked_idx = [(center[0] + y, center[1] + x)for x, y in marked]
    marked_coin = -coin

    num_measurements = n // tau
    Fn = np.zeros(num_measurements)

    psi = np.ones((L, L, 4), dtype=complex)
    psi /= np.linalg.norm(psi)

    attempt = 0
    attempts_since_reset = 0

    survival_prob = 1.0

    for t in range(n):

        psi_prev = psi.copy()
        psi = psi_prev @ coin.T
        for m in marked_idx:
            psi[m] = psi_prev[m] @ marked_coin.T

        psi_prev = psi.copy()
        psi[:] = 0

        psi[:, :, 1] = np.roll(psi_prev[:, :, 0], 1, axis=0)
        psi[:, :, 0] = np.roll(psi_prev[:, :, 1], -1, axis=0)
        psi[:, :, 3] = np.roll(psi_prev[:, :, 2], 1, axis=1)
        psi[:, :, 2] = np.roll(psi_prev[:, :, 3], -1, axis=1)

        if (t + 1) % tau == 0:

            attempt += 1
            attempts_since_reset += 1

            prob = np.sum(np.abs(psi)**2, axis=2)

            marked_probs = np.array([prob[m] for m in marked_idx])
            total_marked_prob = marked_probs.sum()

            Fn[attempt - 1] = (survival_prob * total_marked_prob)

            survival_prob *= (1 - total_marked_prob)

            for m in marked_idx:
                psi[m] = 0
            psi /= np.linalg.norm(psi)

            #norm = np.linalg.norm(psi)
            #if norm > 0:
                #psi /= norm

            if reset == "Uniform Superposition":
                reset_type = "UniSup"
                if attempts_since_reset == r:
                    psi = np.ones((L, L, 4), dtype=complex)
                    psi /= np.linalg.norm(psi)
                    attempts_since_reset = 0


            if reset == "Localised Superposition around Most Probable Position":
                reset_type = "LocalSup_MPP"
                if attempts_since_reset == r:
                    prob = np.sum(np.abs(psi)**2, axis=2)
                    xs, ys = np.where(prob == prob.max())
                    k = np.random.randint(len(xs))
                    i = xs[k]
                    j = ys[k]
                    psi[:] = 0
                    for di in range(-radius, radius + 1):
                        for dj in range(-radius, radius + 1):
                            x = (i + di) % L
                            y = (j + dj) % L
                            psi[x, y, :] = 1
                    psi /= np.linalg.norm(psi)
                    attempts_since_reset = 0

    Pdet = np.cumsum(Fn)

    return Pdet




# --- 2. PARAMETER SCAN SETUP ---
#r_values = range(200)
#radius_values  =range(250)
r_values = [1]
radius_values = range(70)

# --- 3. PARALLEL EXECUTION ---
n_cpus = int(os.environ.get('PBS_NP', 1))
print(f"Running in parallel with {n_cpus} CPUs")

results_flat = Parallel(n_jobs=n_cpus)(
        delayed(qw2d)(r, radius) 
        for r in r_values 
        for radius in radius_values
    )

# Reshape into (len(r_values), len(radius_values), 1000) because each run returns 1000 outputs
results_matrix = np.array(results_flat).reshape(len(r_values), len(radius_values), 1000)

# Get total runtime
runtime = time.time() - start_time
print(f"The runtime is {runtime}")

# Save everything comprehensively
filename = f"qw2_n{n}_L{L}_reset{reset_type}_marked{marked_str}.npz"

np.savez_compressed(
    filename,
    Pdet = results_matrix,
    n=n,
    L=L,
    marked=marked,
    reset_type=reset_type,
    tau=tau,
    runtime=runtime
)

# Get peak memory usage in kilobytes
usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

# Convert to Megabytes or Gigabytes
print(f"--- Resource Usage Report ---")
print(f"Peak Memory Usage: {usage / 1024:.2f} MB")

