## Mackey Glass Task with FN.
# Unoptimized
# Not stabilized

##Imports, Parameters and Definitions
import numpy as np
from Density_matrix import trace_1, mixed_density_matrix
from scipy.linalg import expm
from Models import Z, Ferromagnetic_Heisenberg, Anti_Ferromagnetic_Heisenberg , Mixed_Heisenberg, Ising 
from sklearn.linear_model import LinearRegression


#Parameters
N = 7
J = 1
h = 1/2
tau = 4
z = Z(N)
V=10

def inpt(rho, s, N):
    a = np.array([[0],[1]])
    b = np.array([[1],[0]])
    psi_s = np.sqrt(1-s)*a + np.sqrt(s)*b
    rho_s = np.outer(psi_s, psi_s)
    rho_rest = trace_1(rho,N)
    density_matrix = np.kron(rho_s, rho_rest)
    return density_matrix


def time_evol(rho, H, tau, V):
    den = (rho,)
    U = expm(-1j*H*(tau/V))
    for i in range(V):
        den += (U@den[-1]@U.conj().T,)
    return den

## Data
washout = 0
train = 10000
test = 2000
sigma = 1/10
tau_MG = 17
A = np.zeros(130000)
A[0] = 1.2
for i in range(0,len(A)-1):
    A[i+1] = A[i] + sigma*((0.2*A[i-int(tau_MG/sigma)])/(1 + A[i - int(tau_MG/sigma)]**10) - 0.1*A[i])

## Subsample after washout
A = A[10000:]   ##discarding some initial steps
y_original = np.zeros(washout+train+test)
for i in range(len(y_original)):
    y_original[i] = A[i*10]
print(min(y_original), max(y_original))
y = y_original - 0.4
print(min(y), max(y))

# Training data
s_train = y[washout:washout+train]
y_train = y[washout+1:washout+train+1]

## Hamiltonian and the initial state
Hamiltonian, Jij = Ising(N,J,h)
rho = mixed_density_matrix(10,2,N)

## CODE

# Training phase
X = np.zeros([train, N*V])
for k in range(len(s_train)):
    rho = inpt(rho, s_train[k], N)
    den = time_evol(rho, Hamiltonian, tau, V)
    for v in range(V):
        for n in range(N):
            X[k, n + v*N] = np.real(np.trace(den[v] @ z[n]))
    rho = den[-1]  # Update the density matrix to the last one in the sequence

X = (X+1)/2 # Normalizing the readout data to [0,1]
noise = np.random.uniform(-1e-5,1e-5, X.shape)
X = X + noise
# Linear regression to find the weights
model = LinearRegression()
model.fit(X, y_train)
print('Training is done!')

## Evaluation

y_out = np.zeros(test)
y_out[0] = model.predict(X[9999,:].reshape(1,-1))[0]
for i in range(1,len(y_out)):
    rho = inpt(rho, y_out[i-1],N)   #input 
    den = time_evol(rho, Hamiltonian, tau, V)   #time evolve
    X_eval = np.zeros([1,N*V])  #place to store the readouts
    for v in range(V):
        for n in range(N):
            X_eval[0,n+v*N] = np.real(np.trace(den[v]@z[n]))
    X_eval = (X_eval+1)/2   #readout complete
    y_out[i] = model.predict(X_eval)[0]    #predict
    rho = den[-1]   #Updating the state
    if y_out[i] < 0 or y_out[i] > 1:
        print('break!')
        break

#y_pred = np.zeros(train+test-1)
#y_pred[:train-1] = model.predict(X)[:-1]
#y_pred[train:] = y_out
#np.save('Results/y_pred_N7_T2.npy',y_pred)
#np.save('Results/Mackey_Glass/Reproduction/y_target.npy',y[1:])