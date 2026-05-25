## Imports and data generation
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

## Generating the MG series of 1000 steps
sigma, tau_MG, steps, discard = 0.1, 17, 1000, 1000
A = np.zeros((steps+discard)*10)
A[0] = 1.2
delay_idx = int(tau_MG / sigma)

for i in range((steps+discard)*10 - 1):
    # Using 1.2 as history to avoid the fixed-point at 0
    delayed_val = A[i - delay_idx] if i >= delay_idx else 1.2
    A[i+1] = A[i] + sigma * ((0.2 * delayed_val) / (1 + delayed_val**10) - 0.1 * A[i])

# Subsample with unit time steps
x = A[discard*10::10]

# Reconstructing the Phase space
D = 3
T = 7
N = steps//2
y = np.zeros([N-(D-1)*T,D])
for i in range(N-(D-1)*T):
    for j in range(D):
        y[i,:] = x[i:i+(D-1)*T+1:T]

        ## MLP: Data and definitions

## Defining the MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        # Small MLP Architecture
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # Input: Number of observables
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim), # Optional second layer
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim)  # Output: Task dependent
        )
        
    def forward(self, x):
        return self.network(x)


def predict_single(sample_array, model):
    """
    sample_array: a single reservoir state (or vector)
    """
    model.eval()
    with torch.no_grad():
        # 1. Convert to tensor
        sample_tensor = torch.tensor(sample_array, dtype=torch.float32)
        
        # 2. Add batch dimension if it's a flat 1D array
        if sample_tensor.ndim == 1:
            sample_tensor = sample_tensor.unsqueeze(0)
        
        # 3. Predict
        prediction = model(sample_tensor)
        
        # 4. FIX: Convert the entire tensor to a numpy array and flatten/squeeze it
        # This safely handles 1 output, 3 outputs, or any dimension D.
        return prediction.cpu().numpy().squeeze()
    
## Preparing Training Data
X = y[:-1,:]    #input
Y = y[1:,:]     #target output

# Split data: 80% Training, 20% Validation
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.2, random_state=42)

# Convert to Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)

# Create a dataset and loader for batching
dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# model
model = MLP(input_dim=D, hidden_dim=5, output_dim=D)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

## Genetic Algorithm : Definitions

# Create Population
import random
rng = np.random.default_rng(seed=42)

def create_population(pop_size, chromosome_length):
    population = []
    for _ in range(pop_size):
        # Generates a list of 228 random 0s and 1s
        chromosome = [random.randint(0, 1) for _ in range(chromosome_length)]
        population.append(chromosome)
    return population

# Parameters
POPULATION_SIZE = 100  # Change this to whatever you need
CHROMOSOME_LENGTH = 228
GENE_LENGTH = 6
FRACTIONAL_BITS = 3
GENES = CHROMOSOME_LENGTH//GENE_LENGTH

# Generate
my_population = create_population(POPULATION_SIZE, CHROMOSOME_LENGTH)


# Decode Population

def decode(chromosome, gene_length=GENE_LENGTH):
    decoded_weights = []
    
    for i in range(0, len(chromosome), gene_length):
        gene_bits = chromosome[i:i+gene_length]
        bit_string = "".join(map(str, gene_bits))
        
        # 1. Convert 6 bits to integer (0 to 63)
        integer_val = int(bit_string, 2)
        
        # 2. Map linearly to [-1.0, 1.0]
        # (integer_val / 63.0) gives a number between 0.0 and 1.0
        # Multiplying by 2 and subtracting 1 shifts it to [-1.0, 1.0]
        weight = -1.0 + (integer_val / 63.0) * 2.0
        
        decoded_weights.append(weight)
        
    return decoded_weights

def set_parameters(chromosome):
    # 1. Decode and reshape into NumPy arrays
    w1 = np.array(decode(chromosome[:15*6])).reshape(3, 5).T
    b1 = np.array(decode(chromosome[15*6:20*6]))
    w2 = np.array(decode(chromosome[20*6:35*6])).reshape(5, 3).T
    b2 = np.array(decode(chromosome[35*6:38*6]))
    # 2. Inject weights without tracking gradients
    with torch.no_grad():
        model.network[0].weight.copy_(torch.tensor(w1, dtype=torch.float32))
        model.network[0].bias.copy_(torch.tensor(b1, dtype=torch.float32))
        
        model.network[4].weight.copy_(torch.tensor(w2, dtype=torch.float32))
        model.network[4].bias.copy_(torch.tensor(b2, dtype=torch.float32))


## Assesing fitness of a chromosome
# chromosome to w1,b1,w2,b2
def fitness(chromosome, model=model, criterion=criterion, X_val_tensor=X_val_tensor, Y_val_tensor=Y_val_tensor):
    # 1. Decode and reshape into NumPy arrays
    w1 = np.array(decode(chromosome[:15*6])).reshape(3, 5).T
    b1 = np.array(decode(chromosome[15*6:20*6]))
    w2 = np.array(decode(chromosome[20*6:35*6])).reshape(5, 3).T
    b2 = np.array(decode(chromosome[35*6:38*6]))

    # 2. Inject weights without tracking gradients
    with torch.no_grad():
        model.network[0].weight.copy_(torch.tensor(w1, dtype=torch.float32))
        model.network[0].bias.copy_(torch.tensor(b1, dtype=torch.float32))
        
        model.network[4].weight.copy_(torch.tensor(w2, dtype=torch.float32))
        model.network[4].bias.copy_(torch.tensor(b2, dtype=torch.float32))

        # 3. Evaluate directly on validation data
        model.eval() 
        val_preds = model(X_val_tensor)
        val_loss = criterion(val_preds, Y_val_tensor)

    # Returning loss (Lower is better for fitness evaluation)
    return val_loss.item()

def tournament_selection(population, fitness_scores, tournament_size=3):
    """
    Selects one winning chromosome using a tournament.
    fitness_scores: a list/array of loss values corresponding to the population.
    """
    # Pick random indices from the population to enter the tournament
    chosen_indices = rng.choice(len(population), size=tournament_size, replace=False)
    
    # Find the index with the MINIMUM loss (best fitness)
    best_index = chosen_indices[np.argmin([fitness_scores[i] for i in chosen_indices])]
    
    return population[best_index]

def crossover(parent1, parent2, gene_length = GENE_LENGTH):
    n_genes = len(parent1)//gene_length
    if rng.uniform() < 0.3:
        return parent1, parent2
    else:
        i = rng.choice(n_genes-1)
        child1 = parent1[:(i+1)*gene_length] + parent2[(i+1)*gene_length:]
        child2 = parent2[:(i+1)*gene_length] + parent1[(i+1)*gene_length:]
        return child1, child2

def mutate(child, mutation_rate=0.01):
    for i in range(len(child)):
        if rng.uniform() < mutation_rate:
            child[i] ^= 1
    return child

## Genetic Algorithm: The Algorithm

# 1. Evaluate the entire population
fitness_scores = [fitness(chrom) for chrom in my_population]

# 2. Track your absolute best individual (Elitism)
best_idx = np.argmin(fitness_scores)
best_chromosome = my_population[best_idx]
best_loss = fitness_scores[best_idx]
print(f"Zeroeth Generation Best Loss: {best_loss:.6f}")

n_iterations=30
for i in range(n_iterations):
    # 3. Breed the next generation
    next_generation = [best_chromosome] # Carry over the best individual automatically

    while len(next_generation) < len(my_population):
        # Select two parents using tournament selection
        parent1 = tournament_selection(my_population, fitness_scores, tournament_size=3)
        parent2 = tournament_selection(my_population, fitness_scores, tournament_size=3)
        
        # Crossover
        child1, child2 = crossover(parent1, parent2) # Implement your crossover logic here
        
        # Mutation
        child1 = mutate(child1) # Implement your mutation logic here
        child2 = mutate(child2)
        
        next_generation.append(child1)
        if len(next_generation) < len(my_population):
            next_generation.append(child2)

    my_population = next_generation
    
    # 1. Evaluate the entire population
    fitness_scores = [fitness(chrom) for chrom in my_population]

    # 2. Track your absolute best individual (Elitism)
    best_idx = np.argmin(fitness_scores)
    best_chromosome = my_population[best_idx]
    best_loss = fitness_scores[best_idx]
    print(f"Generation Best Loss: {best_loss:.6f} at index: {best_idx}")

## MLP: Refining the best chromosome using gradient descent

# Hyperparameters
num_epochs = 2000

## SET THE INITIAL WEIGHTS TO THE BEST CHROMOSOME !!!!!!!!!!!!!!!!!!!!!!!!!
set_parameters(my_population[best_idx])

# Train and validate!
for epoch in range(num_epochs):
    model.train() # Set model to training mode
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        # 1. Forward pass
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        # 2. Backward pass (the "learning" part)
        optimizer.zero_grad() # Clear previous gradients
        loss.backward()        # Calculate new gradients
        optimizer.step()       # Update weights
        
        epoch_loss += loss.item()
    
    # Validation Check
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_tensor)
        val_loss = criterion(val_preds, Y_val_tensor)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Train Loss: {epoch_loss/len(train_loader):.4f} | Val Loss: {val_loss.item():.4f}")

## Evaluation
y_pred = np.zeros([500,D])
y_pred[0,:] = predict_single(Y[-1], model)
for k in range(1,500):
    y_pred[k,:] = predict_single(y_pred[k-1,:],model)

plt.figure(figsize=(10,5))
plt.plot(y_pred[:,D-1],label='prediction')
plt.plot(x[500:],label='original')
plt.legend()
plt.show()