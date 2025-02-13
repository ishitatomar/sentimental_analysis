

import numpy as np
import pandas as pd
import pyswarms as ps

# Load the dataset
file_path = "/content/sentiment_tweets3.csv"
data = pd.read_csv(file_path)

# Assume the last column is the target variable (sentiment)
X = data.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').values  # Features
y = data.iloc[:, -1].values    # Target variable
X = np.nan_to_num(X)


def objective_function(weights):
    # Normalize weights
    normalized_weights = weights / np.sum(weights,axis=1, keepdims=True)
    normalized_weights = normalized_weights[:, np.newaxis, :]

    # Calculate a synthetic score based on the weighted sum of features
    # Here we simply sum the weighted features for all samples
    weighted_sum = np.sum(X * normalized_weights, axis=2)

    # For demonstration, we can use the mean of the weighted sum as the objective
    score = np.mean(weighted_sum)

    # Return negative score because PSO minimizes the objective function
    return -score


# Define a function to calculate accuracy based on the best weights
def calculate_accuracy(weights):
    # Normalize weights
    normalized_weights = weights / np.sum(weights)

    # Calculate the weighted sum of features
    weighted_sum = np.dot(X, normalized_weights)

    # Apply a threshold to classify (e.g., threshold = 0.5)
    predictions = (weighted_sum > np.mean(weighted_sum)).astype(int)

    # Calculate accuracy
    accuracy = np.mean(predictions == y)
    return accuracy




# PSO Optimization
def optimize_with_pso():
    # Define bounds for weights (0 to 1 for each feature)
    dimensions = X.shape[1]  # Number of features
    bounds = (np.zeros(dimensions), np.ones(dimensions))  # Weights between 0 and 1

    # Define PSO options
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}  # Example options

    # Define the optimizer
    optimizer = ps.single.GlobalBestPSO(
        n_particles=30,  # Number of particles
        dimensions=dimensions,  # Number of dimensions
        options=options,  # Add the options argument here
        bounds=bounds  # Bounds for the weights
    )
     # Run optimization
    best_cost, best_weights = optimizer.optimize(objective_function, iters=50)

    return best_weights, -best_cost  # Return weights and fitness
# Run PSO
best_weights, best_fitness = optimize_with_pso()

# Calculate accuracy based on the best weights
accuracy = calculate_accuracy(best_weights)

# Output the results
print("Best Weights:", best_weights)
print("Best Fitness Score (Synthetic Score):", best_fitness)
print("Accuracy (%):", accuracy * 100)
