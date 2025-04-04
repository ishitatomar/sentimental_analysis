

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# Load the dataset

data = pd.read_csv('/content/sentiment_tweets3.csv')

# Preprocess the data
X_raw = data['message to examine']  # Text messages
y = data['label (depression result)']  # Labels

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)  # Limit to 1000 features for simplicity
X = vectorizer.fit_transform(X_raw).toarray()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the objective function for GWO
def objective_function(weights):
    """
    Evaluate the performance based on selected feature weights.
    Accuracy is simulated as the sum of weights (proportional).
    """
    # Ensure weights are within bounds
    weights = np.clip(weights, 0, 1)

    # Simulated accuracy as the sum of weights
    fitness = np.sum(weights)
    return -fitness  # Negative for minimization

# GWO Implementation
def gwo(objective_function, dim, n_agents, max_iter, lb, ub):
    alpha_pos = np.zeros(dim)
    beta_pos = np.zeros(dim)
    delta_pos = np.zeros(dim)
    alpha_score = float("inf")
    beta_score = float("inf")
    delta_score = float("inf")
    positions = np.random.uniform(lb, ub, (n_agents, dim))
    for t in range(max_iter):
        for i in range(n_agents):
            positions[i, :] = np.clip(positions[i, :], lb, ub)
            fitness = objective_function(positions[i, :])
            if fitness < alpha_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()
                beta_score = alpha_score
                beta_pos = alpha_pos.copy()
                alpha_score = fitness
                alpha_pos = positions[i, :].copy()
            elif fitness < beta_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()
                beta_score = fitness
                beta_pos = positions[i, :].copy()
            elif fitness < delta_score:
                delta_score = fitness
                delta_pos = positions[i, :].copy()
        a = 2 - t * (2 / max_iter)
        for i in range(n_agents):
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha
                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                X2 = beta_pos[j] - A2 * D_beta
                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                X3 = delta_pos[j] - A3 * D_delta
                positions[i, j] = (X1 + X2 + X3) / 3
        print(f"Iteration {t + 1}, Best Fitness: {-alpha_score}")
    return alpha_pos, -alpha_score

# Run GWO
print("\nOptimizing with GWO...")
dim = X_train.shape[1]  # Number of features
n_agents = 30
max_iter = 50
lb, ub = 0, 1  # Bounds for weights
best_weights, best_fitness = gwo(objective_function, dim, n_agents, max_iter, lb, ub)

# Output the results
print("\nResults:")
print("GWO - Best Fitness Score (Sum of Weights):", best_fitness)
print("GWO - Best Weights:", best_weights)

# Simulated Accuracy
accuracy = (best_fitness / dim) * 100  # Accuracy as percentage of maximum possible score
print("GWO - Simulated Accuracy (%):", accuracy)
