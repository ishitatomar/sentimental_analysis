

!pip install pyswarms
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pyswarms as ps

# Load the dataset
file_path = '/content/sentiment_tweets3.csv'
data = pd.read_csv(file_path)

# Preprocess the data
X_raw = data['message to examine']  # Text messages
y = data['label (depression result)']  # Labels

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(X_raw).toarray()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize lists to store history of weights and accuracies
weights_history = []
accuracies_history = []

# Define the PSO objective function with history tracking
def objective_function_with_history(weights):
    # Reshape weights to (n_particles, 1, n_features)
    weights = weights.reshape(weights.shape[0], 1, weights.shape[1])

    # Initialize an array to store accuracy for each particle
    accuracies = []

    # Track each particle's weights and corresponding accuracy
    for particle_weights in weights:
        # Apply weights to features
        X_train_weighted = X_train * particle_weights
        X_test_weighted = X_test * particle_weights

        # Train a Logistic Regression model
        model = LogisticRegression(max_iter=500)
        model.fit(X_train_weighted, y_train)

        # Predict and calculate accuracy
        y_pred = model.predict(X_test_weighted)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    # Store the history of weights and accuracies
    weights_history.append(weights.flatten())  # Flatten to 1D array for simplicity
    accuracies_history.append(np.mean(accuracies))  # Store the average accuracy for each iteration

    # Return negative accuracies for minimization
    return -np.array(accuracies)

# PSO Optimization with history tracking
def optimize_with_pso_with_history():
    # Define bounds for weights (0 to 1 for each feature)
    dimensions = X_train.shape[1]  # Number of features
    bounds = (np.zeros(dimensions), np.ones(dimensions))

    # Define modified PSO options
    options = {
        'c1': 0.5,  # Cognitive component
        'c2': 0.3,  # Social component
        'w': 0.9,   # Inertia weight
    }

    # Define the optimizer
    optimizer = ps.single.GlobalBestPSO(
        n_particles=30,  # Number of particles
        dimensions=dimensions,  # Number of features
        options=options,  # PSO options
        bounds=bounds  # Bounds for the weights
    )

    # Run optimization with history tracking
    best_cost, best_weights = optimizer.optimize(objective_function_with_history, iters=50)

    return best_weights, -best_cost  # Return weights and accuracy

# Run PSO with history tracking
best_weights, best_accuracy = optimize_with_pso_with_history()

# Output the results
print("Best Accuracy Score:", best_accuracy)
