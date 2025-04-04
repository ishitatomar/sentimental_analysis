

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import re

# Step 1: Data Collection
data = pd.read_csv('/content/sentiment_tweets3.csv')  # Modify the file path as needed
tweets = data['message to examine']
y = data['label (depression result)']  # Assuming this is the label column

# Step 2: Preprocessing
def preprocess_tweet(tweet):
    tweet = tweet.lower()  # Convert to lowercase
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)  # Remove URLs
    tweet = re.sub(r'@\w+', '', tweet)  # Remove mentions
    tweet = re.sub(r'#', '', tweet)  # Remove hashtags
    tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)  # Remove special characters
    return tweet

cleaned_tweets = tweets.apply(preprocess_tweet)

# Step 3: Feature Extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_tweets)

# Step 4: GWO Algorithm Implementation
class GWO:
    def __init__(self, objective_function, num_wolves, max_iter):
        self.objective_function = objective_function
        self.num_wolves = num_wolves
        self.max_iter = max_iter
        self.alpha_pos = None
        self.alpha_score = float("inf")
        self.wolves_pos = np.random.rand(num_wolves, X.shape[1])  # Random positions
        self.error_history = []  # To store error history for plotting

    def optimize(self):
        for iter in range(self.max_iter):
            for i in range(self.num_wolves):
                score = self.objective_function(self.wolves_pos[i])
                if score < self.alpha_score:
                    self.alpha_score = score
                    self.alpha_pos = self.wolves_pos[i]

            # Update positions of wolves
            a = 2 - iter * (2 / self.max_iter)  # Decrease a from 2 to 0
            for i in range(self.num_wolves):
                r1, r2 = np.random.rand(2)
                A = 2 * a * r1 - a
                C = 2 * r2
                D = np.abs(C * self.alpha_pos - self.wolves_pos[i])
                self.wolves_pos[i] = self.alpha_pos - A * D

            # Track the error (1 - accuracy) for the current iteration
            self.error_history.append(self.alpha_score)
        return self.alpha_pos, self.alpha_score

# Example objective function
def objective_function(wolf_position):
    # Here you can define how to evaluate the position
    # For example, using a simple logistic regression model
    model = LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)  # Calculate accuracy
    return 1 - accuracy  # Minimize error (1 - accuracy)

# Step 5: Evaluation
gwo = GWO(objective_function, num_wolves=10, max_iter=100)
best_position, best_score = gwo.optimize()

# Print the best position and the corresponding accuracy
best_accuracy = 1 - best_score  # Convert error to accuracy
print(f'Best Position: {best_position}')
print(f'Best Score (Error): {best_score}')
print(f'Best Accuracy: {best_accuracy}')
