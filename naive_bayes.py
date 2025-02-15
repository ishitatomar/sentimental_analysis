
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA  # Import PCA for dimensionality reduction
import re
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Step 1: Data Collection
data = pd.read_csv('/content/sentiment_tweets3.csv')
tweets = data['message to examine']
y = data['label (depression result)']  # Assuming 'label' column contains sentiment labels

# Step 2: Preprocessing
def preprocess_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)  # Remove URLs
    tweet = re.sub(r'@\w+', '', tweet)  # Remove mentions
    tweet = re.sub(r'#', '', tweet)  # Remove hashtags
    tweet = re.sub(r'[^a-zA-Z\s]', '', tweet)  # Remove special characters
    return tweet

cleaned_tweets = tweets.apply(preprocess_tweet)

# Step 3: Feature Extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_tweets)

# Step 4: Naive Bayes Model
model = MultinomialNB()
model.fit(X, y)

# Step 5: Evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Get the classification report
report = classification_report(y_test, y_pred, output_dict=True)

# Print evaluation metrics including accuracy, precision, recall, and F1-score
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')


print('\nClassification Report:\n')
# Print class-wise precision, recall, F1-score, and support
for class_label, metrics in report.items():
    if class_label not in ['accuracy', 'macro avg', 'weighted avg']:
        print(f'Class: {class_label}')
        print(f'Precision: {metrics["precision"]:.4f}')
        print(f'Recall: {metrics["recall"]:.4f}')
        print(f'F1-Score: {metrics["f1-score"]:.4f}')
        print(f'Support: {metrics["support"]}\n')
