import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
import psutil
import random

def monitor_resources():
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent
    print(f"CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage}%")

# Define your pipeline here (outside the loop, you'll fit it inside the loop)
textclassifier = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('smote', SMOTE(random_state=12)),
    ('nb', MultinomialNB())
])

# Assuming your dataset doesn't fit into memory easily,
# we'll simulate processing chunks of it for training.
chunk_size = 5000  # Adjust based on your Raspberry Pi's memory capacity
chunks_processed = 0

for chunk in pd.read_csv('datasets/IMDB Dataset.csv', chunksize=chunk_size):
    monitor_resources()  # Monitor resources before processing each chunk

    # Simulate random sampling from each chunk
    data_sample = chunk.sample(frac=0.1)  # Adjust sampling rate as needed

    X_train, X_test, y_train, y_test = train_test_split(data_sample['review'], data_sample['sentiment'], random_state=0, train_size=0.8)
    
    # Fit the classifier on the current chunk (you might want to fit it on a cumulated dataset or adjust as needed)
    textclassifier.fit(X_train, y_train)

    # Optional: Evaluate the classifier here or perform other operations

    chunks_processed += 1
    print(f"Processed {chunks_processed} chunks.")

monitor_resources()  # Final resource check

# Note: This script processes each chunk independently for demonstration.
# You might want to accumulate data or perform incremental learning depending on your case.
