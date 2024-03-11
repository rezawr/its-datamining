import threading
import time
import psutil
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
from imblearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

# Simulated duration in seconds for each phase
duration = 60
duration_training = 10

# Load the dataset (adjust the path to where you have your dataset)
df = pd.read_csv('datasets/IMDB Dataset.csv')
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], random_state=0, train_size=0.8)

# Global model and results dictionary
textclassifier = make_pipeline(
    CountVectorizer(),
    TfidfTransformer(),
    SMOTE(random_state=12),
    MultinomialNB()
)
# Initial training
textclassifier.fit(X_train, y_train)

result = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}
performance = {'cpu': [], 'memory': []}

# Lock for thread-safe operations
lock = threading.Lock()

def validation(stop_event):
    while not stop_event.is_set():
        with lock:
            data_sample = df.sample(random.randint(1, 100))
            X_sample = data_sample['review']
            y_true = data_sample['sentiment']

            y_pred = textclassifier.predict(X_sample)

            result['accuracy'].append(accuracy_score(y_true, y_pred))
            result['f1'].append(f1_score(y_true, y_pred, average='weighted'))
            result['precision'].append(precision_score(y_true, y_pred, average='weighted'))
            result['recall'].append(recall_score(y_true, y_pred, average='weighted'))

        time.sleep(1)

def train(stop_event):
    while not stop_event.is_set():
        with lock:
            # Re-train the classifier
            textclassifier.fit(X_train, y_train)
        time.sleep(duration_training)

def record_performance(stop_event):
    while not stop_event.is_set():
        with lock:
            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory().percent
            performance['cpu'].append(cpu)
            performance['memory'].append(memory)
        time.sleep(1)

def main():
    stop_event = threading.Event()

    threads = [
        threading.Thread(target=validation, args=(stop_event,)),
        threading.Thread(target=train, args=(stop_event,)),
        threading.Thread(target=record_performance, args=(stop_event,))
    ]

    for thread in threads:
        thread.start()

    time.sleep(duration)
    stop_event.set()

    for thread in threads:
        thread.join()

    print("Simulation completed.")
    print(performance)
    print(result)

if __name__ == "__main__":
    main()
