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
from sklearn.utils import shuffle

# Simulated duration in seconds for each phase
duration = 3600
duration_training = 300

# Global model, results dictionary, and thread list for validation threads
validation_threads = []
max_validation_threads = 50  # Set a maximum number of validation threads to prevent overloading


chunk_size = 5000  # Adjust based on your Raspberry Pi's memory capacity
chunks_processed = 0

# Global model and results dictionary
textclassifier = make_pipeline(
    CountVectorizer(),
    TfidfTransformer(),
    SMOTE(random_state=12),
    MultinomialNB()
)

for chunk in pd.read_csv('datasets/IMDB Dataset.csv', chunksize=chunk_size):
    # Simulate random sampling from each chunk
    data_sample = chunk.sample(frac=0.1)  # Adjust sampling rate as needed

    X_train, X_test, y_train, y_test = train_test_split(data_sample['review'], data_sample['sentiment'], random_state=0, train_size=0.8)
    
    # Fit the classifier on the current chunk (you might want to fit it on a cumulated dataset or adjust as needed)
    textclassifier.fit(X_train, y_train)

    # Optional: Evaluate the classifier here or perform other operations

    chunks_processed += 1
    print(f"Processed {chunks_processed} chunks.")


# Load the dataset (adjust the path to where you have your dataset)
df = pd.read_csv('datasets/IMDB Dataset.csv')
# X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], random_state=0, train_size=0.8)

# Initial training
# textclassifier.fit(X_train, y_train)

# Assume these are globally defined after the initial fit
global_vect = textclassifier.named_steps['countvectorizer']
global_tfidf = textclassifier.named_steps['tfidftransformer']
# Simplify the model for direct access; this requires re-initialization or adjustment if using pipeline
model = MultinomialNB()

result = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}
performance = {'cpu': [], 'memory': []}
totalThread = []

# Lock for thread-safe operations
lock = threading.Lock()

def preprocess_and_train(X, y, model):
    # Transform the text data using the pre-fitted vectorizer and TF-IDF transformer
    X_vect = global_vect.transform(X)
    X_tfidf = global_tfidf.transform(X_vect)
    
    # Fit the model (assuming y is already prepared)
    model.partial_fit(X_tfidf, y, classes=np.unique(y))

def validation(stop_event):
    global X_train, y_train
    while not stop_event.is_set():
        with lock:
            data_sample = df.sample(random.randint(1, 100))
            X_sample = data_sample['review']
            y_true = data_sample['sentiment']

            y_pred = textclassifier.predict(X_sample)

            X_train = pd.concat([X_train, X_sample])
            y_train = pd.concat([y_train, pd.Series(y_pred)])

            result['accuracy'].append(accuracy_score(y_true, y_pred))
            result['f1'].append(f1_score(y_true, y_pred, average='weighted'))
            result['precision'].append(precision_score(y_true, y_pred, average='weighted'))
            result['recall'].append(recall_score(y_true, y_pred, average='weighted'))

        time.sleep(1)

def train(stop_event, model):
    global X_train, y_train
    chunk_size = 5000  # Define your chunk size here
    while not stop_event.is_set():
        with lock:
            print("=========================================")
            print(len(X_train))
            print(len(y_train))
            print("=========================================")
            # Ensure the data is shuffled before splitting into chunks
            X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train)
            
            num_chunks = int(np.ceil(len(X_train) / chunk_size))
            for i in range(num_chunks):
                start = i * chunk_size
                end = start + chunk_size
                X_chunk = X_train_shuffled.iloc[start:end]
                y_chunk = y_train_shuffled.iloc[start:end]
                
                # Process and train on the chunk
                preprocess_and_train(X_chunk, y_chunk, model)
            
            print(f"Completed training on {num_chunks} chunks.")
        
        time.sleep(duration_training)

def record_performance(stop_event):
    while not stop_event.is_set():
        with lock:
            cpu = psutil.cpu_percent()
            memory = psutil.virtual_memory().percent
            performance['cpu'].append(cpu)
            performance['memory'].append(memory)
        time.sleep(1)
        
def monitor_and_adjust_threads(stop_event):
    while not stop_event.is_set():
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent
        
        # Check if we should add a validation thread
        if cpu < 70 and memory < 70 and len(validation_threads) < max_validation_threads:
            t = threading.Thread(target=validation, args=(stop_event,))
            t.start()
            validation_threads.append(t)
            print(f"Added a validation thread. Total: {len(validation_threads)}")
        # Reduce to 1 thread if over threshold and more than 1 is running
        elif (cpu > 70 or memory > 70) and len(validation_threads) > 1:
            while len(validation_threads) > 1:
                thread_to_stop = validation_threads.pop()
                # Not directly stopping the thread, but could signal it to stop if designed to listen for such a signal
                print(f"Reduced validation threads. Total: {len(validation_threads)}")

        totalThread.append(len(validation_threads))
        time.sleep(1)  # Check every 1 seconds

def main():
    stop_event = threading.Event()

    threads = [
        threading.Thread(target=train, args=(stop_event, model)),
        threading.Thread(target=record_performance, args=(stop_event,))
    ]
    
    # Start initial validation thread
    initial_validation_thread = threading.Thread(target=validation, args=(stop_event,))
    initial_validation_thread.start()
    validation_threads.append(initial_validation_thread)
    
    # Start the monitor and adjust threads function
    monitor_thread = threading.Thread(target=monitor_and_adjust_threads, args=(stop_event,))
    monitor_thread.start()

    for thread in threads:
        thread.start()

    time.sleep(duration)
    stop_event.set()

    for thread in threads:
        thread.join()

    print("Simulation completed.")
    # Convert the dictionaries into DataFrames
    results_df = pd.DataFrame(result)
    performance_df = pd.DataFrame(performance)
    totalThread_df = pd.DataFrame(totalThread)

    # Save the DataFrames to CSV files
    results_df.to_csv('results.csv', index=False)
    performance_df.to_csv('performance.csv', index=False)
    totalThread_df.to_csv('thread_monitoring.csv', index=False)

    print(performance)
    print(result)

if __name__ == "__main__":
    main()
