import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import itertools
import re
import time
import matplotlib.pyplot as plt

# Function to preprocess text data
def preprocess_text(text):
    if isinstance(text, float):
        text = ''
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove special characters and punctuation
    text = text.strip()  # Trim extra spaces
    return text

# Load the data
df = pd.read_csv('S12PX.csv')
truth_df = pd.read_csv('S12PXTruthSet.csv')

# Merge truth data into the main dataframe
df = df.merge(truth_df, on='RecID')

# Preprocessing text fields
for column in ['Name', 'Address', 'City State Zip1', 'City State Zip2', 'PO Box', 'POCity1 State Zip', 'POCity2 State Zip', 'SSN', 'DOB']:
    df[column] = df[column].apply(preprocess_text)

# List of models to test
models_to_test = {
    'DistilBERT': 'distilbert-base-nli-stsb-mean-tokens'
}

# Define neighbors and thresholds to test
neighbor_options = [400, 600, 700, 1000]
threshold_options = [0.2, 0.1]

# Generate true pairs from the ground truth
true_pairs = set()
for _, group in df.groupby('idtruth'):
    if len(group) > 1:
        for pair in itertools.combinations(group['RecID'], 2):
            true_pairs.add(tuple(sorted(pair)))

# Function to combine attribute embeddings into a single vector
def get_combined_embeddings(model, df):
    print("🔄 Generating embeddings for individual attributes...")
    name_embeddings = model.encode(df['Name'].tolist())
    address_embeddings = model.encode(df['Address'].tolist())
    zip1_embeddings = model.encode(df['City State Zip1'].tolist())
    zip2_embeddings = model.encode(df['City State Zip2'].tolist())
    po_box_embeddings = model.encode(df['PO Box'].tolist())
    pocity1_embeddings = model.encode(df['POCity1 State Zip'].tolist())
    pocity2_embeddings = model.encode(df['POCity2 State Zip'].tolist())
    ssn_embeddings = model.encode(df['SSN'].tolist())
    dob_embeddings = model.encode(df['DOB'].tolist())

    # Concatenate embeddings for all attributes to form a single combined embedding for each record
    combined_embeddings = np.concatenate(
        [name_embeddings, address_embeddings, zip1_embeddings, zip2_embeddings, po_box_embeddings,
         pocity1_embeddings, pocity2_embeddings, ssn_embeddings, dob_embeddings], axis=1
    )

    return combined_embeddings

# Function to evaluate a model
def evaluate_model(model_name, model_path):
    print(f"\n🔄 Evaluating model: {model_name}")

    # Load the Sentence BERT model
    model = SentenceTransformer(model_path)

    # Generate combined embeddings for all attributes
    combined_embeddings = get_combined_embeddings(model, df)
    combined_embeddings = np.array(combined_embeddings).astype('float32')  # FAISS requires float32 format

    # Initialize the FAISS index
    index = faiss.IndexFlatL2(combined_embeddings.shape[1])  # L2 distance (Euclidean distance)
    index.add(combined_embeddings)

    # Dictionary to store results
    results = {}

    # Iterating over each combination of neighbors and thresholds
    for num_neighbors in neighbor_options:
        # Perform the nearest neighbor search for the current number of neighbors
        distances, indices = index.search(combined_embeddings, num_neighbors)
        
        # Convert distances to similarities (cosine similarity approximation)
        similarities = 1 - (distances / np.max(distances))
        
        for threshold in threshold_options:
            start_time = time.time()

            candidate_pairs = set()
            for i, rec_id in enumerate(df['RecID']):
                for j in range(num_neighbors):
                    if similarities[i][j] > threshold and i != indices[i][j]:
                        candidate_pairs.add(tuple(sorted((rec_id, df['RecID'].iloc[indices[i][j]]))))

            # Calculate true positives
            true_positives = len(candidate_pairs.intersection(true_pairs))

            # Calculate recall
            recall = true_positives / len(true_pairs) if len(true_pairs) > 0 else 0

            # Store the result
            results[(num_neighbors, threshold)] = recall

            print(f"Model: {model_name} | Neighbors: {num_neighbors}, Threshold: {threshold} -> Recall: {recall:.4f} (Time taken: {time.time() - start_time:.2f} seconds)")

    return results

# Evaluate each model and store results
all_results = {}
for model_name, model_path in models_to_test.items():
    all_results[model_name] = evaluate_model(model_name, model_path)

# Visualization of results for each model
plt.figure(figsize=(14, 8))
for model_name, results in all_results.items():
    for threshold in threshold_options:
        recalls_for_threshold = [results[(n, threshold)] for n in neighbor_options]
        plt.plot(neighbor_options, recalls_for_threshold, marker='o', label=f'{model_name} - Threshold {threshold}')

plt.title('Recall vs. Number of Neighbors for Different Models and Thresholds')
plt.xlabel('Number of Neighbors')
plt.ylabel('Recall')
plt.grid(True)
plt.legend(title='Model and Threshold')
plt.show()

print("\n🔄 Process completed.")
