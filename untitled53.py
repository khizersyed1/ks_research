# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 16:50:07 2024

@author: onais
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import itertools
import re
import time
from sklearn.metrics.pairwise import cosine_similarity

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
df['Name'] = df['Name'].apply(preprocess_text)
df['Address'] = df['Address'].apply(preprocess_text)

# Combine Name and Address into a single string for each record
df['name_address'] = df['Name'] + ' ' + df['Address']

# List of models to test
models_to_test = {
    'DistilBERT': 'distilbert-base-nli-stsb-mean-tokens'
}

# Define thresholds to test
threshold_options = [0.2, 0.1]

# Generate true pairs from the ground truth
true_pairs = set()
for _, group in df.groupby('idtruth'):
    if len(group) > 1:
        for pair in itertools.combinations(group['RecID'], 2):
            true_pairs.add(tuple(sorted(pair)))

# Function to generate embeddings for Name and Address only
def generate_name_address_embeddings(model_name, model_path):
    print(f"\n🔄 Generating embeddings for Name and Address using model: {model_name}")

    # Load the Sentence BERT model
    model = SentenceTransformer(model_path)

    # Generate embeddings
    embeddings = model.encode(df['name_address'].tolist())
    
    print(f"✅ Embeddings generated for {len(df)} records using {model_name}.")
    
    return embeddings

# Function to calculate recall based on cosine similarity
def calculate_recall(embeddings, threshold):
    print(f"\n🔄 Calculating recall with threshold: {threshold}")

    # Calculate cosine similarities between all embeddings
    similarity_matrix = cosine_similarity(embeddings)
    
    # Set up candidate pairs
    candidate_pairs = set()
    num_records = len(df)
    
    # For each record, find candidate pairs based on the similarity threshold
    for i in range(num_records):
        for j in range(i+1, num_records):  # Compare each pair once
            if similarity_matrix[i][j] > threshold:
                candidate_pairs.add(tuple(sorted((df['RecID'].iloc[i], df['RecID'].iloc[j]))))
    
    # Calculate true positives
    true_positives = len(candidate_pairs.intersection(true_pairs))
    
    # Calculate recall
    recall = true_positives / len(true_pairs) if len(true_pairs) > 0 else 0
    print(f"Recall: {recall:.4f} for threshold {threshold}")
    
    return recall

# Main function to evaluate models
def evaluate_models():
    for model_name, model_path in models_to_test.items():
        print(f"\n🔄 Evaluating model: {model_name}")
        
        # Generate embeddings for Name and Address
        embeddings = generate_name_address_embeddings(model_name, model_path)
        
        # Evaluate for each threshold
        for threshold in threshold_options:
            start_time = time.time()
            recall = calculate_recall(embeddings, threshold)
            time_taken = time.time() - start_time
            print(f"Model: {model_name} | Threshold: {threshold} -> Recall: {recall:.4f} (Time taken: {time_taken:.2f} seconds)")

# Run the model evaluations
evaluate_models()
