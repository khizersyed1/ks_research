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
truth_df = pd.read_csv('S12PX_Truth_1_Filled.csv')

# Merge truth data into the main dataframe
df = df.merge(truth_df, on='RecID')

# Preprocessing text fields
for column in ['Name', 'Address', 'City State Zip1', 'City State Zip2', 'PO Box', 'POCity1 State Zip', 'POCity2 State Zip', 'SSN', 'DOB']:
    df[column] = df[column].apply(preprocess_text)

# Combine relevant fields into a single text field for each record
df['combined_text'] = df['Name'].fillna('') + ' ' + df['Address'].fillna('') + ' ' + \
                      df['City State Zip1'].fillna('') + ' ' + df['City State Zip2'].fillna('') + ' ' + \
                      df['PO Box'].fillna('') + ' ' + df['POCity1 State Zip'].fillna('') + ' ' + \
                      df['POCity2 State Zip'].fillna('') + ' ' + df['SSN'].fillna('') + ' ' + df['DOB'].fillna('')

# List of models to test
models_to_test = {
    'MPNet': 'paraphrase-mpnet-base-v2',
    'SBERT': 'all-MiniLM-L6-v2',
    'DistilBERT': 'distilbert-base-nli-stsb-mean-tokens',
}

# Define neighbors and thresholds to test
neighbor_options = [500]  # We will focus on 500 neighbors for analysis
threshold_options = [0.1, 0.01]  # Thresholds to test

# Generate true pairs from the ground truth
true_pairs = set()
for _, group in df.groupby('idtruth'):
    if len(group) > 1:
        for pair in itertools.combinations(group['RecID'], 2):
            true_pairs.add(tuple(sorted(pair)))

# Function to evaluate a model and perform neighbor analysis
def evaluate_model(model_name, model_path):
    print(f"\n🔄 Evaluating model: {model_name}")
    
    # Load the Sentence BERT model
    model = SentenceTransformer(model_path)

    # Use the full string input for generating embeddings
    print("🔄 Generating embeddings...")
    embeddings = model.encode(df['combined_text'].tolist())  # Inputting full combined text as input
    embeddings = np.array(embeddings).astype('float32')  # FAISS requires float32 format

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Initialize FAISS index for cosine similarity
    index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product (equivalent to cosine similarity for normalized embeddings)
    index.add(embeddings)

    # Dictionary to store results
    results = {}
    report_data = []  # List to collect report data for CSV
    percentage_data = []  # List to collect bucketed percentage data for Excel

    for num_neighbors in neighbor_options:
        # Perform the nearest neighbor search for the current number of neighbors
        distances, indices = index.search(embeddings, num_neighbors)

        for threshold in threshold_options:
            start_time = time.time()
            
            candidate_pairs = set()
            true_positive_positions = []  # List to store the rank positions of true positives

            for i, rec_id in enumerate(df['RecID']):
                for j in range(num_neighbors):
                    if i != indices[i][j]:
                        candidate_pair = tuple(sorted((rec_id, df['RecID'].iloc[indices[i][j]])))
                        if candidate_pair in true_pairs:
                            true_positive_positions.append(j + 1)  # Track the position (1-indexed)
                        candidate_pairs.add(candidate_pair)
            
            # Calculate true positives
            true_positives = len(candidate_pairs.intersection(true_pairs))
            
            # Ensure no division by zero when calculating percentages
            if true_positives == 0:
                continue
            
            # Calculate recall
            recall = true_positives / len(true_pairs) if len(true_pairs) > 0 else 0
            # Print the older recall and threshold results
            time_taken = time.time() - start_time
            print(f"Model: {model_name} | Neighbors: {num_neighbors}, Threshold: {threshold} -> Recall: {recall:.4f} (Time taken: {time_taken:.2f} seconds)")
            
            # Collect report data for CSV
            report_data.append([model_name, num_neighbors, threshold, recall, time_taken])
            
            # Analyze TP positions across neighbors
            neighbor_bins = list(range(0, 501, 10))  # Collapsing the buckets to a 10 neighbor range (1-10, 11-20, etc.)
            tp_counts = []
            for start, end in zip(neighbor_bins[:-1], neighbor_bins[1:]):
                tp_counts.append(sum(start < pos <= end for pos in true_positive_positions))
            
            # Calculate percentage for each range, ensuring sum doesn't exceed 100%
            total_tps = sum(tp_counts)
            tp_percentages = [(count / total_tps) * 100 if total_tps > 0 else 0 for count in tp_counts]
            
            # Collect percentage data for Excel
            percentage_data.append([model_name, threshold] + tp_percentages)
            
            # Visualization with percentages on each bar
            plt.figure(figsize=(10, 6))
            bars = plt.bar([f'{start+1}-{end}' for start, end in zip(neighbor_bins[:-1], neighbor_bins[1:])], tp_percentages)
            plt.title(f'True Positives Distribution Across Neighbors ({model_name}, Threshold {threshold})')
            plt.xlabel('Neighbor Ranges')
            plt.ylabel('Percentage of True Positives')
            
            # Display the percentage above each bar
            for bar, percent in zip(bars, tp_percentages):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{percent:.2f}%', ha='center', va='bottom')  # Adjusted placement
            
            # Save the chart as a PNG
            chart_title = f'TP_Distribution_{model_name}_Threshold_{threshold}.png'
            plt.savefig(chart_title, bbox_inches='tight')
            # plt.show()

    return report_data, percentage_data

# Collect data for all models
report_data_all = []
percentage_data_all = []

for model_name, model_path in models_to_test.items():
    report_data, percentage_data = evaluate_model(model_name, model_path)
    report_data_all.extend(report_data)
    percentage_data_all.extend(percentage_data)

# Convert the report data to a DataFrame and save as CSV
report_df = pd.DataFrame(report_data_all, columns=['Model', 'Neighbors', 'Threshold', 'Recall', 'Time (s)'])
report_csv_path = "Model_Recall_Report.csv"
report_df.to_csv(report_csv_path, index=False)

# Convert the percentage data to a DataFrame and save as Excel
percentage_df = pd.DataFrame(percentage_data_all, columns=['Model', 'Threshold'] + [f'{start+1}-{end}' for start, end in zip(range(0, 500, 10), range(10, 510, 10))])
percentage_excel_path = "Neighbor_Percentage_Buckets.xlsx"
percentage_df.to_excel(percentage_excel_path, index=False)

# Output the paths of the saved files
print(f"CSV report saved at: {report_csv_path}")
print(f"Excel report saved at: {percentage_excel_path}")
