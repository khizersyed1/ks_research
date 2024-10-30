import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import re

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

# Load the Household Discovery data
hh_discovery_df = pd.read_csv('HHDiscovery.csv')

# Create a dictionary for fast lookup of HHLink
hh_link_dict = dict(zip(hh_discovery_df['RecID'], hh_discovery_df['LinkedWithHHDiscovery']))

# Load the Sentence BERT model
model = SentenceTransformer('paraphrase-mpnet-base-v2')

# Generate embeddings for the combined text
embeddings = model.encode(df['combined_text'].tolist())  # Inputting full combined text as input
embeddings = np.array(embeddings).astype('float32')  # FAISS requires float32 format

# Normalize embeddings for cosine similarity
faiss.normalize_L2(embeddings)

# Initialize FAISS index for cosine similarity
index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product (equivalent to cosine similarity for normalized embeddings)
index.add(embeddings)

# Perform the nearest neighbor search (for 500 neighbors)
num_neighbors = 500
distances, indices = index.search(embeddings, num_neighbors)

# Create a list to store the nearest neighbors data
neighbor_records = []

# Iterate through each record and store the nearest neighbors
for i, rec_id in enumerate(df['RecID']):
    for rank, j in enumerate(range(num_neighbors), start=1):
        if i != indices[i][j]:  # Skip if the neighbor is the record itself
            neighbor_rec_id = df['RecID'].iloc[indices[i][j]]
            distance = distances[i][j]
            
            # Determine if the neighbor has a household link
            hh_link_flag = 'No'
            if rec_id in hh_link_dict and hh_link_dict[rec_id] == neighbor_rec_id:
                hh_link_flag = 'HHLink'
            
            # Add the record to the list with the rank
            neighbor_record = {
                'RecID': rec_id,
                'Neighbor_RecID': neighbor_rec_id,
                'Distance': distance,
                'Rank': rank,
                'HHLink': hh_link_flag
            }
            neighbor_records.append(neighbor_record)

# Convert the neighbor data to a DataFrame
neighbor_df = pd.DataFrame(neighbor_records)

# Sort the DataFrame by RecID and then by Distance in ascending order
neighbor_df = neighbor_df.sort_values(by=['RecID', 'Rank'])

# Save the sorted neighbors data to a CSV file
neighbor_csv_path = "Nearest_Neighbors_with_HHLink_and_Rank.csv"
neighbor_df.to_csv(neighbor_csv_path, index=False)

# Output the path of the saved file
print(f"Nearest neighbors with HHLink and Rank CSV saved at: {neighbor_csv_path}")
