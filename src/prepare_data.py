import os
import pandas as pd
import numpy as np

# Create required directories
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

print("Adapting the dataset files...")

# 1. Adapt classes.csv
# For classes.csv, we need to convert 'unknown' to a numeric value like -1
try:
    classes_df = pd.read_csv('data/raw/classes.csv')
    # Convert 'unknown' to -1 and other classes to numeric
    classes_df['class'] = classes_df['class'].replace('unknown', -1)
    # Convert remaining strings to integers if any
    classes_df['class'] = pd.to_numeric(classes_df['class'])
    classes_df.to_csv('data/raw/classes_adapted.csv', index=False)
    print(f"Processed classes data: {len(classes_df)} rows")
except FileNotFoundError:
    print("Warning: classes.csv not found. Please place it in data/raw/ directory.")

# 2. Adapt edgelist.csv 
# The edgelist.csv seems correct, but ensuring column names are right
try:
    edges_df = pd.read_csv('data/raw/edgelist.csv')
    # Ensure column names are correct
    if edges_df.columns.tolist() != ['txId1', 'txId2']:
        edges_df.columns = ['txId1', 'txId2']
    edges_df.to_csv('data/raw/edgelist_adapted.csv', index=False)
    print(f"Processed edges data: {len(edges_df)} rows")
except FileNotFoundError:
    print("Warning: edgelist.csv not found. Please place it in data/raw/ directory.")

# 3. Adapt Features.csv 
# Features.csv requires special handling as it appears to have unusual formatting
try:
    # First try to read it normally
    features_df = pd.read_csv('data/raw/Features.csv')
    
    # If the first column doesn't have a name, it might be a transaction ID
    if features_df.columns[0].isdigit() or features_df.columns[0].startswith('2'):
        # This likely means the first row is actually the header
        # Reload with no header
        features_df = pd.read_csv('data/raw/Features.csv', header=None)
        
        # Set first column as txId and the rest as features
        features_df.columns = ['txId'] + [f'feature_{i}' for i in range(1, len(features_df.columns))]
        
        # Check if the second column is all 1's (dummy column)
        if features_df['feature_1'].nunique() == 1:
            # Drop the dummy column
            features_df = features_df.drop(columns=['feature_1'])
    
    features_df.to_csv('data/raw/features_adapted.csv', index=False)
    print(f"Processed features data: {len(features_df)} rows with {len(features_df.columns)-1} features")
except FileNotFoundError:
    print("Warning: Features.csv not found. Please place it in data/raw/ directory.")

print("Data adaptation complete. Adapted files are in data/raw/ with _adapted suffix.")