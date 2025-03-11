import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import networkx as nx
import logging
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(classes_path, edgelist_path, features_path):
    """
    Load blockchain dataset from CSV files with the special format.
    
    Parameters:
    -----------
    classes_path : str
        Path to the classes CSV file with txId and class columns
    edgelist_path : str
        Path to the edgelist CSV file with txId1 and txId2 columns
    features_path : str
        Path to the features CSV file
        
    Returns:
    --------
    df_nodes : pandas.DataFrame
        DataFrame containing node data with txId, class, and features
    df_edges : pandas.DataFrame
        DataFrame containing edge data
    """
    logger.info(f"Loading data from {classes_path}, {edgelist_path}, and {features_path}")
    
    # Load classes data
    df_classes = pd.read_csv(classes_path)
    logger.info(f"Loaded {len(df_classes)} transactions with class information")
    
    # Load edge data
    df_edges = pd.read_csv(edgelist_path)
    logger.info(f"Loaded {len(df_edges)} edges")
    
    # Load features data - this has a non-standard format
    try:
        # Read the features file
        df_features = pd.read_csv(features_path)
        
        # First column should be txId
        df_features = df_features.rename(columns={df_features.columns[0]: 'txId'})
        
        # Second column is a constant (1) - drop it
        df_features = df_features.drop(columns=[df_features.columns[1]])
        
        # Rename remaining columns to feature_0, feature_1, etc.
        feature_cols = [col for col in df_features.columns if col != 'txId']
        feature_rename = {col: f'feature_{i}' for i, col in enumerate(feature_cols)}
        df_features = df_features.rename(columns=feature_rename)
        
        logger.info(f"Loaded {len(df_features.columns)-1} features for {len(df_features)} transactions")
    except Exception as e:
        logger.error(f"Error loading features: {str(e)}")
        raise
    
    # Merge classes and features based on txId
    df_nodes = pd.merge(df_classes, df_features, on='txId', how='inner')
    logger.info(f"Combined data has {len(df_nodes)} transactions with both class and feature information")
    
    return df_nodes, df_edges

def create_node_mapping(df_nodes):
    """
    Create a mapping between transaction IDs and indices for graph construction.
    
    Parameters:
    -----------
    df_nodes : pandas.DataFrame
        DataFrame containing node data
        
    Returns:
    --------
    id2idx : dict
        Dictionary mapping transaction IDs to indices
    """
    # Create mapping of txId to index
    id2idx = {tx_id: i for i, tx_id in enumerate(df_nodes['txId'])}
    logger.info(f"Created mapping for {len(id2idx)} transactions")
    return id2idx

def preprocess_data(df_nodes, df_edges):
    """
    Preprocess the dataset by handling unknown classes and normalizing features.
    
    Parameters:
    -----------
    df_nodes : pandas.DataFrame
        DataFrame containing node data
    df_edges : pandas.DataFrame
        DataFrame containing edge data
        
    Returns:
    --------
    df_processed : pandas.DataFrame
        Processed DataFrame
    df_edges : pandas.DataFrame
        Processed edges DataFrame
    """
    logger.info("Preprocessing data")
    
    # Make a copy to avoid modifying the original
    df_processed = df_nodes.copy()
    
    # Handle unknown classes
    unknown_mask = df_processed['class'] == 'unknown'
    unknown_count = unknown_mask.sum()
    logger.info(f"Found {unknown_count} transactions with unknown class ({unknown_count/len(df_processed):.2%})")
    
    if unknown_count > 0:
        df_processed = df_processed[~unknown_mask]
        logger.info(f"Removed transactions with unknown class. Remaining: {len(df_processed)}")
    
    # Convert non-numeric class labels to integers
    if df_processed['class'].dtype == 'object':
        unique_classes = df_processed['class'].unique()
        class_map = {cls: i for i, cls in enumerate(unique_classes)}
        df_processed['class'] = df_processed['class'].map(class_map)
        logger.info(f"Mapped class values to integers: {class_map}")
    
    # Identify feature columns (exclude txId and class)
    feature_cols = [col for col in df_processed.columns if col not in ['txId', 'class']]
    
    # Check for and remove low-variance features
    variance = df_processed[feature_cols].var()
    low_var_threshold = 0.01
    low_var_cols = variance[variance < low_var_threshold].index.tolist()
    
    if low_var_cols:
        logger.info(f"Removing {len(low_var_cols)} low-variance features")
        df_processed = df_processed.drop(columns=low_var_cols)
        # Update feature columns
        feature_cols = [col for col in df_processed.columns if col not in ['txId', 'class']]
    
    # Normalize features
    scaler = StandardScaler()
    df_processed[feature_cols] = scaler.fit_transform(df_processed[feature_cols])
    
    # Make sure edges only include transactions with known classes
    valid_tx_ids = set(df_processed['txId'])
    edges_before = len(df_edges)
    df_edges = df_edges[df_edges['txId1'].isin(valid_tx_ids) & df_edges['txId2'].isin(valid_tx_ids)]
    edges_after = len(df_edges)
    logger.info(f"Filtered edges to include only known transactions: {edges_before} -> {edges_after}")
    
    return df_processed, df_edges

def build_edge_index(df_edges, id2idx):
    """
    Construct edge_index tensor for PyTorch Geometric.
    
    Parameters:
    -----------
    df_edges : pandas.DataFrame
        DataFrame containing edge data
    id2idx : dict
        Dictionary mapping transaction IDs to indices
        
    Returns:
    --------
    edge_index : torch.LongTensor
        Edge index tensor for PyTorch Geometric
    """
    logger.info("Building edge index tensor")
    
    edges = []
    skipped = 0
    
    for _, row in df_edges.iterrows():
        source_id, target_id = row['txId1'], row['txId2']
        
        # Check if both nodes exist in the mapping
        if source_id in id2idx and target_id in id2idx:
            source_idx = id2idx[source_id]
            target_idx = id2idx[target_id]
            edges.append([source_idx, target_idx])
        else:
            skipped += 1
    
    if skipped > 0:
        logger.warning(f"Skipped {skipped} edges with missing transaction IDs")
    
    if not edges:
        logger.warning("No valid edges found")
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        # Convert to torch tensor
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    logger.info(f"Built edge index with shape {edge_index.shape}")
    
    return edge_index

def create_data_splits(df_processed, edge_index, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """
    Create train/validation/test splits and PyTorch Geometric Data object.
    
    Parameters:
    -----------
    df_processed : pandas.DataFrame
        Processed DataFrame
    edge_index : torch.LongTensor
        Edge index tensor
    train_size, val_size, test_size : float
        Proportions for train/val/test splits
    random_state : int
        Random seed
        
    Returns:
    --------
    data : torch_geometric.data.Data
        PyTorch Geometric Data object
    split_idx : dict
        Dictionary containing indices for train/val/test splits
    """
    logger.info("Creating data splits")
    
    # Get feature matrix
    feature_cols = [col for col in df_processed.columns if col not in ['txId', 'class']]
    features = torch.FloatTensor(df_processed[feature_cols].values)
    
    # Get labels
    labels = torch.LongTensor(df_processed['class'].values)
    
    # Create Data object
    data = Data(x=features, edge_index=edge_index, y=labels)
    
    # Create splits
    indices = np.arange(len(df_processed))
    
    try:
        # First split: train vs. (val+test)
        train_idx, temp_idx = train_test_split(
            indices, 
            train_size=train_size, 
            stratify=df_processed['class'].values,
            random_state=random_state
        )
        
        # Second split: val vs. test
        val_size_adjusted = val_size / (val_size + test_size)
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_size_adjusted,
            stratify=df_processed.iloc[temp_idx]['class'].values,
            random_state=random_state
        )
    except ValueError as e:
        # If stratified split fails (e.g., too few samples in some class),
        # fall back to regular split
        logger.warning(f"Stratified split failed: {str(e)}. Using random split.")
        
        train_idx, temp_idx = train_test_split(
            indices, 
            train_size=train_size, 
            random_state=random_state
        )
        
        val_size_adjusted = val_size / (val_size + test_size)
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_size_adjusted,
            random_state=random_state
        )
    
    # Create split dictionary
    split_idx = {
        'train': train_idx,
        'val': val_idx,
        'test': test_idx
    }
    
    logger.info(f"Created splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    
    return data, split_idx

def save_processed_data(data, split_idx, output_dir='data/processed'):
    """
    Save processed data and splits to disk.
    
    Parameters:
    -----------
    data : torch_geometric.data.Data
        PyTorch Geometric Data object
    split_idx : dict
        Dictionary containing indices for train/val/test splits
    output_dir : str
        Directory to save processed data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Saving processed data to {output_dir}")
    
    # Save data object
    torch.save(data, os.path.join(output_dir, 'data.pt'))
    
    # Save edge index separately
    torch.save(data.edge_index, os.path.join(output_dir, 'edge_index.pt'))
    
    # Save features and labels
    np.save(os.path.join(output_dir, 'features.npy'), data.x.numpy())
    np.save(os.path.join(output_dir, 'labels.npy'), data.y.numpy())
    
    # Save splits
    for split in split_idx:
        np.save(os.path.join(output_dir, f'{split}_idx.npy'), split_idx[split])
    
    logger.info(f"Successfully saved processed data to {output_dir}")

def load_processed_data(input_dir='data/processed'):
    """
    Load processed data from disk.
    
    Parameters:
    -----------
    input_dir : str
        Directory with processed data
        
    Returns:
    --------
    data : torch_geometric.data.Data
        PyTorch Geometric Data object
    split_idx : dict
        Dictionary containing indices for train/val/test splits
    """
    logger.info(f"Loading processed data from {input_dir}")
    
    # Load data object
    data_path = os.path.join(input_dir, 'data.pt')
    
    try:
        # Method 1: Try loading with weights_only=False
        data = torch.load(data_path, weights_only=False)
    except:
        # Method 2: If that fails, try to reconstruct the data object from components
        logger.warning("Failed to load data directly, reconstructing from components")
        features_path = os.path.join(input_dir, 'features.npy')
        labels_path = os.path.join(input_dir, 'labels.npy')
        edge_index_path = os.path.join(input_dir, 'edge_index.pt')
        
        if not all(os.path.exists(p) for p in [features_path, labels_path, edge_index_path]):
            raise FileNotFoundError(f"Missing required files in {input_dir}")
        
        features = torch.FloatTensor(np.load(features_path))
        labels = torch.LongTensor(np.load(labels_path))
        edge_index = torch.load(edge_index_path, weights_only=False)
        
        from torch_geometric.data import Data
        data = Data(x=features, edge_index=edge_index, y=labels)
    
    # Load splits
    split_idx = {}
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(input_dir, f'{split}_idx.npy')
        if os.path.exists(split_path):
            split_idx[split] = np.load(split_path)
    
    logger.info(f"Successfully loaded processed data from {input_dir}")
    
    return data, split_idx
def main():
    """
    Main function to process the blockchain dataset.
    """
    # Paths
    classes_path = 'data/raw/classes.csv'
    edgelist_path = 'data/raw/edgelist.csv'
    features_path = 'data/raw/Features.csv'
    output_dir = 'data/processed'
    
    # Make sure directories exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if files exist
    files_exist = all(os.path.exists(p) for p in [classes_path, edgelist_path, features_path])
    
    if not files_exist:
        logger.error("Required input files not found. Please place them in the data/raw directory.")
        return
    
    try:
        # Load data
        df_nodes, df_edges = load_data(classes_path, edgelist_path, features_path)
        
        # Preprocess data
        df_processed, df_edges = preprocess_data(df_nodes, df_edges)
        
        # Create node mapping
        id2idx = create_node_mapping(df_processed)
        
        # Build edge index
        edge_index = build_edge_index(df_edges, id2idx)
        
        # Create data splits and PyTorch Geometric Data object
        data, split_idx = create_data_splits(df_processed, edge_index)
        
        # Save processed data
        save_processed_data(data, split_idx, output_dir)
        
        logger.info("Data preparation completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise

if __name__ == "__main__":
    main()