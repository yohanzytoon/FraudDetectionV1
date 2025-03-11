import numpy as np
import pandas as pd
import networkx as nx
import torch
from sklearn.preprocessing import StandardScaler
import logging
import time
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_fast_graph_features(df_nodes, df_edges, id2idx):
    """
    Compute optimized graph features that are fast to calculate.
    
    Parameters:
    -----------
    df_nodes : pandas.DataFrame
        Dataframe containing node data
    df_edges : pandas.DataFrame
        Dataframe containing edge data
    id2idx : dict
        Dictionary mapping node IDs to indices
        
    Returns:
    --------
    graph_features : numpy.ndarray
        Array of graph features for each node
    """
    logger.info("Computing optimized graph features (fast version)")
    start_time = time.time()
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes
    logger.info("Adding nodes to graph...")
    for node_id in tqdm(df_nodes['txId'], desc="Adding nodes"):
        G.add_node(node_id)
    
    # Add edges
    logger.info("Adding edges to graph...")
    edge_count = 0
    for _, row in tqdm(df_edges.iterrows(), desc="Adding edges", total=len(df_edges)):
        source_id, target_id = row['txId1'], row['txId2']
        if source_id in id2idx and target_id in id2idx:
            G.add_edge(source_id, target_id)
            edge_count += 1
    
    logger.info(f"Created graph with {G.number_of_nodes()} nodes and {edge_count} edges in {time.time()-start_time:.2f} seconds")
    
    # Compute centrality metrics
    logger.info("Computing centrality metrics (in-degree, out-degree, PageRank)...")
    feature_start = time.time()
    
    # Calculate degrees (fast)
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    
    logger.info(f"Calculated degree centralities in {time.time()-feature_start:.2f} seconds")
    
    # PageRank (reasonably fast)
    logger.info("Computing PageRank centrality...")
    pagerank_start = time.time()
    try:
        pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)
        logger.info(f"Calculated PageRank in {time.time()-pagerank_start:.2f} seconds")
    except nx.PowerIterationFailedConvergence:
        logger.warning("PageRank failed to converge, using simplified calculation")
        pagerank = nx.pagerank(G, alpha=0.85, max_iter=50, tol=1e-3)
        logger.info(f"Calculated simplified PageRank in {time.time()-pagerank_start:.2f} seconds")
    
    # Create feature matrix with 3 features (in-degree, out-degree, pagerank)
    logger.info("Creating feature matrix...")
    graph_features = np.zeros((len(df_nodes), 3))
    
    logger.info("Filling feature matrix...")
    for i, node_id in tqdm(enumerate(df_nodes['txId']), desc="Processing nodes", total=len(df_nodes)):
        # In-degree
        graph_features[i, 0] = in_degree.get(node_id, 0)
        # Out-degree
        graph_features[i, 1] = out_degree.get(node_id, 0)
        # PageRank
        graph_features[i, 2] = pagerank.get(node_id, 0)
    
    logger.info(f"Generated graph features with shape {graph_features.shape} in {time.time()-start_time:.2f} seconds")
    
    return graph_features

def compute_clustering_coefficients(df_nodes, G):
    """
    Compute clustering coefficients for nodes in an efficient way.
    
    Parameters:
    -----------
    df_nodes : pandas.DataFrame
        Dataframe containing node data
    G : networkx.Graph
        NetworkX graph
        
    Returns:
    --------
    clustering_features : numpy.ndarray
        Array of clustering coefficients for each node
    """
    logger.info("Computing clustering coefficients...")
    start_time = time.time()
    
    # Convert to undirected for clustering coefficient calculation
    G_undirected = G.to_undirected()
    
    # Initialize features
    clustering_features = np.zeros((len(df_nodes), 1))
    
    # Calculate clustering coefficients for all nodes at once (more efficient)
    clustering_dict = nx.clustering(G_undirected)
    
    # Fill feature array
    for i, node_id in tqdm(enumerate(df_nodes['txId']), desc="Processing clustering", total=len(df_nodes)):
        clustering_features[i, 0] = clustering_dict.get(node_id, 0)
    
    logger.info(f"Generated clustering features in {time.time()-start_time:.2f} seconds")
    
    return clustering_features

def extract_temporal_features(df_nodes):
    """
    Extract temporal features if time-related columns are available.
    
    Parameters:
    -----------
    df_nodes : pandas.DataFrame
        Dataframe containing node data
        
    Returns:
    --------
    temporal_features : numpy.ndarray or None
        Array of temporal features, or None if no temporal data exists
    """
    # Check if time-related columns exist
    time_columns = [col for col in df_nodes.columns if 'time' in col.lower()]
    
    if not time_columns:
        logger.info("No temporal features found in the dataset")
        return None
    
    logger.info(f"Extracting temporal features from columns: {time_columns}")
    start_time = time.time()
    
    # Extract temporal features
    temporal_features = df_nodes[time_columns].values
    
    logger.info(f"Extracted temporal features in {time.time()-start_time:.2f} seconds")
    
    return temporal_features

def combine_features(transaction_features, graph_features, clustering_features=None, temporal_features=None, normalize=True):
    """
    Combine different feature sets and optionally normalize them.
    
    Parameters:
    -----------
    transaction_features : numpy.ndarray
        Original transaction features
    graph_features : numpy.ndarray
        Graph-based features
    clustering_features : numpy.ndarray or None
        Clustering coefficient features
    temporal_features : numpy.ndarray or None
        Temporal features, if available
    normalize : bool
        Whether to normalize the combined features
        
    Returns:
    --------
    combined_features : numpy.ndarray
        Combined and normalized features
    """
    logger.info("Combining feature sets...")
    start_time = time.time()
    
    feature_list = [transaction_features, graph_features]
    feature_types = ["transaction", "graph"]
    feature_counts = [transaction_features.shape[1], graph_features.shape[1]]
    
    if clustering_features is not None:
        feature_list.append(clustering_features)
        feature_types.append("clustering")
        feature_counts.append(clustering_features.shape[1])
    
    if temporal_features is not None:
        feature_list.append(temporal_features)
        feature_types.append("temporal")
        feature_counts.append(temporal_features.shape[1])
    
    # Combine features
    combined_features = np.hstack(feature_list)
    
    # Create a description of the feature combination
    feature_desc = ", ".join([f"{ftype} ({fcount})" for ftype, fcount in zip(feature_types, feature_counts)])
    logger.info(f"Combined {feature_desc} features")
    
    # Normalize features
    if normalize:
        logger.info("Normalizing combined features...")
        scaler = StandardScaler()
        combined_features = scaler.fit_transform(combined_features)
    
    logger.info(f"Final combined features shape: {combined_features.shape}, completed in {time.time()-start_time:.2f} seconds")
    
    return combined_features

def get_feature_importance(features, labels, method='correlation'):
    """
    Calculate feature importance using the specified method.
    
    Parameters:
    -----------
    features : numpy.ndarray
        Feature matrix
    labels : numpy.ndarray
        Target labels
    method : str
        Method to calculate importance ('correlation', 'mutual_info', etc.)
        
    Returns:
    --------
    feature_importance : numpy.ndarray
        Array of importance scores for each feature
    """
    logger.info(f"Calculating feature importance using {method}...")
    start_time = time.time()
    
    if method == 'correlation':
        # Calculate absolute correlation with target
        correlations = np.zeros(features.shape[1])
        
        for i in tqdm(range(features.shape[1]), desc="Calculating correlations"):
            correlations[i] = abs(np.corrcoef(features[:, i], labels)[0, 1])
        
        logger.info(f"Calculated feature importance in {time.time()-start_time:.2f} seconds")
        return correlations
    
    elif method == 'mutual_info':
        from sklearn.feature_selection import mutual_info_classif
        
        # Calculate mutual information
        logger.info("Computing mutual information...")
        importance = mutual_info_classif(features, labels)
        
        logger.info(f"Calculated feature importance in {time.time()-start_time:.2f} seconds")
        return importance
    
    else:
        logger.warning(f"Unknown importance method: {method}, using correlation")
        return get_feature_importance(features, labels, method='correlation')

def main():
    """
    Main function to extract and combine features.
    """
    import os
    from data_preparation import load_data, create_node_mapping
    
    overall_start = time.time()
    
    # Paths
    classes_path = 'data/raw/classes.csv'
    edgelist_path = 'data/raw/edgelist.csv'
    features_path = 'data/raw/Features.csv'
    output_dir = 'data/processed'
    
    # Load data
    logger.info("Loading data...")
    df_nodes, df_edges = load_data(classes_path, edgelist_path, features_path)
    
    # Filter out unknown classes
    unknown_count = (df_nodes['class'] == 'unknown').sum()
    if unknown_count > 0:
        logger.info(f"Removing {unknown_count} nodes with unknown class")
        df_nodes = df_nodes[df_nodes['class'] != 'unknown']
    
    # Convert class labels to numeric if needed
    if df_nodes['class'].dtype == 'object':
        unique_classes = df_nodes['class'].unique()
        class_map = {cls: i for i, cls in enumerate(unique_classes)}
        df_nodes['class'] = df_nodes['class'].map(class_map)
        logger.info(f"Mapped class values to integers: {class_map}")
    
    # Create node mapping
    logger.info("Creating node mapping...")
    id2idx = create_node_mapping(df_nodes)
    
    # Extract transaction features
    feature_cols = [col for col in df_nodes.columns if col not in ['txId', 'class']]
    transaction_features = df_nodes[feature_cols].values
    logger.info(f"Extracted {transaction_features.shape[1]} transaction features")
    
    # Create NetworkX graph for all graph-based features
    logger.info("Creating graph...")
    G = nx.DiGraph()
    
    # Add nodes
    for node_id in df_nodes['txId']:
        G.add_node(node_id)
    
    # Add edges
    edge_count = 0
    for _, row in df_edges.iterrows():
        source_id, target_id = row['txId1'], row['txId2']
        if source_id in id2idx and target_id in id2idx:
            G.add_edge(source_id, target_id)
            edge_count += 1
    
    logger.info(f"Created graph with {G.number_of_nodes()} nodes and {edge_count} edges")
    
    # Compute graph features (fast version)
    graph_features = compute_fast_graph_features(df_nodes, df_edges, id2idx)
    
    # Compute clustering coefficients (optional - can be somewhat slow)
    # Comment out this block if you want it to run even faster
    try:
        logger.info("Computing clustering coefficients...")
        clustering_features = compute_clustering_coefficients(df_nodes, G)
    except Exception as e:
        logger.warning(f"Error computing clustering coefficients: {str(e)}. Skipping.")
        clustering_features = None
    
    # Extract temporal features if available
    temporal_features = extract_temporal_features(df_nodes)
    
    # Combine features
    combined_features = combine_features(
        transaction_features, 
        graph_features, 
        clustering_features,
        temporal_features
    )
    
    # Save features
    logger.info("Saving features...")
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'combined_features.npy'), combined_features)
    
    # Get feature importance
    labels = df_nodes['class'].values
    importance = get_feature_importance(combined_features, labels)
    np.save(os.path.join(output_dir, 'feature_importance.npy'), importance)
    
    # Save feature names
    feature_names = feature_cols.copy()
    feature_names.extend(['in_degree', 'out_degree', 'pagerank'])
    
    if clustering_features is not None:
        feature_names.append('clustering_coefficient')
    
    if temporal_features is not None:
        time_columns = [col for col in df_nodes.columns if 'time' in col.lower()]
        feature_names.extend(time_columns)
    
    with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    
    total_time = time.time() - overall_start
    logger.info(f"Feature extraction completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

if __name__ == "__main__":
    main()