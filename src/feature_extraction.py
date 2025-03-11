import numpy as np
import pandas as pd
import networkx as nx
import torch
from sklearn.preprocessing import StandardScaler
import logging
import time
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_graph_features(df_nodes, df_edges, id2idx, use_parallel=True, n_jobs=-1):
    """
    Compute comprehensive graph features.
    
    Parameters:
    -----------
    df_nodes : pandas.DataFrame
        Dataframe containing node data
    df_edges : pandas.DataFrame
        Dataframe containing edge data
    id2idx : dict
        Dictionary mapping node IDs to indices
    use_parallel : bool
        Whether to use parallelization for computation
    n_jobs : int
        Number of jobs for parallel processing (-1 for all available cores)
        
    Returns:
    --------
    graph_features : numpy.ndarray
        Array of graph features for each node
    """
    logger.info("Computing enhanced graph features")
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
    logger.info("Computing basic centrality metrics...")
    feature_start = time.time()
    
    # Calculate degrees (fast)
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    
    # Total degree
    total_degree = {node: in_degree.get(node, 0) + out_degree.get(node, 0) for node in G.nodes()}
    
    logger.info(f"Calculated degree centralities in {time.time()-feature_start:.2f} seconds")
    
    # PageRank with different damping factors
    logger.info("Computing PageRank with different damping factors...")
    pagerank_start = time.time()
    
    try:
        # Standard PageRank
        pagerank_085 = nx.pagerank(G, alpha=0.85, max_iter=100)
        # Lower damping factor (gives more weight to direct connections)
        pagerank_050 = nx.pagerank(G, alpha=0.50, max_iter=100)
        logger.info(f"Calculated PageRank variants in {time.time()-pagerank_start:.2f} seconds")
    except nx.PowerIterationFailedConvergence:
        logger.warning("PageRank failed to converge, using simplified calculation")
        pagerank_085 = nx.pagerank(G, alpha=0.85, max_iter=50, tol=1e-3)
        pagerank_050 = nx.pagerank(G, alpha=0.50, max_iter=50, tol=1e-3)
        logger.info(f"Calculated simplified PageRank variants in {time.time()-pagerank_start:.2f} seconds")
    
    # Compute local clustering coefficient for undirected version of the graph
    logger.info("Computing clustering coefficient...")
    G_undirected = G.to_undirected()
    
    if use_parallel and len(G_undirected) > 1000:
        # For large graphs, compute clustering coefficients for chunks of nodes in parallel
        logger.info("Using parallel processing for clustering coefficient...")
        
        def compute_node_clustering(node):
            try:
                neighbors = list(G_undirected.neighbors(node))
                if len(neighbors) < 2:
                    return node, 0.0
                
                # Count edges between neighbors
                edge_count = 0
                for i, n1 in enumerate(neighbors):
                    for n2 in neighbors[i+1:]:
                        if G_undirected.has_edge(n1, n2):
                            edge_count += 1
                
                max_possible = (len(neighbors) * (len(neighbors) - 1)) / 2
                return node, edge_count / max_possible if max_possible > 0 else 0.0
            except Exception as e:
                logger.warning(f"Error computing clustering for node {node}: {str(e)}")
                return node, 0.0
        
        # Use joblib to parallelize
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_node_clustering)(node) for node in tqdm(G_undirected.nodes(), desc="Computing clustering")
        )
        
        clustering = dict(results)
    else:
        # For smaller graphs, use built-in NetworkX function
        clustering = nx.clustering(G_undirected)
        
    logger.info(f"Calculated clustering coefficient in {time.time()-feature_start:.2f} seconds")
    
    # Compute HITS algorithm (hubs and authorities)
    logger.info("Computing HITS (hubs and authorities)...")
    hits_start = time.time()
    
    try:
        hubs, authorities = nx.hits(G, max_iter=100)
        logger.info(f"Calculated HITS in {time.time()-hits_start:.2f} seconds")
    except nx.PowerIterationFailedConvergence:
        logger.warning("HITS failed to converge, using simplified calculation")
        hubs, authorities = nx.hits(G, max_iter=50, tol=1e-3)
        logger.info(f"Calculated simplified HITS in {time.time()-hits_start:.2f} seconds")
    
    # Compute k-core decomposition on undirected graph
    logger.info("Computing k-core decomposition...")
    try:
        core_numbers = nx.core_number(G_undirected)
        logger.info("Calculated k-core decomposition")
    except Exception as e:
        logger.warning(f"Error computing k-core: {str(e)}. Using default values.")
        core_numbers = {node: 0 for node in G.nodes()}
    
    # Compute local efficiency for important nodes
    logger.info("Computing local efficiency for important nodes...")
    efficiency_start = time.time()
    
    # Only compute for nodes with high PageRank (top 10%) to save time
    pagerank_threshold = np.percentile(list(pagerank_085.values()), 90)
    important_nodes = [node for node in G.nodes() if pagerank_085.get(node, 0) >= pagerank_threshold]
    
    local_efficiency = {}
    
    # Define a helper function to compute efficiency for one node
    def compute_local_efficiency(node):
        try:
            # Extract ego network (excluding the node itself)
            neighbors = list(G.neighbors(node))
            if len(neighbors) < 2:
                return node, 0.0
                
            ego = nx.ego_graph(G, node, radius=1)
            ego.remove_node(node)
            
            if len(ego) < 2:
                return node, 0.0
                
            # Compute efficiency
            return node, nx.global_efficiency(ego)
        except Exception as e:
            logger.warning(f"Error computing efficiency for node {node}: {str(e)}")
            return node, 0.0
    
    # Compute efficiency in parallel if requested
    if use_parallel and len(important_nodes) > 100:
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_local_efficiency)(node) for node in tqdm(important_nodes, desc="Computing efficiency")
        )
        local_efficiency_important = dict(results)
    else:
        local_efficiency_important = {}
        for node in tqdm(important_nodes, desc="Computing efficiency"):
            node, eff = compute_local_efficiency(node)
            local_efficiency_important[node] = eff
    
    # Fill in values for remaining nodes
    local_efficiency = {node: local_efficiency_important.get(node, 0.0) for node in G.nodes()}
    
    logger.info(f"Calculated local efficiency in {time.time()-efficiency_start:.2f} seconds")
    
    # Create advanced graph features for each transaction
    logger.info("Computing neighborhood features...")
    neighborhood_start = time.time()
    
    # Precompute some features to speed up neighborhood calculations
    node_in_degree_ratio = {}
    for node in G.nodes():
        total = in_degree.get(node, 0) + out_degree.get(node, 0)
        node_in_degree_ratio[node] = in_degree.get(node, 0) / total if total > 0 else 0.0
    
    # Neighborhood features
    def compute_neighborhood_features(node):
        try:
            # Get in and out neighbors
            in_neighbors = list(G.predecessors(node))
            out_neighbors = list(G.successors(node))
            
            # Feature 1: Ratio of in-neighbors to total neighbors
            total_neighbors = len(in_neighbors) + len(out_neighbors)
            in_ratio = len(in_neighbors) / total_neighbors if total_neighbors > 0 else 0
            
            # Feature 2: Average PageRank of neighbors
            neighbor_pageranks = [pagerank_085.get(n, 0) for n in in_neighbors + out_neighbors]
            avg_neighbor_pagerank = np.mean(neighbor_pageranks) if neighbor_pageranks else 0
            
            # Feature 3: Standard deviation of neighbor degrees
            neighbor_degrees = [total_degree.get(n, 0) for n in in_neighbors + out_neighbors]
            std_neighbor_degree = np.std(neighbor_degrees) if len(neighbor_degrees) > 1 else 0
            
            # Feature 4: Ratio of neighbors with higher PageRank
            node_pagerank = pagerank_085.get(node, 0)
            higher_pagerank_count = sum(1 for n_pr in neighbor_pageranks if n_pr > node_pagerank)
            higher_pagerank_ratio = higher_pagerank_count / len(neighbor_pageranks) if neighbor_pageranks else 0
            
            # Feature 5: Average in-degree ratio of neighbors
            avg_neighbor_in_ratio = np.mean([node_in_degree_ratio.get(n, 0) for n in in_neighbors + out_neighbors]) if total_neighbors > 0 else 0
            
            return node, (in_ratio, avg_neighbor_pagerank, std_neighbor_degree, higher_pagerank_ratio, avg_neighbor_in_ratio)
        except Exception as e:
            logger.warning(f"Error computing neighborhood features for node {node}: {str(e)}")
            return node, (0, 0, 0, 0, 0)
    
    # Compute neighborhood features in parallel if requested
    if use_parallel and len(G) > 1000:
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_neighborhood_features)(node) for node in tqdm(G.nodes(), desc="Computing neighborhood features")
        )
        neighborhood_features = dict(results)
    else:
        neighborhood_features = {}
        for node in tqdm(G.nodes(), desc="Computing neighborhood features"):
            node, features = compute_neighborhood_features(node)
            neighborhood_features[node] = features
    
    logger.info(f"Calculated neighborhood features in {time.time()-neighborhood_start:.2f} seconds")
    
    # Create feature matrix
    logger.info("Creating feature matrix...")
    # We now have 13 graph features per node
    graph_features = np.zeros((len(df_nodes), 13))
    
    logger.info("Filling feature matrix...")
    for i, node_id in tqdm(enumerate(df_nodes['txId']), desc="Processing nodes", total=len(df_nodes)):
        # Basic centrality metrics (3)
        graph_features[i, 0] = in_degree.get(node_id, 0)
        graph_features[i, 1] = out_degree.get(node_id, 0)
        graph_features[i, 2] = total_degree.get(node_id, 0)
        
        # PageRank variations (2)
        graph_features[i, 3] = pagerank_085.get(node_id, 0)
        graph_features[i, 4] = pagerank_050.get(node_id, 0)
        
        # HITS scores (2)
        graph_features[i, 5] = hubs.get(node_id, 0)
        graph_features[i, 6] = authorities.get(node_id, 0)
        
        # Clustering coefficient (1)
        graph_features[i, 7] = clustering.get(node_id, 0)
        
        # K-core decomposition (1)
        graph_features[i, 8] = core_numbers.get(node_id, 0)
        
        # Local efficiency (1)
        graph_features[i, 9] = local_efficiency.get(node_id, 0)
        
        # Neighborhood features (5)
        if node_id in neighborhood_features:
            nf = neighborhood_features[node_id]
            graph_features[i, 10] = nf[0]  # in-neighbor ratio
            graph_features[i, 11] = nf[1]  # avg neighbor PageRank
            graph_features[i, 12] = nf[2]  # std of neighbor degrees
            # We'll exclude features 3 and 4 to keep total at 13
            # graph_features[i, 13] = nf[3]  # higher PageRank ratio
            # graph_features[i, 14] = nf[4]  # avg neighbor in-degree ratio
    
    logger.info(f"Generated graph features with shape {graph_features.shape} in {time.time()-start_time:.2f} seconds")
    
    return graph_features

def extract_temporal_sequences(df_nodes, df_edges, id2idx, sequence_length=5):
    """
    Extract temporal sequence features if time information is available.
    
    Parameters:
    -----------
    df_nodes : pandas.DataFrame
        Dataframe containing node data
    df_edges : pandas.DataFrame
        Dataframe containing edge data
    id2idx : dict
        Dictionary mapping node IDs to indices
    sequence_length : int
        Length of temporal sequences to extract
        
    Returns:
    --------
    sequence_features : numpy.ndarray or None
        Array of sequence features, or None if no temporal data exists
    """
    # Check if time-related columns exist
    time_columns = [col for col in df_nodes.columns if 'time' in col.lower()]
    
    if not time_columns:
        logger.info("No temporal features found in the dataset")
        return None
    
    logger.info(f"Extracting temporal sequence features from columns: {time_columns}")
    start_time = time.time()
    
    # Sort transactions by time if possible
    time_col = time_columns[0]  # Use the first time column
    
    try:
        # Convert time to datetime if possible
        df_nodes['timestamp'] = pd.to_datetime(df_nodes[time_col])
        sorted_nodes = df_nodes.sort_values('timestamp')
    except:
        logger.warning("Could not convert time column to datetime. Using original values for ordering.")
        sorted_nodes = df_nodes.sort_values(time_col)
    
    # Get sorted transaction IDs
    sorted_txids = sorted_nodes['txId'].values
    
    # Initialize sequence features
    sequence_features = np.zeros((len(df_nodes), sequence_length * 3))  # 3 features per sequence step
    
    # Create a graph for traversal
    G = nx.DiGraph()
    for _, row in df_edges.iterrows():
        source_id, target_id = row['txId1'], row['txId2']
        if source_id in id2idx and target_id in id2idx:
            G.add_edge(source_id, target_id)
    
    # Extract features for each node
    for i, node_id in tqdm(enumerate(df_nodes['txId']), desc="Extracting sequences", total=len(df_nodes)):
        # Find position in time sequence
        try:
            node_idx = np.where(sorted_txids == node_id)[0][0]
        except:
            node_idx = -1
        
        # Get previous transactions in the sequence
        seq_features = np.zeros(sequence_length * 3)
        
        for j in range(sequence_length):
            if node_idx - j - 1 >= 0:
                prev_id = sorted_txids[node_idx - j - 1]
                
                # Feature 1: Is there a direct edge from previous to current?
                seq_features[j*3] = 1 if G.has_edge(prev_id, node_id) else 0
                
                # Feature 2: Is there a direct edge from current to previous?
                seq_features[j*3 + 1] = 1 if G.has_edge(node_id, prev_id) else 0
                
                # Feature 3: Temporal distance (if timestamps are available)
                try:
                    if 'timestamp' in df_nodes.columns:
                        current_time = df_nodes.loc[df_nodes['txId'] == node_id, 'timestamp'].values[0]
                        prev_time = df_nodes.loc[df_nodes['txId'] == prev_id, 'timestamp'].values[0]
                        time_diff = (current_time - prev_time).total_seconds()
                        seq_features[j*3 + 2] = np.log1p(abs(time_diff)) if time_diff > 0 else 0
                except:
                    seq_features[j*3 + 2] = 0
        
        sequence_features[i] = seq_features
    
    logger.info(f"Extracted temporal sequence features in {time.time()-start_time:.2f} seconds")
    
    return sequence_features

def create_feature_interactions(transaction_features, graph_features):
    """
    Create interaction features between transaction and graph features.
    
    Parameters:
    -----------
    transaction_features : numpy.ndarray
        Transaction features
    graph_features : numpy.ndarray
        Graph features
        
    Returns:
    --------
    interaction_features : numpy.ndarray
        Interaction features
    """
    logger.info("Creating feature interactions...")
    start_time = time.time()
    
    # Select subset of transaction features for interactions to avoid explosion
    n_trans = min(5, transaction_features.shape[1])
    n_graph = min(5, graph_features.shape[1])
    
    # Get top features by variance
    trans_var = np.var(transaction_features, axis=0)
    graph_var = np.var(graph_features, axis=0)
    
    top_trans_idx = np.argsort(trans_var)[-n_trans:]
    top_graph_idx = np.argsort(graph_var)[-n_graph:]
    
    trans_subset = transaction_features[:, top_trans_idx]
    graph_subset = graph_features[:, top_graph_idx]
    
    # Create interactions (products of features)
    interactions = []
    
    for i in range(n_trans):
        for j in range(n_graph):
            # Multiplicative interaction
            interactions.append(trans_subset[:, i] * graph_subset[:, j])
            
            # Ratio interaction (with safeguards)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = trans_subset[:, i] / (graph_subset[:, j] + 1e-10)
                ratio[~np.isfinite(ratio)] = 0
            interactions.append(ratio)
    
    # Stack interactions
    interaction_features = np.column_stack(interactions)
    
    # Handle NaN or infinite values
    interaction_features = np.nan_to_num(interaction_features)
    
    logger.info(f"Created {interaction_features.shape[1]} interaction features in {time.time()-start_time:.2f} seconds")
    
    return interaction_features

def select_features(features, labels, k=100, method='mutual_info'):
    """
    Select the most important features using various feature selection methods.
    
    Parameters:
    -----------
    features : numpy.ndarray
        Feature matrix
    labels : numpy.ndarray
        Target labels
    k : int
        Number of features to select
    method : str
        Feature selection method
        
    Returns:
    --------
    selected_features : numpy.ndarray
        Selected features
    """
    logger.info(f"Selecting top {k} features using {method}...")
    start_time = time.time()
    
    # Make sure k is not larger than the number of features
    k = min(k, features.shape[1])
    
    if method == 'mutual_info':
        # Use mutual information for classification
        selector = SelectKBest(mutual_info_classif, k=k)
        selected_features = selector.fit_transform(features, labels)
        selected_indices = selector.get_support(indices=True)
        
    elif method == 'pca':
        # Use PCA for dimensionality reduction
        selector = PCA(n_components=k, random_state=42)
        selected_features = selector.fit_transform(features)
        selected_indices = None  # PCA creates new features
        
    else:
        logger.warning(f"Unknown feature selection method: {method}. Using all features.")
        selected_features = features
        selected_indices = np.arange(features.shape[1])
    
    logger.info(f"Selected {selected_features.shape[1]} features in {time.time()-start_time:.2f} seconds")
    
    return selected_features, selected_indices

def combine_features(transaction_features, graph_features, 
                   sequence_features=None, interaction_features=None, 
                   normalize=True, feature_selection=True, k=100):
    """
    Combine different feature sets and optionally normalize them.
    
    Parameters:
    -----------
    transaction_features : numpy.ndarray
        Original transaction features
    graph_features : numpy.ndarray
        Graph-based features
    sequence_features : numpy.ndarray or None
        Temporal sequence features
    interaction_features : numpy.ndarray or None
        Interaction features
    normalize : bool
        Whether to normalize the combined features
    feature_selection : bool
        Whether to perform feature selection
    k : int
        Number of features to select if feature_selection is True
        
    Returns:
    --------
    combined_features : numpy.ndarray
        Combined and possibly selected features
    """
    logger.info("Combining feature sets...")
    start_time = time.time()
    
    feature_list = [transaction_features, graph_features]
    feature_types = ["transaction", "graph"]
    feature_counts = [transaction_features.shape[1], graph_features.shape[1]]
    
    if sequence_features is not None:
        feature_list.append(sequence_features)
        feature_types.append("sequence")
        feature_counts.append(sequence_features.shape[1])
    
    if interaction_features is not None:
        feature_list.append(interaction_features)
        feature_types.append("interaction")
        feature_counts.append(interaction_features.shape[1])
    
    # Combine features
    combined_features = np.hstack(feature_list)
    
    # Create a description of the feature combination
    feature_desc = ", ".join([f"{ftype} ({fcount})" for ftype, fcount in zip(feature_types, feature_counts)])
    logger.info(f"Combined {feature_desc} features with total shape {combined_features.shape}")
    
    # Normalize features
    if normalize:
        logger.info("Normalizing combined features...")
        scaler = StandardScaler()
        combined_features = scaler.fit_transform(combined_features)
    
    logger.info(f"Combined features shape: {combined_features.shape}, completed in {time.time()-start_time:.2f} seconds")
    
    return combined_features

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
    
    # Compute enhanced graph features
    graph_features = compute_graph_features(df_nodes, df_edges, id2idx)
    
    # Extract temporal sequence features if available
    sequence_features = extract_temporal_sequences(df_nodes, df_edges, id2idx)
    
    # Create feature interactions
    interaction_features = create_feature_interactions(transaction_features, graph_features)
    
    # Combine all features
    combined_features = combine_features(
        transaction_features, 
        graph_features, 
        sequence_features,
        interaction_features,
        normalize=True
    )
    
    # Get labels
    labels = df_nodes['class'].values
    
    # Perform feature selection
    selected_features, selected_indices = select_features(
        combined_features, 
        labels, 
        k=min(100, combined_features.shape[1])
    )
    
    # Save features
    logger.info("Saving features...")
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'transaction_features.npy'), transaction_features)
    np.save(os.path.join(output_dir, 'graph_features.npy'), graph_features)
    
    if sequence_features is not None:
        np.save(os.path.join(output_dir, 'sequence_features.npy'), sequence_features)
    
    if interaction_features is not None:
        np.save(os.path.join(output_dir, 'interaction_features.npy'), interaction_features)
    
    np.save(os.path.join(output_dir, 'combined_features.npy'), combined_features)
    np.save(os.path.join(output_dir, 'selected_features.npy'), selected_features)
    
    if selected_indices is not None:
        np.save(os.path.join(output_dir, 'selected_indices.npy'), selected_indices)
    
    # Get feature importance
    try:
        from sklearn.feature_selection import mutual_info_classif
        logger.info("Calculating feature importance...")
        importance = mutual_info_classif(combined_features, labels)
        np.save(os.path.join(output_dir, 'feature_importance.npy'), importance)
        
        # Save top feature indices
        top_indices = np.argsort(importance)[::-1][:100]
        np.save(os.path.join(output_dir, 'top_feature_indices.npy'), top_indices)
    except Exception as e:
        logger.warning(f"Error calculating feature importance: {str(e)}")
    
    # Save feature names and descriptions
    feature_names = feature_cols.copy()
    
    # Add graph feature names
    graph_feature_names = [
        'in_degree', 'out_degree', 'total_degree',
        'pagerank_085', 'pagerank_050',
        'hub_score', 'authority_score',
        'clustering_coefficient',
        'kcore_number',
        'local_efficiency',
        'in_neighbor_ratio',
        'avg_neighbor_pagerank',
        'std_neighbor_degree'
    ]
    feature_names.extend(graph_feature_names)
    
    # Add sequence feature names if available
    if sequence_features is not None:
        sequence_length = sequence_features.shape[1] // 3
        for i in range(sequence_length):
            feature_names.extend([
                f'seq_{i+1}_direct_edge_from_prev',
                f'seq_{i+1}_direct_edge_to_prev',
                f'seq_{i+1}_time_diff'
            ])
    
    # Add interaction feature names if available
    if interaction_features is not None:
        n_trans = min(5, transaction_features.shape[1])
        n_graph = min(5, graph_features.shape[1])
        
        for i in range(n_trans):
            for j in range(n_graph):
                feature_names.append(f'interaction_prod_{i}_{j}')
                feature_names.append(f'interaction_ratio_{i}_{j}')
    
    with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    
    total_time = time.time() - overall_start
    logger.info(f"Feature extraction completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

if __name__ == "__main__":
    main()