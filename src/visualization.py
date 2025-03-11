import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def visualize_graph(G, node_colors=None, node_sizes=None, 
                   node_labels=None, title="Transaction Graph", 
                   output_path=None, figsize=(12, 10)):
    """
    Visualize a transaction graph.
    
    Parameters:
    -----------
    G : networkx.Graph
        The graph to visualize
    node_colors : list, optional
        Colors for each node
    node_sizes : list, optional
        Sizes for each node
    node_labels : dict, optional
        Labels for each node
    title : str
        Plot title
    output_path : str, optional
        Path to save the visualization
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    # Set default node colors and sizes if not provided
    if node_colors is None:
        node_colors = 'skyblue'
    
    if node_sizes is None:
        node_sizes = 50
    
    # Create layout
    if len(G) > 1000:
        logger.info("Large graph detected, using fast layout algorithm")
        pos = nx.spring_layout(G, k=0.3, iterations=20, seed=42)
    else:
        logger.info("Using force-directed layout")
        pos = nx.spring_layout(G, k=0.3, iterations=100, seed=42)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)
    
    if node_labels:
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, ax=ax)
    
    # Set title and remove axis
    plt.title(title, fontsize=16)
    plt.axis('off')
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Graph visualization saved to {output_path}")
    
    return fig

def visualize_embeddings(embeddings, labels, method='tsne', 
                        title=None, output_path=None, figsize=(10, 8)):
    """
    Visualize node embeddings in 2D using dimensionality reduction.
    
    Parameters:
    -----------
    embeddings : numpy.ndarray
        Node embeddings matrix
    labels : numpy.ndarray
        Node labels
    method : str
        Dimensionality reduction method ('tsne' or 'pca')
    title : str, optional
        Plot title
    output_path : str, optional
        Path to save the visualization
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    # Apply dimensionality reduction
    if method.lower() == 'tsne':
        logger.info("Applying t-SNE for dimensionality reduction")
        reducer = TSNE(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
        method_name = "t-SNE"
    elif method.lower() == 'pca':
        logger.info("Applying PCA for dimensionality reduction")
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
        method_name = "PCA"
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'label': labels
    })
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Map labels to colors and names
    unique_labels = np.unique(labels)
    label_names = {0: 'Legitimate', 1: 'Fraudulent'}
    colors = ['#4285F4', '#EA4335']  # Blue for legitimate, red for fraudulent
    
    # Plot each class
    for i, label in enumerate(unique_labels):
        mask = df['label'] == label
        plt.scatter(
            df.loc[mask, 'x'], 
            df.loc[mask, 'y'], 
            c=colors[i] if i < len(colors) else None,
            label=label_names.get(label, f"Class {label}"),
            alpha=0.7,
            edgecolors='w',
            s=50
        )
    
    # Set title and labels
    if title is None:
        title = f"Node Embeddings Visualization using {method_name}"
    plt.title(title, fontsize=15)
    plt.xlabel(f"{method_name} Dimension 1", fontsize=12)
    plt.ylabel(f"{method_name} Dimension 2", fontsize=12)
    
    # Add legend
    plt.legend(fontsize=12)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Embeddings visualization saved to {output_path}")
    
    return fig

def visualize_feature_importance(feature_importance, feature_names=None, 
                               top_n=20, title="Feature Importance", 
                               output_path=None, figsize=(12, 8)):
    """
    Visualize feature importance.
    
    Parameters:
    -----------
    feature_importance : numpy.ndarray
        Feature importance scores
    feature_names : list, optional
        Names of features
    top_n : int
        Number of top features to display
    title : str
        Plot title
    output_path : str, optional
        Path to save the visualization
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    # Create feature indices and names
    n_features = len(feature_importance)
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(n_features)]
    
    # Sort features by importance
    indices = np.argsort(feature_importance)[::-1]
    
    # Select top N features
    top_indices = indices[:top_n]
    top_importances = feature_importance[top_indices]
    top_names = [feature_names[i] for i in top_indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar chart
    bars = ax.barh(range(len(top_indices)), top_importances, align='center', color='skyblue')
    
    # Add feature names as y-axis labels
    ax.set_yticks(range(len(top_indices)))
    ax.set_yticklabels(top_names)
    
    # Add values to the bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{width:.4f}", ha='left', va='center')
    
    # Set title and labels
    ax.set_title(title, fontsize=15)
    ax.set_xlabel('Importance Score', fontsize=12)
    
    # Invert y-axis to show most important at the top
    ax.invert_yaxis()
    
    # Add grid
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance visualization saved to {output_path}")
    
    return fig

def visualize_training_history(history, title="Training History", 
                             output_path=None, figsize=(12, 5)):
    """
    Visualize training history (loss and accuracy).
    
    Parameters:
    -----------
    history : dict
        Training history dictionary
    title : str
        Plot title
    output_path : str, optional
        Path to save the visualization
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot training and validation loss
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Plot training and validation accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history visualization saved to {output_path}")
    
    return fig

def visualize_transaction_subgraph(df_nodes, df_edges, transaction_id, neighborhood_size=2,
                                  output_path=None, figsize=(12, 10)):
    """
    Visualize a subgraph around a specific transaction.
    
    Parameters:
    -----------
    df_nodes : pandas.DataFrame
        DataFrame containing node data
    df_edges : pandas.DataFrame
        DataFrame containing edge data
    transaction_id : int
        ID of the transaction to visualize
    neighborhood_size : int
        Size of the neighborhood to include
    output_path : str, optional
        Path to save the visualization
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    # Create full graph
    G = nx.DiGraph()
    
    # Add all edges
    for _, edge in df_edges.iterrows():
        G.add_edge(edge[0], edge[1])
    
    # Check if transaction exists
    if transaction_id not in G:
        logger.warning(f"Transaction {transaction_id} not found in the graph")
        return None
    
    # Extract subgraph around the transaction
    subgraph_nodes = set([transaction_id])
    frontier = set([transaction_id])
    
    # Expand neighborhood
    for _ in range(neighborhood_size):
        new_frontier = set()
        for node in frontier:
            # Add predecessors
            new_frontier.update(G.predecessors(node))
            # Add successors
            new_frontier.update(G.successors(node))
        
        # Update frontier and subgraph nodes
        frontier = new_frontier - subgraph_nodes
        subgraph_nodes.update(frontier)
    
    # Extract subgraph
    subgraph = G.subgraph(subgraph_nodes)
    
    # Create node attributes
    node_colors = []
    node_sizes = []
    
    # Get node classes
    node_class = {}
    for _, row in df_nodes.iterrows():
        node_id = row[0]
        node_class[node_id] = row[1]
    
    # Set node attributes
    for node in subgraph.nodes():
        if node == transaction_id:
            # Highlighted transaction
            node_colors.append('red')
            node_sizes.append(300)
        elif node in node_class and node_class[node] == 1:
            # Fraudulent transaction
            node_colors.append('orange')
            node_sizes.append(200)
        else:
            # Legitimate transaction
            node_colors.append('skyblue')
            node_sizes.append(100)
    
    # Create visualization
    title = f"Transaction {transaction_id} Neighborhood (Size {neighborhood_size})"
    fig = visualize_graph(
        subgraph, 
        node_colors=node_colors, 
        node_sizes=node_sizes, 
        title=title, 
        output_path=output_path,
        figsize=figsize
    )
    
    return fig

def visualize_fraud_patterns(model, data, split_idx, top_n=10, 
                           output_path=None, figsize=(10, 8)):
    """
    Visualize patterns in transactions classified as fraudulent.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The trained model
    data : torch_geometric.data.Data
        The graph data
    split_idx : dict
        Dictionary containing indices for splits
    top_n : int
        Number of top features to display
    output_path : str, optional
        Path to save the visualization
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    # Set model to evaluation mode
    model.eval()
    
    # Get embeddings before the final layer
    def hook_fn(module, input, output):
        global embeddings
        embeddings = input[0]
    
    # Register hook for the final layer
    for name, module in model.named_modules():
        if name == 'convs.' + str(len(model.convs) - 1):
            hook = module.register_forward_hook(hook_fn)
    
    # Forward pass to get embeddings
    with torch.no_grad():
        _ = model(data.x, data.edge_index)
    
    # Remove hook
    hook.remove()
    
    # Get test indices and corresponding embeddings
    test_idx = split_idx['test']
    test_embeddings = embeddings[test_idx].cpu().numpy()
    
    # Get predictions and true labels
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        preds = out.argmax(dim=1)
    
    test_preds = preds[test_idx].cpu().numpy()
    test_labels = data.y[test_idx].cpu().numpy()
    
    # Filter for correctly predicted fraud
    fraud_mask = (test_labels == 1) & (test_preds == 1)
    fraud_embeddings = test_embeddings[fraud_mask]
    
    # Compute feature contributions
    if len(fraud_embeddings) > 0:
        # Compute average fraud embedding
        avg_fraud_embedding = fraud_embeddings.mean(axis=0)
        
        # Compute feature contributions
        feature_contributions = np.abs(avg_fraud_embedding)
        
        # Sort features by contribution
        sorted_indices = np.argsort(feature_contributions)[::-1]
        
        # Get top N features
        top_indices = sorted_indices[:top_n]
        top_contribs = feature_contributions[top_indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar chart for feature contributions
        bars = ax.barh(range(len(top_indices)), top_contribs, align='center', color='crimson')
        
        # Add indices as y-axis labels
        ax.set_yticks(range(len(top_indices)))
        ax.set_yticklabels([f"Feature {i+1}" for i in top_indices])
        
        # Add values to the bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f"{width:.4f}", ha='left', va='center')
        
        # Set title and labels
        ax.set_title("Top Features Contributing to Fraud Detection", fontsize=15)
        ax.set_xlabel('Contribution Score', fontsize=12)
        
        # Invert y-axis to show most important at the top
        ax.invert_yaxis()
        
        # Add grid
        ax.grid(True, axis='x', linestyle='--', alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if output path is provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Fraud pattern visualization saved to {output_path}")
        
        return fig
    else:
        logger.warning("No correctly predicted fraud cases found in test set")
        return None

def main():
    """
    Main function to generate visualizations.
    """
    import torch
    import json
    from data_preparation import load_data, load_processed_data
    from gnn_model import GCNModel
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    output_dir = 'reports/figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load raw data
    df_nodes, df_edges = load_data('data/raw/nodes.csv', 'data/raw/edges.csv')
    
    # Load processed data
    data, split_idx = load_processed_data()
    
    # 1. Visualize overall graph structure
    logger.info("Generating graph structure visualization")
    G = nx.DiGraph()
    
    # Add all edges (only a subset for large graphs)
    max_edges = 5000  # Limit for visualization
    if len(df_edges) > max_edges:
        logger.info(f"Graph is large, visualizing a subset of {max_edges} edges")
        sampled_edges = df_edges.sample(max_edges, random_state=42)
    else:
        sampled_edges = df_edges
    
    for _, edge in sampled_edges.iterrows():
        G.add_edge(edge[0], edge[1])
    
    # Visualize
    visualize_graph(
        G,
        title="Bitcoin Transaction Graph",
        output_path=os.path.join(output_dir, 'transaction_graph.png')
    )
    
    # 2. Visualize node embeddings if model exists
    model_path = 'models/best_model.pt'
    if os.path.exists(model_path):
        logger.info("Generating node embeddings visualization")
        
        # Load model
        input_dim = data.x.shape[1]
        output_dim = len(torch.unique(data.y))
        
        model = GCNModel(
            input_dim=input_dim,
            hidden_dim=256,
            output_dim=output_dim,
            num_layers=3,
        ).to(device)
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Get embeddings
        def hook_fn(module, input, output):
            global embeddings
            embeddings = input[0]
        
        # Register hook
        for name, module in model.named_modules():
            if name == 'convs.' + str(len(model.convs) - 2):  # Second to last layer
                hook = module.register_forward_hook(hook_fn)
        
        # Forward pass
        with torch.no_grad():
            _ = model(data.x, data.edge_index)
        
        # Remove hook
        hook.remove()
        
        # Visualize test set embeddings
        test_embeddings = embeddings[split_idx['test']].cpu().numpy()
        test_labels = data.y[split_idx['test']].cpu().numpy()
        
        visualize_embeddings(
            test_embeddings,
            test_labels,
            method='tsne',
            title="Test Set Node Embeddings",
            output_path=os.path.join(output_dir, 'node_embeddings_tsne.png')
        )
        
        visualize_embeddings(
            test_embeddings,
            test_labels,
            method='pca',
            title="Test Set Node Embeddings",
            output_path=os.path.join(output_dir, 'node_embeddings_pca.png')
        )
        
        # 3. Visualize fraud patterns
        logger.info("Generating fraud pattern visualization")
        visualize_fraud_patterns(
            model,
            data,
            split_idx,
            output_path=os.path.join(output_dir, 'fraud_patterns.png')
        )
    
    # 4. Visualize feature importance if available
    feature_importance_path = 'data/processed/feature_importance.npy'
    if os.path.exists(feature_importance_path):
        logger.info("Generating feature importance visualization")
        feature_importance = np.load(feature_importance_path)
        
        visualize_feature_importance(
            feature_importance,
            output_path=os.path.join(output_dir, 'feature_importance.png')
        )
    
    # 5. Visualize training history if available
    history_path = 'models/training_history.json'
    if os.path.exists(history_path):
        logger.info("Generating training history visualization")
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        visualize_training_history(
            history,
            output_path=os.path.join(output_dir, 'training_history.png')
        )
    
    # 6. Visualize example transaction subgraph
    logger.info("Generating transaction subgraph visualization")
    
    # Find an example fraud transaction
    fraud_node = None
    for _, node in df_nodes.iterrows():
        if node[1] == 1:  # Fraud label
            fraud_node = node[0]
            break
    
    if fraud_node is not None:
        visualize_transaction_subgraph(
            df_nodes,
            df_edges,
            fraud_node,
            neighborhood_size=2,
            output_path=os.path.join(output_dir, f'transaction_{fraud_node}_subgraph.png')
        )
    
    logger.info("Visualization generation completed")

if __name__ == "__main__":
    main()
