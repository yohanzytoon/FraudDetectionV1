import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
import time
import copy
import logging
import json
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GATv2Conv
from torch_geometric.data import Data
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedGCNModel(nn.Module):
    """
    Enhanced Graph Convolutional Network model with skip connections and deeper architecture.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4,
                dropout=0.5, residual=True, batch_norm=True, layer_norm=True):
        super(EnhancedGCNModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # GCN layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Batch normalization
        self.use_batch_norm = batch_norm
        if batch_norm:
            self.bns = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
            ])
        
        # Layer normalization
        self.use_layer_norm = layer_norm
        if layer_norm:
            self.lns = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_layers)
            ])
        
        # Additional MLPs after graph convolutions for better expressiveness
        self.post_conv_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, hidden_dim)
        )
        
        # Final output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Initialize parameters
        self.reset_parameters()
        
        logger.info(f"Initialized Enhanced GCN model with {num_layers} layers")
        logger.info(f"Input dim: {input_dim}, Hidden dim: {hidden_dim}, Output dim: {output_dim}")
    
    def reset_parameters(self):
        """Reset all parameters for better initialization"""
        gain = nn.init.calculate_gain('relu')
        
        nn.init.xavier_uniform_(self.input_projection.weight, gain=gain)
        nn.init.zeros_(self.input_projection.bias)
        
        for conv in self.convs:
            conv.reset_parameters()
        
        # MLP initialization
        for layer in self.post_conv_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=gain)
                nn.init.zeros_(layer.bias)
        
        nn.init.xavier_uniform_(self.output_layer.weight, gain=gain)
        nn.init.zeros_(self.output_layer.bias)
        
        if self.use_batch_norm:
            for bn in self.bns:
                bn.reset_parameters()
        
        if self.use_layer_norm:
            for ln in self.lns:
                ln.reset_parameters()
    
    def forward(self, x, edge_index):
        # Project input features
        h = self.input_projection(x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Main GCN layers with skip connections
        for i in range(self.num_layers):
            h_prev = h
            h = self.convs[i](h, edge_index)
            
            # Apply normalization
            if self.use_batch_norm:
                h = self.bns[i](h)
            
            if self.use_layer_norm:
                h = self.lns[i](h)
            
            h = F.relu(h)
            
            # Residual connection
            if self.residual:
                h = h + h_prev
                
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Final MLP processing
        h = self.post_conv_mlp(h)
        
        # Output layer
        out = self.output_layer(h)
        
        return F.log_softmax(out, dim=1)
    
    def get_embeddings(self, x, edge_index, layer=-2):
        # Project input features
        h = self.input_projection(x)
        h = F.relu(h)
        
        # Process up to the desired layer
        max_layer = min(self.num_layers, layer if layer >= 0 else self.num_layers + layer)
        
        for i in range(max_layer):
            h_prev = h
            h = self.convs[i](h, edge_index)
            
            if self.use_batch_norm:
                h = self.bns[i](h)
            
            if self.use_layer_norm:
                h = self.lns[i](h)
                
            h = F.relu(h)
            
            if self.residual:
                h = h + h_prev
        
        return h

class EnhancedSAGEModel(nn.Module):
    """
    Enhanced GraphSAGE model with advanced architecture.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, 
                dropout=0.5, residual=True, batch_norm=True, aggr='mean'):
        super(EnhancedSAGEModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # SAGE layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggr))
        
        # Batch normalization
        self.use_batch_norm = batch_norm
        if batch_norm:
            self.bns = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
            ])
        
        # Skip connection adaptation layers (for handling dimension mismatches)
        self.skip_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Additional MLPs after graph convolutions
        self.post_conv_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, hidden_dim)
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Initialize parameters
        self.reset_parameters()
        
        logger.info(f"Initialized Enhanced GraphSAGE model with {num_layers} layers")
        logger.info(f"Input dim: {input_dim}, Hidden dim: {hidden_dim}, Output dim: {output_dim}")
    
    def reset_parameters(self):
        """Reset all parameters for better initialization"""
        gain = nn.init.calculate_gain('relu')
        
        nn.init.xavier_uniform_(self.input_projection.weight, gain=gain)
        nn.init.zeros_(self.input_projection.bias)
        
        for conv in self.convs:
            conv.reset_parameters()
        
        for skip in self.skip_layers:
            nn.init.xavier_uniform_(skip.weight, gain=gain)
            nn.init.zeros_(skip.bias)
        
        for layer in self.post_conv_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=gain)
                nn.init.zeros_(layer.bias)
        
        nn.init.xavier_uniform_(self.output_layer.weight, gain=gain)
        nn.init.zeros_(self.output_layer.bias)
        
        if self.use_batch_norm:
            for bn in self.bns:
                bn.reset_parameters()
    
    def forward(self, x, edge_index):
        # Project input features
        h = self.input_projection(x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Process SAGE layers
        for i in range(self.num_layers):
            # Store previous representation for residual connection
            h_prev = h
            
            # Apply GraphSAGE convolution
            h = self.convs[i](h, edge_index)
            
            # Apply batch normalization
            if self.use_batch_norm:
                h = self.bns[i](h)
            
            # Non-linearity
            h = F.relu(h)
            
            # Residual connection with skip layer adaptation
            if self.residual:
                h_skip = self.skip_layers[i](h_prev)
                h = h + h_skip
                
            # Dropout
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Post-GNN MLP processing
        h = self.post_conv_mlp(h)
        
        # Output layer
        out = self.output_layer(h)
        
        return F.log_softmax(out, dim=1)
    
    def get_embeddings(self, x, edge_index, layer=-2):
        h = self.input_projection(x)
        h = F.relu(h)
        
        max_layer = min(self.num_layers, layer if layer >= 0 else self.num_layers + layer)
        
        for i in range(max_layer):
            h_prev = h
            h = self.convs[i](h, edge_index)
            
            if self.use_batch_norm:
                h = self.bns[i](h)
                
            h = F.relu(h)
            
            if self.residual:
                h_skip = self.skip_layers[i](h_prev)
                h = h + h_skip
        
        return h

class EnhancedGATModel(nn.Module):
    """
    Enhanced Graph Attention Network model.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, 
                heads=4, dropout=0.5, residual=True, batch_norm=True, 
                use_gatv2=True):
        super(EnhancedGATModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        self.heads = heads
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # GAT layers
        self.convs = nn.ModuleList()
        
        # Use GATv2 for better expressiveness if specified
        gat_layer = GATv2Conv if use_gatv2 else GATConv
        
        for i in range(num_layers):
            # Last layer has only 1 head, others have the specified number
            out_heads = 1 if i == num_layers - 1 else heads
            in_channels = hidden_dim if i == 0 else hidden_dim * heads
            out_channels = output_dim if i == num_layers - 1 else hidden_dim
            
            self.convs.append(
                gat_layer(in_channels, out_channels, heads=out_heads)
            )
        
        # Batch normalization
        self.use_batch_norm = batch_norm
        if batch_norm:
            self.bns = nn.ModuleList()
            for i in range(num_layers-1):  # No batch norm for output layer
                channels = hidden_dim * heads
                self.bns.append(nn.BatchNorm1d(channels))
        
        # Attention readout - weighted sum of the last hidden layer
        self.attention_readout = nn.Sequential(
            nn.Linear(hidden_dim * heads, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Final classification layer
        self.classifier = nn.Linear(hidden_dim * heads, output_dim)
        
        # Initialize parameters
        self.reset_parameters()
        
        logger.info(f"Initialized Enhanced GAT model with {num_layers} layers and {heads} heads")
        logger.info(f"Input dim: {input_dim}, Hidden dim: {hidden_dim}, Output dim: {output_dim}")
    
    def reset_parameters(self):
        """Reset all parameters for better initialization"""
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        
        for conv in self.convs:
            conv.reset_parameters()
        
        if self.use_batch_norm:
            for bn in self.bns:
                bn.reset_parameters()
        
        for layer in self.attention_readout:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, x, edge_index):
        # Project input
        h = self.input_projection(x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # GAT layers
        for i in range(self.num_layers - 1):  # Process all but the last layer
            h_prev = h  # Store for residual connection
            
            # Apply GAT convolution
            h = self.convs[i](h, edge_index)
            
            # Reshape for batch norm: [nodes, heads*hidden] -> [nodes*heads, hidden] -> [nodes, heads*hidden]
            if self.use_batch_norm:
                h = self.bns[i](h)
            
            h = F.relu(h)
            
            # Residual connection if dimensions match
            if self.residual and h_prev.size(-1) == h.size(-1):
                h = h + h_prev
            
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Final GAT layer (classification)
        h = self.convs[-1](h, edge_index)
        
        return F.log_softmax(h, dim=1)
    
    def get_embeddings(self, x, edge_index, layer=-2):
        h = self.input_projection(x)
        h = F.relu(h)
        
        max_layer = min(self.num_layers-1, layer if layer >= 0 else self.num_layers + layer)
        
        for i in range(max_layer):
            h_prev = h
            h = self.convs[i](h, edge_index)
            
            if self.use_batch_norm:
                h = self.bns[i](h)
                
            h = F.relu(h)
            
            if self.residual and h_prev.size(-1) == h.size(-1):
                h = h + h_prev
        
        return h

def load_features_from_extraction():
    """
    Load features generated by the feature_extraction.py script.
    
    Returns:
    --------
    features : numpy.ndarray
        Features matrix
    labels : numpy.ndarray
        Labels array
    """
    data_dir = 'data/processed'
    
    # Try to load selected features first (these are the best features from feature selection)
    selected_features_path = os.path.join(data_dir, 'selected_features.npy')
    if os.path.exists(selected_features_path):
        logger.info("Loading selected features from feature extraction")
        features = np.load(selected_features_path)
    else:
        # Fall back to combined features
        combined_features_path = os.path.join(data_dir, 'combined_features.npy')
        if os.path.exists(combined_features_path):
            logger.info("Loading combined features from feature extraction")
            features = np.load(combined_features_path)
        else:
            # Try to load and combine individual feature sets
            logger.info("Loading individual feature sets")
            feature_sets = []
            
            # Transaction features
            trans_path = os.path.join(data_dir, 'transaction_features.npy')
            if os.path.exists(trans_path):
                feature_sets.append(np.load(trans_path))
                
            # Graph features
            graph_path = os.path.join(data_dir, 'graph_features.npy')
            if os.path.exists(graph_path):
                feature_sets.append(np.load(graph_path))
                
            # Sequence features
            seq_path = os.path.join(data_dir, 'sequence_features.npy')
            if os.path.exists(seq_path):
                feature_sets.append(np.load(seq_path))
                
            # Interaction features
            inter_path = os.path.join(data_dir, 'interaction_features.npy')
            if os.path.exists(inter_path):
                feature_sets.append(np.load(inter_path))
            
            if feature_sets:
                features = np.hstack(feature_sets)
                logger.info(f"Combined {len(feature_sets)} feature sets manually")
            else:
                raise FileNotFoundError("No feature files found from feature extraction")
    
    # Load labels from the nodes dataframe
    from data_preparation import load_data
    try:
        classes_path = 'data/raw/classes.csv'
        edgelist_path = 'data/raw/edgelist.csv'
        features_path = 'data/raw/Features.csv'
        
        df_nodes, _ = load_data(classes_path, edgelist_path, features_path)
        
        # Filter out unknown classes
        df_nodes = df_nodes[df_nodes['class'] != 'unknown']
        
        # Convert class labels to numeric if needed
        if df_nodes['class'].dtype == 'object':
            unique_classes = df_nodes['class'].unique()
            class_map = {cls: i for i, cls in enumerate(unique_classes)}
            df_nodes['class'] = df_nodes['class'].map(class_map)
        
        labels = df_nodes['class'].values
        logger.info(f"Loaded {len(labels)} labels from nodes dataframe")
    except Exception as e:
        logger.error(f"Error loading labels from dataframe: {str(e)}")
        raise
    
    logger.info(f"Loaded features with shape {features.shape} and {len(labels)} labels")
    
    return features, labels

def prepare_data_from_features(features, labels, test_size=0.2, val_size=0.15, random_state=42):
    """
    Prepare a PyTorch Geometric Data object from features and labels.
    
    Parameters:
    -----------
    features : numpy.ndarray
        Feature matrix
    labels : numpy.ndarray
        Labels
    test_size : float
        Proportion of data for testing
    val_size : float
        Proportion of data for validation
    random_state : int
        Random seed
        
    Returns:
    --------
    data : torch_geometric.data.Data
        Data object
    split_idx : dict
        Dictionary containing indices for train/val/test splits
    """
    logger.info("Preparing data from features...")
    
    # Convert to PyTorch tensors
    x = torch.FloatTensor(features)
    y = torch.LongTensor(labels)
    
    # Sample a random graph structure if one isn't provided
    # (We're using advanced features that already incorporate graph structure)
    num_nodes = x.size(0)
    
    # Create a simple k-nearest neighbors graph based on feature similarity
    k = min(10, num_nodes - 1)  # Use at most 10 neighbors
    
    logger.info(f"Creating a simple k-nearest neighbors graph with k={k}")
    
    # Compute pairwise distances (this could be memory-intensive for large datasets)
    # Consider using batched computation for very large datasets
    distances = torch.cdist(x, x)
    
    # Get k nearest neighbors for each node (excluding self-loops)
    _, indices = torch.topk(distances, k=k+1, dim=1, largest=False)
    indices = indices[:, 1:]  # Remove self-loop
    
    # Create edge_index
    rows = torch.arange(num_nodes).view(-1, 1).repeat(1, k).view(-1)
    cols = indices.view(-1)
    
    edge_index = torch.stack([rows, cols], dim=0)
    
    # Create Data object
    data = Data(x=x, edge_index=edge_index, y=y)
    
    # Create splits
    indices = np.arange(num_nodes)
    
    # First split: train vs. (val+test)
    train_idx, temp_idx = train_test_split(
        indices, 
        test_size=test_size + val_size, 
        stratify=labels,
        random_state=random_state
    )
    
    # Second split: val vs. test
    relative_val_size = val_size / (test_size + val_size)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1-relative_val_size,
        stratify=labels[temp_idx],
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

def train_model_with_advanced_schedule(
    model, data, split_idx, epochs=100, patience=20, 
    lr=3e-4, weight_decay=1e-4, device='cpu', model_dir='models', 
    model_name='gnn', grad_clip=1.0, use_one_cycle=True):
    """
    Train a GNN model with advanced learning rate scheduling and monitoring.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to train
    data : torch_geometric.data.Data
        The graph data
    split_idx : dict
        Dictionary containing indices for train/val/test splits
    epochs : int
        Maximum number of epochs
    patience : int
        Patience for early stopping
    lr : float
        Maximum learning rate
    weight_decay : float
        Weight decay factor
    device : str
        Device to use ('cpu' or 'cuda')
    model_dir : str
        Directory to save the model
    model_name : str
        Name of the model for saving
    grad_clip : float
        Gradient clipping value
    use_one_cycle : bool
        Whether to use OneCycleLR or CosineAnnealingWarmRestarts
        
    Returns:
    --------
    model : torch.nn.Module
        The trained model
    history : dict
        Training history
    """
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Move data to device
    data = data.to(device)
    
    # Calculate steps per epoch
    steps_per_epoch = 1  # For full-batch training
    
    # Create optimizer with weight decay 
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Create learning rate scheduler
    if use_one_cycle:
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=lr,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            pct_start=0.2,  # Use 20% of iterations for warmup
            div_factor=25,  # Initial lr = max_lr/25
            final_div_factor=10000  # Final lr = max_lr/10000
        )
    else:
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double the restart interval after each restart
            eta_min=lr/100  # Minimum learning rate
        )
    
    # Loss function
    criterion = torch.nn.NLLLoss()
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    # Best model tracking
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    patience_counter = 0
    
    # Training loop
    logger.info(f"Starting training for {epochs} epochs with {model_name} model")
    logger.info(f"Learning rate: {lr}, Weight decay: {weight_decay}")
    logger.info(f"Gradient clipping: {grad_clip}, Scheduler: {'OneCycleLR' if use_one_cycle else 'CosineAnnealing'}")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Train phase
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data.x, data.edge_index)
        train_loss = criterion(out[split_idx['train']], data.y[split_idx['train']])
        
        # Backward pass
        train_loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            history['learning_rates'].append(current_lr)
        
        # Calculate metrics
        with torch.no_grad():
            model.eval()
            out = model(data.x, data.edge_index)
            
            # Validation loss
            val_loss = criterion(out[split_idx['val']], data.y[split_idx['val']])
            
            # Accuracy
            pred = out.argmax(dim=1)
            train_correct = pred[split_idx['train']].eq(data.y[split_idx['train']]).sum().item()
            train_acc = train_correct / len(split_idx['train'])
            
            val_correct = pred[split_idx['val']].eq(data.y[split_idx['val']]).sum().item()
            val_acc = val_correct / len(split_idx['val'])
        
        # Track history
        history['train_loss'].append(train_loss.item())
        history['val_loss'].append(val_loss.item())
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress
        epoch_time = time.time() - epoch_start
        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(f"Epoch {epoch+1}/{epochs} | "
                       f"Train Loss: {train_loss:.4f} | "
                       f"Val Loss: {val_loss:.4f} | "
                       f"Train Acc: {train_acc:.4f} | "
                       f"Val Acc: {val_acc:.4f} | "
                       f"LR: {current_lr:.7f} | "
                       f"Time: {epoch_time:.2f}s")
        
        # Check early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f}s ({total_time/60:.2f}m) | Best epoch: {best_epoch+1}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Save model
    model_path = os.path.join(model_dir, f'{model_name}_best.pt')
    torch.save(best_model_state, model_path)
    logger.info(f"Saved best model to {model_path}")
    
    # Save training history
    history_path = os.path.join(model_dir, f'{model_name}_history.json')
    with open(history_path, 'w') as f:
        # Convert numpy values to float for JSON serialization
        serializable_history = {}
        for key, values in history.items():
            serializable_history[key] = [float(v) for v in values]
        
        json.dump(serializable_history, f)
    
    logger.info(f"Saved training history to {history_path}")
    
    return model, history

def main():
    """
    Main function to load data and train enhanced models.
    """
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directories
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        # Load features from your enhanced feature extraction
        logger.info("Loading features from feature extraction...")
        features, labels = load_features_from_extraction()
        
        # Prepare data
        data, split_idx = prepare_data_from_features(features, labels)
        
        logger.info(f"Prepared PyG data with {data.num_nodes} nodes, {data.num_edges} edges, and {data.x.shape[1]} features")
        
        # Determine number of classes
        num_classes = len(torch.unique(data.y))
        logger.info(f"Dataset has {num_classes} classes")
        
        # Get dimensions
        input_dim = data.x.shape[1]
        hidden_dim = 256  # Larger hidden dimension
        
        # Adjust training parameters based on dataset size
        if data.num_nodes > 10000:
            epochs = 200
            patience = 25
        else:
            epochs = 350  # Longer training for smaller datasets
            patience = 35  # Longer patience for better convergence
        
        # Model parameters
        models_to_train = [
            {
                'name': 'gcn',
                'class': EnhancedGCNModel,
                'params': {
                    'input_dim': input_dim,
                    'hidden_dim': hidden_dim,
                    'output_dim': num_classes,
                    'num_layers': 4,  # Deeper architecture
                    'dropout': 0.5,
                    'residual': True,
                    'batch_norm': True,
                    'layer_norm': True  # Add layer normalization
                },
                'training': {
                    'lr': 5e-4,
                    'weight_decay': 1e-5,
                    'epochs': epochs,
                    'patience': patience,
                    'grad_clip': 1.0,
                    'use_one_cycle': True
                }
            },
            {
                'name': 'sage',
                'class': EnhancedSAGEModel,
                'params': {
                    'input_dim': input_dim,
                    'hidden_dim': hidden_dim,
                    'output_dim': num_classes,
                    'num_layers': 4,
                    'dropout': 0.5,
                    'residual': True,
                    'batch_norm': True,
                    'aggr': 'mean'  # Try 'max' or 'sum' for different aggregations
                },
                'training': {
                    'lr': 1e-3,
                    'weight_decay': 5e-5,
                    'epochs': epochs,
                    'patience': patience,
                    'grad_clip': 1.0,
                    'use_one_cycle': True
                }
            },
            {
                'name': 'gat',
                'class': EnhancedGATModel,
                'params': {
                    'input_dim': input_dim,
                    'hidden_dim': 64,  # Smaller due to multiple heads
                    'output_dim': num_classes,
                    'num_layers': 3,
                    'heads': 4,  # More attention heads
                    'dropout': 0.5,
                    'residual': True,
                    'batch_norm': True,
                    'use_gatv2': True  # Use the improved GATv2
                },
                'training': {
                    'lr': 5e-4,
                    'weight_decay': 1e-5,
                    'epochs': epochs,
                    'patience': patience,
                    'grad_clip': 0.5,  # Smaller clip for GAT stability
                    'use_one_cycle': False  # Use cosine annealing for GAT
                }
            }
        ]
        
        # Train models and track results
        results = {}
        
        for model_config in models_to_train:
            logger.info(f"Training {model_config['name'].upper()} model...")
            
            # Create model
            model = model_config['class'](**model_config['params']).to(device)
            
            # Train model
            trained_model, history = train_model_with_advanced_schedule(
                model=model,
                data=data,
                split_idx=split_idx,
                device=device,
                model_dir=model_dir,
                model_name=model_config['name'],
                **model_config['training']
            )
            
            # Calculate final validation accuracy
            trained_model.eval()
            with torch.no_grad():
                out = trained_model(data.x, data.edge_index)
                val_pred = out.argmax(dim=1)[split_idx['val']]
                val_acc = val_pred.eq(data.y[split_idx['val']]).sum().item() / len(split_idx['val'])
                
                test_pred = out.argmax(dim=1)[split_idx['test']]
                test_acc = test_pred.eq(data.y[split_idx['test']]).sum().item() / len(split_idx['test'])
            
            # Store results
            results[model_config['name']] = {
                'val_acc': val_acc,
                'test_acc': test_acc,
                'best_val_loss': min(history['val_loss']),
                'final_train_acc': history['train_acc'][-1]
            }
            
            logger.info(f"Completed training {model_config['name']} model. "
                        f"Final validation accuracy: {val_acc:.4f}, "
                        f"Test accuracy: {test_acc:.4f}")
        
        # Determine best model
        best_model = max(results.items(), key=lambda x: x[1]['val_acc'])[0]
        
        logger.info(f"Best model based on validation accuracy: {best_model}")
        
        # Copy best model to 'best_model.pt'
        best_model_path = os.path.join(model_dir, f'{best_model}_best.pt')
        best_final_path = os.path.join(model_dir, 'best_model.pt')
        
        import shutil
        shutil.copy(best_model_path, best_final_path)
        
        # Save best model name
        with open(os.path.join(model_dir, 'best_model_name.txt'), 'w') as f:
            f.write(best_model)
        
        # Save comparison results
        with open(os.path.join(model_dir, 'model_comparison.json'), 'w') as f:
            json.dump(results, f, indent=4)
            
        logger.info("Training completed successfully!")
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()