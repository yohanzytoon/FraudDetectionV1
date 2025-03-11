import os
import torch
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import copy
import logging
import json

from gnn_model import GCNModel, SAGEModel, GATModel
from data_preparation import load_processed_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(model, data, split_idx, optimizer, criterion, 
               scheduler=None, epochs=200, patience=20, 
               device='cpu', model_dir='models', model_name='gnn'):
    """
    Train a GNN model with early stopping.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to train
    data : torch_geometric.data.Data
        The graph data
    split_idx : dict
        Dictionary containing indices for train/val/test splits
    optimizer : torch.optim.Optimizer
        The optimizer to use
    criterion : torch.nn.Module
        The loss function
    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        Learning rate scheduler
    epochs : int
        Maximum number of epochs
    patience : int
        Patience for early stopping
    device : str
        Device to use ('cpu' or 'cuda')
    model_dir : str
        Directory to save the model
    model_name : str
        Name of the model for saving
        
    Returns:
    --------
    model : torch.nn.Module
        The trained model
    history : dict
        Training history
    """
    # Prepare model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Move data to device
    data = data.to(device)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_auc': [],
        'val_auc': []
    }
    
    # Best model tracking
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    patience_counter = 0
    
    # Training loop
    logger.info(f"Starting training for {epochs} epochs (early stopping patience: {patience})")
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Train phase
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        train_loss = criterion(out[split_idx['train']], data.y[split_idx['train']])
        train_loss.backward()
        optimizer.step()
        
        # Calculate training metrics
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
            
            # Calculate AUC if binary classification
            num_classes = torch.unique(data.y).shape[0]
            if num_classes == 2:
                try:
                    from sklearn.metrics import roc_auc_score
                    train_probs = torch.exp(out[split_idx['train'], 1]).cpu().numpy()
                    train_labels = data.y[split_idx['train']].cpu().numpy()
                    val_probs = torch.exp(out[split_idx['val'], 1]).cpu().numpy()
                    val_labels = data.y[split_idx['val']].cpu().numpy()
                    
                    train_auc = roc_auc_score(train_labels, train_probs)
                    val_auc = roc_auc_score(val_labels, val_probs)
                    
                    history['train_auc'].append(train_auc)
                    history['val_auc'].append(val_auc)
                except Exception as e:
                    # If AUC calculation fails, skip it
                    logger.warning(f"AUC calculation failed: {str(e)}")
                    history['train_auc'].append(0)
                    history['val_auc'].append(0)
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
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
                       f"Time: {epoch_time:.2f}s")
        
        # Check early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f}s | Best epoch: {best_epoch+1}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Save model
    model_path = os.path.join(model_dir, f'{model_name}_best.pt')
    torch.save(best_model_state, model_path)
    logger.info(f"Saved best model to {model_path}")
    
    # Save training history
    history_path = os.path.join(model_dir, f'{model_name}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f)
    logger.info(f"Saved training history to {history_path}")
    
    return model, history

def train_gcn(data, split_idx, hidden_dim=256, num_layers=3, 
             dropout=0.5, lr=0.01, weight_decay=5e-4, 
             epochs=200, patience=20, device='cpu', model_dir='models'):
    """
    Train a GCN model with the given parameters.
    
    Parameters:
    -----------
    data : torch_geometric.data.Data
        The graph data
    split_idx : dict
        Dictionary containing indices for train/val/test splits
    hidden_dim : int
        Dimension of hidden layers
    num_layers : int
        Number of GCN layers
    dropout : float
        Dropout probability
    lr : float
        Learning rate
    weight_decay : float
        Weight decay factor
    epochs : int
        Maximum number of epochs
    patience : int
        Patience for early stopping
    device : str
        Device to use ('cpu' or 'cuda')
    model_dir : str
        Directory to save the model
        
    Returns:
    --------
    model : torch.nn.Module
        The trained model
    history : dict
        Training history
    """
    # Create model
    input_dim = data.x.shape[1]
    output_dim = len(torch.unique(data.y))
    
    model = GCNModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    # Setup training
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.NLLLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, min_lr=1e-5, verbose=True)
    
    # Train model
    model, history = train_model(
        model=model,
        data=data,
        split_idx=split_idx,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        epochs=epochs,
        patience=patience,
        device=device,
        model_dir=model_dir,
        model_name='gcn'
    )
    
    return model, history

def train_sage(data, split_idx, hidden_dim=256, num_layers=3, 
              dropout=0.5, lr=0.01, weight_decay=5e-4, 
              epochs=200, patience=20, device='cpu', model_dir='models'):
    """
    Train a GraphSAGE model with the given parameters.
    """
    # Create model
    input_dim = data.x.shape[1]
    output_dim = len(torch.unique(data.y))
    
    model = SAGEModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    # Setup training
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.NLLLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, min_lr=1e-5, verbose=True)
    
    # Train model
    model, history = train_model(
        model=model,
        data=data,
        split_idx=split_idx,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        epochs=epochs,
        patience=patience,
        device=device,
        model_dir=model_dir,
        model_name='sage'
    )
    
    return model, history

def train_gat(data, split_idx, hidden_dim=256, num_layers=3, 
             heads=8, dropout=0.5, lr=0.01, weight_decay=5e-4, 
             epochs=200, patience=20, device='cpu', model_dir='models'):
    """
    Train a GAT model with the given parameters.
    """
    # Create model
    input_dim = data.x.shape[1]
    output_dim = len(torch.unique(data.y))
    
    model = GATModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        heads=heads,
        dropout=dropout
    ).to(device)
    
    # Setup training
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.NLLLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, min_lr=1e-5, verbose=True)
    
    # Train model
    model, history = train_model(
        model=model,
        data=data,
        split_idx=split_idx,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        epochs=epochs,
        patience=patience,
        device=device,
        model_dir=model_dir,
        model_name='gat'
    )
    
    return model, history

def main():
    """
    Main function to train models.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # Load data
        data, split_idx = load_processed_data()
    except FileNotFoundError:
        logger.error("Processed data not found. Please run data_preparation.py first.")
        return
    
    # Get number of classes
    num_classes = len(torch.unique(data.y))
    logger.info(f"Dataset has {num_classes} classes")
    
    # Define model parameters
    params = {
        'hidden_dim': 256,
        'num_layers': 3,
        'dropout': 0.5,
        'lr': 0.01,
        'weight_decay': 5e-4,
        'epochs': 200,
        'patience': 200,
        'device': device,
        'model_dir': 'models'
    }
   
    os.makedirs(params['model_dir'], exist_ok=True)
    
    # Train GCN model
    logger.info("Training GCN model")
    gcn_model, gcn_history = train_gcn(data, split_idx, **params)
    
    # Train GraphSAGE model
    logger.info("Training GraphSAGE model")
    sage_model, sage_history = train_sage(data, split_idx, **params)
    
    # Train GAT model (if enough memory)
    try:
        logger.info("Training GAT model")
        gat_model, gat_history = train_gat(data, split_idx, **params)
        has_gat = True
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            logger.warning("Not enough memory for GAT model. Skipping.")
            has_gat = False
        else:
            raise e
    
    logger.info("Model training completed")
    
    # Compare model performance
    logger.info("Model comparison:")
    best_val_gcn = min(gcn_history['val_loss']) if gcn_history['val_loss'] else float('inf')
    best_val_sage = min(sage_history['val_loss']) if sage_history['val_loss'] else float('inf')
    
    logger.info(f"GCN best validation loss: {best_val_gcn:.4f}")
    logger.info(f"SAGE best validation loss: {best_val_sage:.4f}")
    
    # Save combined model comparison
    comparison = {
        'gcn': {'val_loss': best_val_gcn},
        'sage': {'val_loss': best_val_sage}
    }
    
    if has_gat:
        best_val_gat = min(gat_history['val_loss']) if gat_history['val_loss'] else float('inf')
        logger.info(f"GAT best validation loss: {best_val_gat:.4f}")
        comparison['gat'] = {'val_loss': best_val_gat}
    
    with open(os.path.join(params['model_dir'], 'model_comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=4)
    
    # Copy the best model to 'best_model.pt'
    best_model = min(comparison.items(), key=lambda x: x[1]['val_loss'])[0]
    best_model_path = os.path.join(params['model_dir'], f'{best_model}_best.pt')
    
    import shutil
    shutil.copy(best_model_path, os.path.join(params['model_dir'], 'best_model.pt'))
    
    # Save the name of the best model
    with open(os.path.join(params['model_dir'], 'best_model_name.txt'), 'w') as f:
        f.write(best_model)
    
    logger.info(f"Best model was {best_model}, saved as 'best_model.pt'")

if __name__ == "__main__":
    main()