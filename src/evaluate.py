import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, roc_curve, auc, 
    precision_recall_curve, average_precision_score,
    confusion_matrix, f1_score
)
import logging
import json

from gnn_model import GCNModel, SAGEModel, GATModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model, data, split_idx, criterion=None, device='cpu'):
    """
    Evaluate model performance.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The trained model
    data : torch_geometric.data.Data
        The graph data
    split_idx : dict
        Dictionary containing indices for train/val/test splits
    criterion : torch.nn.Module, optional
        Loss function to calculate loss
    device : str
        Device to use ('cpu' or 'cuda')
        
    Returns:
    --------
    metrics : dict
        Dictionary containing evaluation metrics
    raw_data : dict
        Dictionary containing raw predictions
    """
    # Move data to device
    data = data.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Inference
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        
        # Calculate loss if criterion is provided
        loss = {}
        if criterion is not None:
            for split in split_idx:
                loss[split] = criterion(out[split_idx[split]], data.y[split_idx[split]]).item()
        
        # Get predictions and probabilities
        preds = {}
        probs = {}
        
        # Get raw probabilities
        raw_probs = torch.exp(out)
        
        # Number of classes
        num_classes = raw_probs.shape[1]
        
        for split in split_idx:
            preds[split] = out.argmax(dim=1)[split_idx[split]].cpu().numpy()
            
            # For binary classification, use probability of class 1
            # For multi-class, use all probabilities
            if num_classes == 2:
                probs[split] = raw_probs[split_idx[split], 1].cpu().numpy()
            else:
                probs[split] = raw_probs[split_idx[split]].cpu().numpy()
    
    # Collect true labels
    y_true = {}
    for split in split_idx:
        y_true[split] = data.y[split_idx[split]].cpu().numpy()
    
    # Calculate metrics
    metrics = {split: {} for split in split_idx}
    
    for split in split_idx:
        # Add loss if available
        if criterion is not None and split in loss:
            metrics[split]['loss'] = loss[split]
        
        # Classification report
        try:
            report = classification_report(y_true[split], preds[split], output_dict=True)
            
            # Add metrics from report
            for k, v in report.items():
                if isinstance(v, dict):  # Class-specific metrics
                    for metric, value in v.items():
                        metrics[split][f"{k}_{metric}"] = value
                else:  # Overall metrics like accuracy
                    metrics[split][k] = v
        except Exception as e:
            logger.warning(f"Error generating classification report for {split}: {str(e)}")
            metrics[split]['accuracy'] = (y_true[split] == preds[split]).mean()
        
        # Confusion Matrix
        metrics[split]['confusion_matrix'] = confusion_matrix(y_true[split], preds[split]).tolist()
        
        # Calculate macro-average metrics if there are multiple classes
        unique_classes = np.unique(y_true[split])
        if len(unique_classes) > 1:
            # ROC AUC (one-vs-rest for multi-class)
            try:
                if num_classes == 2:
                    from sklearn.metrics import roc_auc_score
                    metrics[split]['roc_auc'] = roc_auc_score(y_true[split], probs[split])
                else:
                    # For multi-class, compute one-vs-rest AUC for each class
                    aucs = []
                    for i in range(num_classes):
                        if i in unique_classes:
                            y_true_bin = (y_true[split] == i).astype(int)
                            try:
                                if isinstance(probs[split], np.ndarray) and probs[split].ndim == 2:
                                    class_probs = probs[split][:, i]
                                    aucs.append(roc_auc_score(y_true_bin, class_probs))
                            except:
                                continue
                    if aucs:
                        metrics[split]['roc_auc'] = np.mean(aucs)
                    else:
                        metrics[split]['roc_auc'] = float('nan')
            except Exception as e:
                logger.warning(f"Error calculating ROC AUC for {split}: {str(e)}")
                metrics[split]['roc_auc'] = float('nan')
                
            # PR AUC (one-vs-rest for multi-class)
            try:
                if num_classes == 2:
                    metrics[split]['pr_auc'] = average_precision_score(y_true[split], probs[split])
                else:
                    # For multi-class, compute one-vs-rest PR AUC for each class
                    pr_aucs = []
                    for i in range(num_classes):
                        if i in unique_classes:
                            y_true_bin = (y_true[split] == i).astype(int)
                            try:
                                if isinstance(probs[split], np.ndarray) and probs[split].ndim == 2:
                                    class_probs = probs[split][:, i]
                                    pr_aucs.append(average_precision_score(y_true_bin, class_probs))
                            except:
                                continue
                    if pr_aucs:
                        metrics[split]['pr_auc'] = np.mean(pr_aucs)
                    else:
                        metrics[split]['pr_auc'] = float('nan')
            except Exception as e:
                logger.warning(f"Error calculating PR AUC for {split}: {str(e)}")
                metrics[split]['pr_auc'] = float('nan')
    
    # Store raw predictions for further analysis
    raw_data = {
        split: {
            'y_true': y_true[split],
            'y_pred': preds[split],
            'probabilities': probs[split]
        } for split in split_idx
    }
    
    return metrics, raw_data

def plot_roc_curve(y_true, y_score, output_path=None):
    """
    Plot ROC curve.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True binary labels
    y_score : numpy.ndarray
        Target scores (probabilities)
    output_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    # Check if it's binary classification
    if len(np.unique(y_true)) != 2:
        logger.warning("ROC curve requires binary classification. Skipping.")
        return None
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (area = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_precision_recall_curve(y_true, y_score, output_path=None):
    """
    Plot precision-recall curve.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True binary labels
    y_score : numpy.ndarray
        Target scores (probabilities)
    output_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    # Check if it's binary classification
    if len(np.unique(y_true)) != 2:
        logger.warning("Precision-Recall curve requires binary classification. Skipping.")
        return None
    
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.step(recall, precision, color='darkorange', lw=2, where='post',
            label=f'AP = {avg_precision:.3f}')
    ax.fill_between(recall, precision, step='post', alpha=0.2, color='darkorange')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_confusion_matrix(y_true, y_pred, output_path=None):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels
    y_pred : numpy.ndarray
        Predicted labels
    output_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    
    # Set x and y tick labels
    classes = sorted(np.unique(np.concatenate((y_true, y_pred))))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def generate_evaluation_report(metrics, raw_data, model_name, output_dir='reports'):
    """
    Generate a comprehensive evaluation report.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing evaluation metrics
    raw_data : dict
        Dictionary containing raw predictions
    model_name : str
        Name of the model
    output_dir : str
        Directory to save report files
        
    Returns:
    --------
    report_path : str
        Path to the generated report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create report subdirectory
    report_dir = os.path.join(output_dir, model_name)
    os.makedirs(report_dir, exist_ok=True)
    
    # Create figures directory
    figures_dir = os.path.join(report_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Generate plots for binary classification
    for split in raw_data:
        y_true = raw_data[split]['y_true']
        y_pred = raw_data[split]['y_pred']
        probs = raw_data[split]['probabilities']
        
        # Confusion matrix (works for any number of classes)
        plot_confusion_matrix(y_true, y_pred, 
                             output_path=os.path.join(figures_dir, f'{split}_confusion_matrix.png'))
        
        # ROC and PR curves (only for binary classification)
        if len(np.unique(y_true)) == 2:
            # ROC curve
            plot_roc_curve(y_true, probs, 
                          output_path=os.path.join(figures_dir, f'{split}_roc_curve.png'))
            
            # Precision-Recall curve
            plot_precision_recall_curve(y_true, probs, 
                                       output_path=os.path.join(figures_dir, f'{split}_pr_curve.png'))
    
    # Save metrics to JSON
    metrics_path = os.path.join(report_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Create markdown report
    report_md = f"# Evaluation Report for {model_name}\n\n"
    
    # Add summary section
    report_md += "## Summary\n\n"
    report_md += "| Metric | Train | Validation | Test |\n"
    report_md += "|--------|-------|------------|------|\n"
    
    # Key metrics to include in summary
    key_metrics = ['accuracy', 'weighted avg_f1-score', 'weighted avg_precision', 'weighted avg_recall']
    
    # Add ROC AUC and PR AUC if available (binary classification)
    if 'roc_auc' in metrics.get('test', {}):
        key_metrics.extend(['roc_auc', 'pr_auc'])
    
    for metric in key_metrics:
        train_val = metrics.get('train', {}).get(metric, 'N/A')
        val_val = metrics.get('val', {}).get(metric, 'N/A')
        test_val = metrics.get('test', {}).get(metric, 'N/A')
        
        # Format values
        if isinstance(train_val, float):
            train_val = f"{train_val:.4f}"
        if isinstance(val_val, float):
            val_val = f"{val_val:.4f}"
        if isinstance(test_val, float):
            test_val = f"{test_val:.4f}"
        
        report_md += f"| {metric} | {train_val} | {val_val} | {test_val} |\n"
    
    # Add class-specific metrics if available
    classes = set()
    for split in metrics:
        for k in metrics[split]:
            if '_' in k and k.split('_')[0].isdigit():
                classes.add(int(k.split('_')[0]))
    
    if classes:
        report_md += "\n## Class-specific Metrics (Test Set)\n\n"
        report_md += "| Class | Precision | Recall | F1-Score | Support |\n"
        report_md += "|-------|-----------|--------|----------|--------|\n"
        
        for cls in sorted(classes):
            precision = metrics.get('test', {}).get(f"{cls}_precision", 'N/A')
            recall = metrics.get('test', {}).get(f"{cls}_recall", 'N/A')
            f1 = metrics.get('test', {}).get(f"{cls}_f1-score", 'N/A')
            support = metrics.get('test', {}).get(f"{cls}_support", 'N/A')
            
            # Format values
            if isinstance(precision, float):
                precision = f"{precision:.4f}"
            if isinstance(recall, float):
                recall = f"{recall:.4f}"
            if isinstance(f1, float):
                f1 = f"{f1:.4f}"
            
            report_md += f"| {cls} | {precision} | {recall} | {f1} | {support} |\n"
    
    # Add detailed metrics section
    report_md += "\n## Detailed Metrics\n\n"
    
    for split in metrics:
        report_md += f"### {split.capitalize()} Set\n\n"
        
        # Add confusion matrix
        report_md += "#### Confusion Matrix\n\n"
        report_md += f"![Confusion Matrix](figures/{split}_confusion_matrix.png)\n\n"
        
        # Add ROC curve and PR curve if available (binary classification)
        if 'roc_auc' in metrics[split]:
            split_metrics = metrics[split]
            unique_classes = set()
            for k in split_metrics:
                if '_' in k and k.split('_')[0].isdigit():
                    unique_classes.add(int(k.split('_')[0]))
            
            if len(unique_classes) == 2:
                # Add ROC curve
                report_md += "#### ROC Curve\n\n"
                report_md += f"![ROC Curve](figures/{split}_roc_curve.png)\n\n"
                
                # Add PR curve
                report_md += "#### Precision-Recall Curve\n\n"
                report_md += f"![PR Curve](figures/{split}_pr_curve.png)\n\n"
    
    # Save report
    report_path = os.path.join(report_dir, 'evaluation_report.md')
    with open(report_path, 'w') as f:
        f.write(report_md)
    
    logger.info(f"Evaluation report generated at {report_path}")
    
    return report_path

def get_embeddings(model, data, layer=-2):
    """
    Extract node embeddings from a specific layer of the model.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The trained model
    data : torch_geometric.data.Data
        The graph data
    layer : int
        Layer to extract embeddings from (negative indices count from end)
        
    Returns:
    --------
    embeddings : numpy.ndarray
        Node embeddings
    """
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'get_embeddings'):
            # Use the model's method if available
            embeddings = model.get_embeddings(data.x, data.edge_index, layer=layer)
        else:
            # Fallback to hooking the forward pass
            embeddings = None
            
            def hook_fn(module, input, output):
                nonlocal embeddings
                embeddings = input[0].detach()
            
            # Register hook on the desired layer
            if layer == -1:
                target_layer = model.convs[-1]
            else:
                target_layer = model.convs[layer]
            
            handle = target_layer.register_forward_hook(hook_fn)
            
            # Forward pass
            _ = model(data.x, data.edge_index)
            
            # Remove hook
            handle.remove()
    
    return embeddings.cpu().numpy()

def main():
    """
    Main function to evaluate trained models.
    """
    import torch
    from data_preparation import load_processed_data
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    try:
        data, split_idx = load_processed_data()
        logger.info(f"Loaded data with {data.num_nodes} nodes and {data.num_edges} edges")
    except FileNotFoundError:
        logger.error("Processed data not found. Please run data_preparation.py first.")
        return
    
    # Loss function
    criterion = torch.nn.NLLLoss()
    
    # Determine number of classes
    num_classes = len(torch.unique(data.y))
    logger.info(f"Dataset has {num_classes} classes")
    
    # Paths to models
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    # Load best model first
    best_model_path = os.path.join(model_dir, 'best_model.pt')
    best_model_name_path = os.path.join(model_dir, 'best_model_name.txt')
    
    if os.path.exists(best_model_name_path) and os.path.exists(best_model_path):
        with open(best_model_name_path, 'r') as f:
            best_model_name = f.read().strip()
        
        logger.info(f"Evaluating best model: {best_model_name}")
        
        # Create appropriate model
        input_dim = data.x.shape[1]
        hidden_dim = 256
        
        if best_model_name.lower() == 'gcn':
            model = GCNModel(input_dim, hidden_dim, num_classes, num_layers=3).to(device)
        elif best_model_name.lower() == 'sage':
            model = SAGEModel(input_dim, hidden_dim, num_classes, num_layers=3).to(device)
        elif best_model_name.lower() == 'gat':
            model = GATModel(input_dim, hidden_dim, num_classes, num_layers=3, heads=8).to(device)
        else:
            logger.error(f"Unknown model type: {best_model_name}")
            return
        
        # Load parameters
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        
        # Evaluate
        metrics, raw_data = evaluate_model(
            model=model,
            data=data,
            split_idx=split_idx,
            criterion=criterion,
            device=device
        )
        
        # Generate report
        generate_evaluation_report(
            metrics=metrics,
            raw_data=raw_data,
            model_name=best_model_name,
            output_dir='reports'
        )
    else:
        logger.warning("Best model not found. Will evaluate individual models if available.")
    
    # Model configurations to evaluate
    model_configs = [
        {'name': 'gcn', 'class': GCNModel, 'path': os.path.join(model_dir, 'gcn_best.pt')},
        {'name': 'sage', 'class': SAGEModel, 'path': os.path.join(model_dir, 'sage_best.pt')},
        {'name': 'gat', 'class': GATModel, 'path': os.path.join(model_dir, 'gat_best.pt')}
    ]
    
    # Track models and metrics for comparison
    available_models = []
    all_metrics = []
    
    # Evaluate each model
    for config in model_configs:
        if os.path.exists(config['path']):
            logger.info(f"Evaluating {config['name']} model")
            
            # Create model
            model = config['class'](
                input_dim=data.x.shape[1],
                hidden_dim=256,
                output_dim=num_classes,
                num_layers=3
            ).to(device)
            
            # Load trained parameters
            model.load_state_dict(torch.load(config['path'], map_location=device))
            
            # Evaluate
            metrics, raw_data = evaluate_model(
                model=model,
                data=data,
                split_idx=split_idx,
                criterion=criterion,
                device=device
            )
            
            # Generate report
            generate_evaluation_report(
                metrics=metrics,
                raw_data=raw_data,
                model_name=config['name'],
                output_dir='reports'
            )
            
            # Save for comparison
            available_models.append(config['name'])
            all_metrics.append(metrics)
    
    # Generate comparison report if multiple models are available
    if len(available_models) > 1:
        logger.info("Generating model comparison report")
        
        # Create comparison report
        report_md = "# Model Comparison Report\n\n"
        
        # Add test performance comparison
        report_md += "## Test Set Performance\n\n"
        
        # Define key metrics for comparison
        key_metrics = ['accuracy', 'weighted avg_f1-score', 'weighted avg_precision', 'weighted avg_recall']
        
        # Check if binary classification metrics are available
        if any('roc_auc' in m.get('test', {}) for m in all_metrics):
            key_metrics.extend(['roc_auc', 'pr_auc'])
        
        # Create table header
        report_md += "| Model | " + " | ".join(key_metrics) + " |\n"
        report_md += "|-------|" + "|".join(["---" for _ in key_metrics]) + "|\n"
        
        for i, model_name in enumerate(available_models):
            metrics = all_metrics[i].get('test', {})
            
            # Add model name
            row = f"| {model_name} |"
            
            # Add metrics
            for metric in key_metrics:
                value = metrics.get(metric, 'N/A')
                if isinstance(value, float):
                    value = f"{value:.4f}"
                row += f" {value} |"
            
            report_md += row + "\n"
        
        # Save report
        comparison_dir = os.path.join('reports', 'comparison')
        os.makedirs(comparison_dir, exist_ok=True)
        report_path = os.path.join(comparison_dir, 'model_comparison.md')
        
        with open(report_path, 'w') as f:
            f.write(report_md)
        
        logger.info(f"Model comparison report generated at {report_path}")
    
    logger.info("Model evaluation completed")

if __name__ == "__main__":
    main()