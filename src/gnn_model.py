import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GCNModel(nn.Module):
    """
    Graph Convolutional Network model for transaction classification.
    
    Features:
    - Multiple GCN layers
    - Batch normalization
    - Residual connections
    - Dropout for regularization
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, 
                dropout=0.5, residual=True, batch_norm=True):
        """
        Initialize the GCN model.
        
        Parameters:
        -----------
        input_dim : int
            Dimension of input features
        hidden_dim : int
            Dimension of hidden layers
        output_dim : int
            Dimension of output (number of classes)
        num_layers : int
            Number of GCN layers
        dropout : float
            Dropout probability
        residual : bool
            Whether to use residual connections
        batch_norm : bool
            Whether to use batch normalization
        """
        super(GCNModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        self.batch_norm = batch_norm
        
        # Input layer
        self.convs = nn.ModuleList([GCNConv(input_dim, hidden_dim)])
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
        # Batch normalization layers
        if batch_norm:
            self.bns = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)
            ])
        
        # Initialize parameters
        self.reset_parameters()
        
        logger.info(f"Initialized GCN model with {num_layers} layers")
        logger.info(f"Input dim: {input_dim}, Hidden dim: {hidden_dim}, Output dim: {output_dim}")
        
    def reset_parameters(self):
        """Reset all parameters for better initialization"""
        for conv in self.convs:
            conv.reset_parameters()
        
        if self.batch_norm:
            for bn in self.bns:
                bn.reset_parameters()
    
    def forward(self, x, edge_index):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        x : torch.Tensor
            Node features [num_nodes, input_dim]
        edge_index : torch.LongTensor
            Graph connectivity [2, num_edges]
            
        Returns:
        --------
        x : torch.Tensor
            Output predictions [num_nodes, output_dim]
        """
        # Input layer
        h = self.convs[0](x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Hidden layers
        for i in range(1, self.num_layers - 1):
            h_prev = h
            h = self.convs[i](h, edge_index)
            
            if self.batch_norm:
                h = self.bns[i-1](h)
            
            h = F.relu(h)
            
            if self.residual:
                h = h + h_prev  # Residual connection
                
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Output layer
        h = self.convs[-1](h, edge_index)
        
        return F.log_softmax(h, dim=1)

    def get_embeddings(self, x, edge_index, layer=-2):
        """
        Get embeddings from an intermediate layer.
        
        Parameters:
        -----------
        x : torch.Tensor
            Node features [num_nodes, input_dim]
        edge_index : torch.LongTensor
            Graph connectivity [2, num_edges]
        layer : int
            Index of the layer to extract embeddings from (negative indices count from end)
            
        Returns:
        --------
        embeddings : torch.Tensor
            Node embeddings
        """
        h = x
        
        # Process up to the desired layer
        max_layer = self.num_layers if layer >= 0 else self.num_layers + layer
        
        for i in range(max_layer):
            h = self.convs[i](h, edge_index)
            
            if i < self.num_layers - 1:  # Not the last layer
                if self.batch_norm and i > 0:
                    h = self.bns[i-1](h)
                
                h = F.relu(h)
                
                if self.residual and i > 0:
                    h_prev = h  # Store for residual connection
                    h = h + h_prev
                    
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        return h

class SAGEModel(nn.Module):
    """
    GraphSAGE model for transaction classification.
    
    Features:
    - GraphSAGE convolutions
    - Batch normalization
    - Residual connections
    - Dropout for regularization
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, 
                dropout=0.5, residual=True, batch_norm=True, aggr='mean'):
        """
        Initialize the GraphSAGE model.
        
        Parameters:
        -----------
        input_dim : int
            Dimension of input features
        hidden_dim : int
            Dimension of hidden layers
        output_dim : int
            Dimension of output (number of classes)
        num_layers : int
            Number of SAGE layers
        dropout : float
            Dropout probability
        residual : bool
            Whether to use residual connections
        batch_norm : bool
            Whether to use batch normalization
        aggr : str
            Aggregation method ('mean', 'max', or 'sum')
        """
        super(SAGEModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        self.batch_norm = batch_norm
        
        # Input layer
        self.convs = nn.ModuleList([SAGEConv(input_dim, hidden_dim, aggr=aggr)])
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggr))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_dim, output_dim, aggr=aggr))
        
        # Batch normalization layers
        if batch_norm:
            self.bns = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)
            ])
        
        # Initialize parameters
        self.reset_parameters()
        
        logger.info(f"Initialized GraphSAGE model with {num_layers} layers")
        logger.info(f"Input dim: {input_dim}, Hidden dim: {hidden_dim}, Output dim: {output_dim}")
    
    def reset_parameters(self):
        """Reset all parameters for better initialization"""
        for conv in self.convs:
            conv.reset_parameters()
        
        if self.batch_norm:
            for bn in self.bns:
                bn.reset_parameters()
    
    def forward(self, x, edge_index):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        x : torch.Tensor
            Node features [num_nodes, input_dim]
        edge_index : torch.LongTensor
            Graph connectivity [2, num_edges]
            
        Returns:
        --------
        x : torch.Tensor
            Output predictions [num_nodes, output_dim]
        """
        # Input layer
        h = self.convs[0](x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Hidden layers
        for i in range(1, self.num_layers - 1):
            h_prev = h
            h = self.convs[i](h, edge_index)
            
            if self.batch_norm:
                h = self.bns[i-1](h)
            
            h = F.relu(h)
            
            if self.residual:
                h = h + h_prev  # Residual connection
                
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Output layer
        h = self.convs[-1](h, edge_index)
        
        return F.log_softmax(h, dim=1)

    def get_embeddings(self, x, edge_index, layer=-2):
        """
        Get embeddings from an intermediate layer.
        
        Parameters:
        -----------
        x : torch.Tensor
            Node features [num_nodes, input_dim]
        edge_index : torch.LongTensor
            Graph connectivity [2, num_edges]
        layer : int
            Index of the layer to extract embeddings from (negative indices count from end)
            
        Returns:
        --------
        embeddings : torch.Tensor
            Node embeddings
        """
        h = x
        
        # Process up to the desired layer
        max_layer = self.num_layers if layer >= 0 else self.num_layers + layer
        
        for i in range(max_layer):
            h = self.convs[i](h, edge_index)
            
            if i < self.num_layers - 1:  # Not the last layer
                if self.batch_norm and i > 0:
                    h = self.bns[i-1](h)
                
                h = F.relu(h)
                
                if self.residual and i > 0:
                    h_prev = h  # Store for residual connection
                    h = h + h_prev
                    
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        return h

class GATModel(nn.Module):
    """
    Graph Attention Network model for transaction classification.
    
    Features:
    - GAT layers with attention
    - Batch normalization
    - Residual connections
    - Dropout for regularization
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, 
                heads=8, dropout=0.5, residual=True, batch_norm=True):
        """
        Initialize the GAT model.
        
        Parameters:
        -----------
        input_dim : int
            Dimension of input features
        hidden_dim : int
            Dimension of hidden layers
        output_dim : int
            Dimension of output (number of classes)
        num_layers : int
            Number of GAT layers
        heads : int
            Number of attention heads
        dropout : float
            Dropout probability
        residual : bool
            Whether to use residual connections
        batch_norm : bool
            Whether to use batch normalization
        """
        super(GATModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        self.batch_norm = batch_norm
        
        # Input layer (with multiple heads)
        self.convs = nn.ModuleList([GATConv(input_dim, hidden_dim // heads, heads=heads)])
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads))
        
        # Output layer (with 1 head)
        self.convs.append(GATConv(hidden_dim, output_dim, heads=1))
        
        # Batch normalization layers
        if batch_norm:
            self.bns = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)
            ])
        
        # Initialize parameters
        self.reset_parameters()
        
        logger.info(f"Initialized GAT model with {num_layers} layers and {heads} heads")
        logger.info(f"Input dim: {input_dim}, Hidden dim: {hidden_dim}, Output dim: {output_dim}")
    
    def reset_parameters(self):
        """Reset all parameters for better initialization"""
        for conv in self.convs:
            conv.reset_parameters()
        
        if self.batch_norm:
            for bn in self.bns:
                bn.reset_parameters()
    
    def forward(self, x, edge_index):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        x : torch.Tensor
            Node features [num_nodes, input_dim]
        edge_index : torch.LongTensor
            Graph connectivity [2, num_edges]
            
        Returns:
        --------
        x : torch.Tensor
            Output predictions [num_nodes, output_dim]
        """
        # Input layer
        h = self.convs[0](x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Hidden layers
        for i in range(1, self.num_layers - 1):
            h_prev = h
            h = self.convs[i](h, edge_index)
            
            if self.batch_norm:
                h = self.bns[i-1](h)
            
            h = F.relu(h)
            
            if self.residual:
                h = h + h_prev  # Residual connection
                
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Output layer
        h = self.convs[-1](h, edge_index)
        
        return F.log_softmax(h, dim=1)

    def get_embeddings(self, x, edge_index, layer=-2):
        """
        Get embeddings from an intermediate layer.
        
        Parameters:
        -----------
        x : torch.Tensor
            Node features [num_nodes, input_dim]
        edge_index : torch.LongTensor
            Graph connectivity [2, num_edges]
        layer : int
            Index of the layer to extract embeddings from (negative indices count from end)
            
        Returns:
        --------
        embeddings : torch.Tensor
            Node embeddings
        """
        h = x
        
        # Process up to the desired layer
        max_layer = self.num_layers if layer >= 0 else self.num_layers + layer
        
        for i in range(max_layer):
            h = self.convs[i](h, edge_index)
            
            if i < self.num_layers - 1:  # Not the last layer
                if self.batch_norm and i > 0:
                    h = self.bns[i-1](h)
                
                h = F.relu(h)
                
                if self.residual and i > 0:
                    h_prev = h  # Store for residual connection
                    h = h + h_prev
                    
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        return h