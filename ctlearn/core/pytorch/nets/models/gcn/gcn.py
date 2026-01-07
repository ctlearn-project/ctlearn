"""
Graph Convolutional Network (GCN) Module

This module implements a Graph Convolutional Network for processing graph-structured
data from Cherenkov telescope observations. The GCN architecture is particularly useful
for analyzing the spatial relationships between triggered pixels in telescope cameras.

The network uses graph convolutions to propagate information between neighboring pixels,
followed by graph pooling and classification/regression heads.

Classes:
    GCN: Graph Convolutional Network for telescope data analysis

References:
    - "Semi-Supervised Classification with Graph Convolutional Networks" (Kipf & Welling, ICLR 2017)
    - PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
"""

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
import pickle
import torch 
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch.nn import Linear, Softmax, PReLU

class GCN(torch.nn.Module):
    """
    Graph Convolutional Network for telescope camera data.
    
    This network processes graph-structured data where nodes represent camera pixels
    and edges connect neighboring pixels. The architecture consists of:
    1. Multiple graph convolutional layers to aggregate information
    2. Global pooling to obtain graph-level representation
    3. Fully connected layers for final prediction
    
    Graph Structure:
        - Nodes: Camera pixels with features (e.g., charge, timing)
        - Edges: Connections between neighboring pixels
        - Graph: Complete telescope camera image
        
    Architecture:
        Input Graph → GraphConv1 → PReLU → Dropout →
        GraphConv2 → PReLU → Dropout → GraphConv3 →
        Global Mean Pool → Linear → Linear → Output
    
    Attributes:
        conv0 (GCNConv): Initial graph convolution (currently unused)
        conv1 (GraphConv): First graph convolutional layer
        conv2 (GraphConv): Second graph convolutional layer
        conv3 (GraphConv): Third graph convolutional layer
        lin_0 (Linear): First fully connected layer
        lin_1 (Linear): Second fully connected layer (output)
        prelu_1 (PReLU): Parametric ReLU activation for conv1
        prelu_2 (PReLU): Parametric ReLU activation for conv2
    """
    
    def __init__(self, hidden_channels, num_node_features=1, num_outputs=1):
        """
        Initialize the Graph Convolutional Network.
        
        Args:
            hidden_channels (int): Number of hidden features in graph conv layers
                Typical values: 64, 128, 256
                Higher values allow more complex representations but increase computation
                
            num_node_features (int, optional): Number of features per node (pixel).
                Defaults to 1 (e.g., charge only)
                Can be 2 for charge + timing, or more for additional features
                
            num_outputs (int, optional): Number of output values. Defaults to 1
                For classification: 2 (gamma vs proton)
                For regression: 1 (energy or direction component)
                
        Network Design Choices:
            - GraphConv vs GCNConv: GraphConv is more general, supports edge features
            - Bias enabled: Helps with different graph sizes and structures
            - 3 conv layers: Balances receptive field vs oversmoothing
            - PReLU: Learnable activation that can adapt to data
            - Dropout (p=0.3): Regularization to prevent overfitting
            
        Example:
            >>> # Create GCN for binary classification
            >>> model = GCN(hidden_channels=128, num_node_features=2, num_outputs=2)
            >>> 
            >>> # Create GCN for energy regression
            >>> model = GCN(hidden_channels=64, num_node_features=1, num_outputs=1)
        """
        super(GCN, self).__init__()
        
        # Set random seed for reproducibility
        torch.manual_seed(12345)
     
        # Graph convolutional layers
        use_bias = True  # Enable bias for better expressiveness
        
        # Initial convolution (currently unused, kept for potential skip connections)
        self.conv0 = GCNConv(num_node_features, hidden_channels, bias=use_bias)
        
        # First graph convolution: node features → hidden_channels
        self.conv1 = GraphConv(num_node_features, hidden_channels, bias=use_bias)
        
        # Second graph convolution: hidden_channels → hidden_channels
        # Aggregates information from 2-hop neighbors
        self.conv2 = GraphConv(hidden_channels, hidden_channels, bias=use_bias)
        
        # Third graph convolution: hidden_channels → hidden_channels
        # Aggregates information from 3-hop neighbors
        self.conv3 = GraphConv(hidden_channels, hidden_channels, bias=use_bias)

        # Fully connected layers for prediction
        # First FC: Processes pooled graph representation
        self.lin_0 = Linear(hidden_channels, hidden_channels)
        
        # Output FC: Maps to final predictions
        self.lin_1 = Linear(hidden_channels, num_outputs)
        
        # Parametric ReLU activations (learnable negative slope)
        self.prelu_1 = PReLU()  # After conv1
        self.prelu_2 = PReLU()  # After conv2
        
    def forward(self, x, edge_index, batch):
        """
        Forward pass through the Graph Convolutional Network.
        
        Process:
        1. Apply graph convolutions to propagate information between neighbors
        2. Use PReLU activation and dropout after each conv layer
        3. Perform global pooling to get graph-level representation
        4. Apply fully connected layers for final prediction
        
        Args:
            x (torch.Tensor): Node feature matrix with shape (num_nodes, num_node_features)
                Each row represents features of one pixel in the camera
                Example: For 1000 triggered pixels with 2 features each: (1000, 2)
                
            edge_index (torch.Tensor): Graph connectivity in COO format
                Shape: (2, num_edges)
                edge_index[0]: Source nodes
                edge_index[1]: Target nodes
                Example: [[0, 1, 1], [1, 0, 2]] represents edges 0→1, 1→0, 1→2
                
            batch (torch.Tensor): Batch vector which assigns each node to a graph
                Shape: (num_nodes,)
                Example: [0, 0, 0, 1, 1, 2] for 3 graphs with 3, 2, and 1 nodes
                
        Returns:
            torch.Tensor: Output predictions with shape (batch_size, num_outputs)
                For classification: logits before softmax
                For regression: predicted values
                
        Graph Convolution Process:
            Each conv layer aggregates information from neighbors:
            h_i^(l+1) = σ(Σ_{j∈N(i)} (h_j^(l) · W^(l)) / √(d_i · d_j))
            where:
            - h_i: features of node i
            - N(i): neighbors of node i
            - W: learnable weight matrix
            - d_i: degree of node i
            - σ: activation function (PReLU)
            
        Example:
            >>> # Process a batch of 3 graphs
            >>> x = torch.randn(100, 2)  # 100 total pixels, 2 features each
            >>> edge_index = torch.randint(0, 100, (2, 300))  # 300 edges
            >>> batch = torch.tensor([0]*30 + [1]*40 + [2]*30)  # 3 graphs
            >>> 
            >>> output = model(x, edge_index, batch)
            >>> print(output.shape)  # torch.Size([3, 1]) for 3 graphs, 1 output each
        """
        # 1. Obtain node embeddings through graph convolutions
        
        # First convolution: Extract local patterns
        x = self.conv1(x, edge_index)  # (num_nodes, hidden_channels)
        x = self.prelu_1(x)  # Parametric ReLU activation
        x = F.dropout(x, p=0.3, training=self.training)  # Regularization
        
        # Second convolution: Aggregate 2-hop neighborhood information
        x = self.conv2(x, edge_index)  # (num_nodes, hidden_channels)
        x = self.prelu_2(x)  # Parametric ReLU activation
        x = F.dropout(x, p=0.3, training=self.training)  # Regularization
        
        # Third convolution: Aggregate 3-hop neighborhood information
        x = self.conv3(x, edge_index)  # (num_nodes, hidden_channels)
        
        # Note: Skip connections commented out
        # x = x + x_ori  # Residual connection with initial features
        # Could help with gradient flow and preserve initial information

        # 2. Readout layer: Aggregate node features to graph-level representation
        # Global mean pooling: Average all node features per graph
        x = global_mean_pool(x, batch=batch)  # (batch_size, hidden_channels)
        
        # Alternative: Global max pooling (commented out)
        # x = global_max_pool(x, batch=batch)
        # Max pooling captures most salient features but loses information

        # 3. Apply fully connected layers for final prediction
        # First FC layer: Further process graph representation
        x = self.lin_0(x)  # (batch_size, hidden_channels)
        
        # Output layer: Map to final predictions
        x = self.lin_1(x)  # (batch_size, num_outputs)
        
        return x