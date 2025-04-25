
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
import pickle
import torch 
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GraphConv
from torch_geometric.nn import global_mean_pool,global_max_pool
from torch.nn import Linear,Softmax,PReLU

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels,num_node_features=1, num_outputs=1):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
     
        # self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        # self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # self.conv3 = GCNConv(hidden_channels, hidden_channels)
        use_bias = True
        self.conv0 = GCNConv(num_node_features, hidden_channels,bias=use_bias)
        self.conv1 = GraphConv(num_node_features, hidden_channels,bias=use_bias)
        self.conv2 = GraphConv(hidden_channels, hidden_channels,bias=use_bias)
        self.conv3 = GraphConv(hidden_channels, hidden_channels,bias=use_bias)

        
        self.lin_0 = Linear(hidden_channels, hidden_channels)
        self.lin_1 = Linear(hidden_channels, num_outputs)
        self.prelu_1 = PReLU()
        self.prelu_2 = PReLU()
    def forward(self, x, edge_index, batch):
        
        # 1. Obtain node embeddings 
        # x_ori = self.conv0(x, edge_index)
        x = self.conv1(x, edge_index)
        
        # x = x.relu()
        x = self.prelu_1(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        # x = x.relu()
        x = self.prelu_2(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv3(x, edge_index)
        # x = x + x_ori
        # x = torch.concat([x,x_ori])
        # batch_aug = torch.concat([batch,(batch+1)*torch.max(batch)])
        # batch_aug = torch.concat([batch,batch])

        # 2. Readout layer
        x = global_mean_pool(x,batch=batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin_0(x)
        x = self.lin_1(x)
        return x