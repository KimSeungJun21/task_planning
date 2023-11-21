
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter, to_dense_batch
from torch_geometric.nn import MetaLayer

class EdgeBlock(torch.nn.Module):
    def __init__(self, device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim):
        super(EdgeBlock, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_action = num_action
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size
        self.global_dim = global_dim


        self.edge_mlp = nn.Sequential(nn.Linear(2*self.node_feature_size + self.edge_feature_size + self.global_dim, self.hidden_dim), 
                            nn.BatchNorm1d(self.hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(self.hidden_dim, self.edge_feature_size)
                            )

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        return self.edge_mlp(out)
    
class NodeBlock(torch.nn.Module):
    def __init__(self, device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim):
        super(NodeBlock, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_action = num_action
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size
        self.global_dim = global_dim

        self.node_mlp_1 = nn.Sequential(nn.Linear(self.node_feature_size+self.edge_feature_size, self.hidden_dim), 
                              nn.BatchNorm1d(self.hidden_dim),
                              nn.ReLU(),
                              nn.Dropout(0.2),
                              nn.Linear(self.hidden_dim, self.hidden_dim))
        
        self.node_mlp_2 = nn.Sequential(nn.Linear(self.node_feature_size+self.hidden_dim+self.global_dim, self.hidden_dim), 
                              nn.BatchNorm1d(self.hidden_dim),
                              nn.ReLU(),
                              nn.Dropout(0.2),
                              nn.Linear(self.hidden_dim, self.node_feature_size))

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter(out, col, dim=0, dim_size=x.size(0), reduce='mean')
        out = torch.cat([x, out, u[batch]], dim=1)
        return self.node_mlp_2(out)

class GlobalBlock(torch.nn.Module):
    def __init__(self, device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim):
        super(GlobalBlock, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_action = num_action
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size
        self.global_dim = global_dim
                                 
        self.global_mlp = nn.Sequential(nn.Linear(self.node_feature_size + self.global_dim, self.hidden_dim),                               
                              nn.BatchNorm1d(self.hidden_dim),
                              nn.ReLU(),
                              nn.Dropout(0.2),
                              nn.Linear(self.hidden_dim, self.global_dim))

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = torch.cat([u,scatter(x, batch, dim=0, reduce='mean')], dim=1)
        return self.global_mlp(out)

class GNNLayer(nn.Module):
    def __init__(self, device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim):
        super(GNNLayer, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_action = num_action
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size
        self.global_dim = global_dim

        self.gnn_layer = MetaLayer(EdgeBlock(self.device,
                                             self.hidden_dim,
                                             self.num_action,
                                             self.node_feature_size,
                                             self.edge_feature_size,
                                             self.global_dim),
                                   NodeBlock(self.device,
                                               self.hidden_dim,
                                               self.num_action,
                                               self.node_feature_size,
                                               self.edge_feature_size,
                                               self.global_dim),
                                   GlobalBlock(self.device,
                                               self.hidden_dim,
                                               self.num_action,
                                               self.node_feature_size,
                                               self.edge_feature_size,
                                               self.global_dim)
                                   )
    
    def forward(self, x, edge_index, edge_attr, u, batch):
        x, edge_attr, u = self.gnn_layer(x, edge_index, edge_attr, u, batch)
        return x, edge_attr, u