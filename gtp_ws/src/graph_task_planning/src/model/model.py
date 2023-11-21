import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter, to_dense_batch
from torch_geometric.nn import MetaLayer
import sys
sys.path.extend("./")
from .GNN_layer import GNNLayer

class ActionModel_Basic(nn.Module):
    def __init__(self, args):
        super(ActionModel_Basic, self).__init__()
        self.args = args
        # graph state encoder
        self.gnn_encoder_1 = GNNLayer(self.args.device,
                                      self.args.hidden_dim,
                                      self.args.num_action,
                                      self.args.node_feature_size,
                                      self.args.edge_feature_size*2,
                                      self.args.global_dim)
        self.gnn_encoder_2 = GNNLayer(self.args.device,
                                      self.args.hidden_dim,
                                      self.args.num_action,
                                      self.args.node_feature_size,
                                      self.args.edge_feature_size*2,
                                      self.args.global_dim)
        
        # node score projection layer
        self.node_score_mlp = nn.Sequential(nn.Linear(self.args.node_feature_size, self.args.hidden_dim),
                                            nn.BatchNorm1d(self.args.hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(self.args.hidden_dim, 1))

        # action prediction layer
        self.action_layers = nn.Sequential(nn.Linear(self.args.node_feature_size, self.args.hidden_dim),
                                           nn.BatchNorm1d(self.args.hidden_dim),
                                           nn.ReLU(),
                                           # nn.Linear(self.args.hidden_dim, self.args.hidden_dim),
                                           # nn.BatchNorm1d(self.args.hidden_dim),
                                           # nn.ReLU(),
                                           nn.Linear(self.args.hidden_dim, self.args.num_action),
                                           #nn.Sigmoid()
                                           )

        # object prediction layer
        self.object_layers = nn.Sequential(nn.Linear(self.args.node_feature_size, self.args.hidden_dim),
                                           nn.BatchNorm1d(self.args.hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(self.args.hidden_dim, self.args.hidden_dim),
                                           nn.BatchNorm1d(self.args.hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(self.args.hidden_dim, self.args.num_object),
                                           #nn.Sigmoid()
                                           )

    def forward(self, current):
        current_x = current['x'].to(self.args.device)
        current_edge_index = current['edge_index'].to(self.args.device).type(torch.long)
        current_edge_attr = current['edge_attr'].to(self.args.device)
        current_batch = current['batch'].to(self.args.device)
        batch_size = current_batch[-1].item()+1
        ## 임시 ##
        current_u = torch.zeros(batch_size, self.args.global_dim).to(self.args.device)

        if self.args.method == 'node_scoring':
            node_score = current_x.clone()
            node_score_ea = current_edge_attr.clone()
            node_score_u = current_u.clone()
            #node scoring
            node_score, node_score_ea, node_score_u = self.gnn_encoder_1(x=node_score,
                                                              edge_index=current_edge_index,
                                                              edge_attr=node_score_ea,
                                                              u=node_score_u,
                                                              batch=current_batch)
            node_score, node_score_ea, node_score_u = self.gnn_encoder_2(x=node_score,
                                                              edge_index=current_edge_index, 
                                                              edge_attr=node_score_ea, 
                                                              u=node_score_u, 
                                                              batch=current_batch)
            node_score = self.node_score_mlp(node_score)

            node_score_batched, node_score_mask = to_dense_batch(node_score, current_batch)
            
        current_x, current_edge_attr, current_u = self.gnn_encoder_1(x=current_x,
                                                             edge_index=current_edge_index,
                                                             edge_attr=current_edge_attr,
                                                             u=current_u,
                                                             batch=current_batch)
        current_x, current_edge_attr, current_u = self.gnn_encoder_2(x=current_x,
                                                             edge_index=current_edge_index, 
                                                             edge_attr=current_edge_attr, 
                                                             u=current_u, 
                                                             batch=current_batch)
        current_x, current_x_mask = to_dense_batch(current_x, current_batch)
        
        if self.args.method == 'mean':
            outputs = current_x.mean(axis=1)
        
        if self.args.method == 'node_scoring':
            object_prob = node_score_batched.squeeze(-1).masked_fill_(current_x_mask==False, -1e9) # -> output -> loss계산(ce)

            node_weight = F.softmax(object_prob.clone(), dim=-1)

            outputs = torch.sum(current_x*node_weight.unsqueeze(dim=-1),dim=-2)/self.args.num_object

        action_prob = self.action_layers(outputs.clone())
        if self.args.method == 'mean':
            object_prob = self.object_layers(outputs.clone())

        return action_prob, object_prob