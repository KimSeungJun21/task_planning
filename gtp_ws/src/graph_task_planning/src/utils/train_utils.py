from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import sys
sys.path.extend("./")

def edge_prediction_loss(pred_ea, target_ea, device):
    ce_loss = nn.CrossEntropyLoss().to(device)
    mse_loss = nn.MSELoss().to(device)
    total_loss = 0
    for idx in range(pred_ea.size(0)):
        target_edge = target_ea[idx,:]
        if torch.any(target_edge): #there is non-zero value in target_edge_attr
            total_loss += ce_loss(pred_ea[idx,:], torch.argmax(target_edge,dim=-1))*10
        else:
            total_loss += mse_loss(pred_ea[idx,:], target_edge)
    return total_loss / pred_ea.size(0)
    
