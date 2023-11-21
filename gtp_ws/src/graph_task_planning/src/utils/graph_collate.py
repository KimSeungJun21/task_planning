import os
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
import pickle
import sys
sys.path.extend("./")
from utils.default_utils import node_feature_loader, root_path

## Basic Model Dataset
class BasicDataset(Dataset):
    def __init__(self, dataset_name, split):
        data_dir_path = os.path.join(root_path(),'demo_generation','datasets', dataset_name)

        self.search_path = os.path.join(data_dir_path, split)
        self.data_list = os.listdir(self.search_path)
    # Data size return
    def __len__(self): 
        return len(self.data_list)

    # Sampling one specific data from dataset
    def __getitem__(self, index): 
        graph_name = self.data_list[index]
        with open(os.path.join(self.search_path, graph_name), "rb") as file:
            loaded_data = pickle.load(file)
        
        x = loaded_data['input']['x']
        ei = loaded_data['input']['edge_index']
        ea = loaded_data['input']['edge_attr']

        action_code = loaded_data['target']['action']
        object_code = loaded_data['target']['object']
        target_data = [action_code, object_code]

        graph_state = Data(x = x,
                           edge_index = ei,
                           edge_attr = ea,
                           y = target_data)
        
        graph_info = loaded_data['info']

        return graph_state, graph_info
        