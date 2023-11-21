import os
import torch
import pandas as pd
import numpy as np
import pickle
import natsort
import pandas as pd
import json
import sys
import random
import shutil
sys.path.extend("./")
from utils.default_utils import create_folder, root_path
from utils.graph_utils import to_dense_graph, concat_current_goal_graphs

dataset_name='1114_tasks'

## load graph data from task dataset
class GraphLoader():
    def __init__(self, dataset_name:str, task_name):
        # Search path
        self.search_path = os.path.join(root_path(), 'demo_generation', 'tasks', dataset_name, task_name)
        # print("\n[Search_path]\n:",self.search_path)
        
        self.node_name_to_id = self.get_node_name_to_id()

    def read_csv_path(self, feature, i):    
        feature_path = os.path.join(self.search_path, feature)
        file_list = natsort.natsorted(os.listdir(feature_path))
        input_path = os.path.join(feature_path, file_list[i])
        return input_path

    def get_node_name_to_id(self):
        edge_inx_path = self.read_csv_path('edge_index', 0)
        ei_csv = pd.read_csv(edge_inx_path, index_col=0)
        ei_index = ei_csv.index.to_list()

        node_name_to_id = {v:k for k, v in enumerate(ei_index)}

        return node_name_to_id
    
    def load_node_feature(self):
        nf_path = self.read_csv_path('node_feature', 0)
        nf_csv = pd.read_csv(nf_path, index_col=0)
        nf = torch.tensor(nf_csv.values)
        return nf

    def load_edge_features(self, state_num):
        # load edge attribute csv
        edge_attr_path = self.read_csv_path('edge_attr', state_num)
        # Read csv file to tensor
        ea_csv = pd.read_csv(edge_attr_path, index_col=0)
        ea = torch.Tensor(ea_csv.values) # dataframe to tensor
        ea = ea.to(dtype = torch.float32)

        # write edge_index from index(ID) of edge_attr dataframe
        ei_list = []
        for ei in ea_csv.index.to_list():
            [src, dest] = ei[2:-2].split('\', \'')
            #ei_list.append(torch.tensor([[int(src)], [int(dest)]]))
            ei_list.append(torch.Tensor([[int(self.node_name_to_id[src])],[int(self.node_name_to_id[dest])]]))

        ei = torch.cat(ei_list, dim=1)
        return ei, ea
    
    
class CollectGraph():
    def __init__(self, dataset_name):        
        # self.action_encoder = {0 :[1, 0], 1 :[0, 1]}     
        self.edge_attr_dim = 4
        self.dataset_name = dataset_name
        self.save_path = os.path.join(root_path(), 'demo_generation','datasets', self.dataset_name)

        self.task_list = self.load_task_list()

        self.graph_num = 0
        self.graph_step_num = 0

    def load_task_list(self):
        data = []  # 읽어온 데이터를 저장할 리스트
        file_path = os.path.join(root_path(), 'demo_generation', 'tasks', self.dataset_name, 'combined_task.json')
        with open(file_path, "r") as file:
            data = json.load(file)
        return data
    
    def get_act_obj_list(self, task_step):
        act_list = []
        obj_list = []
        for steps in task_step:
            act_list.append(steps[0])
            if steps[0] == 0: #pick
                obj_list.append(steps[1])
            elif steps[0] == 1: #place
                obj_list.append(steps[2])
            else:
                raise ValueError("Wrong task step")
        return act_list, obj_list

    def collect_and_save_tasks(self):
        for task_name, task_seq in self.task_list.items():
            # get target action&object list
            graph_loader = GraphLoader(self.dataset_name, task_name)
            act_list, obj_name_list = self.get_act_obj_list(task_seq)
            obj_id_list = [graph_loader.node_name_to_id[obj_name] for obj_name in obj_name_list]
            idx_goal = len(act_list)

            # load graph node feature(constant)
            nf = graph_loader.load_node_feature()

            # load goal graph edge features
            ei_goal, ea_goal = graph_loader.load_edge_features(idx_goal)
            ei_goal, ea_goal = to_dense_graph(ei_goal, ea_goal)
 
            # # sequence 저장 위한 list 선언
            # one_demo = []

            for idx_current in range(idx_goal):
                # load current graph edge features
                ei_current, ea_current = graph_loader.load_edge_features(idx_current)
                ei_current, ea_current = to_dense_graph(ei_current, ea_current)

                # concatenate current and goal state edge features
                ei_cat, ea_cat = concat_current_goal_graphs(ei_current, ea_current, ei_goal, ea_goal)   
                
                # target action code
                act_label = act_list[idx_current]

                # target object code
                obj_label = obj_id_list[idx_current]
                
                # save data
                graph_dict_data = {'input':{},
                                   'target':{'action':[],
                                             'object':[]
                                             },
                                   'info':{
                                           'task_name':str(),
                                           'task_len':int(),
                                           'graph_num':int(),
                                           'step_num':int(),
                                           }
                                           }
                graph_dict_data['input']['x'] = nf
                graph_dict_data['input']['edge_index'] = ei_cat
                graph_dict_data['input']['edge_attr'] = ea_cat

                graph_dict_data['target']['action'] = act_label
                graph_dict_data['target']['object'] = obj_label

                graph_dict_data['info']['task_name'] = task_name
                graph_dict_data['info']['task_len'] = idx_goal
                graph_dict_data['info']['graph_num'] = self.graph_num
                graph_dict_data['info']['step_num'] = idx_current
                # one_demo.append(graph_dict_data)

                # basic dataset 그래프 저장
                create_folder(self.save_path)
                file_path = os.path.join(self.save_path, f"graph_{self.graph_num}_{idx_current}")
                with open(file_path, "wb") as outfile:
                    pickle.dump(graph_dict_data, outfile)

                self.graph_step_num += 1
            self.graph_num+=1
    
    def split_dataset(self, ratio=[0.7, 0.15, 0.15]):
        # List all files to split
        files = os.listdir(self.save_path)
        # Shuffle the files randomly
        random.shuffle(files)

        # Create output folders if they don't exist
        for folder in ['train', 'val', 'test']:
            folder_path = os.path.join(self.save_path, folder)
            create_folder(folder_path)


        # Calculate the number of files for each split
        total_files = len(files)
        train_size = int(total_files * ratio[0])
        val_size = int(total_files * ratio[1])
        test_size = total_files - train_size - val_size

        # Assign files to each split
        train_files = files[:train_size]
        val_files = files[train_size:train_size + val_size]
        test_files = files[train_size + val_size:]

        # move files to the respective output folders
        self.move_files(train_files, 'train')
        self.move_files(val_files, 'val')
        self.move_files(test_files, 'test')

    def move_files(self, file_list, split):
        for file in file_list:
            source_path = os.path.join(self.save_path, file)
            dest_path = os.path.join(self.save_path, split, file)
            shutil.copy(source_path, dest_path)
            os.remove(source_path)

if __name__ == "__main__":
    graph_collector = CollectGraph(dataset_name)
    graph_collector.collect_and_save_tasks()

    graph_step_num = graph_collector.graph_step_num
    graph_num = graph_collector.graph_num

    graph_collector.split_dataset()

    # data num check
