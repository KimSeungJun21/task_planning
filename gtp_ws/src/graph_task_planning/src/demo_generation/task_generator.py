import os
import pandas as pd
import numpy as np
import os
from functools import lru_cache
import torch
import sys
import pandas as pd
from torch.utils.data import Dataset
import natsort
import json
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations, product, permutations
sys.path.extend("./")
from utils.default_utils import create_folder


# @lru_cache()
# def default_path():
#     def_path= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     return sys.path.insert(0, def_path)

# default_path()
# sys.path.extend("./")
# from utils.default_utils  import *

class GenTaskSequence(Dataset):
    def __init__(self, information, problem):
        # Search path
        FILEPATH, _ = os.path.split(os.path.realpath(__file__))
        search_path = os.path.join(FILEPATH, 'tasks', information, problem)
        create_folder(search_path)

        self.FILEPATH = FILEPATH
        self.search_path = search_path
        self.problem = problem # task 종류
        self.input_info = information
        # self.example = example # pose
      
        print("\n==========================================[INIT INFORMATION]======================================================")
        # print("\n[File path]",FILEPATH)
        # print("\n[Search_path]",search_path)
        # print("\n[Example]", example)
        print("\n[Problem]", problem)

    def load_csv_files(self, n: int, feature: str, index_col):    
        feature_path = os.path.join(self.search_path, feature)
        file_list = natsort.natsorted(os.listdir(feature_path))
        input_path = os.path.join(feature_path, file_list[n])
        input_df = pd.read_csv(input_path, index_col= index_col)
        return input_df


    def save_csv_files(self, n: int, features: str, df: pd.DataFrame):
        if features == 'edge_index':
            # file_name = f'{self.example}_ei{n}.csv'
            file_name = 'ei{}.csv'.format(n)
        elif features == 'edge_attr':
            # file_name = f'{self.example}_ea{n}.csv'
            file_name = 'ea{}.csv'.format(n)
        elif features == 'node_feature':
            # file_name = f'{self.example}_nf{n}.csv'
            file_name = 'nf{}.csv'.format(n)
        save_path = os.path.join(self.search_path, features)
        create_folder(save_path)
        df.to_csv(os.path.join(save_path, file_name))        

    ####################################################################################
    ### dataframe functions ###
    def write_df(self, input:tuple):
        #(input: pd.DataFrame, object1:str, object2: str, integer: int)
        if not isinstance(input, tuple):
            raise TypeError("The data must be the type of a tuple.")
        df = input[0]
        object1 = input[1]
        object2 = input[2]
        integer = input[3]
        df.loc[object1, object2] = integer
        df.loc[object2, object1] = integer
        return df

    def iter_write_df(self, input: list):
        for objects in input:
            self.write_df(objects)
    ####################################################################################
    ### node feature functions ###
    def make_nf(self, obj_list):
        # make and save node feature
        self.make_init_node_feature(obj_list)
    
    def make_init_node_feature(self, obj_list: list):
        # make and save initial node feature
        nf0 = self.make_init_node_feature_df(obj_list)
        self.save_csv_files(0, 'node_feature', nf0)
    
    def make_init_node_feature_df(self, obj_list: list):
        if not isinstance(obj_list, list):
            raise TypeError("The data must be the type of a list.")
        property_list = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4', 'ID_5', 'ID_6', 'ID_7', 'ID_8',
                         'Type_Box', 'Type_Robot', 'Type_Region']
        
        nf0 = pd.DataFrame(0, index=obj_list, columns=property_list)
        
        for idx, object in enumerate(nf0.index):
            # write id feature
            nf0.loc[object, property_list[idx]] = 1
            # write type feature
            if "Box" in object:
                nf0.loc[object, 'Type_Box'] = 1
            elif "Robot" in object:
                nf0.loc[object, 'Type_Robot'] = 1
            elif "Region" in object:
                nf0.loc[object, 'Type_Region'] = 1
        return nf0

    ####################################################################################
    ### edge index functions ###
    def make_ei_list(self, task_sequence:list, obj_list):
        # make and save sequence of edge index
        self.make_init_edge_index(obj_list)

        for i, (action, obj1, obj2) in enumerate(task_sequence):            
            if action == 0:
                self.pick_and_save_ei(i+1, obj1)
            elif action == 1:
                self.place_and_save_ei(i+1, obj1, obj2)
            else:
                raise ValueError("Wrong_task")

    def make_init_edge_index(self, obj_list: list):
        # make and save initial edge index 
        ei0 = self.make_init_edge_index_df(obj_list)
        self.save_csv_files(0, 'edge_index', ei0)
   
    def make_init_edge_index_df(self, obj_list: list):
        if not isinstance(obj_list, list):
            raise TypeError("The data must be the type of a list.")
        ei0 = pd.DataFrame(0, index= obj_list, columns= obj_list)
        for object in ei0.index:
            if "Box" in object:
                self.write_df((ei0, object, 'Region_1', 1))    
        return ei0

    def pick_and_save_ei(self, state_num:int, target_obj): 
        # load edge index of previous state
        state_ei = self.load_csv_files(state_num-1, 'edge_index', 0)#이전스텝 불러와서 업데이트하는 방식
        # Remove 'on' relation (Table / other box)
        for obj in state_ei.index:
            if 'Box' in target_obj:
                self.write_df((state_ei, target_obj, obj, 0))
        # write 'grasp' relation with 'robot_hand'
        self.write_df((state_ei, target_obj, 'Robot_hand', 1))
        # save updated edge index
        self.save_csv_files(state_num, 'edge_index', state_ei) 
        # print(f'\n[Pick[{str(object)}].csv] \n') 
        # return state_ei
    
    def place_and_save_ei(self, state_num:int, obj1, obj2):
        # load edge index of previous state
        state_ei = self.load_csv_files(state_num-1, 'edge_index', 0)
        # update state_ei
        self.iter_write_df([(state_ei, obj1, obj2, 1),(state_ei, obj1, 'Robot_hand', 0)])
        # save updated edge index
        self.save_csv_files(state_num, 'edge_index', state_ei)
        # print(f'\n[Place_[{str(object1)}]_on_[{str(object2)}].csv] \n') 
        # return state_ei
    ####################################################################################
    ### edge attr functions ###
    def make_ea_list(self, task_sequence:list):
        # make and save sequence of edge attribute
        self.make_init_edge_attr(len(task_sequence))
        for i, (action, obj1, obj2) in enumerate(task_sequence):
            if action == 0:
                self.pick_and_save_ea(i+1, obj1)
            elif action == 1:
                self.place_and_save_ea(i+1, obj1, obj2)
            else:
                raise ValueError("Wrong_task")

    def make_init_edge_attr(self, len_task: int):
        # make and save initial edge attribute
        ea0 = self.make_init_edge_attr_df(len_task)       
        # write initial edge attribute    
        for node_pair in ea0.index:
            np0 = node_pair[0]
            np1 = node_pair[1]
            if 'Region_1' in np0: 
                if 'Box' in np1:
                    ea0.loc[[node_pair], 'rel_on_left'] = 1
            elif 'Region_1' in np1: 
                if 'Box' in np0:
                    ea0.loc[[node_pair], 'rel_on_right'] = 1    
        # save initial edge attribute   
        self.save_csv_files(0, 'edge_attr', ea0)    
     

    def make_init_edge_attr_df(self, len_task:int):
        index_attr = []
        relation_list = ['rel_on_right','rel_on_left','rel_grasp_right','rel_grasp_left']

        for num in range(len_task+1):
            # load state edge index
            edge_index = self.load_csv_files(num, 'edge_index',0)
            ID_list = list(map(str, edge_index.columns))
            # find connected object id pair
            for index in range(len(ID_list)):
                for column in range(len(ID_list)):
                    if edge_index.iat[index, column] == 1:  
                        index_attr.append((ID_list[index], ID_list[column]))
        # filter duplicated items
        index_attr = sorted(list(set(index_attr)))
        # print("[list_at]\n",index_attr)
        
        # make initial edge attribute
        ea0 = pd.DataFrame(0, index= index_attr, columns= relation_list)
        # print(f'\n[Init_env.csv] \n') 
        # print(init_edge_attr)      
        # input() 
        return ea0    

    def pick_and_save_ea(self, state_num:int, target_obj: str):
        # load edge attr of previous state
        state_ea = self.load_csv_files(state_num-1, 'edge_attr', 0)#이전스텝 불러와서 업데이트하는 방식
        ea_index = state_ea.index.tolist()
        # write 'grasp' relation with 'robot_hand'
        for node_pair in ea_index:
            np0 = node_pair.split(",")[0].lstrip("('").rstrip("'")
            np1 = node_pair.split(",")[1].lstrip(" '").rstrip("')")
            
            if np0 == target_obj:
                if 'Box' in np0:
                    state_ea.loc[[node_pair], :] = 0
                if 'Robot_hand' in np1:
                    state_ea.loc[[node_pair], 'rel_grasp_left'] = 1

            elif np1 == target_obj:
                if 'Box' in np1:
                    state_ea.loc[[node_pair], :] = 0
                if 'Robot_hand' in np0:
                    state_ea.loc[[node_pair], 'rel_grasp_right'] = 1
        
        # save updated edge index
        self.save_csv_files(state_num, 'edge_attr', state_ea) 
        

    def place_and_save_ea(self, state_num:int, obj1: str, obj2: str):
        # load edge attr of previous state
        state_ea = self.load_csv_files(state_num-1, 'edge_attr', 0)#이전스텝 불러와서 업데이트하는 방식
        ea_index = state_ea.index.tolist()
        # write 'grasp' relation with 'robot_hand'
        for node_pair in ea_index:
            np0 = node_pair.split(",")[0].lstrip("('").rstrip("'")
            np1 = node_pair.split(",")[1].lstrip(" '").rstrip("')")
            
            if obj1 == np0:
                if 'Robot_hand' in np1:
                    state_ea.loc[[node_pair], :] = 0
                elif obj2 == np1:
                    if 'Box' in np1 or 'Region' in np1:
                        state_ea.loc[[node_pair], 'rel_on_right'] = 1

            elif obj1 == np1:
                if 'Robot_hand' == np0:
                    state_ea.loc[[node_pair], :] = 0
                elif obj2 == np0:
                    if 'Region' in np0 or 'Box' in np0:
                        state_ea.loc[[node_pair], 'rel_on_left'] = 1 
        # save updated edge index
        self.save_csv_files(state_num, 'edge_attr', state_ea)
    ####################################################################################
    ## graph visualize ##
    def visualize_graph(self, fig_num):
        # Node list
        # nf_csv = self.load_csv_files(fig_num, 'node_features', 0)
        # nf_index = nf_csv.index.to_list()
        
        # Connect edge
        edge_attr_csv = self.load_csv_files(fig_num, 'edge_attr', 0)
        ea_index = edge_attr_csv.index.to_list()
            
        # edge_attr의 column 데이터 list로 가져오기
        col = edge_attr_csv.columns.to_list()
        # edge_attr file에서 'rel'이 들어간 문자열 정보 가져오기 
        ea_col = [col[i] for i in range(len(col)) if col[i].find('rel') == 0]    
        # print("\n[ea col]",ea_col)
        

        # Generate graph
        g = nx.DiGraph() # 방향성
        # g = nx.Graph() # 무방향성
        
        # g.add_nodes_from(nf_index)
        for node_pair in ea_index:
            np0 = node_pair.split(",")[0].lstrip("('").rstrip("'")
            np1 = node_pair.split(",")[1].lstrip(" '").rstrip("')")
            for rel in ea_col:
                # print(node_pair, rel)
                # print(edge_attr_csv.at[node_pair, rel])
                # input()

                if (edge_attr_csv.at[node_pair, rel] == 1).any():
                    if rel == 'rel_on_right' or rel == 'rel_on_left':
                        new_rel = rel.replace(rel, 'On')
                    elif rel == 'rel_grasp_right' or rel == 'rel_grasp_left':
                        new_rel = rel.replace(rel, 'Grasp')

                    networkx_edges = [(np0, np1, {'label': new_rel})]
                    g.add_edges_from(networkx_edges)
               

        ################### Make graph ####################

        # Can manually specify node position
        # pos = nx.spring_layout(g) # 분산된 형태
        pos = nx.shell_layout(g) # 둥근 형태
        
        # # check the position

        
        # Show title
        plt.figure(figsize=(10,8))
        title_font = {'fontsize':14, 'fontweight':'bold'}
        plt.title(f"{self.problem}_task{fig_num}", fontdict = title_font)  
        
        # print("[Graph info]", g)

        egrasp = [(u, v) for (u, v, d) in g.edges(data=True) if d["label"] == "Grasp"]
        eon = [(u, v) for (u, v, d) in g.edges(data=True) if d["label"] == "On"]

        # print("[Grasp]", egrasp)
        # print("[On]", eon)
                
        ## Draw edges from edge attributes
        
        # Task에 해당되는 node의 색깔 다르게 
        # val_map = {node1: 0.5, node2: 0.5}
        # values = [val_map.get(node, 0.25) for node in g.nodes()]
        # val_map = {node1: 0, node2: 0}
        # values = [val_map.get(node, 0) for node in g.nodes()]
        # nx.draw_networkx_nodes(G=g, pos=pos, cmap= plt.get_cmap('rainbow'), node_color= values, node_size=400)

        
        nx.draw_networkx_labels(G=g, pos=pos, labels= {node: node for node in g.nodes()})
        nx.draw_networkx_edges(G=g, pos=pos, edgelist=egrasp, width=3, alpha=0.5, edge_color='blue', style= "dotted")
        nx.draw_networkx_edges(G=g, pos=pos, edgelist=eon, width=3, alpha=0.5, edge_color= 'black')

        lgrasp = {pairs: 'Grasp' for pairs in egrasp}
        lon = {pairs: 'On' for pairs in eon}
        
        # # Draw edge labels from edge attributes
        
        nx.draw_networkx_edge_labels(G= g, pos = pos, edge_labels = lgrasp, font_size = 10)
        nx.draw_networkx_edge_labels(G= g, pos = pos, edge_labels = lon, font_size = 10)
        create_folder(f"Image_graph/{self.problem}")
        plt.savefig(f"Image_graph/{self.problem}/{self.problem}_task{fig_num}")
    ###################################################################################

saved_tasks = {}
def combined_task(input_info, problem, input_list):
    # print(input_list)
    # input()
    # 입력 리스트와 딕셔너리를 비교하여 저장
    if problem not in saved_tasks.keys():
        saved_tasks[problem] = input_list
    
    # return result_dict
    filename = "combined_task.json"  # 저장할 JSON 파일 이름
    file_path = os.path.join(os.getcwd(), 'demo_generation', 'tasks', input_info, filename)
    with open(file_path, "w") as file:
        json.dump(saved_tasks, file)


    
# main(make_data, result, input_info, tasks)
def main(task_generator, task_seq: list, obj_list):
    ######################
    ### 순서 ###
    ## 1. edge index list 생성 ##
    task_generator.make_ei_list(task_seq, obj_list)
    ## 2. edge attr list 생성 ##
    task_generator.make_ea_list(task_seq)
    ## 3. node feature 생성 ##
    task_generator.make_nf(obj_list)
    ## 3. task dictionary 저장 ##
    combined_task(task_generator.input_info, task_generator.problem, task_seq) 

    ######## Check with graphs ####################
    for image_num in range(len(task_seq)+1):
        # print(f"====================================================[Task{image_num}]====================================================")
        task_generator.visualize_graph(fig_num=image_num)
    # print("[[[Graph end]]]")
    ###############################################

###############################################################################################################3

def stacking_tuple(num):
    num_str = str(num)
    stacking_list = [(int(num_str[i]), int(num_str[i+1])) for i in range(len(num_str) - 1)]
    return stacking_list[::-1]

def generate_tasks(task_type: str):
    allowed_task_types = ['stacking', 'clustering']
    if task_type not in allowed_task_types:
        raise ValueError("Invalid task_type. Allowed values are: stacking, clustering")
    
    step_of_task = []

    # fix the number of objects and regions
    num_objects = 5
    num_regions = 3
    objects = range(1, num_objects + 1)
    regions = range(1, num_regions + 1)

    ## generate task name ##
    if task_type == 'clustering':
        cluster_regions = [2,3]
        for r_combination in product(cluster_regions, repeat= num_objects):
            cluster_name = f'{task_type}_{num_objects}_B{"".join(map(str, objects))}_R{"".join(map(str, r_combination))}'
            step_of_task.append(cluster_name)

    elif task_type == 'stacking':
        # stacking_regions = [1]
        for b_combination in permutations(objects, num_objects):
            stack_name = f'{task_type}_{num_objects}_B{"".join(map(str, b_combination))}_R1'
            step_of_task.append(stack_name)
    
    return step_of_task

def run_tasks(make_data, tasks, obj_list):
    task_seq = []
    # print(box_region)
    task_type = tasks.split('_')[0]

    if task_type == 'clustering':

        box_region = tasks.split('_')[2:]
        box_num = box_region[0][1:]
        region_num = box_region[1][1:]
        pairs = [(int(box), int(region)) for box, region in zip(box_num, region_num)]
        # print(pairs)
    
        for box, region in pairs:
            task_seq.extend([(0, f'Box{box}', None), (1, f'Box{box}', f'Region_{region}')])
        print(task_seq)
        

    elif task_type == 'stacking':
        box_region = tasks.split('_')[2:]
        box_num = box_region[0][1:]
        region = box_region[1][1:]
        pairs = stacking_tuple(box_num)
        # print(pairs)

        # task_seq.extend([(0, f'Box{box_num[-1]}', None), (1, f'Box{box_num[-1]}', f'Region_{region}')])
        for box1, box2 in pairs:
            task_seq.extend([(0, f'Box{box1}', None), (1, f'Box{box1}',f'Box{box2}')])
        print(task_seq)

    else:
        raise ValueError("Check the task name")
    
    main(make_data, task_seq, obj_list)


if __name__ == '__main__':
    dataset_name = '1114_tasks'

    obj_list =['Box1','Box2','Box3','Box4','Box5','Robot_hand','Region_1','Region_2','Region_3']

    cluster_tasks_names = generate_tasks(task_type='clustering')
    # print("\n[Cluster tasks name]",cluster_tasks_names, len(cluster_tasks_names))

    stack_tasks_names = generate_tasks(task_type='stacking')
    # print("\n[Stack tasks name]",stack_tasks_names, len(stack_tasks_names))

    # full_tasks = cluster_tasks_names + stack_tasks_names
    # print(full_tasks, len(full_tasks)) #152

    for tasks_names in [cluster_tasks_names, stack_tasks_names]:
        for tasks in tasks_names:
            task_generator = GenTaskSequence(information=dataset_name, problem=tasks)
            run_tasks(task_generator, tasks, obj_list)
