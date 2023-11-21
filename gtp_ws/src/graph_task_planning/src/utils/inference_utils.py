import sys
sys.path.extend("./")
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from torch_geometric.utils import to_dense_adj
from utils.graph_utils import to_dense_graph, concat_edges

class GraphDynamicsUpdate():
    def __init__(self, information, text_basic_nf, debug_mode = False):        
        self.information = information
        self.text_basic_nf = text_basic_nf
        self.ea_col = ['rel_on_right', 'rel_on_left', 'rel_in_right', 'rel_in_left', 'rel_in_grasp', 'rel_grasp']


        self.nf_col = text_basic_nf.columns.to_list()
        # self.obj_list = 
        self.obj_list = self.unique_obj_list(text_basic_nf.index.to_list())
        self.text_basic_nf.index = self.obj_list
        self.num_object = len(self.obj_list)

        # code를 name으로 변환하기 위한 dictionary
        self.obj_idx_to_name = {k:v for k,v in enumerate(self.obj_list)}
        self.act_idx_to_name = {0:"Pick",
                                1:"Place",
                                2:"Pour"}
        self.obj_name_to_idx = {v:k for k,v in self.obj_idx_to_name.items()}
        self.act_name_to_idx = {v:k for k,v in self.act_idx_to_name.items()}

        self.debug_mode = debug_mode
    
    def unique_obj_list(self, lst):
        counts = {}
        result = []

        for item in lst:
            if item in counts:
                result.append(f"{item}_{counts[item]}")
                counts[item] += 1
            else:
                counts[item] = 1
                result.append(item)

        return result
    def split_goal_state(self, concat_graph):
        goal_edge_index = concat_graph['edge_index'].clone()
        goal_edge_attr = concat_graph['edge_attr'][:, self.edge_feature_dim:]
        goal_edge_index, goal_edge_attr = to_dense_graph(goal_edge_index, goal_edge_attr)
        
        state_edge_index = concat_graph['edge_index'].clone()
        state_edge_attr = concat_graph['edge_attr'][:, :self.edge_feature_dim]
        state_edge_index, state_edge_attr = to_dense_graph(state_edge_index, state_edge_attr)

        return goal_edge_index, goal_edge_attr, state_edge_index, state_edge_attr
    
    def find_ea_index(self, src, dest):
        # 주어진 엣지 정보(src, dest) 에 해당하는 edge index값을 리턴
        for idx in range(self.state_ei.size(-1)):
            pair = list(map(int, self.state_ei[:, idx].tolist()))
            if pair == [src, dest]:
                return idx
        return None
    def relaion_split(self, relation):
        # relation 구분
        if 'On' in relation:
            rel_right = "rel_on_right"
            rel_left = "rel_on_left"
        elif 'In' in relation:
            rel_right = "rel_in_right"
            rel_left = "rel_in_left"
        elif 'Grasp' in relation:
            rel_right = "rel_grasp"
            rel_left = "rel_in_grasp"
        else:
            raise TypeError("Wrong Relation")

        if 'reverse' in relation:
            temp = rel_left
            rel_left = rel_right
            rel_right = temp
        
        return rel_right, rel_left
    
    def check_obj_prop(self, target_obj, property):
        if self.text_basic_nf.loc[target_obj, property] == 1:
            return True
        else:
            return False
        
    def find_related_obj(self, target_obj, relation, find_all=False):
        rel_right, rel_left = self.relaion_split(relation)

        target_obj_code = self.obj_name_to_idx[target_obj]

        related_obj_set = set()
        for i in range(self.num_object):
            ea_idx = self.find_ea_index(target_obj_code, i)
            if ea_idx is not None:
                if self.state_ea_df.loc[ea_idx, rel_right] == 1:
                    if find_all:
                        related_obj_set.add(i)
                        continue
                    else:
                        return i
            ea_idx = self.find_ea_index(i, target_obj_code)
            if ea_idx is not None:
                if self.state_ea_df.loc[ea_idx, rel_left] == 1:
                    if find_all:
                        related_obj_set.add(i)
                        continue
                    else:
                        return i
        return list(related_obj_set) if find_all else None
    
    def remove_relation(self, target_obj, dest_obj, relation):
        rel_right, rel_left = self.relaion_split(relation)

        target_obj_code = self.obj_name_to_idx[target_obj]

        if dest_obj is None:#모든 object에 대해 실행
            dest_obj_code_list = [*range(len(self.obj_list))]
            pass
        else:#주어진 object에 대해서만 실행
            dest_obj_code_list = [self.obj_name_to_idx[dest_obj]]
        for dest_obj_code in dest_obj_code_list:
            #right
            ea_idx_right = self.find_ea_index(target_obj_code, dest_obj_code)
            if ea_idx_right is not None:
                self.state_ea_df.loc[ea_idx_right, rel_right] = 0
            #left
            ea_idx_left = self.find_ea_index(dest_obj_code, target_obj_code)
            if ea_idx_left is not None:
                self.state_ea_df.loc[ea_idx_left, rel_left] = 0

    def add_relation(self, target_obj, dest_obj, relation):
        rel_right, rel_left = self.relaion_split(relation)

        target_obj_code = self.obj_name_to_idx[target_obj]

        if dest_obj is None:#모든 object에 대해 실행
            dest_obj_code_list = [*range(len(self.obj_list))]
            pass
        else:#주어진 object에 대해서만 실행
            dest_obj_code_list = [self.obj_name_to_idx[dest_obj]]
        
        for dest_obj_code in dest_obj_code_list:
            # zero_edges 추가
            self.add_zero_ea(target_obj_code, dest_obj_code)
            #right
            ea_idx_right = self.find_ea_index(target_obj_code, dest_obj_code)
            if ea_idx_right is not None:
                self.state_ea_df.loc[ea_idx_right, rel_right] = 1
            #left
            ea_idx_left = self.find_ea_index(dest_obj_code, target_obj_code)
            if ea_idx_left is not None:
                self.state_ea_df.loc[ea_idx_left, rel_left] = 1

    def add_zero_ea(self, target_obj_code, dest_obj_code):
        # state_ea 추가
        ea_size = list(self.state_ea_df.shape)[0]
        self.state_ea_df.loc[ea_size] = [0]*len(self.ea_col)
        self.state_ea_df.loc[ea_size+1] = [0]*len(self.ea_col)
        # state_ei 추가
        new_ei1 = torch.tensor([[target_obj_code],[dest_obj_code]], dtype=torch.long)
        new_ei2 = torch.tensor([[dest_obj_code],[target_obj_code]],dtype=torch.long)
        self.state_ei = torch.cat([self.state_ei, new_ei1, new_ei2], dim=-1)


    def pick_execute(self, object_name):
        #로봇핸드에 물건이 들어있는지 체크 -> 있는 상태면 수행X
        if self.find_related_obj("Robot hand", "Grasp") is not None:
            if self.debug_mode:
                print("Another obj is already grasped in hand\nPick Failed")
            return False
        else: #로봇 핸드가 비어있음
            # object가 graspble 하지 않으면 -> 수행 X
            if not self.check_obj_prop(object_name, "Property_G"):
                if self.debug_mode:
                    print("Target object is not graspable\nPick Failed")
                return False
            else:#object가 graspable한 물체인 경우
                #주어진 object가 이미 다른 bowl안에 들어있으면 -> 수행 X
                if self.find_related_obj(object_name, "In") is not None:
                    if self.debug_mode:
                        print("Target object is in bowl\nPick Falied")
                    return False
                #주어진 object 위에 이미 다른물체가 있으면 -> 수행 X
                elif self.find_related_obj(object_name, "On_reverse") is not None:
                    if self.debug_mode:
                        print("Another object is already on target object\nPick Failed")
                    return False
                else:#not in bowl&on_clean
                    if self.debug_mode:
                        print(f"Pick[{object_name}] is feasible")

                    #target object와 관련된 on relation제거
                    self.remove_relation(object_name, None, "On")
                    #로봇핸드와 target object 사이에 grasp relation 설정
                    self.add_relation("Robot hand", object_name, "Grasp")

                    return True

    def place_execute(self, object_name):
        # 현재 로봇이 잡고 있는 물체 찾기
        in_hand_obj_code = self.find_related_obj("Robot hand", "Grasp")

        #로봇 핸드에 물체 없으면 -> 수행 X
        if in_hand_obj_code is None:
            if self.debug_mode:
                print("There is no obj in robot hand\nPlace Failed")
            return False
        #로봇 핸드에 물체 집혀 있는 상태
        else:
            in_hand_obj = self.obj_idx_to_name[in_hand_obj_code]
            if self.debug_mode:
                print(f"In_hand obj: {in_hand_obj}")
            #in_hand object가 bowl인데, target object가 region이 아닌 경우
            #bowl은 region위에만 place할 수 있음 -> 수행 X
            # if self.check_obj_prop(in_hand_obj, "Type_Bowl") and not self.check_obj_prop(object_name, "Type_Region"):
            #     print("Bowl object can be placed on Region only\nPlace Failed")
            #     return None
            
            # #object가 region이 아니고 위에 이미 다른물체 있으면 -> 수행 X
            # elif (self.check_obj_prop(object_name, "Type_Region") is False) and (self.find_related_obj(object_name, "On_reverse") is not None):
            #     print("Another object is already on target object\nPlace Failed")
            #     return None
            
            #object가 region이 아닌 경우
            if self.check_obj_prop(object_name, "Type_Region") is False:
                # target object 위에 이미 다른물체 있으면 -> 수행 X
                if self.find_related_obj(object_name, "On_reverse") is not None:
                    if self.debug_mode:
                        print("Another object is already on target object\nPlace Failed")
                    return False
                # target object가 로봇 핸드에 잡혀있으면 -> 수행 X
                elif self.find_related_obj(object_name, "Grasp_reverse") is not None:
                    if self.debug_mode:
                        print("Target object is grasped in robot hand\nPlace Failed")
                    return False
                # target object가 다른 bowl안에 들어있으면 -> 수행 X
                elif self.find_related_obj(object_name, "In") is not None:
                    if self.debug_mode:
                        print("Target object is already in another bowl\nPlace Failed")
                    return False
                # in hand object가 bowl인 경우 -> 수행 X
                elif self.check_obj_prop(in_hand_obj, "Type_Bowl"):
                    if self.debug_mode:
                        print("Bowl object can be placed on Region only\nPlace Failed")
                    return False  
            
            #target object가 Robot hand이면 -> 수행 X
            elif self.check_obj_prop(object_name, "Type_Robot"):
                if self.debug_mode:
                    print("The object can't be placed on Robot hand\nPlace Failed")
                return False
            
            # else:
            if self.debug_mode:
                print(f"Place[{object_name}] is feasible")
            #로봇 핸드와 in_hand obj 사이의 grasp relation을 제거
            self.remove_relation("Robot hand", in_hand_obj, "Grasp")
            
            # in_hand obj가 box이면서 target obj가 bowl인 경우
            # -> in relation 추가
            if self.check_obj_prop(in_hand_obj, "Type_Box") and self.check_obj_prop(object_name, "Type_Bowl"):
                if self.debug_mode:
                    print("in hand obj: box, target obj: bowl")
                self.add_relation(in_hand_obj, object_name, "In")

            # 나머지 경우 -> on relation 추가
            else: #on relations
                self.add_relation(in_hand_obj, object_name, "On")
            return True

    def pour_execute(self, object_name):
        # 현재 로봇이 잡고 있는 물체 찾기
        in_hand_obj_code = self.find_related_obj("Robot hand", "Grasp")
        
        # 로봇 핸드에 물체 없는경우 -> 수행 X
        if in_hand_obj_code is None:
            if self.debug_mode:
                print("There is no obj in robot hand\nPour Failed")
            return False
        #로봇 핸드에 물체 집혀 있는 상태
        else:
            in_hand_obj = self.obj_idx_to_name[in_hand_obj_code]
            if self.debug_mode:
                print(f"In_hand obj: {in_hand_obj}")
            # in_hand obj가 bowl이 아닌 경우 -> 수행 X
            if not self.check_obj_prop(in_hand_obj, "Type_Bowl"):
                if self.debug_mode:
                    print("The obj in robot hand is not bowl\nPour Failed")
                return False
            # 로봇 핸드에 bowl이 집혀있는 경우
            else:#in_hand_object = bowl
                #target obj가 bowl이 아닌 경우 -> 수행 X
                if not self.check_obj_prop(object_name, "Type_Bowl"):
                    if self.debug_mode:
                        print("The target object is not bowl\nPour Failed")
                    return False
                # target obj bowl이 로봇 핸드에 잡혀있으면 -> 수행 X
                elif self.find_related_obj(object_name, "Grasp_reverse") is not None:
                    if self.debug_mode:
                        print("Target object is grasped in robot hand\nPour Failed")
                    return False
                #target obj도 bowl인 경우
                else:#object_code = target bowl
                    if self.debug_mode:
                        print(f"Pour[{object_name}] is feasible")
                    #in_hand bowl 안에 있는 box 체크
                    in_bowl_code_list = self.find_related_obj(in_hand_obj, "In_reverse", find_all=True)
                    in_bowl_list = [self.obj_idx_to_name[code] for code in in_bowl_code_list]

                    # in_hand bowl 안에 있는 box들에 대해서 각각 수행
                    for in_bowl_obj in in_bowl_list:
                        # in_bowl box와 in_hand bowl사이에 in relation 제거
                        self.remove_relation(in_bowl_obj, in_hand_obj, "In")
                        # in_bowl box와 target bowl사이에 in relation 추가
                        self.add_relation(in_bowl_obj, object_name, "In")
                    return True

    def graph_update(self, test_input, action_code, object_code):
        self.edge_feature_dim = int(test_input['edge_attr'].size(-1)/2)
        updated_graph = test_input.clone().cpu()
        #goal, state 분리
        self.goal_ei, self.goal_ea, self.state_ei, self.state_ea = self.split_goal_state(updated_graph)

        #state_ea를 df로 변환
        self.state_ea_df = pd.DataFrame(self.state_ea.numpy(), columns=self.ea_col)
        #state_ei를 df로 변환
        # print(self.obj_list)
        # print(to_dense_adj(self.state_ei, max_num_nodes=len(self.obj_list)).squeeze().shape)
        self.state_ei_df = pd.DataFrame(to_dense_adj(self.state_ei, max_num_nodes=len(self.obj_list)).squeeze().numpy(), index=self.obj_list, columns=self.obj_list)

        #action, object code를 name으로 변환
        action_name = self.act_idx_to_name[action_code]
        object_name = self.obj_idx_to_name[object_code]

        #action에 따라 dynamics update
        if self.debug_mode:
            print("Target object:", object_name)
        if action_name == 'Pick':
            if self.debug_mode:
                print("Pick execute")
            is_feasible = self.pick_execute(object_name)

        elif action_name == 'Place':
            if self.debug_mode:
                print("Place execute")
            is_feasible = self.place_execute(object_name)
        elif action_name == 'Pour':
            if self.debug_mode:
                print("Pour execute")
            is_feasible = self.pour_execute(object_name)
        else:
            raise TypeError("Wrong action [Pick, Place, Pour]")
        if not is_feasible:
            return None
        else:
            # state_ea_df를 state_ea tensor로 변환
            self.state_ea = torch.tensor(self.state_ea_df.values)
            # zero edge remove
            self.state_ei, self.state_ea = remove_zero_edges(self.state_ei, self.state_ea)

            # state와 goal을 concat
            updated_ei, updated_ea = concat_edges(self.state_ei, self.state_ea, self.goal_ei, self.goal_ea)
            updated_graph['edge_index'] = updated_ei
            updated_graph['edge_attr'] = updated_ea
            return updated_graph
    def plan_name_to_idx(self, name_plan):
        idx_plan = []
        for plan in name_plan:
            act_name, obj_name = plan
            act_idx = self.act_name_to_idx[act_name]
            obj_idx = self.obj_name_to_idx[obj_name]
            idx_plan.append(tuple([act_idx, obj_idx]))
        return idx_plan

def key_edge_compare(cur_ei, cur_ea, goal_ei, goal_ea):
    cur_edges = []
    goal_edges = []
    key_edges = [] # goal에는 있는데 current에는 없는 edge의 list
    for cur_idx in range(cur_ei.size(-1)):
        ei = cur_ei[:,cur_idx]
        ea = cur_ea[cur_idx,:]
        cur_edges.append(GraphEdge(ei,ea))
    for goal_idx in range(goal_ei.size(-1)):
        ei = goal_ei[:,goal_idx]
        ea = goal_ea[goal_idx,:]
        key_candidate= GraphEdge(ei,ea)
        goal_edges.append(key_candidate)
        is_key = True
        # print('key_edge:\n',key_candidate)
        for comp in cur_edges:
            if edge_equality_checking(key_candidate, comp) is True:
                is_key = False
                break
        if is_key:
            # print('key detected')
            key_edges.append(key_candidate)
            # input()
    
    
    # print((cur_edges[0].src, cur_edges[0].dest, cur_edges[0].ea.tolist()))
    # print((goal_edges[0].src, goal_edges[0].dest, goal_edges[0].ea.tolist()))
    # print(len(key_edges))
    # for key_edge in key_edges:
    #     print(key_edge)
    return len(key_edges)

def edge_equality_checking(edge1, edge2):
    if edge1.src==edge2.src and edge1.dest==edge2.dest:
        if torch.equal(edge1.ea, edge2.ea):
            return True
    return False

def remove_zero_edges(state_edge_index, state_edge_attr):
    #dynamics가 update되면서 생긴 zero edge를 제거
    ze_idx_mask = []
    for idx in range(state_edge_index.size(-1)):
        if torch.equal(state_edge_attr[idx,:],torch.tensor([0]*state_edge_attr.size(-1), dtype=torch.float32)):
            ze_idx_mask.append(False)
        else:
            ze_idx_mask.append(True)
    return state_edge_index[:,ze_idx_mask], state_edge_attr[ze_idx_mask,:]

def add_zero_edges(state_edge_index, state_edge_attr, obj1, obj2):
    new_ei1 = torch.tensor([[obj1],[obj2]], dtype=torch.long)
    new_ei2 = torch.tensor([[obj2],[obj1]],dtype=torch.long)
    # print(state_edge_index)
    # print(state_edge_attr)
    state_edge_index = torch.cat([state_edge_index, new_ei1, new_ei2], dim=-1)
    state_edge_attr = torch.cat([state_edge_attr, torch.zeros(2,6).type(torch.float32)], dim=0)
    # print(state_edge_index)
    # print(state_edge_attr)
    # input()
    return state_edge_index, state_edge_attr


class GraphEdge():
    def __init__(self, ei, ea):
        [self.src, self.dest] = ei.tolist()
        self.ea = ea
    def __str__(self):
        return f"GraphEdge\n  src={self.src}\n  dest={self.dest}\n  ea={self.ea.tolist()}"
    
class GraphDataNode_RNN():
    # 생성자 메소드
    def __init__(self, current, goal, prev_hidden=None):
        self.current = current  # state를 받음
        self.goal = goal
        self.parent = None  # parent를 받음
        self.children = []     # child list
        self.distance = 999   # distance를 초기화
        self.executed_ac = None  #수행된 action type
        self.executed_oc = None  #수행된 target object
        self.prev_hidden = prev_hidden
        self.level = 0

class SearchingTree():
    def __init__(self, root):
        self.root = root
        self.node_list = [root]
    
    def __len__(self):
        return len(self.node_list)
    
    def add_node(self, node_parent, node_new):
        node_new.parent = node_parent
        self.node_list.append(node_new)
        node_parent.children.append(node_new)

class GraphDataNode_Basic():
    # 생성자 메소드
    def __init__(self, current, goal):
        self.current = current  # state를 받음
        self.goal = goal
        self.parent = None  # parent를 받음
        self.children = []     # child list
        self.distance = 999   # distance를 초기화
        self.executed_ac = None  #수행된 action type
        self.executed_oc = None  #수행된 target object
        self.level = 0


def leaf_node_checking(planning_tree):
    leaf_node_list = []
    for node in planning_tree.node_list:
        if len(node.children) == 0:
            leaf_node_list.append(node)
    return leaf_node_list

def dist_checking(current_ea, goal_ea):
    score = torch.sum(current_ea==goal_ea).item()
    return score

def loop_checking(target_node, current_node):
    node_pointer = current_node
    is_loop = False
    while node_pointer.parent is not None:
        if torch.equal(node_pointer.current['edge_attr'], target_node.current['edge_attr']):
            is_loop = True
            return is_loop
        else:
            node_pointer = node_pointer.parent
    return is_loop

def print_planningtree(node_pointer,level):

    branch = '\t|'
    level_indent = '\t'*(level-1) + branch
    if level==0:
        level_indent = ''
    # print(level_indent+'state:'+node_pointer.state)
    print(level_indent+f'action:({node_pointer.executed_ac}, {node_pointer.executed_oc})'+f'dist:{node_pointer.distance}')
    for child in node_pointer.children:
        print_planningtree(child, level+1)

def plot_and_save(model_path, model_param, show_result):
    with open(os.path.join(model_path, "loss_data"), "rb") as file:
        loss_data = pickle.load(file)
    
    epoch_list = loss_data["epoch"]
    
    #Loss Plot
    plt.figure(0)
    plt.figure(figsize=(15, 5))
    plt.suptitle('Loss')

    #Total Loss
    loss_list = []
    val_loss_list = []
    for loss in loss_data['loss']['total']['train']:
        loss_list.append(loss)
    for val_loss in loss_data['loss']['total']['val']:
        val_loss_list.append(val_loss)
    plt.subplot(1,3,1)
    plt.plot(epoch_list, loss_list, label='train')
    plt.plot(epoch_list, val_loss_list , label='val')
    plt.legend(loc='upper right')
    plt.title('total')

    #Action Loss
    act_loss_list = []
    val_act_loss_list = []
    for loss in loss_data['loss']['action']['train']:
        act_loss_list.append(loss)
    for val_loss in loss_data['loss']['action']['val']:
        val_act_loss_list.append(val_loss)
    plt.subplot(1,3,2)
    plt.plot(epoch_list, act_loss_list, label='train')
    plt.plot(epoch_list, val_act_loss_list , label='val')
    plt.legend(loc='upper right')
    plt.title('action')

    #Object Loss
    obj_loss_list = []
    val_obj_loss_list = []
    for loss in loss_data['loss']['object']['train']:
        obj_loss_list.append(loss)
    for val_loss in loss_data['loss']['object']['val']:
        val_obj_loss_list.append(val_loss)
    plt.subplot(1,3,3)
    plt.plot(epoch_list, obj_loss_list, label='train')
    plt.plot(epoch_list, val_obj_loss_list , label='val')
    plt.legend(loc='upper right')
    plt.title('object')

    plt.savefig(os.path.join(model_path, "_".join(list(map(str, model_param))) +'_loss.png'))

    #Accuracy Plot
    plt.figure(1)
    plt.figure(figsize=(10, 5))
    plt.suptitle('Accuracy')

    #Action Accuracy
    act_acc_list = []
    val_act_acc_list = []
    for acc in loss_data['acc']['action']['train']:
        act_acc_list.append(acc)
    for val_acc in loss_data['acc']['action']['val']:
        val_act_acc_list.append(val_acc)
    plt.subplot(1,2,1)
    plt.plot(epoch_list, act_acc_list, label='train')
    plt.plot(epoch_list, val_act_acc_list , label='val')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.title('action')

    #Object Accuracy
    obj_acc_list = []
    val_obj_acc_list = []
    for acc in loss_data['acc']['object']['train']:
        obj_acc_list.append(acc)
    for val_acc in loss_data['acc']['object']['val']:
        val_obj_acc_list.append(val_acc)
    plt.subplot(1,2,2)
    plt.plot(epoch_list, obj_acc_list, label='train')
    plt.plot(epoch_list, val_obj_acc_list , label='val')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.title('object')

    plt.savefig(os.path.join(model_path, "_".join(list(map(str, model_param))) +'_acc.png'))
    if show_result:
        plt.show()
    