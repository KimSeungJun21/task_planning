from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
#for GPU server(headless)
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import os
import copy
import sys
import time
sys.setrecursionlimit(10**6)

sys.path.extend("./")
from model.model import ActionModel_Basic
from utils.graph_collate import BasicDataset
from utils.inference_utils import GraphDynamicsUpdate, GraphDataNode_RNN, GraphDataNode_Basic, GraphEdge, SearchingTree
from utils.inference_utils import plot_and_save, remove_zero_edges, key_edge_compare, leaf_node_checking, loop_checking, print_planningtree
from utils.graph_utils import batch_split
#######
# RNN #
#######
def inference_rnn_sequence(device,
                           text_encoder,
                           gradient_clipping,
                           model_choose, method,
                           information,
                           maximum_plan_length,
                           hidden_dim,
                           num_action,
                           num_object,
                           node_feature_size,
                           edge_feature_size,
                           global_dim, batch_size,
                           lr,
                           train_seed,
                           num_workers,
                           num_epoch,
                           data_dir,
                           show_result,
                           infer_num=None,
                           check_each = False,
                           data_save=True,
                           weight_decay=0):

    model_param = [data_dir,hidden_dim, num_epoch, batch_size, lr]
    model_path = os.path.join(os.getcwd(),
                              "result",
                              information,
                              text_encoder,
                              f"gradient_clipping_{gradient_clipping}",
                              f"weight_decay_{weight_decay}",
                              "_".join(list(map(str, model_param))),
                              f"train_seed_{train_seed}",
                              model_choose,
                              method)

    plot_and_save(model_path, model_param, show_result)

    if infer_num is not None:
        model_name = 'GP_model_{}.pt'.format(infer_num)
    else:
        model_name = 'GP_model_best.pt'

    saved_path = os.path.join(model_path, model_name)
    saved_model = ActionModel_Recurrent(device,
                                        hidden_dim,
                                        num_action, 
                                        num_object, 
                                        node_feature_size, 
                                        edge_feature_size, 
                                        global_dim, 
                                        model_choose, 
                                        method)
    saved_model.load_state_dict(torch.load(saved_path))
    saved_model.to(device)

    for param in saved_model.parameters():
        param.requires_grad = False

    data_test = RecurrentActionDataset(data_dir, text_encoder,'test')
    data_test_loader = DataLoader(data_test, 1)

    obj_list_dir = os.path.join(os.getcwd(),
                                'demo_generation',
                                "datasets",
                                data_dir,
                                'test_obj_list')

    plan_total = 0
    plan_success = 0

    plan_shorten = 0
    plan_shorten_dict = {}

    saved_model.eval()

    plan_result_data = {'data_info':{},
                        'plan_list':[],
                        'plan_len':int(),
                        'plan_reached':bool(),
                        'time':float()}
    
    debug_mode = False

    for test_i, test_data in enumerate(data_test_loader):
        plan_total += 1
        goal_planned = False
        test_current, test_info = test_data
        if debug_mode:
            print("#########################################")
            print("demo type:", test_info['demo'])
            print("graph_num:",test_info['graph_num'])
            print("--------------------------------")
        #text_basic_nf load
        graph_num = test_info['graph_num'][0]
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        text_basic_nf = pd.read_csv(os.path.join(obj_list_dir,f"text_basic_nf_{graph_num}.csv"),index_col=0)
        dynamics_module = GraphDynamicsUpdate(information, text_basic_nf, debug_mode=debug_mode)
        # split batched sequence graph
        split_list = batch_split(test_current.to(device))
        # load initial state graph only
        current_state = split_list[0]
        # time record
        start = time.time()
        # model inference
        pred_action_prob, pred_object_prob, rnn_outputs, status = saved_model(current_state, [1])

        # action X object prob => prob.table
        pred_action_prob = F.softmax(pred_action_prob, dim=-1) #1X1X3
        pred_object_prob = F.softmax(pred_object_prob, dim=-1) #1X1Xnum_object
        act_obj_prob_table = torch.matmul(torch.transpose(pred_action_prob.squeeze(0), 0, 1), pred_object_prob.squeeze(0))
        # read the most probable action&object
        sorted_table, indices = torch.sort(act_obj_prob_table.view(-1), descending=True)
        action_code, object_code = divmod(indices[0].item(),pred_object_prob.size(-1))
        # convert code to name
        action_name = dynamics_module.act_idx_to_name[action_code]
        object_name = dynamics_module.obj_idx_to_name[object_code]

        if debug_mode:
            print(action_name, object_name)

        # graph state update
        updated_graph = dynamics_module.graph_update(current_state, action_code, object_code)
        # plan sequence update
        planned_action = [tuple([action_name, object_name])]

        # sequential planning until maximum plan length limit
        while len(planned_action) < maximum_plan_length:
            if debug_mode:
                print(planned_action)
                print(dynamics_module.plan_name_to_idx(planned_action))
                # input()
            # stop planning if planned action is infeasible
            if updated_graph is None:
                if debug_mode:
                    print('not feasible action is planned')
                break
            else:
                # split graph into current state and goal state
                state_ei = updated_graph['edge_index'].clone()
                state_ea = updated_graph['edge_attr'][:, :int(edge_feature_size)].clone()
                state_ei, state_ea = remove_zero_edges(state_ei, state_ea)

                goal_ei = updated_graph['edge_index'].clone()
                goal_ea = updated_graph['edge_attr'][:, int(edge_feature_size):].clone()
                goal_ei, goal_ea = remove_zero_edges(goal_ei, goal_ea)

                # goal check - key_edge_compare
                if key_edge_compare(state_ei, state_ea, goal_ei, goal_ea) == 0:
                    if debug_mode:
                        print('plan reached')
                    plan_success += 1
                    goal_planned = True
                    break
                else:
                    # if state didn't reach goal - plan again
                    current_state = updated_graph.clone()
                    # model inference with updated state
                    pred_action_prob, pred_object_prob, rnn_outputs, status = saved_model(current_state.clone(),
                                                                                        [1],
                                                                                        True,
                                                                                        status)
                    # action X object prob => prob.table
                    pred_action_prob = F.softmax(pred_action_prob, dim=-1) #1X1X3
                    pred_object_prob = F.softmax(pred_object_prob, dim=-1) #1X1Xnum_object
                    act_obj_prob_table = torch.matmul(torch.transpose(pred_action_prob.squeeze(0), 0, 1), pred_object_prob.squeeze(0))
                    # read the most probable action&object
                    sorted_table, indices = torch.sort(act_obj_prob_table.view(-1), descending=True)
                    action_code, object_code = divmod(indices[0].item(),pred_object_prob.size(-1))
                    # convert code to name
                    action_name = dynamics_module.act_idx_to_name[action_code]
                    object_name = dynamics_module.obj_idx_to_name[object_code]

                    if debug_mode:
                        print(action_name, object_name)
                    # graph state update
                    updated_graph = dynamics_module.graph_update(current_state, action_code, object_code)
                    # plan sequence update
                    planned_action.append(tuple([action_name, object_name]))
            # input()
        end = time.time()

        # shorten plan result checking
        if goal_planned:
            if test_info['goal'].item() > len(planned_action):
                plan_shorten += 1
                shorten_task_name = test_info['demo'][0] + '_' + str(graph_num)
                plan_shorten_dict[shorten_task_name] = planned_action

        # success rate check
        print(f'success rate: {plan_success}/{plan_total} = {plan_success/plan_total:.4f}\t planning time: {end-start:.4f}')

        if debug_mode:
            if goal_planned is False:
                input()
        # inference result save
        plan_result_data['data_info'] = test_info
        plan_result_data['plan_list'] = planned_action
        plan_result_data['plan_len'] = len(planned_action)
        plan_result_data['plan_reached'] = goal_planned
        plan_result_data['time'] = end-start

        if data_save:
            file_dir = os.path.join(os.getcwd(),
                                    "test_result",
                                    information,
                                    text_encoder,
                                    f"gradient_clipping_{gradient_clipping}",
                                    f"weight_decay_{weight_decay}",
                                    "_".join(list(map(str, model_param))),
                                    f"train_seed_{train_seed}",
                                    model_choose,
                                    method,
                                    "Sequential_planning")
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            file_path = os.path.join(file_dir,"plan_"+str(plan_total))
            with open(file_path, "wb") as outfile:
                pickle.dump(plan_result_data, outfile)

    print('#####################################')
    print(f'success rate: {plan_success}/{plan_total} = {plan_success/plan_total}')
    print(f'shorten plan: {plan_shorten}/{plan_success} = {(plan_shorten/plan_success) if plan_success > 0 else 0}')
    print("List:")
    for task, plan in plan_shorten_dict.items():
        print(task)
        print(plan)
    print('#####################################')

    #output text save
    result_path = os.path.join(model_path, f"sequence_output_{infer_num}.txt")
    with open(result_path, 'w') as f:
        print(f'success rate: {plan_success}/{plan_total} = {plan_success/plan_total}')
        print(f'success rate: {plan_success}/{plan_total} = {plan_success/plan_total}', file=f)
        print(f'shorten plan: {plan_shorten}/{plan_success} = {(plan_shorten/plan_success) if plan_success > 0 else 0}', file=f)
        print("List:", file=f)
        print(plan_shorten_dict, file=f)

def inference_rnn_step(device,
                       text_encoder,
                       gradient_clipping,
                       model_choose, method,
                       information,
                       maximum_plan_length,
                       hidden_dim,
                       num_action,
                       num_object,
                       node_feature_size,
                       edge_feature_size,
                       global_dim, batch_size,
                       lr,
                       train_seed,
                       num_workers,
                       num_epoch,
                       data_dir,
                       show_result,
                       infer_num=None,
                       check_each = False,
                       data_save=True,
                       weight_decay=0):   
    
    model_param = [data_dir,hidden_dim, num_epoch, batch_size, lr]
    model_path = os.path.join(os.getcwd(),
                              "result",
                              information,
                              text_encoder,
                              f"gradient_clipping_{gradient_clipping}",
                              f"weight_decay_{weight_decay}",
                              "_".join(list(map(str, model_param))),
                              f"train_seed_{train_seed}",
                              model_choose,
                              method)

    plot_and_save(model_path, model_param, show_result)

    if infer_num is not None:
        model_name = 'GP_model_{}.pt'.format(infer_num)
    else:
        model_name = 'GP_model_best.pt'

    saved_path = os.path.join(model_path, model_name)
    saved_model = ActionModel_Recurrent(device,
                                        hidden_dim,
                                        num_action, 
                                        num_object, 
                                        node_feature_size, 
                                        edge_feature_size, 
                                        global_dim, 
                                        model_choose, 
                                        method)
    saved_model.load_state_dict(torch.load(saved_path))
    saved_model.to(device)

    for param in saved_model.parameters():
        param.requires_grad = False

    data_test = RecurrentActionDataset(data_dir, text_encoder,'test')
    data_test_loader = DataLoader(data_test, 1)

    loss_ce_action = nn.CrossEntropyLoss().to(device)
    loss_ce_object = nn.CrossEntropyLoss().to(device)

    test_num_act_correct = 0
    test_num_obj_correct = 0
    test_num_total = 0

    saved_model.eval()

    plan_result_data = {'data_info':{},
                        'action_corrected':bool(),
                        'object_corrected':bool()}
    
    debug_mode = False

    for test_i, test_data in enumerate(data_test_loader):
        test_current, test_info = test_data
        if debug_mode:
            print("#########################################")
            print("demo type:", test_info['demo'])
            print("graph_num:",test_info['graph_num'])
            print("--------------------------------")

        test_seq_len_list = test_info['goal'].tolist()

        test_pred_action_prob, test_pred_object_prob, _, _ = saved_model(test_current, test_seq_len_list)

        test_target_label_list = [torch.tensor(test_label_seq) for test_label_seq in test_current['y']]
        test_target_label_padded = nn.utils.rnn.pad_sequence(test_target_label_list,
                                                batch_first=True,
                                                padding_value=-1).to(device)

        # action loss & acc
        test_target_action_label = test_target_label_padded[:,:,0]
        test_L_action = loss_ce_action(test_pred_action_prob[test_target_action_label!=-1], test_target_action_label[test_target_action_label!=-1])

        test_pred_action_label = torch.argmax(test_pred_action_prob, dim=-1)
        test_num_act_correct += torch.sum(test_pred_action_label[test_target_action_label!=-1]==test_target_action_label[test_target_action_label!=-1])

        # object loss & acc
        test_target_object_label = test_target_label_padded[:,:,1]
        test_L_object = loss_ce_object(test_pred_object_prob[test_target_object_label!=-1], test_target_object_label[test_target_object_label!=-1])

        test_pred_object_label = torch.argmax(test_pred_object_prob, dim=-1)
        test_num_obj_correct += torch.sum(test_pred_object_label[test_target_object_label!=-1]==test_target_object_label[test_target_object_label!=-1])

        test_num_total += sum(test_seq_len_list)

        test_L_total = 0.1*test_L_action + test_L_object

        action_correct_mask = test_pred_action_label[test_target_action_label!=-1]==test_target_action_label[test_target_action_label!=-1]
        object_correct_mask = test_pred_object_label[test_target_object_label!=-1]==test_target_object_label[test_target_object_label!=-1]
        if debug_mode:
            print("data info:")
            print("demo type:", test_info['demo'])
            print("goal:", test_info['goal'])
            print("--------------------------------")
            print("Loss:")
            print("L_total:", test_L_total.item())
            print("L_action:", test_L_action.item())
            print("L_object:", test_L_object.item())
            print("--------------------------------")
            print("prediced prob:")
            for idx_seq in range(test_seq_len_list[0]):
                print("--------------------------------")
                print(f"{idx_seq} / {test_seq_len_list[0]} ...")
                print("pred_action_score:", F.softmax(test_pred_action_prob[0,idx_seq,:], dim=-1))
                print("target_action_prob:",test_target_action_label[0,idx_seq])
                if torch.argmax(test_pred_action_prob[0,idx_seq], dim=-1) == torch.argmax(test_target_action_label[0,idx_seq]):
                    print("Action Prediction Success!")
                else:
                    print("Action Prediction FailedTT\n")
                print("pred_object_score:", F.softmax(test_pred_object_prob[0,idx_seq], dim=-1))
                print("target_object_prob:",test_target_object_label[0,idx_seq])
                if torch.argmax(test_pred_object_prob[0,idx_seq], dim=-1) == torch.argmax(test_target_object_label[0,idx_seq]):
                    print("Object Prediction Success!")
                else:
                    print("Object Prediction FailedTT")
        if check_each:
            input()

        # inference result save
        plan_result_data['data_info'] = test_info
        plan_result_data['action_corrected'] = action_correct_mask
        plan_result_data['object_corrected'] = object_correct_mask

        if data_save:
            file_dir = os.path.join(os.getcwd(),
                                    "test_result",
                                    information,
                                    text_encoder,
                                    f"gradient_clipping_{gradient_clipping}",
                                    f"weight_decay_{weight_decay}",
                                    "_".join(list(map(str, model_param))),
                                    f"train_seed_{train_seed}",
                                    model_choose,
                                    method,
                                    "accuracy")
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            file_path = os.path.join(file_dir,"plan_"+str(test_i))
            with open(file_path, "wb") as outfile:
                pickle.dump(plan_result_data, outfile)
    print("Accuracy:")
    print("--------------------------------")
    print("Action Result: {}/{} corrected".format(test_num_act_correct.item(), test_num_total))
    print("Action Acc: {:01.4f}".format(test_num_act_correct.item() / test_num_total))
    print("Target Object Result: {}/{} corrected".format(test_num_obj_correct.item(), test_num_total))
    print("Target Object Acc: {:01.4f}".format(test_num_obj_correct.item() / test_num_total))

    #output text save
    result_path = os.path.join(model_path, "step_output.txt")
    with open(result_path, 'w') as f:
        print("Action Result: {}/{} corrected".format(test_num_act_correct.item(), test_num_total),file=f)
        print("Action Acc: {:01.4f}".format(test_num_act_correct.item() / test_num_total),file=f)
        print("{}/{} = {:01.4f}".format(test_num_act_correct.item(), test_num_total,(test_num_act_correct.item() / test_num_total)),file=f)
        print("Target Object Result: {}/{} corrected".format(test_num_obj_correct.item(), test_num_total),file=f)
        print("Target Object Acc: {:01.4f}".format(test_num_obj_correct.item() / test_num_total),file=f)
        print("{}/{} = {:01.4f}".format(test_num_obj_correct.item(), test_num_total,(test_num_obj_correct.item() / test_num_total)),file=f)

def inference_rnn_gbfs(device,
                       text_encoder,
                       gradient_clipping,
                       model_choose, method,
                       information,
                       maximum_plan_length,
                       hidden_dim,
                       num_action,
                       num_object,
                       node_feature_size,
                       edge_feature_size,
                       global_dim, batch_size,
                       lr,
                       train_seed,
                       num_workers,
                       num_epoch,
                       data_dir,
                       show_result,
                       infer_num=None,
                       check_each = False,
                       data_save=True,
                       weight_decay=0):

    model_param = [data_dir,hidden_dim, num_epoch, batch_size, lr]
    model_path = os.path.join(os.getcwd(),
                              "result",
                              information,
                              text_encoder,
                              f"gradient_clipping_{gradient_clipping}",
                              f"weight_decay_{weight_decay}",
                              "_".join(list(map(str, model_param))),
                              f"train_seed_{train_seed}",
                              model_choose,
                              method)

    plot_and_save(model_path, model_param, show_result)

    if infer_num is not None:
        model_name = 'GP_model_{}.pt'.format(infer_num)
    else:
        model_name = 'GP_model_best.pt'

    saved_path = os.path.join(model_path, model_name)
    saved_model = ActionModel_Recurrent(device,
                                        hidden_dim,
                                        num_action, 
                                        num_object, 
                                        node_feature_size, 
                                        edge_feature_size, 
                                        global_dim, 
                                        model_choose, 
                                        method)
    saved_model.load_state_dict(torch.load(saved_path))
    saved_model.to(device)

    for param in saved_model.parameters():
        param.requires_grad = False

    data_test = RecurrentActionDataset(data_dir, text_encoder,'test')
    data_test_loader = DataLoader(data_test, 1)

    obj_list_dir = os.path.join(os.getcwd(),
                                'demo_generation',
                                "datasets",
                                data_dir,
                                'test_obj_list')

    plan_total = 0
    plan_success = 0

    saved_model.eval()

    plan_result_data = {'data_info':{},
                        'plan_list':[],
                        'plan_len':int(),
                        'plan_reached':bool(),
                        'planning_tree':[],
                        'time':float()}
    
    debug_mode = False

    for test_i, test_data in enumerate(data_test_loader):
        plan_total += 1
        plan_list = []
        goal_planned = False
        test_current, test_info = test_data
        if debug_mode:
            print("#########################################")
            print("demo type:", test_info['demo'])
            print("graph_num:",test_info['graph_num'])
            print("--------------------------------")
        #text_basic_nf load
        graph_num = test_info['graph_num'][0]
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        text_basic_nf = pd.read_csv(os.path.join(obj_list_dir,f"text_basic_nf_{graph_num}.csv"),index_col=0)
        dynamics_module = GraphDynamicsUpdate(information, text_basic_nf, debug_mode=debug_mode)
        # split batched sequence graph
        split_list = batch_split(test_current.to(device))
        # load initial state graph only
        test_current = split_list[0].cpu()

        # goal state
        goal_x = test_current['x'].clone()
        goal_ei = test_current['edge_index'].clone()
        goal_ea = test_current['edge_attr'][:, int(edge_feature_size):].clone()
        goal_ei, goal_ea = remove_zero_edges(goal_ei, goal_ea)
        test_goal = Data(goal_x, goal_ei, goal_ea)

        # make root node
        root_node = GraphDataNode_RNN(test_current, test_goal)
        planning_tree = SearchingTree(root_node)

        # time record
        start = time.time()
        # greedy best first search start
        final_node, num_plan_res = gbfs_rnn(root_node,
                                            planning_tree,
                                            dynamics_module,
                                            saved_model,
                                            model_choose,
                                            information, 
                                            num_object,
                                            debug_mode,
                                            num_k=5,
                                            num_plan=1)
        if final_node is None:
            print('plan failed')
        else:
            node_pointer = final_node
            while node_pointer.parent is not None:
                plan_list.append(tuple([node_pointer.executed_ac, node_pointer.executed_oc]))
                node_pointer = node_pointer.parent
            plan_list.reverse()
            print(plan_list)
            print('plan success')
            plan_success+=1
            goal_planned = True
        end = time.time()
        # success rate check
        print(f'success rate: {plan_success}/{plan_total} = {plan_success/plan_total:.4f}\t planning time: {end-start:.4f}')
        
        # inference data save
        plan_result_data['data_info'] = test_info
        plan_result_data['plan_list'] = plan_list
        plan_result_data['plan_len'] = len(plan_list)
        plan_result_data['plan_reached'] = goal_planned
        plan_result_data['planning_tree'] = [planning_tree]
        plan_result_data['time'] = end-start
        if data_save:
            file_dir = os.path.join(os.getcwd(),
                                    "test_result",
                                    information, 
                                    text_encoder, 
                                    f"gradient_clipping_{gradient_clipping}",
                                    f"weight_decay_{weight_decay}",
                                    "_".join(list(map(str, model_param))), 
                                    f"train_seed_{train_seed}",
                                    model_choose, 
                                    method, 
                                    "tree_search")

            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            file_path = os.path.join(file_dir,"plan_"+str(plan_total))
            with open(file_path, "wb") as outfile:
                pickle.dump(plan_result_data, outfile)
    
    #output text로 저장
    result_path = os.path.join(model_path, f"tree_output_{infer_num}.txt")
    with open(result_path, 'w') as f:
        print(f'plan success rate:{plan_success}/{plan_total} ={plan_success/plan_total}',file=f)

def gbfs_rnn(graph_data_node,
             planning_tree, 
             dynamics_module, 
             saved_model, 
             model_choose, 
             information, 
             num_object, 
             debug_mode, 
             num_k, 
             num_plan):
    if debug_mode:
        print('num_plan: ', num_plan)
        print_planningtree(planning_tree.root, level=0)
    #################
    #planning limit #
    #tree height 기준#
    #################
    if (graph_data_node.level > 33) or( num_plan > 50):
        if debug_mode:
            print('plan limit reached')
        return None, None

    else:
        if debug_mode:
            print('gbsf plan start')
        if graph_data_node.level==0:#root node이면
            pred_action_prob, pred_object_prob, rnn_outputs, status = saved_model(graph_data_node.current, [1])
        else:
            pred_action_prob, pred_object_prob, rnn_outputs, status = saved_model(graph_data_node.current,
                                                                                [1],
                                                                                True,
                                                                                graph_data_node.prev_hidden)
        pred_action_prob = F.softmax(pred_action_prob, dim=-1) #1X1X3
        pred_object_prob = F.softmax(pred_object_prob, dim=-1) #1X1Xnum_object
        #####################################
        #action X object prob => table 만들기#
        #####################################
        act_obj_prob_table = torch.matmul(torch.transpose(pred_action_prob.squeeze(0), 0, 1), pred_object_prob.squeeze(0))
        sorted_table, indices = torch.sort(act_obj_prob_table.view(-1), descending=True)
        idx_table = 0
        num_feasible = 0
        while (num_feasible < num_k) and (idx_table < pred_action_prob.size(-1)*pred_object_prob.size(-1)):
            temp_graph_data = graph_data_node.current.clone()
            action_code, object_code = divmod(indices[idx_table].item(),pred_object_prob.size(-1))
            
            action_name = dynamics_module.act_idx_to_name[action_code]
            object_name = dynamics_module.obj_idx_to_name[object_code]

            idx_table+=1
            if debug_mode:
                print("Predicted action and object:")
                print(action_name)
                print(object_name)
            updated_graph = dynamics_module.graph_update(temp_graph_data, action_code, object_code)

            if updated_graph is not None:#feasible
                updated_graph_node = GraphDataNode_RNN(updated_graph,
                                                       graph_data_node.goal,
                                                       status
                                                       )
                ############
                #loop check#
                ############
                is_loop = loop_checking(updated_graph_node, graph_data_node)
                if is_loop == False:
                    planning_tree.add_node(graph_data_node, updated_graph_node)
                    num_feasible+=1

                    ####################
                    #distance값 업데이트 #
                    ####################
                    goal_ei = updated_graph_node.goal['edge_index']
                    goal_ea = updated_graph_node.goal['edge_attr']
                    goal_ei, goal_ea = remove_zero_edges(goal_ei, goal_ea)

                    edge_feature_size = goal_ea.size(-1)
                    cur_ei = updated_graph_node.current['edge_index']
                    cur_ea = updated_graph_node.current['edge_attr'][:,:edge_feature_size]
                    cur_ei, cur_ea = remove_zero_edges(cur_ei, cur_ea)

                    updated_graph_node.distance = key_edge_compare(cur_ei, cur_ea, goal_ei, goal_ea)
                    ##################
                    #action정보 업데이트#
                    ##################
                    updated_graph_node.executed_ac = action_name
                    updated_graph_node.executed_oc = object_name
                    ###################
                    #node level 업데이트#
                    ###################
                    updated_graph_node.level = updated_graph_node.parent.level+1
                    ###############
                    #goal checking#
                    ###############
                    # if torch.equal(updated_graph_node.state, updated_graph_node.goal):
                    if updated_graph_node.distance == 0:
                        if debug_mode:
                            print("goal reached")
                        return updated_graph_node, num_plan
            else:
                continue

        #################
        #leaf node check#
        #################
        leaf_node_list = leaf_node_checking(planning_tree)
        if debug_mode:
            print('num of leaf node:', len(leaf_node_list))

        ########################
        #minimum distnace check#
        ########################
        expansion_node = leaf_node_list[0]
        for leaf_node in leaf_node_list[1:]:
            #if leaf_node.distance > expansion_node.distance:
            #################
            if leaf_node.distance < expansion_node.distance:
            #################
                expansion_node = leaf_node
        if debug_mode:
            print('expansion node:')
            print(f'action: ({expansion_node.executed_ac},{expansion_node.executed_oc}) dist:{expansion_node.distance}')
            # input()
        
        ###########
        #expansion#
        ###########
        return gbfs_rnn(expansion_node, planning_tree, dynamics_module, saved_model, model_choose, information, num_object, debug_mode, num_k, num_plan+1)

#########
# Basic #
#########
def inference_basic_sequence(device,
                             text_encoder,
                             gradient_clipping,
                             model_choose, method,
                             information,
                             maximum_plan_length,
                             hidden_dim,
                             num_action,
                             num_object,
                             node_feature_size,
                             edge_feature_size,
                             global_dim, batch_size,
                             lr,
                             train_seed,
                             num_workers,
                             num_epoch,
                             data_dir,
                             show_result,
                             infer_num=None,
                             check_each = False,
                             data_save=True,
                             weight_decay=0):

    model_param = [data_dir,hidden_dim, num_epoch, batch_size, lr]
    model_path = os.path.join(os.getcwd(),
                              "result",
                              information,
                              text_encoder,
                              f"gradient_clipping_{gradient_clipping}",
                              f"weight_decay_{weight_decay}",
                              "_".join(list(map(str, model_param))),
                              f"train_seed_{train_seed}",
                              model_choose,
                              method)

    plot_and_save(model_path, model_param, show_result)

    if infer_num is not None:
        model_name = 'GP_model_{}.pt'.format(infer_num)
    else:
        model_name = 'GP_model_best.pt'

    saved_path = os.path.join(model_path, model_name)
    saved_model = ActionModel_Basic(device,
                                    hidden_dim,
                                    num_action, 
                                    num_object, 
                                    node_feature_size, 
                                    edge_feature_size, 
                                    global_dim, 
                                    model_choose, 
                                    method)
    saved_model.load_state_dict(torch.load(saved_path))
    saved_model.to(device)

    for param in saved_model.parameters():
        param.requires_grad = False

    data_test = BasicDataset(data_dir, text_encoder,'test')
    data_test_loader = DataLoader(data_test, 1)

    obj_list_dir = os.path.join(os.getcwd(),
                                'demo_generation',
                                "datasets",
                                data_dir,
                                'test_obj_list')

    plan_total = 0
    plan_success = 0

    plan_shorten = 0
    plan_shorten_dict = {}

    saved_model.eval()

    plan_result_data = {'data_info':{},
                        'plan_list':[],
                        'plan_len':int(),
                        'plan_reached':bool(),
                        'time':float()}
    
    debug_mode = False

    for test_i, test_data in enumerate(data_test_loader):
        test_current, test_info = test_data
        # planning only for initial state
        if test_info['step'][0] == 0:
            plan_total += 1
            goal_planned = False
            if debug_mode:
                print("demo type:", test_info['demo'])
                print("graph_num:",test_info['graph_num'])
                print("--------------------------------")
            #text_basic_nf load
            graph_num = test_info['graph_num'][0]
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            text_basic_nf = pd.read_csv(os.path.join(obj_list_dir,f"text_basic_nf_{graph_num}.csv"),index_col=0)
            dynamics_module = GraphDynamicsUpdate(information, text_basic_nf, debug_mode=debug_mode)
            # load state graph
            current_state = test_current
            # time record
            start = time.time()
            # model inference
            pred_action_prob, pred_object_prob = saved_model(current_state)

            # action X object prob => prob.table
            pred_action_prob = F.softmax(pred_action_prob, dim=-1) #1X1X3
            pred_object_prob = F.softmax(pred_object_prob, dim=-1) #1X1Xnum_object
            act_obj_prob_table = torch.matmul(torch.transpose(pred_action_prob, 0, 1), pred_object_prob)
            # read the most probable action&object
            sorted_table, indices = torch.sort(act_obj_prob_table.view(-1), descending=True)
            action_code, object_code = divmod(indices[0].item(),pred_object_prob.size(-1))
            # convert code to name
            action_name = dynamics_module.act_idx_to_name[action_code]
            object_name = dynamics_module.obj_idx_to_name[object_code]

            if debug_mode:
                print(action_name, object_name)

            # graph state update
            updated_graph = dynamics_module.graph_update(current_state, action_code, object_code)
            # plan sequence update
            planned_action = [tuple([action_name, object_name])]

            # sequential planning until maximum plan length limit
            while len(planned_action) < maximum_plan_length:
                if debug_mode:
                    print(planned_action)
                    print(dynamics_module.plan_name_to_idx(planned_action))
                    # input()
                # stop planning if planned action is infeasible
                if updated_graph is None:
                    if debug_mode:
                        print('not feasible action is planned')
                    break
                else:
                    # split graph into current state and goal state
                    state_ei = updated_graph['edge_index'].clone()
                    state_ea = updated_graph['edge_attr'][:, :int(edge_feature_size)].clone()
                    state_ei, state_ea = remove_zero_edges(state_ei, state_ea)

                    goal_ei = updated_graph['edge_index'].clone()
                    goal_ea = updated_graph['edge_attr'][:, int(edge_feature_size):].clone()
                    goal_ei, goal_ea = remove_zero_edges(goal_ei, goal_ea)

                    # goal check - key_edge_compare
                    if key_edge_compare(state_ei, state_ea, goal_ei, goal_ea) == 0:
                        if debug_mode:
                            print('plan reached')
                        plan_success += 1
                        goal_planned = True
                        break
                    else:
                        # if state didn't reach goal - plan again
                        current_state = updated_graph.clone()
                        # model inference with updated state
                        pred_action_prob, pred_object_prob = saved_model(current_state.clone())
                        # action X object prob => prob.table
                        pred_action_prob = F.softmax(pred_action_prob, dim=-1) #1X1X3
                        pred_object_prob = F.softmax(pred_object_prob, dim=-1) #1X1Xnum_object
                        act_obj_prob_table = torch.matmul(torch.transpose(pred_action_prob, 0, 1), pred_object_prob)
                        # read the most probable action&object
                        sorted_table, indices = torch.sort(act_obj_prob_table.view(-1), descending=True)
                        action_code, object_code = divmod(indices[0].item(),pred_object_prob.size(-1))
                        # convert code to name
                        action_name = dynamics_module.act_idx_to_name[action_code]
                        object_name = dynamics_module.obj_idx_to_name[object_code]

                        if debug_mode:
                            print(action_name, object_name)
                        # graph state update
                        updated_graph = dynamics_module.graph_update(current_state, action_code, object_code)
                        # plan sequence update
                        planned_action.append(tuple([action_name, object_name]))
                # input()
            end = time.time()

            # shorten plan result checking
            if goal_planned:
                if test_info['goal'].item() > len(planned_action):
                    plan_shorten += 1
                    shorten_task_name = test_info['demo'][0] + '_' + str(graph_num)
                    plan_shorten_dict[shorten_task_name] = planned_action

            # success rate check
            print(f'success rate: {plan_success}/{plan_total} = {plan_success/plan_total:.4f}\t planning time: {end-start:.4f}')

            if debug_mode:
                if goal_planned is False:
                    input()
            # inference result save
            plan_result_data['data_info'] = test_info
            plan_result_data['plan_list'] = planned_action
            plan_result_data['plan_len'] = len(planned_action)
            plan_result_data['plan_reached'] = goal_planned
            plan_result_data['time'] = end-start

            if data_save:
                file_dir = os.path.join(os.getcwd(),
                                        "test_result",
                                        information,
                                        text_encoder,
                                        f"gradient_clipping_{gradient_clipping}",
                                        f"weight_decay_{weight_decay}",
                                        "_".join(list(map(str, model_param))),
                                        f"train_seed_{train_seed}",
                                        model_choose,
                                        method,
                                        "Sequential_planning")
                if not os.path.exists(file_dir):
                    os.makedirs(file_dir)
                file_path = os.path.join(file_dir,"plan_"+str(plan_total))
                with open(file_path, "wb") as outfile:
                    pickle.dump(plan_result_data, outfile)

    print('#####################################')
    print(f'success rate: {plan_success}/{plan_total} = {plan_success/plan_total}')
    print(f'shorten plan: {plan_shorten}/{plan_success} = {(plan_shorten/plan_success) if plan_success > 0 else 0}')
    print("List:")
    for task, plan in plan_shorten_dict.items():
        print(task)
        print(plan)
    print('#####################################')

    #output text save
    result_path = os.path.join(model_path, f"sequence_output_{infer_num}.txt")
    with open(result_path, 'w') as f:
        print(f'success rate: {plan_success}/{plan_total} = {plan_success/plan_total}')
        print(f'success rate: {plan_success}/{plan_total} = {plan_success/plan_total}', file=f)
        print(f'shorten plan: {plan_shorten}/{plan_success} = {(plan_shorten/plan_success) if plan_success > 0 else 0}', file=f)
        print("List:", file=f)
        print(plan_shorten_dict, file=f)

def inference_basic_step(device,
                         text_encoder,
                         gradient_clipping,
                         model_choose, method,
                         information,
                         maximum_plan_length,
                         hidden_dim,
                         num_action,
                         num_object,
                         node_feature_size,
                         edge_feature_size,
                         global_dim, batch_size,
                         lr,
                         train_seed,
                         num_workers,
                         num_epoch,
                         data_dir,
                         show_result,
                         infer_num=None,
                         check_each = False,
                         data_save=True,
                         weight_decay=0):   
    
    model_param = [data_dir,hidden_dim, num_epoch, batch_size, lr]
    model_path = os.path.join(os.getcwd(),
                              "result",
                              information,
                              text_encoder,
                              f"gradient_clipping_{gradient_clipping}",
                              f"weight_decay_{weight_decay}",
                              "_".join(list(map(str, model_param))),
                              f"train_seed_{train_seed}",
                              model_choose,
                              method)

    plot_and_save(model_path, model_param, show_result)

    if infer_num is not None:
        model_name = 'GP_model_{}.pt'.format(infer_num)
    else:
        model_name = 'GP_model_best.pt'

    saved_path = os.path.join(model_path, model_name)
    saved_model = ActionModel_Basic(device,
                                    hidden_dim,
                                    num_action, 
                                    num_object, 
                                    node_feature_size, 
                                    edge_feature_size, 
                                    global_dim, 
                                    model_choose, 
                                    method)
    saved_model.load_state_dict(torch.load(saved_path))
    saved_model.to(device)

    for param in saved_model.parameters():
        param.requires_grad = False

    data_test = BasicDataset(data_dir, text_encoder,'test')
    data_test_loader = DataLoader(data_test, 1)

    loss_ce_action = nn.CrossEntropyLoss().to(device)
    loss_ce_object = nn.CrossEntropyLoss().to(device)

    test_num_act_correct = 0
    test_num_obj_correct = 0
    test_num_total = 0

    saved_model.eval()

    plan_result_data = {'data_info':{},
                        'action_corrected':bool(),
                        'object_corrected':bool()}
    
    debug_mode = False

    for test_i, test_data in enumerate(data_test_loader):
        test_current, test_info = test_data
        if debug_mode:
            print("#########################################")
            print("demo type:", test_info['demo'])
            print("graph_num:",test_info['graph_num'])
            print("--------------------------------")

        test_pred_action_prob, test_pred_object_prob = saved_model(test_current)

        test_target_label = torch.tensor(test_current['y']).to(device)
        # action loss & acc
        test_target_action_label = test_target_label[:,0]
        test_L_action = loss_ce_action(test_pred_action_prob, test_target_action_label)


        test_pred_action_label = torch.argmax(test_pred_action_prob, dim=-1)
        test_num_act_correct += torch.sum(test_pred_action_label==test_target_action_label)

        # object loss & acc
        test_target_object_label = test_target_label[:,1]
        test_L_object = loss_ce_object(test_pred_object_prob, test_target_object_label)

        test_pred_object_label = torch.argmax(test_pred_object_prob, dim=-1)
        test_num_obj_correct += torch.sum(test_pred_object_label==test_target_object_label)

        test_num_total += test_target_label.size(0)

        test_L_total = 0.1*test_L_action + test_L_object

        if debug_mode:
            print("Loss:")
            print("L_total:", test_L_total.item())
            print("L_action:", test_L_action.item())
            print("L_object:", test_L_object.item())
            print("--------------------------------")
            print("prediced prob:")
            print("pred_action_score:", F.softmax(test_pred_action_prob, dim=-1))
            print("target_action_prob:",test_target_action_label)
            if torch.argmax(test_pred_action_prob, dim=-1) == torch.argmax(test_target_action_label):
                print("Action Prediction Success!")
            else:
                print("Action Prediction FailedTT\n")
            print("pred_object_score:", F.softmax(test_pred_object_prob, dim=-1))
            print("target_object_prob:",test_target_object_label)
            if torch.argmax(test_pred_object_prob, dim=-1) == torch.argmax(test_target_object_label):
                print("Object Prediction Success!")
            else:
                print("Object Prediction FailedTT")
        if check_each:
            input()

        # inference result save
        plan_result_data['data_info'] = test_info

        if data_save:
            file_dir = os.path.join(os.getcwd(),
                                    "test_result",
                                    information,
                                    text_encoder,
                                    f"gradient_clipping_{gradient_clipping}",
                                    f"weight_decay_{weight_decay}",
                                    "_".join(list(map(str, model_param))),
                                    f"train_seed_{train_seed}",
                                    model_choose,
                                    method,
                                    "accuracy")
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            file_path = os.path.join(file_dir,"plan_"+str(test_i))
            with open(file_path, "wb") as outfile:
                pickle.dump(plan_result_data, outfile)
    print("Accuracy:")
    print("--------------------------------")
    print("Action Result: {}/{} corrected".format(test_num_act_correct.item(), test_num_total))
    print("Action Acc: {:01.4f}".format(test_num_act_correct.item() / test_num_total))
    print("Target Object Result: {}/{} corrected".format(test_num_obj_correct.item(), test_num_total))
    print("Target Object Acc: {:01.4f}".format(test_num_obj_correct.item() / test_num_total))

    #output text save
    result_path = os.path.join(model_path, "step_output.txt")
    with open(result_path, 'w') as f:
        print("Action Result: {}/{} corrected".format(test_num_act_correct.item(), test_num_total),file=f)
        print("Action Acc: {:01.4f}".format(test_num_act_correct.item() / test_num_total),file=f)
        print("{}/{} = {:01.4f}".format(test_num_act_correct.item(), test_num_total,(test_num_act_correct.item() / test_num_total)),file=f)
        print("Target Object Result: {}/{} corrected".format(test_num_obj_correct.item(), test_num_total),file=f)
        print("Target Object Acc: {:01.4f}".format(test_num_obj_correct.item() / test_num_total),file=f)
        print("{}/{} = {:01.4f}".format(test_num_obj_correct.item(), test_num_total,(test_num_obj_correct.item() / test_num_total)),file=f)


def inference_basic_gbfs(device,
                         text_encoder, 
                         gradient_clipping, 
                         model_choose, 
                         method, 
                         information, 
                         maximum_plan_length, 
                         hidden_dim, 
                         num_action, 
                         num_object, 
                         node_feature_size, 
                         edge_feature_size, 
                         global_dim, 
                         batch_size, 
                         lr, 
                         num_epoch, 
                         train_seed,
                         num_workers,
                         data_dir, 
                         show_result, 
                         infer_num=None, 
                         check_each = False, 
                         data_save=True, 
                         weight_decay=0):
    model_param = [data_dir, hidden_dim, num_epoch, batch_size, lr]
    model_path = os.path.join(os.getcwd(), 
                              "result", 
                              information, 
                              text_encoder, 
                              f"gradient_clipping_{gradient_clipping}",
                              f"weight_decay_{weight_decay}",
                              "_".join(list(map(str, model_param))),
                              f"train_seed_{train_seed}",
                              model_choose, 
                              method)

    plot_and_save(model_path, model_param, show_result)

    if infer_num is not None:
        model_name = 'GP_model_{}.pt'.format(infer_num)
    else:
        model_name = 'GP_model_best.pt'

    saved_path = os.path.join(model_path, model_name)
    saved_model = ActionModel_Basic(device, 
                                    hidden_dim, 
                                    num_action, 
                                    num_object, 
                                    node_feature_size, 
                                    edge_feature_size, 
                                    global_dim, 
                                    model_choose, 
                                    method)
    saved_model.load_state_dict(torch.load(saved_path))
    saved_model.to(device)

    for param in saved_model.parameters():
        param.requires_grad = False

    data_test = BasicDataset(data_dir, text_encoder,'test')
    data_test_loader = DataLoader(data_test, 1)

    obj_list_dir = os.path.join(os.getcwd(), 
                                'demo_generation', 
                                "datasets",
                                data_dir, 
                                'test_obj_list')

    plan_total = 0
    plan_success = 0

    saved_model.eval()

    plan_result_data = {'data_info':{},
                    'plan_list':[],
                    'plan_len':int(),
                    'plan_reached':bool(),
                    'planning_tree':[],
                    'time':float()}

    debug_mode = False

    for test_i, test_data in enumerate(data_test_loader):
        test_current, test_info = test_data
        # planning only for initial state
        if test_info['step'][0] == 0:
            plan_total += 1
            plan_list = []
            goal_planned = False
            if debug_mode:
                print("#########################################")
                print("demo type:", test_info['demo'])
                print("graph_num:",test_info['graph_num'])
                print("--------------------------------")

            #text_basic_nf load
            graph_num = test_info['graph_num'][0]
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            text_basic_nf = pd.read_csv(os.path.join(obj_list_dir,f"text_basic_nf_{graph_num}.csv"),index_col=0)
            dynamics_module = GraphDynamicsUpdate(information, text_basic_nf, debug_mode=debug_mode)

            # goal state
            goal_x = test_current['x'].clone()
            goal_ei = test_current['edge_index'].clone()
            goal_ea = test_current['edge_attr'][:, int(edge_feature_size):].clone()
            goal_ei, goal_ea = remove_zero_edges(goal_ei, goal_ea)
            test_goal = Data(goal_x, goal_ei, goal_ea)

            # make root node
            root_node = GraphDataNode_Basic(test_current, test_goal)
            planning_tree = SearchingTree(root_node)

            # time record
            start = time.time()
            # greedy best first search start
            final_node, num_plan_res = gbfs_basic(root_node, 
                                                  planning_tree, 
                                                  dynamics_module, 
                                                  saved_model, 
                                                  model_choose, 
                                                  information, 
                                                  num_object, 
                                                  debug_mode,
                                                  num_k=5, 
                                                  num_plan=1)
            if final_node is None:
                print('plan failed')
            else:
                node_pointer = final_node
                while node_pointer.parent is not None:
                    plan_list.append(tuple([node_pointer.executed_ac, node_pointer.executed_oc]))
                    node_pointer = node_pointer.parent
                plan_list.reverse()
                print(plan_list)
                print('plan success')
                plan_success+=1
                goal_planned = True
            end = time.time()
            # success rate check
            print(f'success rate: {plan_success}/{plan_total} = {plan_success/plan_total:.4f}\t planning time: {end-start:.4f}')
            
            # inference data save
            plan_result_data['data_info'] = test_info
            plan_result_data['plan_list'] = plan_list
            plan_result_data['plan_len'] = len(plan_list)
            plan_result_data['plan_reached'] = goal_planned
            plan_result_data['planning_tree'] = [planning_tree]
            plan_result_data['time'] = end-start
            if data_save:
                file_dir = os.path.join(os.getcwd(),
                                        "test_result",
                                        information, 
                                        text_encoder, 
                                        f"gradient_clipping_{gradient_clipping}",
                                        f"weight_decay_{weight_decay}",
                                        "_".join(list(map(str, model_param))), 
                                        f"train_seed_{train_seed}",
                                        model_choose, 
                                        method, 
                                        "tree_search")

                if not os.path.exists(file_dir):
                    os.makedirs(file_dir)
                file_path = os.path.join(file_dir,"plan_"+str(plan_total))
                with open(file_path, "wb") as outfile:
                    pickle.dump(plan_result_data, outfile)

    #output text로 저장
    result_path = os.path.join(model_path, f"tree_output_{infer_num}.txt")
    with open(result_path, 'w') as f:
        print(f'plan success rate:{plan_success}/{plan_total} ={plan_success/plan_total}',file=f)

def gbfs_basic(graph_data_node,
               planning_tree,
               dynamics_module, 
               saved_model, 
               model_choose, 
               information, 
               num_object,
               debug_mode,
               num_k, 
               num_plan):
    if debug_mode:
        print('num_plan: ', num_plan)
        print_planningtree(planning_tree.root, level=0)
    #################
    #planning limit #
    #tree height 기준#
    #################
    if (graph_data_node.level > 33) or(num_plan > 50):
        if debug_mode:
            print('plan limit reached')
        return None, None

    else:
        if debug_mode:
            print('gbsf plan start')

        pred_action_prob, pred_object_prob = saved_model(graph_data_node.current)
        
        pred_action_prob = F.softmax(pred_action_prob, dim=-1) #1X3
        pred_object_prob = F.softmax(pred_object_prob, dim=-1) #1Xnum_object
        #####################################
        #action X object prob => table 만들기#
        #####################################
        act_obj_prob_table = torch.matmul(torch.transpose(pred_action_prob, 0, 1), pred_object_prob)
        sorted_table, indices = torch.sort(act_obj_prob_table.view(-1), descending=True)
        idx_table = 0
        num_feasible = 0
        while (num_feasible < num_k) and (idx_table < pred_action_prob.size(-1)*pred_object_prob.size(-1)):
            temp_graph_data = graph_data_node.current.clone()
            action_code, object_code = divmod(indices[idx_table].item(),pred_object_prob.size(-1))
            
            action_name = dynamics_module.act_idx_to_name[action_code]
            object_name = dynamics_module.obj_idx_to_name[object_code]

            idx_table+=1
            if debug_mode:
                print("Predicted action and object:")
                print(action_name)
                print(object_name)
            updated_graph = dynamics_module.graph_update(temp_graph_data, action_code, object_code)

            if updated_graph is not None:#feasible
                updated_graph_node = GraphDataNode_Basic(updated_graph,
                                                        graph_data_node.goal,
                                                        )
                ############
                #loop check#
                ############
                is_loop = loop_checking(updated_graph_node, graph_data_node)
                if is_loop == False:
                    planning_tree.add_node(graph_data_node, updated_graph_node)
                    num_feasible+=1

                    ####################
                    #distance값 업데이트 #
                    ####################
                    goal_ei = updated_graph_node.goal['edge_index']
                    goal_ea = updated_graph_node.goal['edge_attr']
                    goal_ei, goal_ea = remove_zero_edges(goal_ei, goal_ea)

                    edge_feature_size = goal_ea.size(-1)
                    cur_ei = updated_graph_node.current['edge_index']
                    cur_ea = updated_graph_node.current['edge_attr'][:,:edge_feature_size]
                    cur_ei, cur_ea = remove_zero_edges(cur_ei, cur_ea)

                    updated_graph_node.distance = key_edge_compare(cur_ei, cur_ea, goal_ei, goal_ea)
                    ##################
                    #action정보 업데이트#
                    ##################
                    updated_graph_node.executed_ac = action_name
                    updated_graph_node.executed_oc = object_name
                    ###################
                    #node level 업데이트#
                    ###################
                    updated_graph_node.level = updated_graph_node.parent.level+1
                    ###############
                    #goal checking#
                    ###############
                    # if torch.equal(updated_graph_node.state, updated_graph_node.goal):
                    if updated_graph_node.distance == 0:
                        if debug_mode:
                            print("goal reached")
                        return updated_graph_node, num_plan
            else:
                continue

        #################
        #leaf node check#
        #################
        leaf_node_list = leaf_node_checking(planning_tree)
        if debug_mode:
            print('num of leaf node:', len(leaf_node_list))

        ########################
        #maximum score check#
        ########################
        expansion_node = leaf_node_list[0]
        for leaf_node in leaf_node_list[1:]:
            #if leaf_node.distance > expansion_node.distance:
            #################
            if leaf_node.distance < expansion_node.distance:
            #################
                expansion_node = leaf_node
        if debug_mode:
            print('expansion node:')
            print(f'action: ({expansion_node.executed_ac},{expansion_node.executed_oc}) dist:{expansion_node.distance}')
            # input()

        ###########
        #expansion#
        ###########
        return gbfs_basic(expansion_node, planning_tree, dynamics_module, saved_model, model_choose, information, num_object, debug_mode, num_k, num_plan+1)