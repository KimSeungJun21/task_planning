#! /usr/bin/env python

import rospy
from math import pi
import os
import logging

import numpy as np
import pickle
import pandas as pd
import sys
import time
import natsort
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from model.model import ActionModel_Basic
from utils.graph_collate import BasicDataset
from utils.inference_utils import GraphDynamicsUpdate
from utils.inference_utils import remove_zero_edges
from utils.default_utils import root_path
from utils.graph_utils import to_dense_graph, concat_current_goal_graphs, get_edge_index_from_edge_attr, key_edge_compare
from utils.arg_utils import parse_default_args
from utils.ros_msg_utils import float64_multiarray_to_numpy

from std_msgs.msg import Float64MultiArray
from graph_task_planning.msg import Plan_msg
from graph_task_planning.srv import PlanPrediction, PlanPredictionResponse

logging.basicConfig(level=logging.INFO)

class PlannerService:
    def __init__(self, args, goal_task_name):
        # self.pretrained_model_path = pretrained_model_path
        self.args = args
        self.goal_task_name = goal_task_name

        # set model parameters
        model_param = [self.args.dataset_name,
                       self.args.hidden_dim,
                       self.args.num_epoch,
                       self.args.batch_size,
                       self.args.lr]
        model_path = os.path.join(root_path(),
                                  "pretrained_model",
                                  self.args.dataset_name,
                                  f"gradient_clipping_{self.args.gradient_clipping}",
                                  f"weight_decay_{self.args.weight_decay}",
                                  "_".join(list(map(str, model_param))),
                                  self.args.method)
        self.saved_path = os.path.join(model_path, 'GP_model_best.pt')

        # load pretrained model
        self.saved_model = ActionModel_Basic(self.args)
        self.saved_model.load_state_dict(torch.load(self.saved_path))
        self.saved_model.to(self.args.device)
        for param in self.saved_model.parameters():
            param.requires_grad = False
        self.saved_model.eval()

        # set goal state graph
        # 임시 - test dataset에서 load
        self.load_goal_graph_from_task_dataset()
        # remove zero edges
        self.goal_ei, self.goal_ea = to_dense_graph(self.goal_ei, self.goal_ea)
        # tensor data type setting
        self.goal_ei = self.goal_ei.to(torch.int)
        self.goal_ea = self.goal_ea.to(torch.float32)

        # start ros service
        rospy.Service('/pred_plan', PlanPrediction, self.compute_service_handler)
    
    def load_goal_graph_from_task_dataset(self):
        task_data_path = os.path.join(root_path(), 'demo_generation', 'tasks', self.args.dataset_name, self.goal_task_name)

        nf_name = natsort.natsorted(os.listdir(os.path.join(task_data_path, 'node_feature')))[-1]
        nf = pd.read_csv(os.path.join(task_data_path, 'node_feature', nf_name), index_col=0)
        self.goal_nf = torch.tensor(nf.values)

        ei_name = natsort.natsorted(os.listdir(os.path.join(task_data_path, 'edge_index')))[-1]
        ei_csv = pd.read_csv(os.path.join(task_data_path, 'edge_index', ei_name), index_col=0)

        ea_name = natsort.natsorted(os.listdir(os.path.join(task_data_path, 'edge_attr')))[-1]
        ea_csv = pd.read_csv(os.path.join(task_data_path, 'edge_attr', ea_name), index_col=0)
        self.goal_ei, self.goal_ea = get_edge_index_from_edge_attr(ei_csv, ea_csv)

    def plan_inference(self, cat_graph):
        # 실험 할 test data => cat_graph
        pred_action_prob, pred_object_prob = self.saved_model(cat_graph)
        # get pred act-obj labels
        pred_action_label = torch.argmax(pred_action_prob, dim=-1)
        pred_object_label = torch.argmax(pred_object_prob, dim=-1)

        return pred_action_label, pred_object_label

    def goal_check(self, state_ei, state_ea):
        # check if given state is reached to goal state
        if key_edge_compare(state_ei, state_ea, self.goal_ei, self.goal_ea) == 0:
            return True
        else:
            return False

    def compute_service_handler(self, req):
        # req -> sim에서 받아온 state graph 정보
        # model data type setting
        state_ei = torch.tensor(float64_multiarray_to_numpy(req.edge_index))
        state_ea = torch.tensor(float64_multiarray_to_numpy(req.edge_attr))
        # remove zero edges
        state_ei, state_ea = to_dense_graph(state_ei, state_ea)
        # set data type
        state_ei = state_ei.to(torch.int)
        state_ea = state_ea.to(torch.float32)

        # goal check
        if self.goal_check(state_ei, state_ea):
            print('goal reached!')
            pred_plan = Plan_msg()
            pred_plan.action = -1
            pred_plan.object = -1

            return pred_plan
        else:
            # goal_graph랑 concat
            cat_ei, cat_ea = concat_current_goal_graphs(state_ei, state_ea, self.goal_ei, self.goal_ea)

            # data batch -> set to 1
            cat_graph = Data(x = self.goal_nf,
                            edge_index = cat_ei,
                            edge_attr = cat_ea,
                            batch = torch.zeros((self.goal_nf.size(0)),dtype=torch.long))

            # concat된 graph를 pretrained model에 inference
            pred_action_label, pred_object_label = self.plan_inference(cat_graph)
            
            # write predicted action&object to ros message
            pred_plan = Plan_msg()
            pred_plan.action = pred_action_label
            pred_plan.object = pred_object_label

            return pred_plan

if __name__ == '__main__':
    rospy.init_node('planner_service')

    args = parse_default_args()
    # ## 임시 ##
    # # planning 하려는 goal task - 임의로 입력
    # # goal_task = 'stacking_5_B13542_R1'
    # goal_task = 'clustering_5_B12345_R23233'

    try:
        task_type = input("enter the task type to plan: [stacking, clustering]\n")
        if (task_type != 'stacking') and (task_type != 'clustering'):
            raise ValueError
    except:
        print("task type should be stacking or clustering")
    
    if task_type == 'stacking':
        try:
            region_list = 'w'
            region_id_list = '1'
            box_order = input("enter the order of boxes to stack: [permutation of 1~5]\n")
            
            if len(box_order) != 5:
                raise ValueError

            for box_id in box_order:
                if box_id not in ['1', '2', '3', '4', '5']:
                    raise ValueError
            
        except:
            print("box id should be in 1~5")
    
    elif task_type == 'clustering':
        try:
            box_order = '12345'
            region_list = input("enter the list of regions(red/blue) to cluster 5 boxes: [combination of [r, b] 5 times in total]\n")

            if len(region_list) != 5:
                raise ValueError
            
            for region in region_list:
                if region not in ['r', 'b']:
                    raise ValueError
            color_to_id = {'b': '2', 'r': '3'}
            # region_id_list = []
            # for region_color in region_list:
            #     region_id_list.append(color_to_id[region_color])
            region_id_list = "".join([color_to_id[region_color] for region_color in region_list])
            
        except:
            print("region should be 'r' or 'b'")

    print('given goal task is:', f'{task_type}_5_B{box_order}_R{region_list}')
    goal_task = f'{task_type}_5_B{box_order}_R{region_id_list}'

    TaskPlanner = PlannerService(args, goal_task)


    print('planner server started!')
    print('waiting state input from sim...')
    rospy.spin()
