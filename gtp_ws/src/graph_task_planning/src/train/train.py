from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import pickle
import os
import sys
from tqdm import tqdm
sys.path.extend("./")
from utils.graph_collate import BasicDataset
from model.model import ActionModel_Basic
from utils.default_utils import create_folder, root_path

def train_basic(args):
    model = ActionModel_Basic(args)

    model.to(args.device)

    model_name = [args.dataset_name, args.hidden_dim, args.num_epoch, args.batch_size, args.lr]

    model_path_base = os.path.join(root_path(),
                                   "result",
                                   args.dataset_name,
                                   f"gradient_clipping_{args.gradient_clipping}", 
                                   f"weight_decay_{args.weight_decay}",
                                   "_".join(list(map(str, model_name))))
    
    model_path = os.path.join(model_path_base,
                              args.method)
    create_folder(model_path)

    # Dataset 불러오는 위치
    train_dataset = BasicDataset(args.dataset_name, 'train')
    val_dataset = BasicDataset(args.dataset_name, 'val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay= args.weight_decay)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                            lr_lambda=lambda epoch: 0.75**epoch)

    loss_ce_action = nn.CrossEntropyLoss().to(args.device)
    loss_ce_object = nn.CrossEntropyLoss().to(args.device)

    for param in model.parameters():
        param.requires_grad = True

    best_loss = 10000
    
    loss_data = {"epoch":[],
                 "loss":{"total":{"train":[],
                                  "val":[]},
                         "action":{"train":[],
                                   "val":[]},
                         "object":{"train":[],
                                   "val":[]}},
                 "acc":{"action":{"train":[],
                                   "val":[]},
                         "object":{"train":[],
                                   "val":[]}}}

    #train
    for epoch in range(args.num_epoch):
        print("#############################")
        print("epoch number {}".format(epoch+1))
        model.train()

        running_loss = 0.0
        last_loss = 0.0
        act_running_loss = 0.0
        act_last_loss = 0.0
        obj_running_loss = 0.0
        obj_last_loss = 0.0

        num_act_correct = 0
        num_act_total = 0
        num_obj_correct = 0
        num_obj_total = 0

        for i, data in enumerate(tqdm(train_loader,
                                      desc="train",
                                      leave=False)):
            current, info = data

            pred_action_prob, pred_object_prob = model(current)

            target_label = torch.tensor(current['y']).to(args.device)
            # action loss & acc
            target_action_label = target_label[:,0]
            L_action = loss_ce_action(pred_action_prob, target_action_label)

            pred_action_label = torch.argmax(pred_action_prob.detach(), dim=-1)
            num_act_correct += torch.sum(pred_action_label==target_action_label)
            num_act_total += target_action_label.size(0)

            # object loss & acc
            target_object_label = target_label[:,1]
            L_object = loss_ce_object(pred_object_prob, target_object_label)

            pred_object_label = torch.argmax(pred_object_prob.detach(), dim=-1)
            num_obj_correct += torch.sum(pred_object_label==target_object_label)
            num_obj_total += target_object_label.size(0)

            L_total = 0.1*L_action + L_object
            optimizer.zero_grad()
            L_total.backward()

            if args.gradient_clipping == True:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            else:
                pass
            optimizer.step()
            
            running_loss += L_total.item()
            last_loss = running_loss/(i+1)
            act_running_loss += L_action.item()
            act_last_loss = act_running_loss/(i+1)
            obj_running_loss += L_object.item()
            obj_last_loss = obj_running_loss/(i+1)

        val_running_loss = 0.0
        val_last_loss = 0.0
        val_act_running_loss = 0.0
        val_act_last_loss = 0.0
        val_obj_running_loss = 0.0
        val_obj_last_loss = 0.0

        val_num_act_correct = 0
        val_num_act_total = 0
        val_num_obj_correct = 0
        val_num_obj_total = 0

        model.eval()
        with torch.no_grad():
            for val_i, val_data in enumerate(tqdm(val_loader,
                                                  desc="val",
                                                  leave=False)):
                val_current, val_info = val_data

                val_pred_action_prob, val_pred_object_prob = model(val_current)

                val_target_label = torch.tensor(val_current['y']).to(args.device)
                # action loss & acc
                val_target_action_label = val_target_label[:,0]
                val_L_action = loss_ce_action(val_pred_action_prob, val_target_action_label)


                val_pred_action_label = torch.argmax(val_pred_action_prob, dim=-1)
                val_num_act_correct += torch.sum(val_pred_action_label==val_target_action_label)
                val_num_act_total += val_target_action_label.size(0)

                # object loss & acc
                val_target_object_label = val_target_label[:,1]
                val_L_object = loss_ce_object(val_pred_object_prob, val_target_object_label)

                val_pred_object_label = torch.argmax(val_pred_object_prob, dim=-1)
                val_num_obj_correct += torch.sum(val_pred_object_label==val_target_object_label)
                val_num_obj_total += val_target_object_label.size(0)

                val_L_total = 0.1*val_L_action + val_L_object

                val_running_loss += val_L_total.item()
                val_last_loss = val_running_loss/(val_i+1)
                val_act_running_loss += val_L_action.item()
                val_act_last_loss = val_act_running_loss/(val_i+1)
                val_obj_running_loss += val_L_object.item()
                val_obj_last_loss = val_obj_running_loss/(val_i+1)


        act_acc = num_act_correct.item()/num_act_total
        obj_acc = num_obj_correct.item()/num_obj_total
        
        val_act_acc = val_num_act_correct.item()/val_num_act_total
        val_obj_acc = val_num_obj_correct.item()/val_num_obj_total
        
        print("Action Acc\ttrain:{:01.4f}\tval:{:01.4f}".format(act_acc, val_act_acc))
        print("Object Acc\ttrain:{:01.4f}\tval:{:01.4f}".format(obj_acc, val_obj_acc))  

        print("\nTotal Loss\ttrain:{:01.4f}\tval:{:01.4f}".format(last_loss, val_last_loss))
        print("Action Loss\ttrain:{:01.4f}\tval:{:01.4f}".format(act_last_loss, val_act_last_loss))
        print("Object Loss\ttrain:{:01.4f}\tval:{:01.4f}".format(obj_last_loss, val_obj_last_loss))

        loss_data['epoch'].append(epoch)
        
        loss_data['acc']['action']['train'].append(act_acc)
        loss_data['acc']['action']['val'].append(val_act_acc)
        loss_data['acc']['object']['train'].append(obj_acc)
        loss_data['acc']['object']['val'].append(val_obj_acc)

        loss_data['loss']['total']['train'].append(last_loss)
        loss_data['loss']['total']['val'].append(val_last_loss)
        loss_data['loss']['action']['train'].append(act_last_loss)
        loss_data['loss']['action']['val'].append(val_act_last_loss)
        loss_data['loss']['object']['train'].append(obj_last_loss)
        loss_data['loss']['object']['val'].append(val_obj_last_loss)

        ###########
        #loss save#
        ###########
        # 1. save validation loss가 낮아질 때만 저장
        if val_last_loss < best_loss:
            best_loss = val_last_loss
            torch.save(model.state_dict(), model_path + '/GP_model_{}.pt'.format(epoch))
            torch.save(model.state_dict(), model_path + '/GP_model_best.pt')
        # # 2. 모든 epoch 저장
        # torch.save(model.state_dict(), model_path + '/GP_model_{}.pt'.format(epoch))
        # if val_last_loss < best_loss:
        #     best_loss = val_last_loss
        #     torch.save(model.state_dict(), model_path + '/GP_model_best.pt')

        #save loss record
        file_path = os.path.join(model_path,'loss_data')
        # createFolder(file_path)
        with open(file_path, "wb") as outfile:
            pickle.dump(loss_data, outfile)