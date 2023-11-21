import os
import matplotlib.pyplot as plt
import pickle
import time
import torch
import numpy as np
from collections import Counter

def read_result(device, text_encoder, gradient_clipping, model_choose, method, information, maximum_plan_length, hidden_dim, num_workers,
                                    num_action, num_object, node_feature_size, edge_feature_size, global_dim, batch_size, lr, train_seed,
                                    num_epoch, data_dir, show_result, infer_num=None, check_each = False, data_save=True, weight_decay=0):
    model_param = [data_dir, hidden_dim, num_epoch, batch_size, lr]


    file_dir = os.path.join(os.getcwd(),"test_result", 
                            information, text_encoder, 
                            f"gradient_clipping_{gradient_clipping}",
                            f"weight_decay_{weight_decay}","_".join(list(map(str, model_param))),
                            f"train_seed_{train_seed}", 
                            model_choose, method, 
                            "Sequential_planning")
    file_list = os.listdir(file_dir)

    plan_result_list = []
    failed_list = []
    task_dict = {}
    task_type_list = ['clustering_1_B1_R1','clustering_2_B12_R11','clustering_2_B12_R12', 'clustering_3_B123_R111', 'clustering_3_B123_R112', 'clustering_3_B123_R123', 'clustering_4_B1234_R1111', 'clustering_4_B1234_R1112', 'clustering_4_B1234_R1122', 'clustering_4_B1234_R1123', 'clustering_4_B1234_R1234', 'clustering_5_B12345_R11111', 'clustering_5_B12345_R11112', 'clustering_5_B12345_R11122', 'clustering_5_B12345_R11123', 'clustering_5_B12345_R11223', 'clustering_5_B12345_R11234', 'stacking_1_B1_R1', 'stacking_2_B12_R1', 'stacking_3_B123_R1', 'stacking_4_B1234_R1', 'stacking_5_B12345_R1', 'collecting_1_BW1_BW6', 'collecting_2_BW12_BW6', 'collecting_3_BW123_BW6', 'collecting_4_BW1234_BW6', 'collecting_5_BW12345_BW6', 'pouring_1_BW67_R1_INIT_BW1_BW6', 'pouring_2_BW67_R1_INIT_BW12_BW6', 'pouring_3_BW67_R1_INIT_BW123_BW6', 'pouring_4_BW67_R1_INIT_BW1234_BW6', 'pouring_5_BW67_R1_INIT_BW12345_BW6', 'cluster-stacking_3_B123_R12_C12', 'cluster-stacking_4_B1234_R12_C13', 'cluster-stacking_4_B1234_R12_C22', 'cluster-stacking_4_B1234_R123_C112', 'cluster-stacking_5_B12345_R12_C14', 'cluster-stacking_5_B12345_R12_C23', 'cluster-stacking_5_B12345_R123_C113', 'cluster-stacking_5_B12345_R123_C122', 'cluster-stacking_5_B12345_R1234_C1112', 'collect-pouring_1_BW1_BW67_R1', 'collect-pouring_2_BW12_BW67_R1', 'collect-pouring_3_BW123_BW67_R1', 'collect-pouring_4_BW1234_BW67_R1', 'collect-pouring_5_BW12345_BW67_R1']
    for task_type in task_type_list:
        task_dict[task_type] = []

    plan_len_dict = {}
    for i in range(20):
        plan_len_dict[str(i+1)] = []

    max_step_num = 0
    num_success = 0
    for plan_result in file_list:
        if 'plan_' in plan_result:
            with open(os.path.join(file_dir, plan_result), "rb") as file:
                plan_result_data = pickle.load(file)
                # print(plan_result_data)
                # input()

                data_info = plan_result_data['data_info']


                plan_result_list.append(plan_result_data)
                task_name = plan_result_data['data_info']['demo'][0]
                # task_type = task_name.split('_')[0]
                # task_size = task_name.split('_')[1]
                
                task_dict[task_name].append(plan_result_data)

                step_num = plan_result_data['data_info']['goal'].item() - plan_result_data['data_info']['step'].item()
                plan_len_dict[str(step_num)].append(plan_result_data)
                #성공률
                if plan_result_data['plan_reached']:
                    num_success+=1
                    if step_num > max_step_num:
                        max_step_num = step_num
                else:
                    failed_list.append(plan_result_data)

    print('success rate:',num_success/len(plan_result_list))
    print(max_step_num)

    ########
    # plot #
    ########

    #################
    # planning time #
    #################
    # 5개씩
    xs_len = ['1-5', '6-10', '11-15', '16-20', '21~']
    time_5 = [0, 0, 0, 0, 0]
    success_5 = [0,0,0,0,0]
    total_5 = [0,0,0,0,0]
    # ys_len = []
    print("####################")
    print("planning time result")
    print("####################")
    for k,v in plan_len_dict.items():
        cls = (int(k)-1)//5
        if cls>4:
            cls=4
        for data in v:
            total_5[cls] += 1
            if data['plan_reached']:
                success_5[cls] += 1
                # time
                planning_time = data['time']
                time_5[cls] += planning_time
    
    time_avg = [0 ,0, 0, 0, 0]
    
    for i in range(4):
        time_avg[i] = round((time_5[i] / (success_5[i] if success_5[i]>0 else 1)), 5)
    ys_len = time_avg
    print(ys_len)

    print(success_5)
    print(total_5)


    plt.figure(figsize=(15, 10))
    plt.title('Planning Time for Demo Length',fontweight='bold', fontsize=25)
    plt.ylim([0, 1])
    # cmap = ['y']*len(xs_len)
    # cmap[:5] = ['#e35f62']*5
    # cmap[5:10] = ['C2']*5
    # cmap[10:15] = ['dodgerblue']*5
    cmap = ['#e35f62', 'C2', 'dodgerblue', 'y', 'c']
    plt.bar(np.arange(len(xs_len)), ys_len, color = cmap)
    plt.xticks(np.arange(len(xs_len)), xs_len,fontweight='bold', fontsize=20)
    for i in range(len(xs_len)):
        rate = ys_len[i]

        if rate>0.95:
            plt.text(np.arange(len(xs_len))[i] - 0.03, ys_len[i] - 0.025,round(rate, 2),fontweight='bold', fontsize=15)
        else:
            plt.text(np.arange(len(xs_len))[i] - 0.03, ys_len[i],round(rate, 2),fontweight='bold', fontsize=15)

    plt.savefig(os.path.join(file_dir, 'time_fig.png'))
    #output text로 저장
    result_path = os.path.join(file_dir, 'time_result.txt')
    with open(result_path, 'w') as f:
        print("time: ",ys_len, file=f)
        print("success: ",success_5, file=f)
        print("total: ", total_5, file=f)
    # plt.show()

