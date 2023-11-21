import os
import matplotlib.pyplot as plt
import pickle
import time
import torch
import numpy as np

data_name = 'stack_mix_clean_0525_pose5_RNN_catX_OFGX_zeX_512_100_128_0.0001_tree'
plot_task = True #or length
plot_group = True

data_dir = os.path.join(os.getcwd(), 'test_result',data_name)
file_list = os.listdir(data_dir)

plan_result_list = []
failed_list = []
task_dict = {'stacking_5':[],
             'stacking_init2':[],
             'stacking_init2_reverse':[],
             'stacking_init3':[],
             'stacking_init3_replace':[],
             'stacking_init3_reverse':[],
             'stacking_init4_reverse':[],
             'mixing_3':[],
             'mixing_2':[],
             'mixing_withbox3':[],
             'mixing_withbox2':[],
             'mixing_withbox1':[],
             'cleaning_stacking_5':[],
             'cleaning_stacking_init2':[],
             'cleaning_stacking_init2_reverse':[],
             'cleaning_stacking_init3':[],
             'cleaning_stacking_init3_replace':[],
             'cleaning_stacking_init3_reverse':[],
             'cleaning_stacking_init4_reverse':[],
             'cleaning_mixing_3':[],
             'cleaning_mixing_2':[],
             'cleaning_mixing_withbox3':[],
             'cleaning_mixing_withbox2':[],
             'cleaning_mixing_withbox1':[]}

plan_len_dict = {}
for i in range(24):
    plan_len_dict[str(i+1)] = []

max_step_num = 0
num_success = 0
for plan_result in file_list:
    if 'plan_' in plan_result:
        with open(os.path.join(data_dir, plan_result), "rb") as file:
            plan_result_data = pickle.load(file)
            plan_result_list.append(plan_result_data)
            task_dict[plan_result_data['data_info']['demo'][0]].append(plan_result_data)
            step_num = plan_result_data['data_info']['goal'].item() - plan_result_data['data_info']['step'].item()
            plan_len_dict[str(step_num)].append(plan_result_data)
            #성공률
            if plan_result_data['plan_reached']:
                num_success+=1
                if step_num > max_step_num:
                    max_step_num = step_num
            else:
                failed_list.append(plan_result_data)
                # print("#######################")
                # print('task:', plan_result_data['data_info']['demo'][0])
                # print('start step:',plan_result_data['data_info']['step'].item())
                # print('plan list:')
                # print(plan_result_data['plan_list'])
                # print('planning time:',plan_result_data['time'])
print('success rate:',num_success/len(plan_result_list))
print(max_step_num)

########
# plot #
########
if plot_task:
########
# task #
########
    if plot_group:
        # stack / mix / cleaning
        xs_task = ['stacking', 'mixing', 'cleaning']

        task_success = [0,0,0]
        task_total = [0,0,0]

        for k,v in task_dict.items():
            # print('success rate of task',k)
            task_idx = 0
            if 'stacking' in k:
                task_idx = 0
            elif 'mixing' in k:
                task_idx = 1
            elif 'cleaning' in k:
                task_idx = 2
            
            for data in v:
                task_total[task_idx] += 1
                if data['plan_reached']:
                    task_success[task_idx] +=1
                else: #실패
                    pass
        ys_task = [task_success[i] / task_total[i] if task_total[i]!=0 else 1 for i in range(len(task_success))]
        print(ys_task)

        plt.figure(figsize=(15, 10))
        plt.title('Success Rate for Demo Task',fontweight='bold', fontsize=25)
        plt.ylim([0, 1])
        # cmap = ['y']*len(xs_len)
        # cmap[:5] = ['#e35f62']*5
        # cmap[5:10] = ['C2']*5
        # cmap[10:15] = ['dodgerblue']*5
        cmap = ['#e35f62', 'C2', 'dodgerblue']
        plt.bar(np.arange(len(xs_task)), ys_task, color = cmap)
        plt.xticks(np.arange(len(xs_task)), xs_task,fontweight='bold', fontsize=25)
        for i in range(len(xs_task)):
            rate = ys_task[i]
            plt.text(np.arange(len(xs_task))[i], ys_task[i],round(rate, 2) ,fontweight='bold',fontsize=20)
    else:
        # 각각
        xs_task = list(task_dict.keys())
        ys_task = []

        for k,v in task_dict.items():
            # print('success rate of task',k)
            n = 0
            for data in v:
                if data['plan_reached']:
                    n+=1
                else: #실패
                    pass
                    # print("#######################")
                    # print('task:', k)
                    # print('start step:',data['data_info']['step'].item())
                    # print('plan list:')
                    # print(data['plan_list'])
                    # print('planning time:',data['time'])
            # input()
            print(f'num:\t{n}/{len(v)}')
            print('rate:\t{:.2f}'.format(n/len(v)if len(v)!=0 else 0))
            ys_task.append(n/len(v)if len(v)!=0 else 0)

        xs_task_stacking = xs_task[:5]
        xs_task_mixing = xs_task[5:13]
        xs_task_cleaning = xs_task[13:]

        ys_task_stacking = ys_task[:5]
        ys_task_mixing = ys_task[5:13]
        ys_task_cleaning = ys_task[13:]


        plt.figure(0)
        plt.figure(figsize=(15, 10))
        # plt.suptitle('Success Rate')
        # plt.subplot(1,3,1)
        plt.title('Stacking')
        plt.ylim([0, 1])
        plt.bar(np.arange(len(xs_task_stacking)), ys_task_stacking, color = '#e35f62')
        plt.xticks(np.arange(len(xs_task_stacking)), xs_task_stacking)
        for i in range(len(xs_task_stacking)):
            rate = ys_task_stacking[i]
            plt.text(np.arange(len(xs_task_stacking))[i], ys_task_stacking[i], round(rate, 2))

        # plt.subplot(1,3,2)
        plt.figure(1)
        plt.figure(figsize=(15, 10))
        plt.title('Mixing')
        plt.ylim([0, 1])
        plt.bar(np.arange(len(xs_task_mixing)), ys_task_mixing, color = 'C2')
        plt.xticks(np.arange(len(xs_task_mixing)), xs_task_mixing)
        for i in range(len(xs_task_mixing)):
            rate = ys_task_mixing[i]
            plt.text(np.arange(len(xs_task_mixing))[i], ys_task_mixing[i], round(rate, 2))

        # plt.subplot(1,3,3)
        plt.figure(2)
        plt.figure(figsize=(15, 10))
        plt.title('Cleaning')
        plt.ylim([0, 1])
        plt.bar(np.arange(len(xs_task_cleaning)), ys_task_cleaning, color = 'dodgerblue')
        plt.xticks(np.arange(len(xs_task_cleaning)), xs_task_cleaning)
        for i in range(len(xs_task_cleaning)):
            rate = ys_task_cleaning[i]
            plt.text(np.arange(len(xs_task_cleaning))[i], ys_task_cleaning[i],round(rate, 2))

else:
##########
# length #
##########
    if plot_group:
        # 5개씩
        xs_len = ['1-5', '6-10', '11-15', '16-20', '21~']
        success_5 = [0,0,0,0,0]
        total_5 = [0,0,0,0,0]
        # ys_len = []
        print("##############")
        print("Length result")
        print("##############")
        for k,v in plan_len_dict.items():
            cls = (int(k)-1)//5
            if cls>4:
                cls=4
            for data in v:
                total_5[cls] += 1
                if data['plan_reached']:
                    success_5[cls] += 1

        ys_len = [success_5[i] / total_5[i] if total_5[i]!=0 else 1 for i in range(len(success_5))]
        print(ys_len)



        plt.figure(figsize=(15, 10))
        plt.title('Success Rate for Demo Length',fontweight='bold', fontsize=25)
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
    else:
        xs_len = []
        ys_len = []
        print("##############")
        print("Length result")
        print("##############")
        for k,v in plan_len_dict.items():
            print('success rate of step length',k)
            xs_len.append(int(k))
            n = 0
            for data in v:
                if data['plan_reached']:
                    n+=1
            print(f'num:\t{n}/{len(v)}')
            print('rate:\t{:.2f}'.format(n/len(v) if len(v)!=0 else 0))
            ys_len.append(n/len(v) if len(v)!=0 else 0)

        plt.figure(figsize=(15, 10))
        plt.title('Success Rate for Demo Length')
        plt.ylim([0, 1])
        cmap = ['y']*len(xs_len)
        cmap[:5] = ['#e35f62']*5
        cmap[5:10] = ['C2']*5
        cmap[10:15] = ['dodgerblue']*5
        plt.bar(np.arange(len(xs_len)), ys_len, color = cmap)
        plt.xticks(np.arange(len(xs_len)), xs_len)
        for i in range(len(xs_len)):
            rate = ys_len[i]
            plt.text(np.arange(len(xs_len))[i], ys_len[i],round(rate, 2))

plt.show()
print('success rate:',num_success/len(plan_result_list))



