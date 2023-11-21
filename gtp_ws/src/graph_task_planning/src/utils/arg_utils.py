import argparse
import torch

def parse_default_args(print_arg=False):
    # Create an ArgumentParser object with values
    parser = argparse.ArgumentParser(description='Launch file')

    # Add common args
    parser.add_argument("--device", dest='device', type=str, choices=['cpu','cuda'], default=('cuda:0' if torch.cuda.is_available() else 'cpu'),
                        help="Select the device ('cpu' or 'cuda')")
    
    ## model output parameter ##
    parser.add_argument("--num_action", dest='num_action', type=int, default=2)
    parser.add_argument("--num_object", dest='num_object', type=int, default=9)
    
    ## model dimension ##
    parser.add_argument("--node_feature_size", dest='node_feature_size', type=int, default=12) 
    parser.add_argument("--edge_feature_size", dest='edge_feature_size', type=int, default=4)
    parser.add_argument("--hidden_dim", dest='hidden_dim', type=int, default=32)
    parser.add_argument("--global_dim", dest='global_dim', type=int, default=32)

    ## training hyper parameter ##
    parser.add_argument("--num_epoch", dest='num_epoch', type= int, default= 50)
    parser.add_argument("--lr", dest='lr', type=float, default=1e-04)
    parser.add_argument("--weight_decay", dest='weight_decay', type=float, default=0)
    parser.add_argument("--gradient_clipping", dest='gradient_clipping', type=bool, default=True)
    parser.add_argument("--batch_size", dest='batch_size', type=int, default=8)
    
    ## model type ##
    parser.add_argument("--method", dest='method', type=str, default='node_scoring', choices=['mean', 'node_scoring'], 
                        help="Choose one of the available options ('mean', 'node_scoring')")

    ## training dataset name ##
    parser.add_argument("--dataset_name", dest='dataset_name', type= str, default='1114_tasks') 

    #### inference setting ####
    parser.add_argument("--maximum_plan_length", dest='maximum_plan_length', type=int, default=25)
    parser.add_argument("--show_result", dest='show_result', type=bool, default=False)
    parser.add_argument("--check_each", dest='check_each', type=bool, default=False)
    parser.add_argument("--data_save", dest='data_save', type=bool, default=True)
    parser.add_argument("--only_final_goal", dest='only_final_goal', type=bool, default=False)
    parser.add_argument("--infer_num", dest='infer_num', type=str, default=None)


    args = parser.parse_args()

    if print_arg:
        for arg_name in vars(args):
            arg_value = getattr(args, arg_name)
            print(f'{arg_name}: {arg_value}')

    return args
