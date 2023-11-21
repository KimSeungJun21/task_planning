from train.train import train_basic
# from inference.inference import inference
# from inference.inference import inference_rnn_sequence, inference_rnn_step, inference_rnn_gbfs
# from inference.inference import inference_basic_sequence, inference_basic_step, inference_basic_gbfs
from utils.arg_utils import parse_default_args
import time
import argparse
import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def run_train(train_args):
    start = time.time()
    # train start
    train_basic(train_args)

    end = time.time()
    print(f'training time: {end - start:.3f} sec')



# def run_inference(inference_args):

    # inference(inference_args)

    # # train start
    # if rnn_type == 'Basic':
    #     inference_basic_sequence(inference_args)
    #     # inference_basic_step(inference_args)
    #     # inference_basic_gbfs(inference_args)
    # else:
    #     inference_rnn_sequence(inference_args)
    #     # inference_rnn_step(inference_args)
    #     # inference_rnn_gbfs(inference_args)


if __name__ == '__main__':
    ## arguments setting ##
    args = parse_default_args()

    ############################
    ## input argument setting ##
    ############################
    ## setting needed arguments ##
    args.dataset_name = '1114_tasks'

    ## run train & inference ##
    run_train(args)
    # run_inference(args)
