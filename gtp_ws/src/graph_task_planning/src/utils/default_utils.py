import os
from functools import lru_cache
import torch
from pathlib import Path
import glob
import argparse
import numpy as np
import random

# @lru_cache() 
# # Least Recently Used
# #함수의 입출력을 동시에 저장하여 이전에 호출된 입력에 대한 결과를 재사용하는 것 (다시 계산 X, 캐시에 있는 것 가져와 성능 향상 시킬 수 있음)
# def default_seq_data_path():
#     return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'seq_dataset')
# # return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz") # CLIP에서 이렇게 불러옴!

@lru_cache()
def abs_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)))

@lru_cache()
def root_path():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@lru_cache()
def default_data_path():
    return os.path.join(os.path.abspath(os.getcwd()),'demo_generation', 'embeddings_results') #main 폴더에서 실행했을 때 기준
    # return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'seq_dataset', 'embeddings_results')

# # Start with seq_dataset
# @lru_cache()
# def default_embedding_data_path():
#     return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'embeddings_results')


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            pass
    except OSError:
        print ('Error: Creating directory.'  +  directory)

def set_seed(seed_value):
    # NumPy 시드 설정
    np.random.seed(seed_value)
    # Torch 시드 설정
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    # 파이썬 기본 random 모듈 시드 설정
    random.seed(seed_value)

def select_method(input: torch.Tensor, method):
    if method == 'mean':
        np_mean = input.mean(dim=0)
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor.")
    return np_mean


def save_tensor(input: torch.Tensor, directory: Path, object_name: str):
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor.")
    torch.save(input, f'{directory}/{object_name}.pt')


def load_tensor(path: Path):
    tensor_list = []
    print(f'======Loading data from embeddings_results======')
    while True:
        files = glob.glob(f'{path}/*.pt')
        for file in files:
            load_tensor = torch.load(file)
            # print(load_tensor.shape)
            tensor_list.append(load_tensor)
            output = torch.stack(tensor_list, dim=0)
            # print(output.shape)
            return output

def node_feature_loader(text_embedding_path,obj_list):
    tensor_list = []
    for obj in obj_list:
        loaded_tensor = torch.load(f'{text_embedding_path}/{obj}.pt')
        tensor_list.append(loaded_tensor)
    # print(obj_list)
    # input()

    output = torch.stack(tensor_list, dim=0)
    return output