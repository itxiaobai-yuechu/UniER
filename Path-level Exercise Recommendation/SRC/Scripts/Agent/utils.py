import torch
import numpy as np

def pl_loss(pro, reward):

    return -torch.mean(reward * torch.log(pro + 1e-9))


def generate_path(batch_size, skill_num, path_type, n):

    if path_type in (0, 1):
        origin_path = np.argsort(np.random.rand(batch_size, n))
        
        if path_type == 1:
            offset = n * np.random.randint(0, skill_num // n, (batch_size, 1))
            origin_path += offset
            
    else:
        origin_path = np.argsort(np.random.rand(batch_size, skill_num))
        
        if path_type == 2:
            origin_path = origin_path[:, :n]
            
    return torch.from_numpy(origin_path.astype(np.int32))