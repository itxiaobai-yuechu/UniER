
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import sys



def get_raw_data_path(proj_name='KnowLP'):
    cur_path = os.path.abspath(os.path.dirname(__file__))
    return cur_path[:cur_path.find(proj_name)] + proj_name + '/data'


def get_proj_path(proj_name='KnowLP'):
    cur_path = os.path.abspath(os.path.dirname(__file__))
    return cur_path[:cur_path.find(proj_name)] + proj_name


def mean_entropy_cal(dist):
    return torch.mean(torch.sum(dist * torch.log(dist), dim=1))


def batch_cat_targets(KT_states, targets, num_skills):
    target_tensor = torch.zeros((KT_states.shape[0], num_skills), dtype=torch.float32)

    for i, sample in enumerate(targets):
        index = torch.tensor(sample, dtype=torch.int64)
        updates = torch.ones(index.shape, dtype=torch.float32)
        target_tensor[i].scatter_(0, index, updates)

    return torch.cat((KT_states, target_tensor), 1)


def get_feature_matrix(sequence, action_dim, embedding_dim, max_sequence_length=20):
    input_data = torch.zeros((max_sequence_length, embedding_dim))

    if sequence:
        index = torch.tensor([int(item[0]) if item[1] == 0 else int(item[0]) + action_dim for item in sequence],
                                 dtype=torch.int64)
        input_data[:len(sequence)] = F.one_hot(index, num_classes=embedding_dim).float()

    return input_data


def episode_reward_reshape(episode_log, episode_reward):
    items = [log[0] for i, log in enumerate(episode_log[-1][3])]
    for i, el in enumerate(episode_log):
        el[2] = 0.0

        if i == len(episode_log) - 1:
            el[2] = episode_reward
            if episode_reward <= 0:
                el[2] = -1.0 / (len(list(set(items))))


def sample_reward_reshape(score, next_score, targets):
    reward = 0
    for i in targets:
        if score[i] == 0 and next_score[i] == 1:
            reward += 1
    return reward


def compute_KT_loss(output, batch_data):
    loss_f = nn.BCEWithLogitsLoss()
    num_skills = int(batch_data[0].shape[1] / 2)
    sequence_lengths = [int(torch.sum(sample)) for sample in batch_data]
    target_corrects = torch.tensor([])
    target_ids = torch.tensor([])
    output = output.permute(1, 0, 2)
    for episode in range(batch_data.shape[0]):
        tmp_target_id = torch.argmax(batch_data[episode, :, :], dim=-1)
        ones = torch.ones(tmp_target_id.shape, dtype=torch.float32)
        zeros = torch.zeros(tmp_target_id.shape, dtype=torch.float32)
        target_correct = torch.where(tmp_target_id > num_skills - 1, ones, zeros).unsqueeze(1).unsqueeze(0)
        target_id = torch.where(tmp_target_id > num_skills - 1, tmp_target_id - num_skills, tmp_target_id)

        target_id = torch.roll(target_id, -1, 0).unsqueeze(1).unsqueeze(0)

        target_ids = trc_concat((target_ids, target_id), 0)
        target_corrects = trc_concat((target_corrects, target_correct), 0)
    logits = output.gather(dim=2, index=target_ids.long())
    loss = torch.tensor([0.0])
    for i, sequence_length in enumerate(sequence_lengths):
        if sequence_length <= 1:
            continue
        a = logits[i, 0:sequence_length - 1]
        b = target_corrects[i, 1:sequence_length]
        loss = loss + loss_f(a, b)
    return loss


def get_graph_embeddings(env_name):
    if env_name == 'KSS':
        data_path = f'{get_proj_path()}/EduSim/Envs/meta_data/KSSGraphEmbedding.npy'
    elif env_name == 'KES_junyi':
        data_path = f'{get_proj_path()}/EduSim/Envs/meta_data/junyiGraphEmbedding.npy'
    elif env_name == 'KES_assist09':
        data_path = f'{get_proj_path()}/data/dataProcess/assist09/ASSIST09GraphEmbedding.npy'
    elif env_name == 'KES_assist12':
        data_path = f'{get_proj_path()}/data/dataProcess/assist12/ASSIST12GraphEmbedding.npy'
    elif env_name == 'KES_assist17':
        data_path = f'{get_proj_path()}/data/dataProcess/assist17/ASSIST17GraphEmbedding.npy'
    elif env_name == 'KES_algebra2005':
        data_path = f'{get_proj_path()}/data/dataProcess/algebra2005/algebra2005GraphEmbedding.npy'
    elif env_name == 'KES_bridge2006':
        data_path = f'{get_proj_path()}/data/dataProcess/bridge2006/bridge2006GraphEmbedding.npy'
    elif env_name == 'KES_ednet':
        data_path = f'{get_proj_path()}/data/dataProcess/ednet/ednetGraphEmbedding.npy'
    elif env_name == 'KES_junyi':
        data_path = f'{get_proj_path()}/data/dataProcess/junyi/junyiGraphEmbedding.npy'
    elif env_name == 'KES_xes3g5m':
        data_path = f'{get_proj_path()}/data/dataProcess/xes3g5m/xes3g5mGraphEmbedding.npy' 
    elif env_name == 'KES_nips34':
        data_path = f'{get_proj_path()}/data/dataProcess/nips34/nips34GraphEmbedding.npy'
    else:
        raise ValueError('Wrong graph embedding data path')

    try:
        embeddings = np.load(data_path)
    except FileNotFoundError:
        embeddings = None
        print(f'no {data_path} yet')
    return embeddings


def trc_concat(tensor_list, axis):
    if tensor_list[0].shape[0] == 0:
        return_thing = tensor_list[1]
    else:
        return_thing = torch.cat(tensor_list, dim=axis)
    return return_thing
