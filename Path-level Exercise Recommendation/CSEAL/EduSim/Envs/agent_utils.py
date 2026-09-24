import torch
import numpy as np
import time
import torch.nn as nn
import os


def get_raw_data_path(proj_name='CSEAL'):
    cur_path = os.path.abspath(os.path.dirname(__file__))

    return cur_path[:cur_path.find(proj_name)] + proj_name + '/data'


def get_proj_path(proj_name='CSEAL'):

    cur_path = os.path.abspath(os.path.dirname(__file__))

    return cur_path[:cur_path.find(proj_name)] + proj_name


def batch_cat_targets(DKT_states, targets, num_skills, device, graph_embedding=False, graph_embeddings=None, agg_model=None):
    target_tensor = torch.zeros([DKT_states.size(0), num_skills], dtype=torch.float).to(device)
    if not graph_embedding:
        for i, sample in enumerate(targets):
            index = torch.LongTensor(sample).to(device)
            target_tensor[i].scatter_(dim=0, index=index, value=1)
    else:

        device = DKT_states.device
        graph_embeddings = torch.tensor(graph_embeddings).to(device)
        embedding_dim = graph_embeddings.size(-1)
        batch_size = len(targets)
        max_targets_num = num_skills
        targets_tensor = torch.zeros((batch_size, max_targets_num, embedding_dim), dtype=torch.float).to(device)

        for i, sample in enumerate(targets):
            for j, target_id in enumerate(sample):
                targets_tensor[i][j] = graph_embeddings[target_id]

        result = agg_model(targets_tensor)
        result = torch.mean(result, dim=1)
        target_tensor = result
    return torch.cat((DKT_states, target_tensor), dim=1)


def get_feature_matrix(sequence, targets, action_dim, embedding_dim, max_sequence_length=20, device=None, graph_embedding=False,
                       graph_embeddings=None):
    input_data = torch.FloatTensor(max_sequence_length, embedding_dim)

    if device is not None:
        input_data = input_data.to(device)

    input_data.zero_()

    if not graph_embedding:
        index = torch.tensor([int(item[0]) if item[1] == 0 else int(item[0]) + action_dim for item in sequence]).view(-1, 1).to(input_data.device)
        input_data.scatter_(1, index, 1)

        if embedding_dim > 2 * action_dim and len(sequence) > 0:
            index = torch.tensor(np.array(targets) + 2 * action_dim).view(1, -1).expand(len(sequence), -1).to(input_data.device)
            input_data.scatter_(1, index, 1)

    else:
        graph_embeddings = torch.tensor(graph_embeddings).to(input_data.device)
        for i, log in enumerate(sequence):
            if log[1] == 0:
                input_data[i][:int(graph_embeddings.size(-1))] = graph_embeddings[int(log[0])]
            else:
                input_data[i][-int(graph_embeddings.size(-1)):] = graph_embeddings[int(log[0])]

    return input_data


def episode_reward_reshape(episode_log, episode_reward, episode_targets, episode_dkt_states=None, targets=None,
                           intrin_sic_reward_flag=False, args={}):
    items = [log[0] for i, log in enumerate(episode_log[-1][3])]
    answers = [log[1] for i, log in enumerate(episode_log[-1][3])]
    for i, el in enumerate(episode_log):
        el[2] = 0.0

        if i == len(episode_log) - 1:
            el[2] = episode_reward
            if episode_reward <= 0:
                el[2] = -1.0 / (len(list(set(items))))





def compute_dkt_loss(output, batch_data, device):
    loss_f = nn.BCEWithLogitsLoss()
    num_skills = int(batch_data[0].size(1) / 2)
    sequence_lengths = [int(torch.sum(sample).detach()) for sample in batch_data]
    target_corrects = torch.tensor([], dtype=torch.int64).to(device)
    target_ids = torch.tensor([], dtype=torch.int64).to(device)
    output = output.permute(1, 0, 2)
    for episode in range(batch_data.size(0)):
        tmp_target_id = torch.argmax(batch_data[episode, :, :], -1).to(device)
        ones = torch.ones(tmp_target_id.size()).to(device)
        zeros = torch.zeros(tmp_target_id.size()).to(device)
        target_correct = torch.where(torch.tensor(tmp_target_id > num_skills - 1, dtype=torch.bool), ones, zeros).to(device).unsqueeze(
            1).unsqueeze(0)
        target_id = torch.where(torch.tensor(tmp_target_id > num_skills - 1, dtype=torch.bool), tmp_target_id - num_skills, tmp_target_id) \
            .to(device)
        target_id = torch.roll(target_id, -1, 0).unsqueeze(1).unsqueeze(0)
        target_ids = torch.cat((target_ids, target_id), 0)
        target_corrects = torch.cat((target_corrects, target_correct))
    logits = output.gather(2, target_ids)
    preds = torch.sigmoid(logits)
    loss = torch.tensor([0.0], requires_grad=True).to(device)
    for i, sequence_length in enumerate(sequence_lengths):
        if sequence_length <= 1:
            continue
        loss = loss + loss_f(logits[i, 0:sequence_length - 1], target_corrects[i, 1:sequence_length])
    return loss


def get_graph_embeddings(env_name):
    if env_name == 'KSS':
        data_path = f'{get_proj_path()}/EduSim/Envs/meta_data/MyTransitionGraph.npy'
    elif env_name == 'KES_junyi':
        data_path = f'{get_proj_path()}/EduSim/dataProcess/junyi/MyTransitionGraph.npy'
    elif env_name == 'KES_ASSIST15':
        data_path = f'{get_proj_path()}/data/dataProcess/assist15/MyTransitionGraph.npy'
    elif env_name == 'KES_ASSIST09':
        data_path = f'{get_proj_path()}/data/dataProcess/assist09/MyTransitionGraph.npy'
    elif env_name == 'KES_ASSIST12':
        data_path = f'{get_proj_path()}/data/dataProcess/assist12/MyTransitionGraph.npy'
    elif env_name == 'KES_ASSIST17':
        data_path = f'{get_proj_path()}/data/dataProcess/assist17/MyTransitionGraph.npy'
    elif env_name == 'KES_algebra2005':
        data_path = f'{get_proj_path()}/data/dataProcess/algebra2005/MyTransitionGraph.npy'
    elif env_name == 'KES_bridge2006':
        data_path = f'{get_proj_path()}/data/dataProcess/bridge2006/MyTransitionGraph.npy'
    elif env_name == 'KES_nips34':
        data_path = f'{get_proj_path()}/data/dataProcess/nips34/MyTransitionGraph.npy'
    elif env_name == 'KES_ednet':
        data_path = f'{get_proj_path()}/data/dataProcess/ednet/MyTransitionGraph.npy'
    elif env_name == 'KES_junyi':
        data_path = f'{get_proj_path()}/data/dataProcess/junyi/MyTransitionGraph.npy'
    elif env_name == 'KES_mooccube':
        data_path = f'{get_proj_path()}/data/dataProcess/mooccube/MyTransitionGraph.npy'
    elif env_name == 'KES_xes3g5m':
        data_path = f'{get_proj_path()}/data/dataProcess/xes3g5m/MyTransitionGraph.npy'
    else:
        raise ValueError('Wrong graph embedding data path')


    try:
        embeddings = np.load(data_path)
    except:
        embeddings = None
        print(f'no {data_path} yet')
    return embeddings


def repe_action_constrain(ori_can_probs, candidates, args):
    id = torch.argmax(ori_can_probs, dim=1)
    action = candidates[id]
    repe_num_thresh = max(min(args['max_steps'] - 2, int(0.3 * args['max_steps'])), 3)
    prob_thresh = 5 * 1 / len(candidates)
    if int(action) in args['item_count_dict'].keys():
        if args['item_count_dict'][int(action)] >= repe_num_thresh and ori_can_probs[0, id] >= prob_thresh:
            dif = ori_can_probs[0, id] - prob_thresh
            ori_can_probs[0, id] = ori_can_probs[0, id] - 2 * dif
            new_can_probs = ori_can_probs + dif / len(candidates)
            return new_can_probs
    return ori_can_probs


def mds_concat(tensor_list, axis):
    """Concatenate tensors while handling the empty accumulator used by envDKT."""
    if tensor_list[0].shape[0] == 0:
        return tensor_list[1]
    return torch.cat(tensor_list, axis)
