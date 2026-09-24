import random
from gensim.models import Word2Vec

import math
import networkx as nx
import random
import numpy as np
import sys
import os
import torch.nn as nn

sys.path.append(os.path.dirname(sys.path[0]))
import warnings
import torch
from EduSim.Envs.agent_utils import *

warnings.filterwarnings('ignore')


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):

        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = (x - mean).pow(2).mean(-1, keepdim=True)
        x = (x - var) / torch.sqrt(var + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob, device=None):
        super(SelfAttention, self).__init__()
        self.device = device
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def get_attn_pad_mask(self, input_tensor):

        batch_size = input_tensor.size(0)
        max_target_num = input_tensor.size(1)

        mask_tensor = torch.sum(torch.abs(input_tensor), dim=-1, keepdim=False).view(batch_size, 1, 1, max_target_num)
        paddings = (torch.ones(size=[batch_size, 1, 1, max_target_num]) * (-2 ** 32 + 1)).to(self.device)
        zeros = torch.zeros(size=[batch_size, 1, 1, max_target_num]).to(self.device)
        add_mask_tensor = torch.where(torch.tensor(mask_tensor == 0), paddings, zeros)

        mask_tensor = mask_tensor.view(batch_size, max_target_num, 1).to(self.device)
        ones = torch.ones(size=mask_tensor.size()).to(self.device)
        zeros = torch.zeros(size=mask_tensor.size()).to(self.device)
        dot_mask_tensor = torch.where(torch.tensor(mask_tensor == 0), zeros, ones)
        return add_mask_tensor, dot_mask_tensor

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        add_attention_mask, dot_mask = self.get_attn_pad_mask(input_tensor)
        attention_scores = attention_scores + add_attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        hidden_states = hidden_states * dot_mask
        return hidden_states


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    print('cuda_ava:' + str(torch.cuda.is_available()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    targets = [[1, 2, 3], [3, 4, 5, 6]]
    num_skills = 10
    embedding_dims = 10
    graph_embeddings = get_graph_embeddings('KSS')
    graph_embeddings = torch.tensor(graph_embeddings).to(device)

    batch_size = 2
    max_targets_num = num_skills
    targets_tensor = torch.zeros((batch_size, max_targets_num, embedding_dims), dtype=torch.float).to(device)

    for i, sample in enumerate(targets):
        for j, target_id in enumerate(sample):
            targets_tensor[i][j] = graph_embeddings[target_id]

    sel_attn = SelfAttention(num_attention_heads=2,
                             input_size=embedding_dims,
                             hidden_size=embedding_dims,
                             hidden_dropout_prob=0.0,
                             device=device).to(device)
    result = sel_attn(targets_tensor)
    result = torch.mean(result, dim=1)
    print(result)
