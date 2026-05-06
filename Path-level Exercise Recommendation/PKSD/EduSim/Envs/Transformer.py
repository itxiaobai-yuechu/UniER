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

import torch
import torch.nn.functional as F


def SequenceMask(X, X_len, value=-1e6):
    maxlen = X.size(1)
    X_len = X_len.to(X.device)
    mask = torch.arange((maxlen), dtype=torch.float, device=X.device)
    mask = mask[None, :] < X_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X, valid_length):
    softmax = nn.Softmax(dim=-1)
    if valid_length is None:
        return softmax(X)
    else:
        shape = X.shape
        if valid_length.dim() == 1:
            try:
                valid_length = torch.FloatTensor(valid_length.numpy().repeat(shape[1], axis=0))
            except:
                valid_length = torch.FloatTensor(valid_length.cpu().numpy().repeat(shape[1], axis=0))
        else:
            valid_length = valid_length.reshape((-1,))
        X = SequenceMask(X.reshape((-1, shape[-1])), valid_length)

        return softmax(X).reshape(shape)


class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, valid_length=None):
        d = query.shape[-1]
        scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(d)
        attention_weights = self.dropout(masked_softmax(scores, valid_length))
        return torch.bmm(attention_weights, value)


class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, dropout, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(input_size, hidden_size, bias=False)
        self.W_k = nn.Linear(input_size, hidden_size, bias=False)
        self.W_v = nn.Linear(input_size, hidden_size, bias=False)
        self.W_o = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, query, key, value, valid_length=None):


        query = transpose_qkv(self.W_q(query), self.num_heads)
        key = transpose_qkv(self.W_k(key), self.num_heads)
        value = transpose_qkv(self.W_v(value), self.num_heads)

        if valid_length is not None:
            device = valid_length.device
            valid_length = valid_length.cpu().numpy() if valid_length.is_cuda else valid_length.numpy()
            if valid_length.ndim == 1:
                valid_length = torch.FloatTensor(np.tile(valid_length, self.num_heads))
            else:
                valid_length = torch.FloatTensor(np.tile(valid_length, (self.num_heads, 1)))

            valid_length = valid_length.to(device)

        output = self.attention(query, key, value, valid_length)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


def transpose_qkv(X, num_heads):
    X = X.view(X.shape[0], X.shape[1], num_heads, -1)

    X = X.transpose(2, 1).contiguous()

    output = X.view(-1, X.shape[2], X.shape[3])
    return output


def transpose_output(X, num_heads):
    X = X.view(-1, num_heads, X.shape[1], X.shape[2])
    X = X.transpose(2, 1).contiguous()
    return X.view(X.shape[0], X.shape[1], -1)


class PositionWiseFFN(nn.Module):
    def __init__(self, input_size, ffn_hidden_size, hidden_size_out, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.ffn_1 = nn.Linear(input_size, ffn_hidden_size)
        self.ffn_2 = nn.Linear(ffn_hidden_size, hidden_size_out)

    def forward(self, X):
        return self.ffn_2(F.relu(self.ffn_1(X)))


class AddNorm(nn.Module):
    def __init__(self, hidden_size, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, X, Y):
        return self.norm(self.dropout(Y) + X)


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, dropout, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = np.zeros((1, max_len, embedding_size))
        X = np.arange(0, max_len).reshape(-1, 1) / np.power(10000, np.arange(0, embedding_size, 2) / embedding_size)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)
        self.P = torch.FloatTensor(self.P)

    def forward(self, X):
        if X.is_cuda and not self.P.is_cuda:
            self.P = self.P.cuda()
        X = X + self.P[:, :X.shape[1], :]
        return self.dropout(X)


class EncoderBlock(nn.Module):
    def __init__(self, embedding_size, ffn_hidden_size, num_heads, dropout, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(embedding_size, embedding_size, num_heads, dropout)
        self.addnorm_1 = AddNorm(embedding_size, dropout)
        self.ffn = PositionWiseFFN(embedding_size, ffn_hidden_size, embedding_size)
        self.addnorm_2 = AddNorm(embedding_size, dropout)

    def forward(self, X, valid_length):
        Y = self.addnorm_1(X, self.attention(X, X, X, valid_length))
        return self.addnorm_2(Y, self.ffn(Y))


class TransformerEncoder(nn.Module):
    def __init__(self, input_size, embedding_size, ffn_hidden_size, num_heads, num_layers, dropout, in_order, max_seq_length, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.in_order = in_order
        self.embedding_size = embedding_size
        self.embed = nn.Linear(input_size, embedding_size, bias=False)
        self.pos_encoding = PositionalEncoding(embedding_size, dropout, max_len=max_seq_length)
        self.blks = nn.ModuleList()
        for i in range(num_layers):
            self.blks.append(EncoderBlock(embedding_size, ffn_hidden_size, num_heads, dropout))

    def forward(self, X, valid_length, *args):
        if self.in_order:
            X = self.pos_encoding(self.embed(X) * math.sqrt(self.embedding_size))
        else:
            X = self.embed(X)
        for blk in self.blks:
            X = blk(X, valid_length)
        return X


class Transformer(nn.Module):
    def __init__(self,
                 input_dim,
                 embedding_size,
                 out_dim,
                 in_order,
                 num_heads=8,
                 num_encoder_layers=6,
                 dim_feedforward=512,
                 dropout=0.1,
                 max_seq_length=50,
                 encoder_only=True,
                 num_decoder_layers=6, ):
        super(Transformer, self).__init__()
        self.name = 'transformer'
        self.in_order = in_order
        while embedding_size % num_heads != 0:
            embedding_size = embedding_size + 1
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        while embedding_size % self.num_heads != 0:
            self.num_heads = self.num_heads - 1
        self.TransformerEncoder = TransformerEncoder(input_size=input_dim,
                                                     embedding_size=embedding_size,
                                                     ffn_hidden_size=dim_feedforward,
                                                     num_heads=num_heads,
                                                     num_layers=num_encoder_layers,
                                                     dropout=dropout,
                                                     in_order=in_order,
                                                     max_seq_length=max_seq_length)
        self.output_layer = nn.Linear(embedding_size, out_dim)
        self._reset_parameters()

    def forward(self, X, valid_length=None):
        X = self.TransformerEncoder(X, valid_length)
        X = self.output_layer(X)
        return X

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    print('cuda_ava:' + str(torch.cuda.is_available()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_skills = 835
    test_input = torch.randn(16, 20, 200).to(device)
    model = Transformer(input_dim=200,
                        embedding_size=128,
                        out_dim=64,
                        in_order=True,
                        num_heads=8,
                        num_encoder_layers=6,
                        dim_feedforward=512,
                        dropout=0.1,
                        max_seq_length=num_skills).to(device)
    result = model(test_input)
    print(result)
