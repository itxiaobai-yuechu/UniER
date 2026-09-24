from torch import nn
import torch
import time
import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
import json
import re

import os


class DKTRec(nn.Module):
    def __init__(self, DKTnet, max_length, env, device=None, use_cuda=False):
        super(DKTRec, self).__init__()
        self.name = "dktrec"
        self.policy_mode = "on_policy"
        self.max_sequence_length = max_length

        if device is None:
            self.use_cuda = use_cuda
            self.device = torch.device('cuda' if use_cuda else 'cpu')
        else:
            self.device = torch.device(device)
            self.use_cuda = self.device.type == 'cuda'

        self.DKTnet = DKTnet
        self = self.to(self.device)

    def step(self, input, candidates, step):
        logit = torch.sigmoid(self.DKTnet(input))[max(0, step - 1), 0, :].unsqueeze(0)
        candidate_preds = torch.gather(logit, 1, torch.tensor(candidates).view(1, -1).to(self.device))
        zeros = torch.zeros(candidate_preds.size()).to(self.device)
        candidate_preds = torch.where(candidate_preds < 0.5, candidate_preds, zeros)
        max_value, idx = torch.max(candidate_preds, dim=1)

        return candidates[idx.item()]

    def learn(self, RL_states, actions, RL_next_states, rewards, dones):
        pass

