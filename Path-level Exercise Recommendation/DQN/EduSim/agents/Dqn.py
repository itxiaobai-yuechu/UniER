import torch
import numpy as np
import os
import torch.nn as nn
from EduSim.Envs.buffer import *
from EduSim.Envs.deep_model import *


class DQNAgent(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim1,
                 hidden_dim2,
                 advance_types,
                 target_update,
                 env,
                 lr_rate=0.0001,
                 gamma=0.98,
                 device='cuda:0',
                 lr_sche=False,
                 args={}):
        super(DQNAgent, self).__init__()
        self.name = 'dqn'
        self.policy_mode = 'off_policy'
        self.pre_learn_node = '-1'
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = lr_rate
        self.device = device
        self.advance_types = advance_types
        self.gamma = gamma
        self.lr_sche = lr_sche
        self.count = 0
        self.epsilon = 0.01
        self._n_train_steps_total = 0
        self.target_update = target_update
        self.tau = 0.001
        self.env = env
        self.args = args

        self.q_net = Qnet(self.input_dim,
                          hidden_dim1,
                          hidden_dim2,
                          self.output_dim,
                          dueling_dqn=self.advance_types['dueling dqn'],
                          noisy_dqn=self.advance_types['noisy dqn'],
                          args=args).to(self.device)
        self.target_q_net = copy.deepcopy(self.q_net)

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.95)


    def step(self, states, candidates):
        preds = self.q_net(states)
        candidate_preds = preds.gather(1, torch.tensor(candidates).view(1, -1).to(self.device))
        max_value, idx = torch.max(candidate_preds, dim=1)
        idx = candidates[idx]

        if random.random() < self.epsilon:
            idx = np.random.randint(self.output_dim)
        return idx

    def learn(self, RL_states, actions, RL_next_states, rewards, dones, tree_idxs=None, ISweights=None, buffer=None):
        if self.lr_sche:
            self.scheduler.step()

        with torch.no_grad():
            if self.advance_types['double dqn']:
                max_actions = self.q_net(RL_next_states).max(1)[1].view(-1, 1)
                max_next_q_values = self.target_q_net(RL_next_states).gather(1, max_actions)
            else:
                max_next_q_values = self.target_q_net(RL_next_states).max(1)[0].view(-1, 1)
            q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
            if self.advance_types['multi_step dqn']:
                q_targets = rewards + (self.gamma ** self.advance_types['n_multi_step']) * max_next_q_values * (1 - dones)
        q_values = self.q_net(RL_states).gather(1, actions)

        if self.advance_types['prioritized dqn']:
            dqn_loss = torch.mean(ISweights * F.mse_loss(q_values, q_targets))
            abs_errors = torch.sum(torch.abs(q_values - q_targets), dim=1)
            buffer.batch_update(tree_idxs, abs_errors.cpu().detach().numpy())
        else:
            dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))

        self.optimizer.zero_grad()
        dqn_loss.backward(retain_graph=True)
        self.optimizer.step()

        self.update_target_net()

        self._n_train_steps_total += 1


    def update_target_net(self, tau=None):
        if tau is None:
            tau = self.tau
        for target_param, local_param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)