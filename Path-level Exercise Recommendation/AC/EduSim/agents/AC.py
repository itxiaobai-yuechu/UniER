import torch
import numpy as np
import os
import torch.nn as nn
from EduSim.Envs.deep_model import *


class ActorCritic(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim1,
                 hidden_dim2,
                 env,
                 lr_rate=0.0001,
                 gamma=0.98,
                 device='cuda:0',
                 lr_sche=False,
                 args={}):
        super(ActorCritic, self).__init__()
        self.name = 'ac'
        self.policy_mode = 'on_policy'
        self.pre_learn_node = '0'
        self.device = device
        self.gamma = gamma
        self.learning_rate = lr_rate
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr_sche = lr_sche
        self.env = env
        self.args = args

        self.policy_net = PolicyNet(state_dim=self.input_dim,
                                    hidden_dim1=hidden_dim1,
                                    hidden_dim2=hidden_dim2,
                                    action_dim=self.output_dim,
                                    args=args).to(self.device)
        self.value_net = ValueNet(state_dim=self.input_dim,
                                  hidden_dim1=hidden_dim1,
                                  hidden_dim2=hidden_dim2,
                                  args=args).to(self.device)

        self.policy_optimizer = torch.optim.Adam(params=self.policy_net.parameters(), lr=self.learning_rate)
        self.value_opitimizer = torch.optim.Adam(params=self.value_net.parameters(), lr=self.learning_rate * 8)

        self.policy_scheduler = torch.optim.lr_scheduler.StepLR(self.policy_optimizer, step_size=500, gamma=0.95)
        self.value_scheduler = torch.optim.lr_scheduler.StepLR(self.value_opitimizer, step_size=500, gamma=0.95)

    def step(self, states, candidates):
        if self.env.type == "KES_junyi" and self.env.feasible_action_flag:
            candidates = list(set(candidates) & set(self.env.feasible_actions))
            if len(candidates) == 0:
                candidates = self.env.feasible_actions

        probs = self.policy_net(states)
        action_dist = torch.distributions.Categorical(probs=probs)

        candidate_probs = probs.gather(1, torch.tensor(candidates).view(1, -1).to(self.device))
        candidate_dist = torch.distributions.Categorical(probs=candidate_probs)
        id = candidate_dist.sample().item()
        action = candidates[id]

        return action

    def learn(self, RL_states, actions, RL_next_states, rewards, dones):
        if self.lr_sche:
            self.value_scheduler.step()
            self.policy_scheduler.step()

        td_target = rewards + self.gamma * self.value_net(RL_next_states) * (1 - dones)
        G = 0
        for i in reversed(range(rewards.shape[0])):
            G = self.gamma * G + rewards[i]
            td_target[i] = G
        advantages = td_target - self.value_net(RL_states)
        advantages = advantages.detach()
        td_target = td_target.detach()
        probs = torch.clamp(self.policy_net(RL_states), min=1e-8, max=1)

        log_probs = torch.log(probs).gather(1, actions)
        actor_loss = torch.mean(-log_probs * advantages.detach())
        critic_loss = torch.mean(F.mse_loss(self.value_net(RL_states), td_target.detach()))
        self.policy_optimizer.zero_grad()
        self.value_opitimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        critic_loss.backward(retain_graph=True)
        self.policy_optimizer.step()
        self.value_opitimizer.step()
