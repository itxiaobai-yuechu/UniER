import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
from EduSim.Envs.shared.KSS_KES.KS import influence_control
from EduSim.Envs.buffer import *
from EduSim.Envs.deep_model import *
from .GoalNav import *

class CSEAL(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim1, hidden_dim2, baseline_settings, env, lr_rate=0.0001, gamma=0.98, device='cuda:1',
                 lr_sche = False, args={}):
        super(CSEAL, self).__init__()
        self.name = 'CSEAL'
        self.policy_mode = 'on_policy'
        self.pre_learn_node = '-1'
        self.device = device
        self.gamma = gamma
        self.learning_rate = lr_rate
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.baseline_settings = baseline_settings
        self.env = env
        self.args=args

        self.policy_net = PolicyNet(state_dim=self.input_dim,
                                    hidden_dim1=hidden_dim1,
                                    hidden_dim2=hidden_dim2,
                                    action_dim=self.output_dim,
                                    args=self.args).to(self.device)
        self.value_net = ValueNet(state_dim=self.input_dim,
                                  hidden_dim1=hidden_dim1,
                                  hidden_dim2=hidden_dim2,
                                  args=args).to(self.device)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.policy_optimizer = torch.optim.Adam(params=self.policy_net.parameters(), lr=self.learning_rate, eps=1e-8)
        self.value_optimizer = torch.optim.Adam(params=self.value_net.parameters(), lr=self.learning_rate, eps=1e-8)

        self.policy_scheduler = None
        self.value_scheduler = None

        self.batch_size = 16

    def step(self, states, _):
        preds = self.policy_net(states)

        if not self.baseline_settings['CSEAL_NCN_baseline']:
            goal_candidates = get_goal_neighbors(self.env.knowledge_structure, self.env._learner.target)

            if self.pre_learn_node != '-1':
                learning_item = self.env.learning_item_base[self.pre_learn_node]
                candidates = influence_control(
                    self.env.knowledge_structure,
                    self.env._learner._state,
                    learning_item.knowledge,
                    allow_shortcut=False,
                    target=self.env._learner.target
                )[0]
            else:
                if hasattr(self.env,'type') and self.env.type == 'xxx':
                    self.pre_learn_node = self.env._learner._logs[-1][0]
                    learning_item = self.env.learning_item_base[self.pre_learn_node]
                    candidates = influence_control(
                        self.env.knowledge_structure,
                        self.env._learner._state,
                        learning_item.knowledge,
                        allow_shortcut=False,
                        target=self.env._learner.target
                    )[0]
                else:
                    candidates = [i for i in range(self.output_dim)]

            candidates = list(set(candidates) & set(goal_candidates))
            if not candidates:
                candidates = list(range(self.output_dim))

            CN_candidates_probs = preds.gather(1, torch.tensor(candidates).unsqueeze(0).to(self.device))

            if self.baseline_settings['Cog_baseline']:
                knowledge_prof = states[:, :self.output_dim]
                master_rate = torch.sigmoid(knowledge_prof)
                weight = 1.0 - master_rate
                weight = weight.gather(1, torch.tensor(candidates).unsqueeze(0).to(self.device))
                weight = torch.softmax(weight, dim=-1)
                action_dist = torch.distributions.Categorical(weight)
                id = action_dist.sample().item()
                action = candidates[id]
                item = action

            elif self.baseline_settings['CN_random']:
                item = random.choice(candidates)

            else:
                if torch.isnan(CN_candidates_probs).any():
                    item = random.choice(candidates)
                else:
                    action_dist = torch.distributions.Categorical(CN_candidates_probs)
                    id = action_dist.sample().item()
                    action = candidates[id]
                    item = action

        else:
            action_dist = torch.distributions.Categorical(probs=preds)
            item = action_dist.sample().item()

        return item

    def learn(self, RL_states, actions, RL_next_states, rewards, dones):
        if self.baseline_settings['CN_random']:
            return

        G_list = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            G_list.append(G)
        G_list.reverse()
        td_target = torch.tensor(G_list, dtype=torch.float32).to(self.device).unsqueeze(1)

        td_delta = td_target - self.value_net(RL_states).detach()

        probs = torch.clamp(self.policy_net(RL_states), min=1e-8, max=1)
        log_probs = torch.log(probs).gather(1, actions)

        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1, keepdim=True)
        actor_loss = torch.mean(-log_probs * td_delta.detach() - 0.01 * entropy)
        critic_loss = torch.mean(F.mse_loss(self.value_net(RL_states), td_target.detach()))

        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        total_loss = actor_loss + critic_loss
        total_loss.backward()

        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=40)
        nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=40)

        self.policy_optimizer.step()
        self.value_optimizer.step()