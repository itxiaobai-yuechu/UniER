import torch
import numpy as np
import os
import torch.nn as nn
from EduSim.Envs.shared.KSS_KES.KS import influence_control
from EduSim.Envs.buffer import *
from EduSim.Envs.deep_model import *
from .GoalNav import *


class CSEAL(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim1, hidden_dim2, baseline_settings, env, lr_rate=0.0001, gamma=0.98, device='cuda:0',
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
        self.lr_sche = lr_sche
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

        self.policy_optimizer = torch.optim.Adam(params=self.policy_net.parameters(), lr=self.learning_rate)
        self.value_opitimizer = torch.optim.Adam(params=self.value_net.parameters(), lr=self.learning_rate * 8)

        self.policy_scheduler = torch.optim.lr_scheduler.StepLR(self.policy_optimizer, step_size=500, gamma=0.95)
        self.value_scheduler = torch.optim.lr_scheduler.StepLR(self.value_opitimizer, step_size=500, gamma=0.95)


    def step(self, states, _):
        preds = self.policy_net(states)

        if not self.baseline_settings['CSEAL_NCN_baseline']:
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
                if self.env.type == 'xxx':
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

            CN_candidates_probs = preds.gather(1, torch.tensor(candidates).unsqueeze(0).to(self.device))

            if self.baseline_settings['Cog_baseline']:
                weight = torch.tensor([1] * self.output_dim).to(self.device) - torch.sigmoid(states[-1, :self.output_dim])
                CN_candidates_probs = weight.gather(0, torch.tensor(candidates).to(self.device)).to(self.device)
                action_dist = torch.distributions.Categorical(CN_candidates_probs)
                id = action_dist.sample().item()
                action = candidates[id]
                item = action

            elif self.baseline_settings['CN_random']:
                item = candidates[random.randint(0, len(candidates) - 1)]

            else:
                if np.isnan(CN_candidates_probs[0][0].cpu().detach().numpy()):
                    item = candidates[random.randint(0, len(candidates) - 1)]
                    print()
                    print('Nan accur')
                else:
                    action_dist = torch.distributions.Categorical(CN_candidates_probs)
                    id = action_dist.sample().item()
                    action = candidates[id]
                    item = action

        else:
            action_dist = torch.distributions.Categorical(probs=preds)
            item = action_dist.sample().item()

        if random.randint(0, 500) < 1:
            print('pred:' + str(preds))
        return item

    def learn(self, RL_states, actions, RL_next_states, rewards, dones):
        if self.baseline_settings['CN_random']:
            return
        if self.lr_sche:
            self.policy_scheduler.step()
            self.value_scheduler.step()

        td_target = rewards + self.gamma * self.value_net(RL_next_states) * (1 - dones)
        G = 0
        for i in reversed(range(rewards.shape[0])):
            G = self.gamma * G + rewards[i]
            td_target[i] = G
        td_delta = td_target - self.value_net(RL_states)

        probs = torch.clamp(self.policy_net(RL_states), min=1e-8, max=1)

        log_probs = torch.log(probs).gather(1, actions)
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        critic_loss = torch.mean(F.mse_loss(self.value_net(RL_states), td_target.detach()))
        self.policy_optimizer.zero_grad()
        self.value_opitimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        critic_loss.backward()
        self.policy_optimizer.step()
        self.value_opitimizer.step()
