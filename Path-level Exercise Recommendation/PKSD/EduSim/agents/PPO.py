import torch
import numpy as np
import os
import torch.nn as nn
from EduSim.Envs.deep_model import *
import torch.nn.functional as F
from torch.distributions import Normal
from EduSim.Envs.agent_utils import *


class PPO(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim1, hidden_dim2, env, policy_clip=0.1, gae_lambda=0.95, k_epochs=5, mini_batch_size=5,
                 lr_rate=0.0001, gamma=0.98, c2=0.001, device='cuda', lr_sche=False,
                 has_continuous_action_sapce=False, init_log_std=0.0, action_bound=0, args={}):
        super(PPO, self).__init__()
        self.name = 'ppo'
        self.policy_mode = 'on_policy'
        self.pre_learn_node = '-1'
        self.device = device
        self.gamma = gamma
        self.learning_rate = lr_rate
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.action_dim = output_dim
        self.lr_sche = lr_sche
        self.env = env
        self.policy_clip = policy_clip
        self.k_epochs = k_epochs
        self.gae_lambda = gae_lambda
        self.mini_batch_size = mini_batch_size
        self.c2 = c2
        self.args = args

        self.has_continuous_action_sapce = has_continuous_action_sapce
        self.init_log_std = init_log_std
        self.action_bound = action_bound

        self.ep_old_logprobs = []
        self.continuous_normal_sample = torch.tensor([]).to(self.device)

        self.policy_net = PolicyNet(state_dim=self.input_dim,
                                    hidden_dim1=hidden_dim1,
                                    hidden_dim2=hidden_dim2,
                                    action_dim=self.output_dim,
                                    has_continuous_action_sapce=self.has_continuous_action_sapce,
                                    action_bound=self.action_bound,
                                    args=self.args
                                    ).to(self.device)

        self.value_net = ValueNet(state_dim=self.input_dim,
                                  hidden_dim1=hidden_dim1,
                                  hidden_dim2=hidden_dim2,
                                  args=self.args).to(self.device)

        self.MSEloss = nn.MSELoss()

        self.policy_optimizer = torch.optim.Adam(params=self.policy_net.parameters(), lr=self.learning_rate)
        self.value_opitimizer = torch.optim.Adam(params=self.value_net.parameters(), lr=self.learning_rate * 10)

    def step(self, states, candidates):
        if self.env.type == "KES_junyi" and self.env.feasible_action_flag:
            candidates = list(set(candidates) & set(self.env.feasible_actions))
            if len(candidates) == 0:
                candidates = self.env.feasible_actions
        if self.has_continuous_action_sapce:
            dist = self.policy_net(states)

            normal_sample = dist.rsample()
            sample_action_embedding = torch.tanh(normal_sample)
            sample_action_embedding = sample_action_embedding * self.action_bound

            graph_embeddings = torch.tensor(self.env.graph_embeddings[candidates]).to(self.device)
            distance_item = [[torch.dist(sample_action_embedding, embedding, p=2), i] for i, embedding in enumerate(graph_embeddings)]
            distance_item_sorted = sorted(distance_item, key=lambda x: x[0])
            action_id = candidates[distance_item_sorted[0][1]]

            log_prob = dist.log_prob(normal_sample) - torch.log(1 - torch.tanh(normal_sample).pow(2) + 1e-7)
            log_prob = torch.sum(log_prob)
            self.continuous_normal_sample = torch.cat((self.continuous_normal_sample, normal_sample), dim=0)
            self.ep_old_logprobs.append(log_prob)

            action = action_id
        else:
            probs = self.policy_net(states)
            action_dist = torch.distributions.Categorical(probs=probs)

            candidate_probs = probs.gather(1, torch.tensor(candidates).view(1, -1).to(self.device))
            candidate_dist = torch.distributions.Categorical(probs=candidate_probs)
            id = candidate_dist.sample().item()
            action = candidates[id]

            action_logprob = action_dist.log_prob(torch.tensor(action).to(self.device))
            self.ep_old_logprobs.append(action_logprob)
        if len(self.ep_old_logprobs) > 100:
            self.ep_old_logprobs = []

        return action

    def learn(self, RL_states, actions, RL_next_states, rewards, dones, multiepi_sample_ids: list = None, multiepi_done=False):
        if multiepi_sample_ids is not None:
            tmp_epi_old_logprobs = self.ep_old_logprobs[multiepi_sample_ids[0]:multiepi_sample_ids[1]]
        else:
            tmp_epi_old_logprobs = self.ep_old_logprobs

        if self.env.env_name == 'xxx':
            returns = torch.zeros(rewards.shape).to(self.device)
            discounted_return = 0
            for i in reversed(range(rewards.shape[0])):
                if dones[i]:
                    discounted_return = 0
                discounted_return = self.gamma * discounted_return + rewards[i]
                returns[i] = discounted_return
            advantages = returns - self.value_net(RL_states).detach()
        else:
            with torch.no_grad():
                values = self.value_net(RL_states).detach()
            advantages, returns = self.estimate_advantages(rewards, dones, values, self.gamma, self.gae_lambda)
        advantages = advantages.detach()
        returns = returns.detach()
        ep_length = len(rewards)
        batch_starts = np.arange(0, ep_length, self.mini_batch_size)
        indicies = np.arange(ep_length, dtype=np.int64)
        mini_batch_indicies = [indicies[i:i + self.mini_batch_size] for i in batch_starts]

        for _ in range(self.k_epochs):
            for batch_indicies in mini_batch_indicies:
                batch_states = RL_states[batch_indicies]
                batch_old_logprobs = torch.tensor(tmp_epi_old_logprobs)[batch_indicies].detach().to(self.device)
                batch_actions = actions[batch_indicies]

                batch_state_values = self.value_net(batch_states)
                if self.has_continuous_action_sapce:
                    batch_dist = self.policy_net(batch_states)


                    batch_normal_samples = self.continuous_normal_sample[batch_indicies].detach().to(self.device)
                    batch_new_logprobs = batch_dist.log_prob(batch_normal_samples) - torch.log(1 - torch.tanh(batch_normal_samples).pow(2) + 1e-7)
                    batch_new_logprobs = torch.sum(batch_new_logprobs, dim=1)
                else:
                    batch_dist = torch.distributions.Categorical(self.policy_net(batch_states))
                    batch_new_logprobs = batch_dist.log_prob(torch.squeeze(batch_actions))

                ratios = torch.exp(batch_new_logprobs - batch_old_logprobs.detach())
                batch_advantages = torch.squeeze(advantages[batch_indicies])
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * batch_advantages

                actor_loss = -torch.min(surr1, surr2).mean()

                entropy_loss = self.c2 * batch_dist.entropy().mean()
                loss = actor_loss + 0.5 * self.MSEloss(batch_state_values, returns[batch_indicies]) - entropy_loss
                self.policy_optimizer.zero_grad()
                self.value_opitimizer.zero_grad()
                loss.backward(retain_graph=True)
                for name, param in self.policy_net.named_parameters():
                    if 'weight' in name and param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print('policy gradient nan')
                for name, param in self.value_net.named_parameters():
                    if 'weight' in name and param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print('value gradient nan')
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=40)
                self.policy_optimizer.step()
                self.value_opitimizer.step()
        if (multiepi_sample_ids is not None and multiepi_done) or multiepi_sample_ids is None:
            self.ep_old_logprobs = []

        self.continuous_normal_sample = torch.tensor([]).to(self.device)

    def estimate_advantages(self, rewards, dones, values, gamma, lam):
        tensor_type = type(rewards)
        deltas = tensor_type(rewards.size(0), 1).to(self.device)
        advantages = tensor_type(rewards.size(0), 1).to(self.device)

        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(rewards.size(0))):
            deltas[i] = rewards[i] + gamma * prev_value * (1.0 - dones[i]) - values[i]
            advantages[i] = deltas[i] + gamma * lam * prev_advantage * (1.0 - dones[i])

            prev_value = values[i, 0]
            prev_advantage = advantages[i, 0]

        returns = values + advantages

        return advantages, returns
