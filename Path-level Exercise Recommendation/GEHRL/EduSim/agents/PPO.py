import random
import numpy as np
import torch
from torch import nn
from EduSim.deep_model import PolicyNetWithOutterEncoder, ValueNetWithOutterEncoder
from EduSim.utils import mean_entropy_cal


class PPO(nn.Module):
    def __init__(self, ppo_para_dict):
        super(PPO, self).__init__()
        input_dim = ppo_para_dict['input_dim']
        output_dim = ppo_para_dict['output_dim']
        hidden_dim1 = ppo_para_dict['hidden_dim1']
        hidden_dim2 = ppo_para_dict['hidden_dim2']
        env = ppo_para_dict['env']
        policy_clip = ppo_para_dict['policy_clip']
        gae_lambda = ppo_para_dict['gae_lambda']
        k_epochs = ppo_para_dict['k_epochs']
        mini_batch_size = ppo_para_dict['mini_batch_size']
        lr_rate = ppo_para_dict['lr_rate']
        gamma = ppo_para_dict['gamma']
        c2 = ppo_para_dict['c2']
        args = ppo_para_dict['args']
        outer_encoder = ppo_para_dict['outer_encoder']
        RLInputCompo = ppo_para_dict['RLInputCompo']

        self.name = 'ppo'
        self.policy_mode = 'on_policy'
        self.pre_learn_node = '-1'
        self.gamma = gamma
        self.learning_rate = lr_rate
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.action_dim = output_dim
        self.env = env
        self.policy_clip = policy_clip
        self.k_epochs = k_epochs
        self.gae_lambda = gae_lambda
        self.mini_batch_size = mini_batch_size
        self.c2 = c2
        self.args = args

        self.ep_old_logprobs = []
        self.continuous_normal_sample = torch.tensor([])

        if outer_encoder is not None:
            para_dict = {
                'state_dim': self.input_dim,
                'hidden_dim1': hidden_dim1,
                'hidden_dim2': hidden_dim2,
                'action_dim': self.output_dim,
                'outer_encoder': outer_encoder,
                'RLInputCompo': RLInputCompo
            }
            self.policy_net = PolicyNetWithOutterEncoder(para_dict)
            self.value_net = ValueNetWithOutterEncoder(para_dict)
        else:
            raise ValueError('outer encoder missed!')
        self.MSEloss = nn.MSELoss()

        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.learning_rate * 8)

    def step(self, state_input_dict, candidates):
        probs = self.policy_net(state_input_dict)
        candidate_probs = probs.gather(dim=1, index=torch.tensor(candidates, dtype=torch.int64).view(1, -1))


        action = random.choices(candidates, weights=candidate_probs.view(-1).detach().cpu().numpy(), k=1)[0]
        action_logprob = torch.log(probs[0][action])
        self.ep_old_logprobs.append(action_logprob.detach().cpu().numpy().item())

        return action

    def forward_fn_policy(self, ffn_input_dict):
        batch_states_dict = ffn_input_dict['batch_states_dict']
        batch_actions = ffn_input_dict['batch_actions']
        batch_old_logprobs = ffn_input_dict['batch_old_logprobs']
        advantages = ffn_input_dict['advantages']
        batch_indices = ffn_input_dict['batch_indices']

        probs = self.policy_net(batch_states_dict)
        batch_new_logprobs = torch.log(probs.gather(dim=1, index=batch_actions.to(torch.int64))).squeeze(1)

        ratios = torch.exp(batch_new_logprobs - batch_old_logprobs)
        batch_advantages = torch.squeeze(advantages[batch_indices])
        surr1 = ratios * batch_advantages
        surr2 = torch.clamp(ratios, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * batch_advantages

        a = torch.cat((surr1.view(1, -1), surr2.view(1, -1)), dim=0)
        a, _ = torch.min(a, dim=0)
        actor_loss = -a.mean()

        entropy_loss = -self.c2 * mean_entropy_cal(probs)
        return actor_loss, entropy_loss

    def forward_fn_value(self, batch_states_dict, returns, batch_indices):
        batch_state_values = self.value_net(batch_states_dict)
        value_loss = 0.5 * self.MSEloss(batch_state_values, returns[batch_indices])
        return value_loss

    def learn(self, ac_learn_dict, multiepi_sample_ids: list = None, multiepi_done=False, forOutterRNN=False):
        RL_states_dict = ac_learn_dict['RL_states_dict']
        actions = ac_learn_dict['actions']
        rewards = ac_learn_dict['rewards']
        dones = ac_learn_dict['dones']

        if multiepi_sample_ids is not None:
            tmp_epi_old_logprobs = self.ep_old_logprobs[multiepi_sample_ids[0]:multiepi_sample_ids[1]]
        else:
            tmp_epi_old_logprobs = self.ep_old_logprobs

        values = self.value_net(RL_states_dict).detach()
        ad_input_dict = {
            'rewards': rewards,
            'dones': dones,
            'values': values,
            'gamma': self.gamma,
            'lam': self.gae_lambda
        }
        advantages, returns = self.estimate_advantages(ad_input_dict=ad_input_dict)

        ep_length = len(rewards)
        batch_starts = np.arange(0, ep_length, self.mini_batch_size)
        indices = np.arange(ep_length, dtype=np.int64)
        mini_batch_indices = [list(indices[i:i + self.mini_batch_size]) for i in batch_starts]

        loss = torch.tensor([0.0])
        for _ in range(self.k_epochs):
            for batch_indices in mini_batch_indices:
                batch_indices = torch.tensor(batch_indices)
                batch_states_dict = {
                    'states': RL_states_dict['states'][batch_indices],
                    'states_lengths_ids': RL_states_dict['states_lengths_ids'][batch_indices],
                    'targets': [RL_states_dict['targets'][i] for i in batch_indices]
                }
                if 'RNN_concate_vector' in RL_states_dict.keys():
                    batch_states_dict['RNN_concate_vector'] = RL_states_dict['RNN_concate_vector'][batch_indices]
                batch_old_logprobs = torch.tensor(tmp_epi_old_logprobs)[batch_indices]
                batch_actions = actions[batch_indices]

                self.value_optimizer.zero_grad()
                value_loss = self.forward_fn_value(batch_states_dict, returns, batch_indices)
                value_loss.backward()
                self.value_optimizer.step()

                self.policy_optimizer.zero_grad()
                ffn_para_dict = {
                    'batch_states_dict': batch_states_dict,
                    'batch_actions': batch_actions,
                    'batch_old_logprobs': batch_old_logprobs,
                    'advantages': advantages,
                    'batch_indices': batch_indices
                }
                actor_loss, entropy_loss = self.forward_fn_policy(ffn_para_dict)
                policy_loss = actor_loss + entropy_loss
                policy_loss.backward()
                self.policy_optimizer.step()

        if ((multiepi_sample_ids is not None and multiepi_done) or multiepi_sample_ids is None) and forOutterRNN:
            self.ep_old_logprobs = []
        return loss

    def estimate_advantages(self, ad_input_dict):
        rewards = ad_input_dict['rewards']
        dones = ad_input_dict['dones']
        values = ad_input_dict['values']
        gamma = ad_input_dict['gamma']
        lam = ad_input_dict['lam']

        deltas = torch.zeros((rewards.shape[0], 1))
        advantages = torch.zeros((rewards.shape[0], 1))

        prev_value = 0.0
        prev_advantage = 0.0
        for i in reversed(range(rewards.shape[0])):
            deltas[i] = rewards[i] + gamma * prev_value * (1.0 - dones[i]) - values[i]
            advantages[i] = deltas[i] + gamma * lam * prev_advantage * (1.0 - dones[i])

            prev_value = values[i, 0]
            prev_advantage = advantages[i, 0]

        returns = values + advantages

        return advantages, returns
