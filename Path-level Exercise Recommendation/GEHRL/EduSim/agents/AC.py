import random
import torch
from torch import nn
from EduSim.deep_model import PolicyNetWithOutterEncoder, ValueNetWithOutterEncoder


class ActorCritic(nn.Module):
    def __init__(self, ac_para_dict):
        super(ActorCritic, self).__init__()
        input_dim = ac_para_dict['input_dim']
        output_dim = ac_para_dict['output_dim']
        hidden_dim1 = ac_para_dict['hidden_dim1']
        hidden_dim2 = ac_para_dict['hidden_dim2']
        env = ac_para_dict['env']
        lr_rate = ac_para_dict['lr_rate']
        gamma = ac_para_dict['gamma']
        lr_sche = ac_para_dict['lr_sche']
        args = ac_para_dict['args']
        outer_encoder = ac_para_dict['outer_encoder']
        RLInputCompo = ac_para_dict['RLInputCompo']

        self.name = 'ac'
        self.policy_mode = 'on_policy'
        self.pre_learn_node = '0'
        self.gamma = gamma
        self.learning_rate = lr_rate
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr_sche = lr_sche
        self.env = env
        self.args = args

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

        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.learning_rate * 8)

    def step(self, state_input_dict, candidates):
        probs = self.policy_net(state_input_dict)

        candidate_probs = probs.gather(dim=1, index=torch.tensor(candidates, dtype=torch.int64).view(1, -1))

        action = random.choices(candidates, weights=candidate_probs.view(-1).detach().cpu().numpy(), k=1)[0]

        return action

    def forward_fn_policy(self, advantages, RL_states_dict, actions):
        probs = self.policy_net(RL_states_dict)
        log_probs = probs.gather(dim=1, index=actions.to(torch.int64)).view(-1, 1).log()
        actor_loss = torch.mean(-log_probs * advantages)
        return actor_loss

    def forward_fn_value(self, td_target, RL_states_dict):
        output = self.value_net(RL_states_dict)
        critic_loss = torch.nn.functional.mse_loss(output, td_target)
        return critic_loss

    def learn(self, ac_learn_dict):
        RL_states_dict = ac_learn_dict['RL_states_dict']
        actions = ac_learn_dict['actions']
        RL_next_states_dict = ac_learn_dict['RL_next_states_dict']
        rewards = ac_learn_dict['rewards']
        dones = ac_learn_dict['dones']

        td_target = rewards + self.gamma * self.value_net(RL_next_states_dict) * (1 - dones)
        G = 0
        for i in reversed(range(rewards.shape[0])):
            G = self.gamma * G + rewards[i]
            td_target[i] = G
        advantages = td_target - self.value_net(RL_states_dict)

        total_loss = torch.tensor([0.0])
        
        self.value_optimizer.zero_grad()
        critic_loss = self.forward_fn_value(td_target.detach(), RL_states_dict)
        critic_loss.backward()
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        actor_loss = self.forward_fn_policy(advantages.detach(), RL_states_dict, actions)
        actor_loss.backward()
        self.policy_optimizer.step()
        
        return total_loss
