import random
import numpy as np
import torch
import torch.nn.functional as F
class DQN(torch.nn.Module):
    def __init__(self, n_input, n_output, args):
        super(DQN, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features=n_input, out_features=args.hidden_size),
            torch.nn.Linear(in_features=args.hidden_size, out_features=n_output),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return (self.net(x))

    def act(self, args, obs):
        device = next(self.parameters()).device
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        q_value = self(obs_tensor)
        v = torch.cat(((1 - q_value).unsqueeze(2), q_value.unsqueeze(2)), dim=2)
        max_values = v.max(dim=2, keepdim=True)[0].squeeze()
        max_q_idx = torch.argmax(input=v, dim=2)
        action = (v[:,:,1] > 0.5) * torch.ones_like(max_values)
        return action, max_values, max_q_idx, v[:,:, 1]


class MFNet(torch.nn.Module):
    def __init__(self, args):
        super(MFNet, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=args.cpt_num,
                            out_features=args.hidden_size),
            torch.nn.Linear(in_features=args.hidden_size,
                            out_features=args.cpt_num),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


class RankNet(torch.nn.Module):
    def __init__(self, n_input, n_output):
        super(RankNet, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=n_input,
                            out_features=int(n_input / 2)),
            torch.nn.Linear(in_features=int(n_input / 2),
                            out_features=n_output),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)
