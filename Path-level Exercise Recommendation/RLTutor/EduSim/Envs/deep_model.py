import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal

class RnnEncoder(nn.Module):
    def __init__(self, input_size, emb_dim, hidden_size, num_skills, nlayers, dropout=0, out_activation='Tanh'):
        super().__init__()
        self.name = 'normal_rnn_encoder'
        self.nhid = hidden_size
        self.nlayers = nlayers
        self.dropout = dropout

        self.embedding_layer = nn.Linear(input_size, emb_dim, bias=False)
        self.rnn = nn.GRU(input_size, hidden_size, nlayers)
        self.fc_out = nn.Linear(hidden_size, num_skills)
        self.dropout = nn.Dropout(p=self.dropout)
        self.out_activation = out_activation

    def forward(self, x):
        x = x.permute(1, 0, 2)
        h_0, c_0 = self.init_hidden_state(x.size(1))
        h_0, c_0 = h_0.to(x.device), c_0.to(x.device)

        self.rnn.flatten_parameters()

        output, _ = self.rnn(x, h_0)
        out = self.fc_out(output)
        if self.out_activation == 'Tanh':
            out = torch.tanh(out)
        elif self.out_activation == 'Sigmoid':
            out = torch.sigmoid(out)
        return out

    def init_hidden_state(self, batch_size):
        h_0 = torch.rand(self.nlayers, batch_size, self.nhid)
        c_0 = torch.rand(self.nlayers, batch_size, self.nhid)
        return h_0, c_0


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim1, hidden_dim2, action_dim, has_continuous_action_sapce=False, init_log_std=0.0, action_bound=1.0, args={}):
        super(PolicyNet, self).__init__()
        if args.get('wid', False):
            hidden_dim2 = 2 * hidden_dim2
            hidden_dim1 = 2 * hidden_dim1
        self.has_continuous_action_sapce = has_continuous_action_sapce
        self.action_bound = action_bound

        if has_continuous_action_sapce:
            self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * init_log_std)
            self.encode_net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim1),
                nn.ReLU(),
                nn.Linear(hidden_dim1, hidden_dim2),
                nn.ReLU()
            )
            self.fc_mu = nn.Linear(hidden_dim2, action_dim)
            self.fc_std = nn.Linear(hidden_dim2, action_dim)
        else:
            if args.get('wid', False):
                self.actor_net = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim1),
                    nn.ReLU(),
                    nn.Linear(hidden_dim1, hidden_dim2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim2, hidden_dim2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim2, hidden_dim2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim2, action_dim),
                    nn.Softmax(dim=1)
                )
            else:
                self.actor_net = nn.Sequential(
                    nn.Linear(state_dim, hidden_dim1),
                    nn.ReLU(),
                    nn.Linear(hidden_dim1, hidden_dim2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim2, action_dim),
                    nn.Softmax(dim=1)
                )

    def forward(self, state):
        if self.has_continuous_action_sapce:
            state = self.encode_net(state)
            mu = self.fc_mu(state)
            std = F.softplus(self.fc_std(state))
            dist = Normal(mu, std)
        else:
            dist = self.actor_net(state)
        return dist


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim1, hidden_dim2, args={}):
        super(ValueNet, self).__init__()
        if args.get('wid', False):
            hidden_dim2 = 2 * hidden_dim2
            hidden_dim1 = 2 * hidden_dim1

            self.value_net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim1),
                nn.ReLU(),
                nn.Linear(hidden_dim1, hidden_dim2),
                nn.ReLU(),
                nn.Linear(hidden_dim2, hidden_dim2),
                nn.ReLU(),
                nn.Linear(hidden_dim2, hidden_dim2),
                nn.ReLU(),
                nn.Linear(hidden_dim2, 1),
            )
        else:
            self.value_net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim1),
                nn.ReLU(),
                nn.Linear(hidden_dim1, hidden_dim2),
                nn.ReLU(),
                nn.Linear(hidden_dim2, 1),
            )

    def forward(self, state):
        value = self.value_net(state)
        return value


class DKTnet(nn.Module):
    def __init__(self,
                 input_size,
                 emb_dim,
                 hidden_size,
                 num_skills,
                 nlayers,
                 dropout=0):
        super().__init__()
        self.name = 'DKT'
        self.nhid = hidden_size
        self.nlayers = nlayers
        self.dropout = dropout

        self.embedding_layer = nn.Linear(input_size, emb_dim, bias=False)
        self.rnn = nn.LSTM(emb_dim, hidden_size, nlayers)
        self.fc_out = nn.Linear(hidden_size, num_skills)

        self.dropout = nn.Dropout(p=self.dropout)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        h_0, c_0 = self.init_hidden_state(x.size(1))
        h_0, c_0 = h_0.to(x.device), c_0.to(x.device)

        self.rnn.flatten_parameters()

        embed = self.embedding_layer(x)
        output, _ = self.rnn(embed, (h_0, c_0))
        out = self.fc_out(output)
        out = self.dropout(out)
        return out

    def init_hidden_state(self, batch_size):
        h_0 = torch.rand(self.nlayers, batch_size, self.nhid)
        c_0 = torch.rand(self.nlayers, batch_size, self.nhid)
        return h_0, c_0