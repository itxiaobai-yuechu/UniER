import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torch.distributions import Normal
import numpy as np
from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv


class MLPNet(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim1,
                 hidden_dim2,
                 output_dim):
        super(MLPNet, self).__init__()
        self.name = 'MLP'
        self.MLP = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.Tanh(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.Tanh(),
            nn.Linear(hidden_dim2, output_dim)
        )

    def forward(self, input):
        out = self.MLP(input)
        return torch.sigmoid(out)


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


class Qnet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_skills, dueling_dqn, noisy_dqn, args={}):
        super(Qnet, self).__init__()
        if args['wid']:
            hidden_size1 = hidden_size1 * 2
            hidden_size2 = hidden_size2 * 2
        self.dueling_dqn = dueling_dqn
        self.noisy_dqn = noisy_dqn

        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU()
        )
        if noisy_dqn:
            self.input_layer = nn.Sequential(
                NoisyLinear(input_size, hidden_size1),
                nn.ReLU()
            )

        self.mid_layers = nn.ModuleList([nn.Linear(hidden_size1, hidden_size2), nn.ReLU()])
        self.mid_layer_num = 0
        if args['wid']:
            self.mid_layer_num = 2
        for i in range(self.mid_layer_num):
            self.mid_layers.append(nn.Linear(hidden_size2, hidden_size2))
            self.mid_layers.append(nn.ReLU())
        if noisy_dqn:
            self.mid_layers = nn.ModuleList([NoisyLinear(hidden_size1, hidden_size2), nn.ReLU()])

        self.out_layer = nn.Linear(hidden_size2, num_skills)
        self.out_layer_A = nn.Linear(hidden_size2, num_skills)
        self.out_layer_V = nn.Linear(hidden_size2, 1)
        if noisy_dqn:
            self.out_layer = NoisyLinear(hidden_size2, num_skills)
            self.out_layer_A = NoisyLinear(hidden_size2, num_skills)
            self.out_layer_V = NoisyLinear(hidden_size2, 1)

    def forward(self, x):
        x = self.input_layer(x)
        for mid_layer in self.mid_layers:
            x = mid_layer(x)
        if self.dueling_dqn:
            A = self.out_layer_A(x)
            V = self.out_layer_V(x)
            Q = V + A - A.mean(1).view(-1, 1)
            return Q
        else:
            out = self.out_layer(x)
            return out


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim1, hidden_dim2, action_dim, has_continuous_action_sapce=False, init_log_std=0.0, action_bound=1.0, args={}):
        super(PolicyNet, self).__init__()
        if args['wid']:
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
            if args['wid']:
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
        if args['wid']:
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


class DKTnetWithEmb(nn.Module):


    def __init__(self, input_size, emb_dim, hidden_size, num_skills, nlayers, dropout=0):
        super(DKTnetWithEmb).__init__()
        self.name = 'DKT'
        self.nhid = hidden_size
        self.nlayers = nlayers
        self.dropout = dropout
        self.embedding_layer = nn.Embedding(num_embeddings=input_size, embedding_dim=emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim, hidden_size, nlayers)
        self.fc_out = nn.Linear(hidden_size, num_skills)
        self.dropout = nn.Dropout(p=self.dropout)

    def forward(self, x):
        h_0, c_0 = self.init_hidden_state(x.size(0))
        h_0, c_0 = h_0.to(x.device), c_0.to(x.device)

        self.rnn.flatten_parameters()

        embed = self.embedding_layer(x)
        embed = embed.permute(1, 0, 2)
        output, _ = self.rnn(embed, (h_0, c_0))
        out = self.fc_out(output)
        out = self.dropout(out)
        return out

    def init_hidden_state(self, batch_size):
        h_0 = torch.rand(self.nlayers, batch_size, self.nhid)
        c_0 = torch.rand(self.nlayers, batch_size, self.nhid)
        return h_0, c_0


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


class DisQnet(nn.Module):
    def __init__(self, input_size, hidden_size, num_skills, atom_size, support: torch.Tensor):

        super(DisQnet, self).__init__()
        self.fc1 = nn.Linear(input_size, num_skills * atom_size)
        self.fc2 = nn.Linear(num_skills * atom_size, num_skills * atom_size)


        self.nhid = hidden_size

        self.support = support
        self.out_dim = num_skills
        self.atom_size = atom_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        dist = self.dist(input)
        q = torch.sum(dist * self.support, dim=2)
        return q

    def dist(self, x: torch.Tensor) -> torch.Tensor:


        out = self.fc1(x)
        q_atoms = out.view(-1, self.out_dim, self.atom_size)
        dist = F.softmax(q_atoms, dim=-1)

        return dist


class NoisyLinear(nn.Module):


    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):

        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):

        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):

        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:

        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())




class Regularization(torch.nn.Module):
    def __init__(self, model, weight_decay, p=2):

        super(Regularization, self).__init__()
        if weight_decay < 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)
        self.weight_info(self.weight_list)

    def to(self, device):

        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):

        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):

        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):

        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)
        print("---------------------------------------------------")






class GCNNet(nn.Module):
    def __init__(self, feat_dim, num_class, num_node=None, GCN_hidden_dim=128):
        super(GCNNet, self).__init__()
        hidden_dim = GCN_hidden_dim
        self.conv1 = GCNConv(feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_class)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x


class GATNet(torch.nn.Module):
    def __init__(self, in_channel, out_channel, node_num=None):
        super(GATNet, self).__init__()
        self.gat1 = GATConv(in_channel, 16, 8)
        self.gat2 = GATConv(128, out_channel, 1)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = self.gat2(x, edge_index)
        return x
