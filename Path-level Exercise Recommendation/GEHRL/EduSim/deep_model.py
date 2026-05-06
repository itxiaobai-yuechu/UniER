import torch
from torch import nn
from EduSim.utils import batch_cat_targets



class RnnEncoder(nn.Module):
    def __init__(self, rnn_encoder_para_dict):
        super().__init__()
        self.name = 'normal_rnn_encoder'
        hidden_size = rnn_encoder_para_dict['hidden_size']
        input_size = rnn_encoder_para_dict['input_size']
        emb_dim = rnn_encoder_para_dict['emb_dim']
        num_skills = rnn_encoder_para_dict['num_skills']
        nlayers = rnn_encoder_para_dict['nlayers']
        dropout = rnn_encoder_para_dict['dropout']
        out_activation = rnn_encoder_para_dict['out_activation']

        self.nhid = hidden_size
        self.nlayers = nlayers
        self.dropout = dropout

        self.embedding_layer = nn.Linear(input_size, emb_dim)
        self.rnn = nn.GRU(input_size, hidden_size, nlayers)
        self.fc_out = nn.Linear(hidden_size, num_skills)
        self.dropout = nn.Dropout(p=self.dropout)
        self.out_activation = out_activation

    def forward(self, x):
        x = x.permute(1, 0, 2)
        h_0, _ = self.init_hidden_state(x.shape[1])

        output, _ = self.rnn(x, h_0)
        out = self.fc_out(output)
        if self.out_activation == 'Tanh':
            out = torch.tanh(out)
        elif self.out_activation == 'Sigmoid':
            out = torch.sigmoid(out)
        return out

    def init_hidden_state(self, batch_size):
        h_0 = torch.rand((self.nlayers, batch_size, self.nhid))
        c_0 = torch.rand((self.nlayers, batch_size, self.nhid))
        return h_0, c_0


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim1, hidden_dim2, action_dim):
        super().__init__()
        self.actor_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, action_dim),
        )

        self.fc1 = nn.Linear(state_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc_out = nn.Linear(hidden_dim2, action_dim)

    def forward(self, state):
        dist = self.actor_net(state)
        dist = torch.softmax(dist, dim=1)
        return dist


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim1, hidden_dim2):
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1),
        )

        self.fc1 = nn.Linear(state_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc_out = nn.Linear(hidden_dim2, 1)

    def forward(self, state):
        value = self.value_net(state)
        return value


class PolicyNetWithOutterEncoder(nn.Module):
    def __init__(self, para_dict):
        super().__init__()
        state_dim = para_dict['state_dim']
        hidden_dim1 = para_dict['hidden_dim1']
        hidden_dim2 = para_dict['hidden_dim2']
        action_dim = para_dict['action_dim']
        outer_encoder = para_dict['outer_encoder']
        RLInputCompo = para_dict['RLInputCompo']

        self.outer_encoder = outer_encoder
        self.actor_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, action_dim),
        )
        self.action_dim = action_dim
        self.RLInputCompo = RLInputCompo

    def forward(self, state_input_dict):
        states = state_input_dict['states']
        states_lengths_ids = state_input_dict['states_lengths_ids']
        targets = state_input_dict['targets']

        encoded_states = self.outer_encoder(states)

        sequence_length = states_lengths_ids.shape[0]
        encoded_states = (encoded_states.gather(0, states_lengths_ids.to(torch.int64).view(1, -1, 1).
                                                         broadcast_to((1, sequence_length, encoded_states.shape[2]))).
                          squeeze(0))
        if self.RLInputCompo == 'RNNwithTarget':
            RL_states = batch_cat_targets(encoded_states, targets, self.action_dim)
        elif self.RLInputCompo == 'RNNSubgoalTarget':
            RNN_concate_vector = state_input_dict['RNN_concate_vector']
            RL_states = torch.cat((encoded_states, RNN_concate_vector), -1)
            RL_states = batch_cat_targets(RL_states, targets, self.action_dim)
        else:
            raise ValueError('wrong setting of RLInputCompo')


        dist = self.actor_net(RL_states)
        dist = torch.softmax(dist, dim=1)
        return dist


class ValueNetWithOutterEncoder(nn.Module):
    def __init__(self, para_dict):
        super().__init__()
        state_dim = para_dict['state_dim']
        hidden_dim1 = para_dict['hidden_dim1']
        hidden_dim2 = para_dict['hidden_dim2']
        action_dim = para_dict['action_dim']
        outer_encoder = para_dict['outer_encoder']
        RLInputCompo = para_dict['RLInputCompo']

        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1),
        )
        self.outer_encoder = outer_encoder
        self.RLInputCompo = RLInputCompo
        self.action_dim = action_dim

    def forward(self, state_input_dict):
        states = state_input_dict['states']
        states_lengths_ids = state_input_dict['states_lengths_ids']
        targets = state_input_dict['targets']

        encoded_states = self.outer_encoder(states)

        sequence_length = states_lengths_ids.shape[0]
        encoded_states = encoded_states.gather(0, states_lengths_ids.to(torch.int64).view(1, -1, 1).broadcast_to(
            (1, sequence_length, encoded_states.shape[2]))).squeeze(0)
        if self.RLInputCompo == 'RNNwithTarget':
            RL_states = batch_cat_targets(encoded_states, targets, self.action_dim)
        elif self.RLInputCompo == 'RNNSubgoalTarget':
            RNN_concate_vector = state_input_dict['RNN_concate_vector']
            RL_states = torch.cat((encoded_states, RNN_concate_vector), -1)
            RL_states = batch_cat_targets(RL_states, targets, self.action_dim)
        else:
            raise ValueError('wrong setting of RLInputCompo')


        value = self.value_net(RL_states)
        return value


class KTnet(nn.Module):
    def __init__(self, dkt_para_dict):
        super().__init__()
        input_size = dkt_para_dict['input_size']
        emb_dim = dkt_para_dict['emb_dim']
        hidden_size = dkt_para_dict['hidden_size']
        num_skills = dkt_para_dict['num_skills']
        nlayers = dkt_para_dict['nlayers']
        dropout = dkt_para_dict['dropout']

        self.name = 'DKT'
        self.nhid = hidden_size
        self.nlayers = nlayers
        self.dropout = dropout

        self.embedding_layer = nn.Linear(input_size, emb_dim)
        torch.nn.init.normal_(self.embedding_layer.weight)
        torch.nn.init.zeros_(self.embedding_layer.bias)

        self.rnn = nn.LSTM(emb_dim, hidden_size, nlayers)
        self.fc_out = nn.Linear(hidden_size, num_skills)

        self.dropout = nn.Dropout(p=self.dropout)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        h_0, c_0 = self.init_hidden_state(x.shape[1])

        embed = self.embedding_layer(x)
        output, _ = self.rnn(embed, (h_0, c_0))
        out = self.fc_out(output)
        out = self.dropout(out)
        return out

    def init_hidden_state(self, batch_size):
        h_0 = torch.rand((self.nlayers, batch_size, self.nhid))
        c_0 = torch.rand((self.nlayers, batch_size, self.nhid))
        return h_0, c_0

