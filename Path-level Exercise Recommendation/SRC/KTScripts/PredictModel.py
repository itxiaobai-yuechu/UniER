import torch
import torch.nn as nn
import torch.nn.functional as F

from KTScripts.BackModels import MLP, Transformer, CoKT


class PredictModel(nn.Module):
    def __init__(self, feat_nums, embed_size, hidden_size, pre_hidden_sizes, dropout, output_size=1, with_label=True,
                 model_name='DKT'):
        super(PredictModel, self).__init__()
        self.item_embedding = nn.Embedding(feat_nums, embed_size)
        self.mlp = MLP(hidden_size, pre_hidden_sizes + [output_size], dropout=dropout)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.with_label = with_label
        self.move_label = True
        input_size_label = embed_size + 1 if with_label else embed_size
        self.model_name = model_name
        self.return_tuple = True
        
        if model_name == 'DKT':
            self.rnn = nn.LSTM(input_size_label, hidden_size, batch_first=True)
        elif model_name == 'Transformer':
            self.rnn = Transformer(input_size_label, hidden_size, dropout, head=4, b=1, position=True)
            self.return_tuple = False
        elif model_name == 'GRU4Rec':
            self.rnn = nn.GRU(input_size_label, hidden_size, batch_first=True)
            self.move_label = False

    def forward(self, x, y, mask=None):
        x = self.item_embedding(x.long())
        if self.with_label:
            if self.move_label:
                y_ = torch.cat((torch.zeros_like(y[:, 0:1]), y[:, :-1]), dim=1)
            else:
                y_ = y
            x = torch.cat((x, y_.unsqueeze(-1).float()), dim=-1)
            
        o = self.rnn(x)
        if self.return_tuple:
            o = o[0]
            
        if mask is not None:
            mask_bool = mask.bool()
            o = o[mask_bool]
            y = y[mask_bool]
            
        o = o.view(-1, self.hidden_size)
        y = y.view(-1)
        o = self.mlp(o)
        
        if self.model_name == 'GRU4Rec':
            o = F.softmax(o, dim=-1)
        else:
            o = torch.sigmoid(o).squeeze(-1)
        return o, y

    def learn_lstm(self, x, states1=None, states2=None, get_score=True):
        if states1 is None:
            states = None
        else:
            states = (states1, states2)
        return self.learn(x, states, get_score)

    def learn(self, x, states=None, get_score=True):
        x = self.item_embedding(x.long())
        o = torch.zeros_like(x[:, 0:1, 0:1])
        os = [None] * x.size(1)
        with_label, rnn, mlp = self.with_label, self.rnn, self.mlp
        
        for i in range(x.size(1)):
            x_i = x[:, i:i + 1]
            if with_label and get_score:
                x_i = torch.cat((x_i, o), dim=-1)
            o, states = rnn(x_i, states)
            if get_score:
                o = torch.sigmoid(mlp(o.squeeze(1))).unsqueeze(1)
            os[i] = o
            
        os = torch.cat(os, dim=1)
        if self.output_size == 1:
            os = os.squeeze(-1)
        return os, states

    def GRU4RecSelect(self, origin_paths, n, skill_num, initial_logs):
        ranked_paths = [None] * n
        selected_paths = torch.ones((origin_paths.size(0), skill_num), dtype=torch.bool, device=origin_paths.device)
        
        selected_paths.scatter_(1, origin_paths.long(), False)
        
        path, states = initial_logs, None
        for i in range(n):
            o, states = self.learn(path, states)
            o = o[:, -1]
            o[selected_paths] = -1
            path = torch.argmax(o, dim=-1)
            ranked_paths[i] = path
            
            selected_paths.scatter_(1, path.unsqueeze(1), True)
            path = path.unsqueeze(1)
            
        ranked_paths = torch.stack(ranked_paths, dim=-1)
        return ranked_paths


class PredictRetrieval(PredictModel):
    def __init__(self, feat_nums, input_size, hidden_size, pre_hidden_sizes, dropout, with_label=True,
                 model_name='CoKT'):
        super(PredictRetrieval, self).__init__(feat_nums, input_size, hidden_size, pre_hidden_sizes, dropout, 1,
                                               with_label, model_name)
        if model_name == 'CoKT':
            self.rnn = CoKT(input_size + 1, hidden_size, dropout, head=2)

    def forward(self, intra_x, inter_his, inter_r, y, mask, inter_len):
        intra_x = self.item_embedding(intra_x.long())
        if self.with_label:
            y_ = torch.cat((torch.zeros_like(y[:, 0:1, None]), y[:, :-1, None]), dim=1).float()
            intra_x = torch.cat((intra_x, y_), dim=-1)
            
        inter_his = torch.cat((self.item_embedding(inter_his[:, :, :, 0].long()),
                               inter_his[:, :, :, 1:].float()), dim=-1)
        inter_r = torch.cat((self.item_embedding(inter_r[:, :, :, 0].long()), 
                             inter_r[:, :, :, 1:].float()), dim=-1)
        
        o = self.rnn(intra_x, inter_his, inter_r, mask, inter_len)
        o = torch.sigmoid(self.mlp(o)).squeeze(-1)
        y = y[mask.bool()].view(-1)
        return o, y

    def learn(self, intra_x, inter_his, inter_r, inter_len, states=None):
        his_len, seq_len = 0, intra_x.size(1)
        intra_x = self.item_embedding(intra_x.long())
        intra_h = None
        
        if states is not None:
            his_len = states[0].size(1)
            intra_x = torch.cat((intra_x, states[0]), dim=1)
            intra_h = states[1]
            
        o = torch.zeros_like(intra_x[:, 0:1, 0:1])
        inter_his = torch.cat((self.item_embedding(inter_his[:, :, :, 0].long()),
                               inter_his[:, :, :, 1:].float()), dim=1)
        inter_r = torch.cat((self.item_embedding(inter_r[:, :, :, 0].long()), 
                             inter_r[:, :, :, 1:].float()), dim=-1)
        
        M_rv, M_pv = self.rnn.deal_inter(inter_his, inter_r, inter_len)
        os = []
        for i in range(seq_len):
            o, intra_h = self.rnn.step(M_rv[:, i], M_pv[:, i], intra_x[:, :i + his_len + 1], o, intra_h)
            o = torch.sigmoid(self.mlp(o))
            os.append(o)
            
        o = torch.cat(os, dim=1)
        return o, (intra_x, intra_h)


class ModelWithLoss(nn.Module):
    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, *data):
        output_data = self.model(*data)
        return self.criterion(*output_data), output_data

    def output(self, *data):
        output_data = self.model(*data)
        return self.criterion(*output_data), output_data


class ModelWithLossMask(ModelWithLoss):
    def forward(self, *data):
        output_data = self.model(*data[:-1])
        return self.criterion(*output_data, data[-1]), output_data

    def output(self, *data):
        output_data = self.model(*data[:-1])
        return self.criterion(*output_data, data[-1]), self.mask_fn(*output_data, data[-1].view(-1))

    @staticmethod
    def mask_fn(o, y, mask):
        mask_bool = mask.bool()
        o_mask = o[mask_bool].view(-1, o.size(-1))
        y_mask = y[mask_bool]
        return o_mask, y_mask


class ModelWithOptimizer(nn.Module):
    def __init__(self, model_with_loss, optimizer, mask=False):
        super().__init__()
        self.mask = mask
        self.model_with_loss = model_with_loss
        self.optimizer = optimizer

    def forward(self, *data):
        self.optimizer.zero_grad()
        loss, output_data = self.model_with_loss(*data)
        loss.backward()
        self.optimizer.step()
        
        if self.mask:
            output_data = self.model_with_loss.mask_fn(*output_data, data[-1].view(-1))
        return loss, output_data