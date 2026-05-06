import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: list[int],
                 norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm1d,
                 activation_layer: Optional[Callable[..., nn.Module]] = nn.LeakyReLU,
                 bias: bool = True,
                 dropout: float = 0.0):

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, hidden_channels[-1], bias=bias))

        super().__init__(*layers)


def nll_loss(y_, y, mask):
    mask = mask.view(-1)
    loss = F.nll_loss(torch.log(y_ + 1e-9), y, reduction='none')
    return torch.sum(loss * mask) / torch.sum(mask)


class MultiHeadedAttention(nn.Module):
    def __init__(self, head, hidden_sizes, dropout_rate, input_sizes=None):
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes] * 4
        if input_sizes is None:
            input_sizes = hidden_sizes
        for hidden_size in hidden_sizes:
            assert hidden_size % head == 0
            
        self.head = head
        self.head_size = hidden_sizes[0] // head
        self.hidden_size = hidden_sizes[-1]
        self.d_k = math.sqrt(hidden_sizes[0] // head)
        self.linear_s = nn.ModuleList([
            nn.Linear(input_size, hidden_size) 
            for (input_size, hidden_size) in zip(input_sizes, hidden_sizes)
        ])
        self.dropout = nn.Dropout(p=dropout_rate)

    def attention(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-1, -2)) / self.d_k
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        tensors = [query, key, value]
        q, k, v = [
            l(x.contiguous().view(-1, x.size(-1))).view(*x.shape[:2], self.head, self.head_size).transpose(1, 2)
            for l, x in zip(self.linear_s, tensors)
        ]
        x, _ = self.attention(q, k, v, mask)
        x = x.transpose(1, 2).contiguous()
        return self.linear_s[-1](x.view(-1, self.head * self.head_size)).view(*x.shape[:2], self.hidden_size)


class FeedForward(nn.Module):
    def __init__(self, head, input_size, dropout_rate):
        super(FeedForward, self).__init__()
        self.mh = MultiHeadedAttention(head, input_size, dropout_rate)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.activate = nn.LeakyReLU()
        self.ln1 = nn.LayerNorm(input_size)
        self.ln2 = nn.LayerNorm(input_size)
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)

    def forward(self, s, mask):
        s = s + self.dropout1(self.mh(s, s, s, mask))
        s = self.ln1(s)
        s_ = self.activate(self.fc1(s))
        s_ = self.dropout2(self.fc2(s_))
        s = self.ln2(s + s_)
        return s


class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate, head=1, b=1, position=False, transformer_mask=True):
        super(Transformer, self).__init__()
        self.position = position
        if position:
            self.pe = PositionalEncoding(input_size, 0.5)
        self.fc = nn.Linear(input_size, hidden_size)
        self.SAs = nn.ModuleList([MultiHeadedAttention(head, hidden_size, dropout_rate) for _ in range(b)])
        self.FFNs = nn.ModuleList([FeedForward(head, hidden_size, dropout_rate) for _ in range(b)])
        self.b = b
        self.transformer_mask = transformer_mask

    def forward(self, inputs, mask=None):
        if self.position:
            inputs = self.pe(inputs)
        inputs = self.fc(inputs)
        max_len = inputs.shape[1]
        
        if self.transformer_mask:
            mask = torch.tril(torch.ones(1, max_len, max_len, dtype=torch.bool, device=inputs.device))
        elif mask is not None:
            mask = mask.unsqueeze(1)
            
        if mask is not None:
            mask = mask.unsqueeze(1)
            
        for i in range(self.b):
            inputs = self.SAs[i](inputs, inputs, inputs, mask)
            inputs = self.FFNs[i](inputs, mask)
        return inputs


class PositionalEncoding(nn.Module):


    def __init__(self, d_model, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)[:, :d_model // 2]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CoKT(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate, head=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.ma_inter = MultiHeadedAttention(head, hidden_size, dropout_rate, input_sizes=(
            hidden_size + input_size - 1, hidden_size + input_size - 1, hidden_size + input_size, hidden_size))
        self.ma_intra = MultiHeadedAttention(head, hidden_size, dropout_rate, input_sizes=(
            input_size - 1, input_size - 1, hidden_size + 1, hidden_size))
        self.wr = nn.Parameter(torch.randn(1, 1, 2))
        self.ln = nn.Linear(2 * hidden_size + input_size - 1, hidden_size)

    def forward(self, intra_x, inter_his, inter_r, intra_mask, inter_len):
        intra_mask_bool = intra_mask.bool()

        intra_h, _ = self.rnn(intra_x)
        intra_h_mask = intra_h[intra_mask_bool]
        intra_x_mask = intra_x[intra_mask_bool]

        inter_his, _ = self.rnn(
            inter_his.view(inter_his.size(0) * inter_his.size(1), *inter_his.shape[2:]))
        
        inter_his = inter_his[torch.arange(inter_his.size(0), device=inter_his.device), inter_len.view(-1) - 1]
        inter_his = inter_his.view(*inter_len.shape, self.hidden_size)
        
        inter_his_mask = inter_his[intra_mask_bool]
        inter_r_mask = inter_r[intra_mask_bool]
        
        M_rv = torch.cat((inter_his_mask, inter_r_mask), dim=-1)
        M_pv = M_rv[:, :, :-1].contiguous().view(M_rv.size(0), M_rv.size(1), self.input_size + self.hidden_size - 1)
        
        m_pv = torch.cat((intra_h_mask, intra_x_mask[:, :-1]), dim=1).unsqueeze(1)
        
        v_v = self.ma_inter(m_pv, M_pv, M_rv)[0].squeeze(1)

        intra_x_p = intra_x[:, :, :-1]
        intra_h_p = torch.cat((intra_h, intra_x[:, :, -1:]), dim=-1)
        intra_mask_attn = torch.tril(torch.ones(1, 1, intra_x_p.size(1), intra_x_p.size(1), dtype=torch.bool, device=intra_x.device))
        
        v_h = self.ma_intra(intra_x_p, intra_x_p, intra_h_p, mask=intra_mask_attn)[0]
        v_h_mask = v_h[intra_mask_bool]
        
        v = torch.sum(F.softmax(self.wr, dim=-1) * torch.stack((v_v, v_h_mask), dim=-1), dim=-1)
        return self.ln(torch.cat((v, intra_h_mask, intra_x_mask[:, :-1]), dim=1))

    def deal_inter(self, inter_his, inter_r, inter_len):
        inter_his, _ = self.rnn(
            inter_his.view(inter_his.size(0) * inter_his.size(1), *inter_his.shape[2:]))
        inter_his = inter_his[torch.arange(inter_his.size(0), device=inter_his.device), inter_len.view(-1) - 1]
        inter_his = inter_his.view(*inter_len.shape, self.hidden_size)
        M_rv = torch.cat((inter_his, inter_r), dim=-1)
        M_pv = M_rv[:, :, :-1].contiguous().view(*M_rv.shape[:3], self.input_size + self.hidden_size - 1)
        return M_rv, M_pv

    def step(self, m_rv, M_pv, intra_x, o, intra_h_p=None):
        
        h_0 = None if intra_h_p is None else intra_h_p[:, -1, :-1].unsqueeze(0).contiguous()
        intra_h_next, _ = self.rnn(torch.cat((intra_x[:, -1:], o), dim=-1), h_0)
        m_pv = torch.cat((intra_h_next, intra_x[:, -1:]), dim=-1)
        v_v = self.ma_inter(m_pv, M_pv, m_rv)[0]

        intra_x_p = intra_x
        intra_h_next = torch.cat((intra_h_next, o), dim=-1)
        intra_h_p = intra_h_next if intra_h_p is None else torch.cat((intra_h_p, intra_h_next), dim=1)
        
        v_h = self.ma_intra(intra_x_p[:, -1:], intra_x_p, intra_h_p)[0]
        v = torch.sum(F.softmax(self.wr, dim=-1) * torch.stack((v_v, v_h), dim=-1), dim=-1)
        
        out = self.ln(torch.cat((v, intra_h_p[:, -1:, :-1], intra_x[:, -1:]), dim=-1))
        return out, intra_h_p


if __name__ == '__main__':
    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CoKT(16, 14, 0.5).to(device)

    seq_len_ = list(range(100, 139))
    max_len_ = 200
    for j in range(len(seq_len_)):
        t0 = time.perf_counter()
        seq_len = seq_len_[:j + 1]
        seq_sum = sum(seq_len)
        
        seq_len_tensor = torch.tensor(seq_len, device=device)
        mask_ = torch.arange(max_len_, device=device).unsqueeze(0) < seq_len_tensor.unsqueeze(1)
        
        x_ = torch.rand(len(seq_len), max_len_, 16, device=device)
        his = torch.rand(len(seq_len), max_len_ * 5, max_len_, 16, device=device)
        r = torch.rand(len(seq_len), max_len_, 5, 16, device=device)
        inter_len_ = torch.randint(1, 200, r.shape[:3], device=device)
        
        t1 = time.perf_counter()
        print(f"Prep time: {t1 - t0:.6f}s")
        
        output = model(x_, his, r, mask_, inter_len_)
        print(output)
        print(f"Forward time: {time.perf_counter() - t1:.6f}s")