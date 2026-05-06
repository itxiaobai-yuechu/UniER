import torch
import torch.nn as nn
import torch.nn.functional as F

from KTScripts.BackModels import MLP, Transformer

class SRC(nn.Module):
    def __init__(self, skill_num, input_size, weight_size, hidden_size, dropout, allow_repeat=False,
                 with_kt=False):
        super(SRC, self).__init__()
        self.embedding = nn.Embedding(skill_num, input_size)
        self.l2 = nn.Linear(input_size, hidden_size)
        self.state_encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.path_encoder = Transformer(hidden_size, hidden_size, 0.0, head=1, b=1, transformer_mask=False)
        
        self.W1 = nn.Linear(hidden_size, weight_size, bias=False)
        self.W2 = nn.Linear(hidden_size, weight_size, bias=False)
        self.vt = nn.Linear(weight_size, 1, bias=False)
        
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        self.withKt = with_kt
        if with_kt:
            self.ktRnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
            self.ktMlp = MLP(hidden_size, [hidden_size // 2, hidden_size // 4, 1], dropout=dropout)
            
        self.allow_repeat = allow_repeat
        self.skill_num = skill_num

    def begin_episode(self, targets, initial_logs, initial_log_scores):
        target_embeds = self.embedding(targets.long())
        targets_mean = torch.mean(target_embeds, dim=1, keepdim=True)
        targets = self.l2(targets_mean)
        
        if initial_logs is not None:
            states = self.step(initial_logs, initial_log_scores, None)
        else:
            h0 = torch.zeros(1, targets.size(0), targets.size(2)).to(targets.device)
            c0 = torch.zeros(1, targets.size(0), targets.size(2)).to(targets.device)
            states = (h0, c0)
        return targets, states

    def step(self, x, score, states):
        x = self.embedding(x.long())
        x = torch.cat((x, score.unsqueeze(-1).float()), dim=-1)
        x = self.l1(x)
        _, states = self.state_encoder(x, states)
        return states

    def forward(self, targets, initial_logs, initial_log_scores, origin_path, n):
        targets, states = self.begin_episode(targets, initial_logs, initial_log_scores)
        
        inputs = self.l2(self.embedding(origin_path.long()))
        encoder_states = self.path_encoder(inputs)
        encoder_states = encoder_states + inputs
        
        blend1 = self.W1(encoder_states + torch.mean(encoder_states, dim=1, keepdim=True) + targets)
        
        decoder_input = torch.zeros_like(inputs[:, 0:1])
        probs, paths, selecting_s = [], [], []
        
        batch_idx = torch.arange(inputs.size(0)).to(inputs.device)
        selected = torch.zeros((inputs.size(0), inputs.size(1)), dtype=torch.bool).to(inputs.device)
        minimum_fill = torch.full_like(selected, -1e9, dtype=torch.float32)
        
        hidden_states_list = []
        
        for i in range(n):
            hidden, states = self.decoder(decoder_input, states)
            if self.withKt and i > 0:
                hidden_states_list.append(hidden)
            
            blend2 = self.W2(hidden)
            blend_sum = blend1 + blend2
            out = self.vt(blend_sum).squeeze(-1)
            
            if not self.allow_repeat:
                out = torch.where(selected, minimum_fill, out)
                out_probs = F.softmax(out, dim=-1)
                if self.training:
                    selecting = torch.multinomial(out_probs, 1).squeeze(-1)
                else:
                    selecting = torch.argmax(out_probs, dim=1)
                selected[batch_idx, selecting] = True
            else:
                out_probs = F.softmax(out, dim=-1)
                selecting = torch.multinomial(out_probs, 1).squeeze(-1)
            
            selecting_s.append(selecting)
            path = origin_path[batch_idx, selecting]
            decoder_input = encoder_states[batch_idx, selecting].unsqueeze(1)
            
            selected_prob = out_probs[batch_idx, selecting]
            paths.append(path)
            probs.append(selected_prob)
            
        probs = torch.stack(probs, dim=1)
        paths = torch.stack(paths, dim=1)
        selecting_s = torch.stack(selecting_s, dim=1)
        
        if self.withKt and self.training:
            last_hidden, _ = self.decoder(decoder_input, states)
            hidden_states_list.append(last_hidden)
            hidden_states = torch.cat(hidden_states_list, dim=1)
            kt_output = torch.sigmoid(self.ktMlp(hidden_states))
            return [paths, probs, selecting_s, kt_output]
            
        return paths, probs, selecting_s

    def backup(self, targets, initial_logs, initial_log_scores, origin_path, selecting_s):
        targets, states = self.begin_episode(targets, initial_logs, initial_log_scores)
        
        inputs = self.l2(self.embedding(origin_path.long()))
        encoder_states = self.path_encoder(inputs)
        encoder_states = encoder_states + inputs
        
        blend1 = self.W1(encoder_states + torch.mean(encoder_states, dim=1, keepdim=True) + targets)
        
        batch_size, n_steps = selecting_s.shape
        idx = torch.arange(batch_size).unsqueeze(1).expand(-1, n_steps)
        selecting_states = encoder_states[idx, selecting_s]
        
        zeros = torch.zeros_like(selecting_states[:, 0:1])
        decoder_inputs = torch.cat((zeros, selecting_states[:, :-1]), dim=1)
        
        hidden_states, _ = self.decoder(decoder_inputs, states)
        blend2 = self.W2(hidden_states)
        
        blend_sum = blend1.unsqueeze(1) + blend2.unsqueeze(2)
        out = self.vt(blend_sum).squeeze(-1)
        
        mask = selecting_s.unsqueeze(1).repeat(1, n_steps, 1)
        final_out = out.clone()
        for b in range(batch_size):
            for t in range(1, n_steps):
                prev_indices = selecting_s[b, :t]
                final_out[b, t, prev_indices] = -1e9

        out_softmax = F.softmax(final_out, dim=-1)
        probs = torch.gather(out_softmax, 2, selecting_s.unsqueeze(-1)).squeeze(-1)
        return probs