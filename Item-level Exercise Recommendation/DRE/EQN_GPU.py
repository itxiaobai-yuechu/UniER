import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import csv
import collections

class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)  
        out, hn = self.gru(x, h0)  
        out = hn[-1] 
        return out

class ExerciseEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(ExerciseEmbedding, self).__init__()
        self.padded_concept_embedding = nn.Embedding(vocab_size, embed_size)
        self.fc = nn.Linear(embed_size + 1, embed_size)
                                               
    def forward(self, padded_concept, difficulty):
        padded_concept = padded_concept.long()
        v_k = self.padded_concept_embedding(padded_concept)
        combined_features = torch.cat((v_k, difficulty.expand(len(padded_concept)).unsqueeze(1)), dim=1)
        exercise_embedding = self.fc(combined_features)
        return exercise_embedding

def load_file_data(file_path):
    data_dict = {}
    with open(file_path, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        for row in reader:
            if row[0] not in data_dict:
                data_dict[row[0]] = []
            data_dict[row[0]].append(float(row[1]))
    return data_dict

def exercise_embedding(padded_c, exercise, ex_embed_dim, data_dict, device):  
    vocab_size = int(padded_c.max().item() + 1) 
    filtered_data = data_dict.get(exercise, []) 
    diff = torch.tensor(filtered_data, dtype=torch.float).to(device)
    padded_c = padded_c.to(device)
    
    
    embedding_weight = torch.randn(vocab_size, ex_embed_dim, device=device)
    fc_weight = torch.randn(ex_embed_dim, ex_embed_dim + 1, device=device)
    fc_bias = torch.randn(ex_embed_dim, device=device)
    
    padded_concept = padded_c.long()
    v_k = F.embedding(padded_concept, embedding_weight)
    combined_features = torch.cat((v_k, diff.expand(len(padded_concept)).unsqueeze(1)), dim=1)
    xt = F.linear(combined_features, fc_weight, fc_bias)
    
    return xt

class Net(nn.Module):
    def __init__(self, n_states, n_hidden, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)  
        self.fc3 = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class EQN(nn.Module):
    def __init__(self, all_problem_d, n_states,
                 n_hidden, n_actions,learning_rate,
                 gamma, epsilon,target_update, device, dataset):
        super(EQN, self).__init__()
        self.epsilon = epsilon
        self.p_data = all_problem_d
        self.n_states = n_states  
        self.n_hidden = n_hidden  
        self.n_actions = n_actions  
        self.learning_rate = learning_rate  
        self.gamma = gamma  
        self.epsilon = epsilon  
        self.target_update = target_update  
        self.device = device
        self.dataset = dataset
        self.gru_network = GRUNet(80, 80, self.n_actions, 2)
        self.count = 0
        self.q_net = Net(self.n_states, self.n_hidden, self.n_actions)
        self.target_q_net = Net(self.n_states, self.n_hidden, self.n_actions)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.to(self.device)

    def update(self, transition_dict, dataset):  
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)

        actions = torch.tensor(transition_dict['actions']).view(-1,1).to(torch.int64).to(self.device) 

        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device)

        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1,1)
        q_targets = rewards + self.gamma * max_next_q_values * 1
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())     
            torch.save(self.target_q_net.state_dict(), f'Model/{dataset}/model_parameters.pth')
        self.count += 1

    def forward(self, padded_concept, performance, exercise, A, P,exercise_module_dim, data_dict, istrain, pattern, dataset):       
        if not istrain:
            self.target_q_net.load_state_dict(torch.load(f'Model/{dataset}/model_parameters.pth'))
        if  pattern == "M" or pattern == "m":    
            xt = exercise_embedding(padded_concept, exercise, exercise_module_dim,data_dict, self.device).view(-1)
            if performance == 1:
                st = torch.cat((xt, torch.zeros_like(xt)), dim=0)
            else:
                st = torch.cat((torch.zeros_like(xt), xt), dim=0)
        elif pattern == "R" or pattern == "r":              
            pxt_list = []
            for c, p, e in zip(padded_concept, performance, exercise):
                xt = exercise_embedding(c, e, exercise_module_dim,data_dict, self.device).view(-1)
                if p == 1:
                    pxt = torch.cat((xt, torch.zeros_like(xt)), dim=0)
                else:
                    pxt = torch.cat((torch.zeros_like(xt), xt), dim=0)
                pxt_list.append(pxt.tolist())
            st = self.gru_network(torch.tensor(pxt_list).unsqueeze(1).to(self.device))[-1]
            
        Q = [] 
        tensors = []
        for index, ai in enumerate(A): 
            ai = list(map(int,ai)) 
            ai_tensor = torch.tensor(ai, dtype=torch.int).long().to(self.device)
            padding_length = 4 - ai_tensor.size(0)
            padded_ai_tensor = F.pad(ai_tensor, (0, padding_length), "constant", 0) 
            xai = exercise_embedding(padded_ai_tensor, P[index], exercise_module_dim, data_dict, self.device).view(-1)
            combined_tensor = torch.cat((st, xai), dim=0)
            tensors.append(combined_tensor)
            if index == len(A) - 1:
                for i in range(0, len(tensors), 64):
                    batch = tensors[i:i + 64]
                    batch_tensor = torch.stack(batch).to(self.device)
                    if istrain:
                        h_t_n = self.q_net(batch_tensor)
                    else:
                        h_t_n = self.target_q_net(batch_tensor)
                    q_values = 1 / torch.exp(-h_t_n) 
                    q_values_sum = q_values.sum(dim=1)
                    q_values_flat_list = q_values_sum.tolist()
                    Q.extend(q_values_flat_list)
        indexed_data = list(enumerate(Q))        
        sorted_indexed_data = sorted(indexed_data, key=lambda x: x[1], reverse=True)
        top_twenty_indices = [index for index, value in sorted_indexed_data[:20]]
        max_q_values = max(Q)
        max_q_values_index = Q.index(max_q_values)
        return st,max_q_values_index,top_twenty_indices
    
    def take_action(self, Recommended_exercise):   
        if np.random.random() < self.epsilon:  
            action = Recommended_exercise
        else:
            action = random.choice(self.p_data)
        return action
    
    def take_next_state(self, padded_concept, performance, exercise, exercise_module_dim, data_dict):   
        xt = exercise_embedding(padded_concept, exercise, exercise_module_dim,data_dict, self.device)
        if performance == 1:
            st = torch.cat((xt, torch.zeros_like(xt)), dim=1)
        else:
            st = torch.cat((torch.zeros_like(xt), xt), dim=1)
        return st
    
def batch_iterator(file_path, batch_size):
    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        batch = []
        for i, row in enumerate(csv_reader):
            batch.append(row)
            if (i + 1) % 4 == 0:
                if len(batch) == 4 * batch_size:
                    yield batch[:4 * batch_size]
                    batch = []     
        if batch:
            yield batch

class ReplayBuffer():
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state):
        state = state.detach().cpu().numpy()  
        state_padding_length = 120 - len(state)
        state = np.concatenate((state, np.zeros(state_padding_length)))
        state = state.astype(float)
        next_state = next_state.detach().cpu().numpy() 
        next_state_padding_length = 120 - len(next_state)
        next_state = np.concatenate((next_state, np.zeros(next_state_padding_length)))
        next_state = next_state.astype(float)
        self.buffer.append((state, action, reward, next_state))
    
    def sample(self, batch_size):  
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state)
    
    def size(self):
        return len(self.buffer)