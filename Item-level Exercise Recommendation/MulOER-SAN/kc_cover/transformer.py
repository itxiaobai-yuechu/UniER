import numpy as np
import pandas as pd
import os
import sys
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import torch.optim as optim

np.random.seed(int(os.environ.get('UNIER_SEED', '42')))
torch.manual_seed(int(os.environ.get('UNIER_SEED', '42')))
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(int(os.environ.get('UNIER_SEED', '42')))

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=64):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransAm(nn.Module):
    def __init__(self, feature_size,out_size,d_model=512, num_layers=1, dropout=0):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.embedding=nn.Linear(feature_size,512)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, out_size)
        self.init_weights()
        self.src_key_padding_mask = None

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_padding):
        if self.src_key_padding_mask is None:
            mask_key = src_padding
            self.src_key_padding_mask = mask_key
        src=self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask, self.src_key_padding_mask)
        output = self.decoder(output)
        self.src_key_padding_mask = None
        return output
class dataset(Dataset):
    def __init__(self,path,seq_len):
        self.path = path
        self.seq_len = seq_len
        self.data = pd.read_csv(self.path, sep='\t')
        self.data = self.data.groupby('orirow')['concepts'].apply(list).reset_index()
        self.data['concepts'] = self.data['concepts'].apply(lambda x: self.adjust_length(x, self.seq_len))
        
    def adjust_length(self, skill_list, target_length):
        if len(skill_list) > target_length:
            return skill_list[:target_length]
        else:
            return skill_list + [0] * (target_length - len(skill_list))
    
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        skill_data = self.data.iloc[idx]['concepts']
        
        input_sequence = np.array(skill_data[:-1])  
        target_sequence = np.array(skill_data[1:]) 
        input_sequence = torch.tensor(input_sequence, dtype=torch.float).unsqueeze(0)
        target_sequence = torch.tensor(target_sequence, dtype=torch.float).unsqueeze(0)
                
        return input_sequence, target_sequence

dataset_name = os.environ.get("UNIER_DATASET", "assist2017")
base_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(base_dir,f'../PYKT/data/{dataset_name}/qid_test_question_predictions.txt')
seq_len = 6500
batch = 1
data_set = dataset(path,seq_len)
dataloader = DataLoader(data_set, batch_size=batch, shuffle=True)

model = TransAm(6499,6499)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.99)
m = nn.Sigmoid()

def train(epochs):
    for epoch in range(epochs):
        epoch_loss = 0
        y_pre = []
        y_true = []
        for X, y in dataloader:  
            enc_inputs = X.permute([1,0,2])
            y=y.permute([1,0,2])
            key_padding_mask = torch.ones(enc_inputs.shape[1], enc_inputs.shape[0])
            optimizer.zero_grad()
            output = model(enc_inputs, key_padding_mask)
            output=output[-5:]
            y=y[-5:]
            loss=criterion(output,y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            epoch_loss+=loss.item()
            pres=output.detach().numpy()
            pres=pres.transpose(0,1,2).reshape(-1,6499)
            tru=y.detach().numpy()
            tru=tru.transpose(0,1,2).reshape(-1,6499)
            y_pre.append(pres)
            y_true.append(tru)
        pre=np.concatenate(y_pre,axis=0)
        true=np.concatenate(y_true,axis=0)
        pre_tensor = torch.tensor(pre)
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(epoch_loss))
    return m(pre_tensor).numpy()

pre = train(50)
df = pd.read_csv(path, sep='\t')
group_lengths = df.groupby('orirow').size()
max_group_length = group_lengths.max()
max_group_orirow = group_lengths.idxmax()

print(f"The maximum group length is: {max_group_length}, corresponding orirow is: {max_group_orirow}")

print(group_lengths)
adjusted_list = []
for i, length in group_lengths.items():
    group_data = pre[i][:length]
    adjusted_list.append(group_data)

merged_data = np.concatenate(adjusted_list)
print(len(merged_data))

df['merged_data'] = pd.Series(merged_data[:len(df)])
path = os.path.join(base_dir, f'../select/{dataset_name}/new_data0.txt')
df.to_csv(path, sep='\t', index=False)
