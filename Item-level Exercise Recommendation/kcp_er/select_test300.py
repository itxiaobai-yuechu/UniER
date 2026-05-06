import numpy as np
import pandas as pd
import torch
import random
import os



datasets = 'assist2017'
kc_num = {'assist2009':123, 'nips34':57, 'assist2012':265,'assist2017':102,'algebra2005':112,'bridge2006':493,'ednet':188, 'junyi':39,'xes3g5m':865}
test_raw = pd.read_csv(f'./datasets/{datasets}/select/test_sequences.csv')
test_raw_len = len(test_raw)
select_index = np.array(random.sample(range(0, len(test_raw)), len(test_raw)))
test_300 = test_raw.iloc[select_index]
pkc = torch.load(f'kcpl_data/{datasets}/pkc.pth')
pkm = torch.load(f'kcpl_data/{datasets}/pkm.pth')
select_index_list = select_index.tolist()
pkc_300 = pkc[select_index_list]
pkm_300 = pkm[select_index_list]
save_dir = f'./datasets/{datasets}/select'
try:
    os.mkdir(save_dir)
except:
    pass
test_300.to_csv(f'./datasets/{datasets}/select/test_sequences.csv', index=False)
torch.save(pkc_300, f'./datasets/{datasets}/select/pkc.pth')
torch.save(pkm_300, f'./datasets/{datasets}/select/pkm.pth')


df = pd.read_csv(f'./datasets/{datasets}/select/test_sequences.csv')
df = df[['questions', 'concepts']]
df['questions'] = df['questions'].apply(lambda x: ','.join(map(str.strip, x.split(','))))
df['concepts'] = df['concepts'].apply(lambda x: ','.join(map(str.strip, x.split(','))))
all_questions = [int(q) for q_list in df['questions'].values for q in q_list.split(',') if int(q) != -1]
max_qidx = max(all_questions)
Q = np.zeros((max_qidx + 1, kc_num[datasets]))
for i in range(len(df)):
    questions = df.iloc[i]['questions'].split(',')
    concepts = df.iloc[i]['concepts'].split(',')
    for que, kc in zip(questions, concepts):
        Q[int(que)][int(kc)] = 1
print(Q.shape)
np.save(f'./datasets/{datasets}/select/Q.npy', Q)

