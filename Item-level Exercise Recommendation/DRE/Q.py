import numpy as np
import pandas as pd
import re

dataset = 'nips34'
df_train = pd.read_csv(f'data/{dataset}/train_valid.csv')
df_test = pd.read_csv(f'data/{dataset}/test.csv')
merged_df = pd.concat([df_train[['questions', 'concepts']], df_test[['questions', 'concepts']]], ignore_index=True)

que_set = set()
kc_set = set()
for i in range(len(merged_df)):
    que = [int(x.strip()) for x in str(merged_df['questions'][i]).split(',')]
    kc = [int(x.strip()) for x in str(merged_df['concepts'][i]).split(',')]
    que_set.update(que)
    kc_set.update(kc)

Q = [[0 for _ in range(max(kc_set)+1)] for _ in range(max(que_set)+1)]
for i in range(len(merged_df)):
    questions = str(merged_df.iloc[i]['questions']).split(',')
    concepts = str(merged_df.iloc[i]['concepts']).split(',')
    for que, kc in zip(questions, concepts):
        if int(que) != -1:
            Q[int(que)][int(kc)] = 1

np.save(f'data/{dataset}/Q.npy', Q)
