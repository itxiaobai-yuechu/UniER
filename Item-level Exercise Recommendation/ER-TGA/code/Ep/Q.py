import numpy as np
import pandas as pd
import re
import os

dataset = os.environ.get('UNIER_DATASET', 'assist2017')


merged_df = pd.read_csv(f'../../evaluate/{dataset}/test_sequences.csv')

que_set = set()
kc_set = set()
for i in range(len(merged_df)):
    que = [int(x.strip()) for x in str(merged_df['questions'][i]).split(',')]
    kc = [int(x.strip()) for x in str(merged_df['concepts'][i]).split(',')]
    que_set.update(que)
    kc_set.update(kc)

print(f"min_kc: {min(kc_set)}, max_kc: {max(kc_set)}, kc_num: {len(kc_set)}")
print(f"min_que: {min(que_set)}, max_que: {max(que_set)}, que_num: {len(que_set)}")

Q = [[0 for _ in range(max(kc_set)+1)] for _ in range(max(que_set)+1)]
for i in range(len(merged_df)):
    questions = str(merged_df.iloc[i]['questions']).split(',')
    concepts = str(merged_df.iloc[i]['concepts']).split(',')
    for que, kc in zip(questions, concepts):
        if int(que) != -1:
            Q[int(que)][int(kc)] = 1

np.save(f'../../evaluate/{dataset}/Q.npy', Q)
