import os
import sys
import json
import random
import pandas as pd
from longling import wf_open
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

dataset = os.environ.get('UNIER_PATH_DATASET', 'assist17')
base_dir = f'data/dataProcess/{dataset}'

train_file = os.path.join(base_dir, 'train_valid.csv')
test_file = os.path.join(base_dir, 'test.csv')

df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)
data = pd.concat([df_train, df_test], ignore_index=True)

questions_list = data['questions'].apply(lambda x: x.split(","))
concepts_list = data['concepts'].apply(lambda x: x.split(","))
responses_list = data['responses'].apply(lambda x: x.split(","))

data_kt = []
data_q = []

for qs, cs, ans in zip(questions_list, concepts_list, responses_list):
    seq_q = [[int(q), int(a)] for q, a in zip(qs, ans)]
    seq_kt = [[int(c), int(a)] for c, a in zip(cs, ans)]
    
    data_q.append(seq_q)
    data_kt.append(seq_kt)

random.seed(42)
combined = list(zip(data_q, data_kt))
random.shuffle(combined)
data_q_shuffled, data_kt_shuffled = zip(*combined)

split_idx = len(data_q_shuffled) // 2

with open(os.path.join(base_dir, 'student_log_kt_None'), 'w', encoding='utf-8') as f:
    for seq in data_kt_shuffled:
        f.write(json.dumps(seq) + '\n')

with open(os.path.join(base_dir, 'student_log_kt_None_q'), 'w', encoding='utf-8') as f:
    for seq in data_q_shuffled:
        f.write(json.dumps(seq) + '\n')

with wf_open(os.path.join(base_dir, 'dataOff')) as wf_kt_off, \
     wf_open(os.path.join(base_dir, 'dataRec')) as wf_kt_rec, \
     wf_open(os.path.join(base_dir, 'dataOff_q')) as wf_q_off, \
     wf_open(os.path.join(base_dir, 'dataRec_q')) as wf_q_rec:

    for i, (seq_q, seq_kt) in enumerate(zip(data_q_shuffled, data_kt_shuffled)):
        if i <= split_idx:
            print(json.dumps(seq_kt), file=wf_kt_off)
            print(json.dumps(seq_q), file=wf_q_off)
        else:
            print(json.dumps(seq_kt), file=wf_kt_rec)
            print(json.dumps(seq_q), file=wf_q_rec)
