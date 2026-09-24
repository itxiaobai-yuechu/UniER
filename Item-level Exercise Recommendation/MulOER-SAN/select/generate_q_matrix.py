import pandas as pd
import numpy as np
import os

dataset = os.environ.get('UNIER_DATASET', 'assist2017')
print(f"Processing dataset: {dataset}")
input_file = f'../PYKT/data/{dataset}/qid_test_question_predictions.txt'
output_file = f'./{dataset}/Q.npy'
df = pd.read_csv(input_file, sep='\t')
questions = df['questions'].astype(str)
concepts = df['concepts'].astype(str)
all_qids = [int(q) for qs in questions for q in qs.split(',')]
all_kcs = [int(kc) for kcs in concepts for kc in kcs.split(',')]
max_qid = max(all_qids)
max_kcid = max(all_kcs)
print(max_qid, max_kcid)

Q = np.zeros((max_qid + 1, max_kcid + 1))
for qs, kcs in zip(questions, concepts):
    q_list = list(map(int, qs.split(',')))
    kc_list = list(map(int, kcs.split(',')))
    for q, kc in zip(q_list, kc_list):
        Q[q][kc] = 1
np.save(output_file, Q)
