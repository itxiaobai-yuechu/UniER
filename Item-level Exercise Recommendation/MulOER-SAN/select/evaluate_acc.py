import os
import numpy as np
import pandas as pd
from numpy.linalg import norm

dataset="xes3g5m"
print("dataset:", dataset)

if dataset == "assist2009":
    kc_num = 123
    group_size = 13392
elif dataset == "assist2012":
    kc_num = 261
    group_size = 44545
elif dataset == "assist2017":
    kc_num = 102
    group_size = 2594
elif dataset == "algebra2005":
    kc_num = 111
    group_size = 35694
elif dataset == "bridge2006":
    kc_num = 493
    group_size = 74276
elif dataset == "ednet":
    kc_num = 189
    group_size = 11311
elif dataset == "nips34":
    kc_num = 57
    group_size = 919
elif dataset == "mooccubex":
    kc_num = 191
    group_size = 199
elif dataset == "xes3g5m":
    kc_num = 865
    group_size = 7650

base_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(base_dir,f'{dataset}/new_data1.txt')


def R_q(path):
    data = pd.read_csv(path, sep='\t')
    grouped = data.groupby('orirow').agg({
        'questions': list,
        'concepts': list,
        'concept_preds': list,
        'merged_data': list
    }).reset_index()
    df = pd.DataFrame(grouped)
    def process_row(row):
        unique_concepts = []
        unique_min_preds = []
        unique_merged = []
        seen_concepts = set()
        concept_data = zip(row['concepts'], row['concept_preds'], row['merged_data'])
        for concept, pred, merged in concept_data:
            if concept not in seen_concepts:
                seen_concepts.add(concept)
                unique_concepts.append(concept)
                unique_min_preds.append(pred)
                unique_merged.append(merged)
            else:
                index = unique_concepts.index(concept)
                if pred < unique_min_preds[index]:
                  unique_min_preds[index] = pred
                  unique_merged[index] = merged
        return pd.Series({
            'orirow': row['orirow'],
            'questions': row['questions'],
            'unique_concepts': unique_concepts,
            'unique_preds': unique_min_preds,
            'pkc': unique_merged
        })
    processed_df = df.apply(process_row, axis=1)
    return processed_df
def Que_cp(path):
    df = pd.read_csv(path, delimiter='\t')
    df_sorted = df.sort_values(by='questions')
    grouped = df_sorted.groupby('questions').agg({
        'concepts': lambda x: list(set(x))
    }).reset_index()
    return grouped
r_q = R_q(path)
que_cp = Que_cp(path)
q_concepts = que_cp['concepts']
orirow_data = r_q['orirow']
u_concepts = r_q['unique_concepts']
u_preds = r_q['unique_preds']
all_scores = []
for use_id in range(len(orirow_data)):
    for q in q_concepts:
        q_r = 1
        for i in q:
            if i in u_concepts[use_id]:
                q_r *= u_preds[use_id][u_concepts[use_id].index(i)]
            else:
                q_r *= 0
                dr = 1
                dis = (0.7 - dr)**2
                d = abs(0.7-dr)
                break
            dr = 1 - q_r
            dis = (0.7 - dr)**2
            d = abs(0.7-dr)
        all_scores.append(dr)
def split_list(data, group_size):
    grouped_data = [data[i:i + group_size] for i in range(0, len(data), group_size)]
    return grouped_data
D_q = split_list(all_scores,group_size)
q_kc = []
ac = []


new_ex = []
with open(f'{dataset}/new_ex.txt', 'r') as f:
    for line in f:
        student, exercises = line.strip().split('\t')
        exercise_list = list(map(int, exercises.split(',')))
        new_ex.append(exercise_list)

for kc in q_concepts:
  concepts_list = [0] * kc_num
  for concept in kc:
    concepts_list[concept] = 1
  q_kc.append(concepts_list)

for index,exx in enumerate(new_ex):
  ac_list = []
  for e in exx:
     acy = abs(0.7-D_q[index][e])
     ac_list.append(acy)
  ac.append(ac_list)
accy = []
for b in ac:
     acc = 0
     for c in b:
         acc += (1-c)
     acc = acc/len(b)
     accy.append(acc)
auc = 0
for i in accy:
   auc += i
acc = auc/len(new_ex)
print('Accuracy:',acc)
print('acc std',np.std(accy))

df = pd.read_csv(path, sep='\t')
grouped = df.groupby('orirow')
def process_group(group):
    filtered = group[group['late_trues'] == 1]
    unique_questions = filtered['questions'].unique()
    return unique_questions
def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    
    intersection = set1.intersection(set2)
    
    union = set1.union(set2)
    
    similarity = len(intersection) / len(union)
    
    return similarity
result = grouped.apply(process_group).reset_index(name='unique_questions')

ex_old = result['unique_questions']
jc = []
for old,new in zip(ex_old,new_ex):
    jca = 1 - jaccard_similarity(old,new)
    jc.append(jca)
print('Novelty:', np.mean(jc))
print('jc std',np.std(jc))

d = []
dim = 20
for e in new_ex:
  di = 0
  for i in range(len(e)):
    for j in range(len(e)):
      cosine = np.dot(q_kc[e[i]],q_kc[e[j]])/(norm(q_kc[e[i]])*norm(q_kc[e[j]]))
      div = 1 - cosine
      di += div
  di = di/(dim*(dim-1))
  d.append(di)
divt = 0
for i in d:
  divt += i
divt = divt/len(new_ex)
print('Diversity:',divt)
print('div std',np.std(d))

def volatility(pkm, Q, rec_Q, test_raw, u_concepts_dict):
    vol = []
    for i in range(len(test_raw)):
        uid = test_raw.iloc[i]['uid']
        if uid not in rec_Q:
            continue
        pkm_i = pkm[uid]
        concepts = u_concepts_dict[uid]
        rec_qs = rec_Q[uid]
        if len(rec_qs) < 2:
            vol.append(0.0)
            continue
            
        vol_i = 0
        for j in range(len(rec_qs) - 1):
            q1 = rec_qs[j]
            q2 = rec_qs[j + 1]
            
            idx1 = np.where(Q[q1] == 1)[0]
            prob1 = 1.0
            for k in idx1:
                if k in concepts:
                    prob1 *= pkm_i[concepts.index(k)]
                else:
                    prob1 *= 0.5
            
            idx2 = np.where(Q[q2] == 1)[0]
            prob2 = 1.0
            for k in idx2:
                if k in concepts:
                    prob2 *= pkm_i[concepts.index(k)]
                else:
                    prob2 *= 0.5
            
            vol_i += np.abs(prob1 - prob2)
            
        vol.append(vol_i / len(rec_qs))
    return 1-np.mean(vol), np.std(vol)

def coverage(test_raw, Q, rec_Q):

    cov = []
    
    for i in range(len(test_raw)):
        uid = test_raw.iloc[i]['uid']
        if uid not in rec_Q:
            continue
        kcs = [int(kc) for kc in test_raw.iloc[i]['concepts'].split(',')]
        responses = [int(r) for r in test_raw.iloc[i]['responses'].split(',')]
        
        try:
            index = kcs.index(-1)
            kcs = kcs[:index]
            responses = responses[:index]
        except ValueError:
            pass
            
        kc_last_response = {}
        for kc, r in zip(kcs, responses):
            kc_last_response[kc] = r
            
        weak_kcs = {kc for kc, r in kc_last_response.items() if r == 0}
        
        if len(weak_kcs) == 0:
            continue
            
        rec_qs = rec_Q[uid]
        covered_kcs = set()
        for q in rec_qs:
            idx = np.where(Q[q] == 1)[0]
            covered_kcs.update(idx)
            
        covered_weak_kcs = weak_kcs.intersection(covered_kcs)
        cov.append(len(covered_weak_kcs) / len(weak_kcs))
        
    if len(cov) == 0:
        return 0.0, 0.0
    return np.mean(cov), np.std(cov)

rec_Q = {}
pkm_dict = {}
u_concepts_dict = {}
for idx, exercises in enumerate(new_ex):
    uid = result.iloc[idx]['orirow']
    rec_Q[uid] = exercises
    pkm_dict[uid] = u_preds[idx]
    u_concepts_dict[uid] = u_concepts[idx]

Q = np.array(q_kc)

test_raw = df

grouped_df = df.groupby('orirow').agg({
    'concepts': lambda x: ','.join(map(str, x)),
    'late_trues': lambda x: ','.join(map(str, x))
}).reset_index()
grouped_df.rename(columns={'orirow': 'uid', 'late_trues': 'responses'}, inplace=True)

vol, vol_std = volatility(pkm_dict, Q, rec_Q, grouped_df, u_concepts_dict)
print(f'Volatility: {vol:.3f}')
print(f'vol std: {vol_std:.3f}')

cov, cov_std = coverage(grouped_df, Q, rec_Q)
print(f'Coverage: {cov:.3f}')
print(f'cov std: {cov_std:.3f}')