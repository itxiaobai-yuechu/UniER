import os
import numpy as np
import pandas as pd
import torch
import random

def accuracy(pkm, Q, rec_Q, test_raw, dielta=0.7):
    acc = []
    for i in range(len(test_raw)):
        uid = test_raw.iloc[i]['uid']
        if uid not in rec_Q:
            continue
        pkm_i = pkm[i]
        rec_qs = rec_Q[uid]
        D_qk = 0
        for q in rec_qs:
            index_1 = np.where(Q[q] == 1)[0]
            select_pkm_i = [pkm_i[k] for k in index_1]
            dis = dielta - np.prod(select_pkm_i)
            D_qk += 1 - np.abs(dis)
        acc.append(D_qk / len(rec_qs))
    return np.mean(acc), np.std(acc)

def novlty(test_raw, Q, rec_Q):
    nov = []
    kc_responses = test_raw[['concepts', 'responses']]
    for i in range(len(test_raw)):
        uid = test_raw.iloc[i]['uid']
        if uid not in rec_Q:
            continue
        kc_res_yes_set = set()
        kcs = kc_responses.iloc[i]['concepts'].split(',')
        res = kc_responses.iloc[i]['responses'].split(',')
        for kc, res in zip(kcs, res):
            if res == '1':
                kc_res_yes_set.add(int(kc))
        rec_qs = rec_Q[uid]
        dis = 0
        valid_len = 0
        for q in rec_qs:
            index_1 = np.where(Q[q] == 1)[0]
            index_1 = set(index_1)
            union_len = len(kc_res_yes_set.union(index_1))
            if union_len == 0:
                continue  
            dis += 1 - len(kc_res_yes_set.intersection(index_1)) / union_len
            valid_len += 1
        if valid_len > 0:
            nov.append(dis / valid_len)
        else:
            nov.append(0)
    return np.mean(nov), np.std(nov)

def diversity(Q, rec_Q):
    diversity = []
    for uid, rec_q in rec_Q.items():
        different = 0
        valid_pairs = 0
        for i in range(len(rec_q)):
            rec_qi = Q[rec_q[i]]
            norm_i = np.linalg.norm(rec_qi)
            for j in range(i + 1, len(rec_q)):
                rec_qj = Q[rec_q[j]]
                norm_j = np.linalg.norm(rec_qj)
                if norm_i == 0 or norm_j == 0:
                    continue
                different += 1 - np.dot(rec_qi, rec_qj) / (norm_i * norm_j)     
                valid_pairs += 1
        if len(rec_q) > 1:
            if valid_pairs > 0:
                diversity.append(different / valid_pairs)
            else:
                diversity.append(0)
    return np.mean(diversity), np.std(diversity)

def volatility(pkm, Q, rec_Q, test_raw):
    vol = []
    for i in range(len(test_raw)):
        uid = test_raw.iloc[i]['uid']
        if uid not in rec_Q:
            continue
        pkm_i = pkm[i]
        rec_qs = rec_Q[uid]
        if len(rec_qs) < 2:
            vol.append(0.0)
            continue
            
        vol_i = 0
        for j in range(len(rec_qs) - 1):
            q1 = rec_qs[j]
            q2 = rec_qs[j + 1]
            
            idx1 = np.where(Q[q1] == 1)[0]
            prob1 = np.prod([pkm_i[k] for k in idx1])
            
            idx2 = np.where(Q[q2] == 1)[0]
            prob2 = np.prod([pkm_i[k] for k in idx2])
            
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

if __name__ == '__main__':
    dataset = os.environ.get('UNIER_DATASET', 'assist2017')
    test_raw = pd.read_csv(f'./data/{dataset}/test.csv')
    pkm = torch.load(f'./data/{dataset}/pkm.pth',weights_only=True)
    Q = np.load(f'./data/{dataset}/Q.npy')
    pkm = pkm.to('cpu').numpy()
    rec_Q = {}
    with open(f'./data/{dataset}/recommended_problems.txt', 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            uid = int(line[0])
            rec_Q[uid] = np.array([int(q) for q in line[1].split(',')])
    acc, acc_std = accuracy(pkm, Q, rec_Q, test_raw)
    print('dataset:', dataset)
    print(f"acc\tmean:{acc:.3f}\tstd:{acc_std:.3f}")
    nov, nov_std = novlty(test_raw, Q, rec_Q)
    print(f"nov\tmean:{nov:.3f}\tstd:{nov_std:.3f}")
    div, div_std = diversity(Q, rec_Q)
    print(f"div\tmean:{div:.3f}\tstd:{div_std:.3f}")
    vol, vol_std = volatility(pkm, Q, rec_Q, test_raw)
    print(f"vol\tmean:{vol:.3f}\tstd:{vol_std:.3f}")
    cov, cov_std = coverage(test_raw, Q, rec_Q)
    print(f"cov\tmean:{cov:.3f}\tstd:{cov_std:.3f}")
