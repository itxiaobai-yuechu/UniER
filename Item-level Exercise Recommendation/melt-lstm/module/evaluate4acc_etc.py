import numpy as np
import pandas as pd
import torch
import random

def accuracy(pkm, Q, rec_Q, dielta=0.7):
    student_num = pkm.shape[0]
    acc = []
    for i in range(student_num):
        pkm_i = pkm[i]
        rec_qs = rec_Q[i]
        D_qk = 0
        for q in rec_qs:
            index_1 = np.where(Q[q] == 1)[0]
            select_pkm_i = [pkm_i[k] for k in index_1]
            dis = dielta - np.prod(select_pkm_i)
            D_qk += 1 - np.abs(dis)
        acc.append(D_qk / len(rec_qs))
    return np.mean(acc), np.std(acc)


def novlty(test_raw, Q, rec_Q):
    student_num = len(test_raw)
    nov = []
    kc_responses = test_raw[['concepts', 'responses']]
    for i in range(student_num):
        kc_res_yes_set = set()
        kcs = kc_responses.iloc[i]['concepts'].split(',')
        res = kc_responses.iloc[i]['responses'].split(',')
        for kc, res in zip(kcs, res):
            if res == '1':
                kc_res_yes_set.add(int(kc))
        rec_qs = rec_Q[i]
        dis = 0
        for q in rec_qs:
            index_1 = np.where(Q[q] == 1)[0]
            index_1 = set(index_1)
            union_len = len(kc_res_yes_set.union(index_1))
            if union_len == 0:
                continue  
            dis += 1 - len(kc_res_yes_set.intersection(index_1)) / union_len
        nov.append(dis / len(rec_qs))
    return np.mean(nov), np.std(nov)


def diversity(Q, rec_Q):
    student_num = len(rec_Q)
    diversity = []
    for k in range(student_num):
        different = 0
        rec_q = rec_Q[k]
        for i in range(len(rec_q)):
            rec_qi = Q[rec_q[i]]
            for j in range(i + 1, len(rec_q)):
                rec_qj = Q[rec_q[j]]
                norm_i = np.linalg.norm(rec_qi)
                norm_j = np.linalg.norm(rec_qj)
                if norm_i == 0 or norm_j == 0:
                    continue
                different += 1 - np.dot(rec_qi, rec_qj) / (norm_i * norm_j)     
        if len(rec_q) > 1:
            diversity.append((2 * different) / (len(rec_q) * (len(rec_q) - 1)))
    return np.mean(diversity), np.std(diversity)



def volatility(pkm, Q, rec_Q, test_raw):
    vol = []
    for i in range(len(test_raw)):
        pkm_i = pkm[i]
        rec_qs = rec_Q[i]
        if len(rec_qs) < 2:
            vol.append(0.0)
            continue
            
        vol_i = 0
        for j in range(len(rec_qs) - 1):
            q1 = rec_qs[j]
            q2 = rec_qs[j + 1]
            
            idx1 = np.where(Q[q1] == 1)[0]
            prob1 = np.prod([pkm_i[k] for k in idx1]) if len(idx1) > 0 else 1.0
            
            idx2 = np.where(Q[q2] == 1)[0]
            prob2 = np.prod([pkm_i[k] for k in idx2]) if len(idx2) > 0 else 1.0
            
            vol_i += np.abs(prob1 - prob2)
            
        vol.append(vol_i / len(rec_qs))
    if len(vol) == 0:
        return 0.0, 0.0
    return 1-np.mean(vol), np.std(vol)


def coverage(test_raw, Q, rec_Q):

    cov = []
    
    for i in range(len(test_raw)):
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
            
        rec_qs = rec_Q[i]
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
    dataset = 'junyi'
    print(f"dataset: {dataset}")
    test_raw = pd.read_csv(f'../dataset/data_200/{dataset}/test_sequences.csv')

    pkm = torch.load(f'../dataset/data_200/{dataset}/pkm.pth',weights_only=True)

    Q = np.load(f'../dataset/data_200/{dataset}/Q.npy')

    pkm = pkm.to('cpu').numpy()
    rec_Q = []
    with open(f'../dataset/data_200/{dataset}/Reranking_result_melt.txt', 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            rec_Q.append(np.array([int(q) for q in line[1].split(',')]))
    acc, acc_std = accuracy(pkm, Q, rec_Q)
    print(f'dataset: {dataset}')
    print(f"acc\tmean:{acc:.3f}\tstd:{acc_std:.3f}")
    nov, nov_std = novlty(test_raw, Q, rec_Q)
    print(f"nov\tmean:{nov:.3f}\tstd:{nov_std:.3f}")
    div, div_std = diversity(Q, rec_Q)
    print(f"div\tmean:{div:.3f}\tstd:{div_std:.3f}")
    vol, vol_std = volatility(pkm, Q, rec_Q, test_raw)
    print(f"vol\tmean:{vol:.3f}\tstd:{vol_std:.3f}")
    cov, cov_std = coverage(test_raw, Q, rec_Q)
    print(f"cov\tmean:{cov:.3f}\tstd:{cov_std:.3f}")