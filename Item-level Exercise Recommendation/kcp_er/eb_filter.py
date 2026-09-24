import os
import numpy as np
import pandas as pd
import torch
import random
from tqdm import tqdm


def get_pkc(test_raw, pkc_300, kc_num):
    device = pkc_300.device
    pkc_300_final = []
    for i in range(len(test_raw)):
        pkc = pkc_300[i]
        kc_cnt = {i: 0. for i in range(kc_num)}
        kc_yes = {i: 0. for i in range(kc_num)}
        kcs = [int(kc) for kc in test_raw.iloc[i]['concepts'].split(',')]
        responses = [int(res) for res in test_raw.iloc[i]['responses'].split(',')]
        try:
            index = kcs.index(-1)
        except:
            index = len(kcs)
        kcs = kcs[:index]
        responses = responses[:index]
        for index, kc in enumerate(kcs):
            kc_cnt[kc] += 1.
            if responses[index] == 1:
                kc_yes[kc] += 1.
        w_kc = {}
        for kc in kc_cnt.keys():
            if kc_cnt[kc] == 0.:
                w_kc[kc] = 1.
            else:
                w_kc[kc] = 1 - kc_yes[kc] / kc_cnt[kc]
        w_k = torch.tensor([value for value in w_kc.values()]).to(device)
        pkc_300_final.append(pkc * w_k)
    return torch.stack(pkc_300_final)


def EB_filter(Q, pkm, pkc, dielta=0.7, N=150):
    ES = []
    student_num = pkm.shape[0]
    N = min(N, Q.shape[0])
    for i in tqdm(range(student_num)):
        pkm_i = pkm[i].to('cpu').numpy()
        pkc_i = pkc[i].to('cpu').numpy()
        Q_score = []
        is_select = []
        cnt = 0
        while cnt < N:
            j = random.randint(0, Q.shape[0] - 1)
            if j in is_select:
                continue
            cnt += 1
            is_select.append(j)
            denominator = np.linalg.norm(pkc_i) * np.linalg.norm(Q[j])
            cos_sim = 0.0 if denominator == 0 else np.dot(pkc_i, Q[j]) / denominator
            index_1 = np.where(Q[j] == 1)[0]
            select_pkm_i = [pkm_i[k] for k in index_1]
            dis = dielta - np.prod(select_pkm_i)
            Omega = np.sqrt(cos_sim ** 2 + dis ** 2)

            Q_score.append((j, Omega))

        Q_sort_score = sorted(Q_score, key=lambda x: x[1])
        Q_sort = [i[0] for i in Q_sort_score]
        ES.append(Q_sort)
        
    return ES


if __name__ == '__main__':
    dataset = os.environ.get('UNIER_DATASET', 'assist2017')
    data_dir = f'datasets/{dataset}/select/selected'
    kc_num = {'assist2009':123, 'nips34':57, 'assist2012':265,'assist2017':102,'algebra2005':112,'bridge2006':493,'ednet':188,'junyi':39,'xes3g5m':865}
    test_raw = pd.read_csv(f'{data_dir}/test_sequences.csv')
    pkc_300 = torch.load(f'{data_dir}/pkc.pth')
    pkc_300_new = get_pkc(test_raw, pkc_300, kc_num=kc_num[dataset])
    torch.save(pkc_300_new, f'{data_dir}/pkc_final.pth')

    pkm = torch.load(f'{data_dir}/pkm.pth')
    pkc = torch.load(f'{data_dir}/pkc_final.pth')
    Q = np.load(f'{data_dir}/Q.npy')
    random.seed(int(os.environ.get('UNIER_SEED', '42')))
    print(f"start EB-filter:")
    N = 150
    ES = EB_filter(Q, pkm, pkc, dielta=0.7, N=N)

    test_raw = pd.read_csv(f'{data_dir}/test_sequences.csv')
    uid = test_raw['uid'].tolist()

    with open(f'{data_dir}/EB_filter_{N}.txt', 'w') as f:
        for i in range(len(ES)):
            es = ES[i]
            filter_q = ','.join([str(q) for q in es])
            f.write(f"{uid[i]}\t{filter_q}\n")
