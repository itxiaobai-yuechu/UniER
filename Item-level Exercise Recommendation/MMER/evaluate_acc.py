import torch
import numpy as np
import pandas as pd
import statistics
import os


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
            for j in range(i+1, len(rec_q)):
                rec_qj = Q[rec_q[j]]
                different += 1 - np.dot(rec_qi, rec_qj) / (np.linalg.norm(rec_qi) * np.linalg.norm(rec_qj))
        diversity.append((2 * different) / (len(rec_Q[0]) * (len(rec_Q[0]) - 1)))
    return np.mean(diversity), np.std(diversity)

def volatility(pkm, Q, rec_Q):
    student_num = pkm.shape[0]
    vol = []
    for i in range(student_num):
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
            prob1 = np.prod([pkm_i[k] for k in idx1])
            idx2 = np.where(Q[q2] == 1)[0]
            prob2 = np.prod([pkm_i[k] for k in idx2])
            vol_i += np.abs(prob1 - prob2)  
        vol.append(vol_i / len(rec_qs))
    return 1-np.mean(vol), np.std(vol)


class ProximityCalculator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_assist_data(self, csv_path):

        df = pd.read_csv(csv_path)

        uid_col = None
        for c in df.columns:
            low = c.lower()
            if low in ('uid', 'user_id', 'userid', 'user', 'student_id', 'student', 'username', 'user_name'):
                uid_col = c
                break

        if uid_col is None:
            index_user_ids = [f'row_{i}' for i in range(len(df))]
        else:
            index_user_ids = df[uid_col].astype(str).tolist()

        self.index_to_user = index_user_ids
        self.user_to_index = {uid: idx for idx, uid in enumerate(index_user_ids)}
        self.total_csv_users = len(index_user_ids)

        user_list = []
        problem_list = []
        correct_list = []

        for row_idx, row in df.iterrows():
            uid_orig = index_user_ids[row_idx]
            uid = self.user_to_index[uid_orig]
            qs = []
            rs = []
            if 'questions' in row and not pd.isna(row['questions']):
                qs = list(map(int, str(row['questions']).strip().split(',')))
            if 'responses' in row and not pd.isna(row['responses']):
                rs = list(map(int, str(row['responses']).strip().split(',')))
            for q, r in zip(qs, rs):
                user_list.append(uid)
                problem_list.append(q)
                correct_list.append(r)

        data = pd.DataFrame({
            'user_id': user_list,
            'problem_id': problem_list,
            'correct': correct_list
        })
        return data

    def load_recommend_dict(self, rec_path):

        rec_dict = {}
        with open(rec_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) == 0:
                    continue
                uid_str = parts[0].strip()
                prob_str = parts[1].strip() if len(parts) > 1 else ''
                if prob_str == '':
                    prob_list = []
                else:
                    prob_list = [int(x) for x in prob_str.split(',') if x.strip() != '']
                rec_dict[uid_str] = prob_list
        return rec_dict

    def compute_de_ds(self, data, max_problem_id, total_csv_users):

        device = self.device
        E_num = max_problem_id + 1
        S_num = total_csv_users

        user_id = torch.tensor(data['user_id'].values, device=device, dtype=torch.long)
        problem_id = torch.tensor(data['problem_id'].values, device=device, dtype=torch.long)
        correct = torch.tensor(data['correct'].values, device=device, dtype=torch.float32)

        SF = torch.zeros(E_num, device=device)
        num = torch.zeros(E_num, device=device)
        st = torch.zeros((S_num, E_num), device=device, dtype=torch.bool)
        ET = [set() for _ in range(S_num)]

        for si, ej, ans in zip(user_id, problem_id, correct):
            if not st[si, ej]:
                num[ej] += 1
                st[si, ej] = True
                if ans == 0:
                    SF[ej] += 1
            if ans == 1:
                ET[si.item()].add(ej.item())

        de = torch.where(num != 0, SF / num, torch.zeros_like(SF))
        ds = torch.zeros(S_num, device=device)
        for i in range(S_num):
            et = list(ET[i])
            if et:
                ds[i] = de[torch.tensor(et, device=device)].mean()
        return de, ds

    def calculate_final_proximity(self, csv_path, rec_path):
        data = self.load_assist_data(csv_path)
        max_problem_id = data['problem_id'].max()
        de, ds = self.compute_de_ds(data, max_problem_id, self.total_csv_users)

        user_proximity_list = []
        with open(rec_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(maxsplit=1)
                if len(parts) < 2:
                    continue

                uid_str = parts[0].strip()
                q_str = parts[1].strip()
                if not q_str:
                    continue

                rec_probs = [int(q) for q in q_str.split(',') if q.strip() != '']
                if not rec_probs:
                    continue

                if uid_str in self.user_to_index:
                    uid = self.user_to_index[uid_str]
                else:
                    try:
                        n = int(uid_str)
                    except Exception:
                        continue

                    if 0 <= n < self.total_csv_users:
                        uid = n
                    else:
                        row_key = f'row_{n}'
                        if row_key in self.user_to_index:
                            uid = self.user_to_index[row_key]
                        else:
                            continue

                sum_de = torch.sum(de[torch.tensor(rec_probs, device=self.device)])
                avg_de = sum_de / len(rec_probs)
                user_prox = (1.0 - torch.abs(avg_de - ds[uid])).item()

                user_proximity_list.append(user_prox)

        if len(user_proximity_list) >= 2:
            mean_prox = sum(user_proximity_list) / len(user_proximity_list)
            std_prox = statistics.stdev(user_proximity_list)
        elif len(user_proximity_list) == 1:
            mean_prox = user_proximity_list[0]
            std_prox = 0.0
        else:
            mean_prox = 0.0
            std_prox = 0.0

        return mean_prox, std_prox

if __name__ == '__main__':
    dataset = os.environ.get('UNIER_DATASET', 'assist2017')
    file_dir = f'./datasets/{dataset}'
    test_raw = pd.read_csv(f'{file_dir}/test.csv')
    pkm = torch.load(f'{file_dir}/pkm.pth')
    pkm = pkm.to('cpu').numpy()
    Q = np.load(f'{file_dir}/Q.npy')
    rec_Q = []
    with open(f'{file_dir}/score_rec_exer_topk_stu_all.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) < 2 or parts[1].strip() == '':
                continue
            q_str = parts[1].strip()
            arr = np.array([int(q) for q in q_str.split(',') if q.strip() != ''], dtype=int)
            if arr.size == 0:
                continue
            rec_Q.append(arr)

    metric_n = min(pkm.shape[0], len(test_raw), len(rec_Q))
    pkm_metric = pkm[:metric_n]
    test_raw_metric = test_raw.iloc[:metric_n].reset_index(drop=True)
    rec_Q_metric = rec_Q[:metric_n]

    acc, acc_std = accuracy(pkm_metric, Q, rec_Q_metric)
    nov, nov_std = novlty(test_raw_metric, Q, rec_Q_metric)
    div, div_std = diversity(Q, rec_Q_metric)
    vol, vol_std = volatility(pkm_metric, Q, rec_Q_metric)
    print(f"acc\tmean:{acc:.3f}\tstd:{acc_std:.3f}")
    print(f"nov\tmean:{nov:.3f}\tstd:{nov_std:.3f}")
    print(f"div\tmean:{div:.3f}\tstd:{div_std:.3f}")
    print(f"vol\tmean:{vol:.3f}\tstd:{vol_std:.3f}")

    calc = ProximityCalculator()
    csv_file = f"./datasets/{dataset}/test.csv"
    rec_file = f"./datasets/{dataset}/score_rec_exer_topk_stu_all.txt"
    mean_prox, std_prox = calc.calculate_final_proximity(csv_file, rec_file)
    print(f"pro\tmean:{mean_prox:.3f}\tstd:{std_prox:.3f}")
