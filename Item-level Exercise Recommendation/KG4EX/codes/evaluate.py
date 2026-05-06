import json
import numpy as np
import pandas as pd
import utils as eval
import csv

dataset = "xes3g5m"
dict_path = f"../data/{dataset}"
with open(f"{dict_path}/uid_ex_scores.json", "r", encoding="utf-8") as json_file:
    uid_ex_scores = json.load(json_file)
uid_ex_scores = [(item["uid"], item["scores"]) for item in uid_ex_scores]

kg_test_path = f"{dict_path}/Test20/kg_test.txt"
uid_mlkc_dict = {}
with open(kg_test_path, 'r', encoding="UTF-8") as load_file:
    for line in load_file:
        item1, item2, uid = line.strip().split('\t')
        print("item1:",f'{item1}')
        print("item2:",f'{item1}')
        print("uid:",f'{uid}')
        if item2[0] == 'm':
            kc, mlkc, uid = item1, item2, uid
            if uid not in uid_mlkc_dict.keys():
                uid_mlkc_dict[uid] = {}
            uid_mlkc_dict[uid][kc] = 'mlkc' + str(round(float(mlkc[4:]), 2))

test_data_path = f"{dict_path}/Test20.csv"
test_data = pd.read_csv(test_data_path)

Q = np.load(f'{dict_path}/Q.npy')

def ACC(uid_mlkc_dict, uid_ex_scores, Q, r1, n):
    acc = []
    for item in uid_ex_scores:
        uid, scores = item[0], item[1]
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        uid_ex_score = [item[0] for item in sorted_scores][:n]
        user_mlkc = uid_mlkc_dict[uid]
        diff = 0
        for ex_id in uid_ex_score:
            kc_list = [index for index, value in enumerate(Q[ex_id]) if value == 1]
            ex_ml = 1.0
            for kc in kc_list:
                ex_ml = ex_ml * float(user_mlkc['kc' + str(kc)][4:])
            diff += 1 - np.abs(r1 - (ex_ml))
        acc.append(diff / n)
    return np.mean(acc), np.std(acc)

def NOV(uid_kc_response, uid_ex_scores, Q, n):
    jaccsim = []
    for item in uid_ex_scores:
        uid, scores = item[0], item[1]
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        uid_ex_score = [item[0] for item in sorted_scores][:n]
        kc_response = set(uid_kc_response[uid])
        jaccard_similarity = 0
        for ex_id in uid_ex_score:
            rec_ex_kc_set = set()
            kc_list = [index for index, value in enumerate(Q[ex_id]) if value == 1]
            rec_ex_kc_set.update(kc_list)
            intersection = len(kc_response.intersection(rec_ex_kc_set))
            union = len(kc_response.union(rec_ex_kc_set))
            jaccard_similarity += 1 - intersection / union
        jaccsim.append(jaccard_similarity / n)
    return np.mean(jaccsim), np.std(jaccsim)

def DIV(uid_ex_scores, Q, n):
    diversity_scores = []
    for item in uid_ex_scores:
        uid, scores = item[0], item[1]
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        uid_ex_score = [item[0] for item in sorted_scores][:n]
        different = 0
        for i in range(len(uid_ex_score)):
            rec_qi = Q[uid_ex_score[i]]
            for j in range(i + 1, len(uid_ex_score)):
                rec_qj = Q[uid_ex_score[j]]
                norm_i = np.linalg.norm(rec_qi)
                norm_j = np.linalg.norm(rec_qj)
                if norm_i == 0 or norm_j == 0:
                    continue
                different += 1 - np.dot(rec_qi, rec_qj) / (norm_i * norm_j)
        
        if len(uid_ex_score) > 1:
            diversity_scores.append((2 * different) / (len(uid_ex_score) * (len(uid_ex_score) - 1)))
    
    if len(diversity_scores) == 0:
        return 0.0, 0.0
    return np.mean(diversity_scores), np.std(diversity_scores)

def diversity(rec_Q, Q, k):
    div_scores = []
    for uid, rec_qs in rec_Q.items():
        rec_qs = rec_qs[:k]
        if len(rec_qs) < 2:
            continue
        similarities = []
        for i in range(len(rec_qs)):
            for j in range(i+1, len(rec_qs)):
                q1 = rec_qs[i]
                q2 = rec_qs[j]
                vec1 = Q[q1]
                vec2 = Q[q2]
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                if norm1 == 0 or norm2 == 0:
                    sim = 0
                else:
                    sim = np.dot(vec1, vec2) / (norm1 * norm2)
                similarities.append(sim)
        if similarities:
            avg_sim = np.mean(similarities)
            div = 1 - avg_sim
            div_scores.append(div)
    return np.mean(div_scores) if div_scores else 0.0

def ndcg_at_k(hits, k):
    if len(hits) < k:
        return -1
    dcg = np.sum((2 ** np.array(hits[:k]) - 1) / np.log2(np.arange(2, k + 2)))
    sorted_hits = sorted(hits, reverse=True)
    idcg = np.sum((2 ** np.array(sorted_hits[:k]) - 1) / np.log2(np.arange(2, k + 2)))
    if idcg == 0:
        return 0.0
    ndcg = dcg / idcg
    return ndcg

def calculate_metrics(stu_true_response, stu_rec_weak_kc, stu_rec_ex, Q_matrix, k=1):
    hit, ndcg_list = 0, []
    precision_sum, recall_sum, f1_sum = 0.0, 0.0, 0.0
    mrr_list, ap_list = [], []
    hits = []
    valid_test_stu_num = 0

    div = diversity(stu_rec_ex, Q_matrix, k)

    for uid, rec_ex in stu_rec_ex.items():
        if uid not in stu_true_response:
            continue
        kc_true_score = {kc: 1 - stu_true_response[uid][kc] for kc in stu_true_response[uid]}
        rank_kc = stu_rec_weak_kc[uid][:k]
        if len(rank_kc) == 0:
            continue
        valid_test_stu_num += 1

        hit_list = [kc_true_score[kc] for kc in rank_kc]
        hits.append(hit_list)
        hit += sum(hit_list) / len(hit_list)

        temp_ndcg = ndcg_at_k(hit_list, k)
        if temp_ndcg != -1:
            ndcg_list.append(temp_ndcg)

        rr = 0.0
        for idx, h in enumerate(hit_list):
            if h == 1:
                rr = 1.0 / (idx + 1)
                break
        mrr_list.append(rr)

        total_weak = sum(kc_true_score.values())
        if total_weak > 0:
            relevant = 0
            prec_sum = 0.0
            for idx, h in enumerate(hit_list):
                if h == 1:
                    relevant += 1
                    prec_sum += relevant / (idx + 1)
            ap_list.append(prec_sum / min(k, total_weak))
        else:
            ap_list.append(0.0)

        hits_count = sum(hit_list)
        len_rank = len(rank_kc)
        total_weak_stu = sum(kc_true_score.values())

        precision = hits_count / len_rank if len_rank > 0 else 0.0
        recall_stu = hits_count / total_weak_stu if total_weak_stu > 0 else 0.0
        if precision + recall_stu == 0:
            f1_stu = 0.0
        else:
            f1_stu = 2 * (precision * recall_stu) / (precision + recall_stu)

        precision_sum += precision
        recall_sum += recall_stu
        f1_sum += f1_stu

    hit = hit / valid_test_stu_num if valid_test_stu_num > 0 else 0.0
    ndcg = np.mean(ndcg_list) if len(ndcg_list) > 0 else 0.0
    mrr = np.mean(mrr_list) if len(mrr_list) > 0 else 0.0
    map_score = np.mean(ap_list) if len(ap_list) > 0 else 0.0
    precision = precision_sum / valid_test_stu_num if valid_test_stu_num > 0 else 0.0
    recall = recall_sum / valid_test_stu_num if valid_test_stu_num > 0 else 0.0
    f1 = f1_sum / valid_test_stu_num if valid_test_stu_num > 0 else 0.0

    print(f'k = {k}, ndcg len = {len(ndcg_list)}')
    return hit, ndcg, map_score, mrr, precision, f1, recall, div, valid_test_stu_num

def volatility(pkm, Q, rec_Q, test_raw):
    vol = []
    for i in range(len(test_raw)):
        uid = test_raw.iloc[i]['uid']
        if uid not in rec_Q:
            continue
        pkm_i = pkm.get(uid, [0.5] * Q.shape[1])
        rec_qs = rec_Q[uid]
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

u_rerank_list = []
n = 150
for item in uid_ex_scores:
    uid, scores = item[0], item[1]
    sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    uid_ex_score = [item[0] for item in sorted_scores][:n]
    u_rerank_list.append(uid_ex_score)
u_rerank_list = np.array(u_rerank_list)

rec_Q = {}
for i, item in enumerate(uid_ex_scores):
    uid = item[0]
    if isinstance(uid, str) and uid.startswith('uid'):
        uid = int(uid[3:])
    rec_Q[uid] = u_rerank_list[i]

stu_rec_ex = rec_Q


with open(f'{dict_path}/raw.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    uid_kc_response = {}
    for row in reader:
        uid = row['uid']
        concepts = row['concepts']
        concepts_list = list(set(concepts.split(',')))
        uid_kc_response[uid] = list(map(int, concepts_list))
with open(f'{dict_path}/uid_kc_response.txt', 'w') as f:
    for uid, concepts in uid_kc_response.items():
        f.write(f"uid{uid}\t{','.join(map(str, concepts))}\n")

stu_true_response = eval.preprocess_test_data(test_data)
stu_rec_weak_kc = eval.preprocess_stu_rec_ex(stu_rec_ex, Q, stu_true_response)

hits, f1s, ndcgs = {}, {}, {}
for k in [1, 3, 5, 10]:
    print(f'****************** Evaluating top{k} ******************')
    hit, f1, ndcg, _ = eval.calculate_metrics_ndcg(stu_true_response, stu_rec_weak_kc, k)
    hits[f'@{k}'], f1s[f'@{k}'], ndcgs[f'@{k}'] = round(hit, 4), round(f1, 4), round(ndcg, 4)
    print(f'NDCG@{k}: {ndcg}')
    print(f'HIT@{k}: {hit}')
    print(f'F1@{k}: {f1}')

print("-----------------------------------------------Start calculating ACC-----------------------------------------------")
r1 = 0.7
for n in [20]:
    mean_acc, std_acc = ACC(uid_mlkc_dict, uid_ex_scores, Q, r1, n)
    print(f"The recommendation list length is n = {n}, the mean ACC = {mean_acc}, the std ACC = {std_acc}")


print("-----------------------------------------------Start calculating NOV-----------------------------------------------")
all_uid_kc_response = {}
with open(f"{dict_path}/uid_kc_response.txt", 'r') as file:
    for line in file:
        line = line.strip().split('\t')
        uid = line[0]
        correct_kc_response = [int(x) for x in line[1].split(',')]
        all_uid_kc_response[uid] = correct_kc_response

test_uid_kc_response = {}
for uid in uid_mlkc_dict.keys():
    test_uid_kc_response[uid] = all_uid_kc_response[uid]

for n in [20]:
    mean_nov, std_nov = NOV(test_uid_kc_response, uid_ex_scores, Q, n)
    print(f"The recommendation list length is n = {n}, the mean NOV = {mean_nov}, the std NOV = {std_nov}")

print("-----------------------------------------------Start calculating DIV-----------------------------------------------")
for n in [20]:
    mean_div, std_div = DIV(uid_ex_scores, Q, n)
    print(f"The recommendation list length is n = {n}, the mean DIV = {mean_div}, the std DIV = {std_div}")

print("dataset:", dataset)
print("-----------------------------------------------Start calculating VOL-----------------------------------------------")
pkm_dict = {}
for uid in uid_mlkc_dict:
    uid_int = int(uid) if uid.isdigit() else uid
    pkm_dict[uid_int] = [float(uid_mlkc_dict[uid].get('kc' + str(k), 'mlkc0.5')[4:]) for k in range(Q.shape[1])]

vol_mean, vol_std = volatility(pkm_dict, Q, rec_Q, test_data)
print(f"Volatility: mean={vol_mean:.3f}, std={vol_std:.3f}")

print("-----------------------------------------------Start calculating COV-----------------------------------------------")
cov_mean, cov_std = coverage(test_data, Q, rec_Q)
print(f"Coverage: mean={cov_mean:.3f}, std={cov_std:.3f}")

print("-----------------------------------------------Start calculating comprehensive metrics-----------------------------------------------")
stu_rec_ex = rec_Q
for k in [1,3,5,10]:
    hit, ndcg, map_score, mrr, precision, f1, recall, div, valid_num = calculate_metrics(stu_true_response, stu_rec_weak_kc, stu_rec_ex, Q, k)
    print(f"k={k}, hit={hit:.3f}, ndcg={ndcg:.3f}, map={map_score:.3f}, mrr={mrr:.3f}, precision={precision:.3f}, f1={f1:.3f}, recall={recall:.3f}, div={div:.4f}, valid_users={valid_num}")