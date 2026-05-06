import random
import pandas as pd
import numpy as np
import torch
import ast
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import torch.nn as nn


def ACC(uid_mlkc_dict, uid_ex_scores, Q, r1, n):
    acc = []
    for item in uid_ex_scores:
        uid, scores = item[0], item[1]
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        uid_ex_score = [item[0] for item in sorted_scores][:n]
        user_mlkc = uid_mlkc_dict[uid]
        diff = 0
        for ex_id in uid_ex_score:
            kc_list = np.where(Q[ex_id] == 1)[0]
            ex_ml = 1.0
            for kc in kc_list:
                ex_ml = ex_ml * float(user_mlkc['kc' + str(kc)][4:])
            diff += 1 - np.abs(r1 - (ex_ml))
        acc.append(diff / n)
    return np.mean(acc), np.std(acc)

def Nov(uid_kc_response, uid_ex_scores, Q, n):
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
def preprocess_test_data(test_data):
    stu_true_response = {}
    for i in range(test_data.shape[0]):
        uid = test_data.iloc[i]['uid']
        kcs = [int(kc) for kc in test_data.iloc[i]['concepts'].split(',')]
        responses = [int(r) for r in test_data.iloc[i]['responses'].split(',')]
        kc_last_response = {}
        for kc, r in zip(kcs, responses):
            kc_last_response[kc] = r
        stu_true_response[uid] = kc_last_response

    return stu_true_response


def preprocess_stu_rec_ex(stu_rec_ex, Q_matrix, stu_true_response):
    stu_rec_weak_kc = {}
    for uid, rec_ex in stu_rec_ex.items():
        if uid not in stu_true_response:
            continue
        had_done_kc = stu_true_response[uid]
        rec_weak_kc = []
        seen_kc = set()

        for qid in rec_ex:
            kc_list = np.where(Q_matrix[qid, :] == 1)[0].tolist()
            for kc in kc_list:
                if kc not in seen_kc and kc in had_done_kc:
                    rec_weak_kc.append(kc)
                    seen_kc.add(kc)

        stu_rec_weak_kc[uid] = rec_weak_kc

    return stu_rec_weak_kc

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


def calculate_metrics_ndcg(stu_true_response, stu_rec_weak_kc, k=1):
    hit, ndcg_list, f1 = 0, [], 0
    hits = []
    valid_test_stu_num = 0

    for uid in stu_rec_weak_kc:
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

        t = [1] * len(hit_list)
        f1 += f1_score(t, hit_list)

    hit = hit / valid_test_stu_num
    ndcg = np.mean(ndcg_list)
    f1 = f1 / valid_test_stu_num

    return hit, f1, ndcg, valid_test_stu_num


def calculate_metrics_acc(predictions, targets, mask, concepts, kc_num, is_pkc):
    predictions = (predictions * nn.functional.one_hot(concepts.to(torch.long), kc_num)).sum(-1)
    predictions = torch.masked_select(predictions, mask.to(torch.bool))

    targets = torch.masked_select(targets, mask.to(torch.bool))

    if is_pkc == False:
        auc = roc_auc_score(targets.detach().numpy(), predictions.detach().numpy())
    else:
        auc = 0

    predict_label = (predictions > 0.5).long().numpy()
    acc = accuracy_score(targets.numpy(), predict_label)
    return (auc, acc)




if __name__ == '__main__':
    dataset = 'assist2009'
    test_file = f'./datasets/{dataset}/test_300/test_300.csv'
    REL_generator_file = f'./datasets/{dataset}/test_300/REL_generator_result.txt'
    Q_matrix = np.load(f'./datasets/{dataset}/test_300/Q.npy')

    test_data = pd.read_csv(test_file)
    stu_true_response = preprocess_test_data(test_data)

    stu_rec_ex = {}
    with open(REL_generator_file, 'r') as f:
        for line in f.readlines():
            stu_id, rec_ex = line.strip().split('\t')
            rec_ex = [int(qid) for qid in rec_ex.split(',')]
            stu_rec_ex[int(stu_id)] = rec_ex

    stu_rec_weak_kc = preprocess_stu_rec_ex(stu_rec_ex, Q_matrix, stu_true_response)


    for k in [1, 3, 5, 10]:
        hit, ndcg, f1, valid_test_stu_num = calculate_metrics_ndcg(stu_true_response, stu_rec_weak_kc, k=k)
        print(f'k = {k}, ndcg: {ndcg:.3f}, f1: {f1:.3f}, hit: {hit:.3f}, valid_test_stu_num: {valid_test_stu_num}')
