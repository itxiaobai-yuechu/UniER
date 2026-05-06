
import random
import pandas as pd
import numpy as np
import torch
import ast
from sklearn.metrics import f1_score,recall_score



def preprocess_test_data(test_data):
    stu_true_response = {}
    test_data['uid'] = test_data['orirow'] if 'orirow' in test_data.columns else [i for i in range(test_data.shape[0])]
    for i in range(test_data.shape[0]):
        uid = test_data.iloc[i]['uid']
        if uid < 200:
            concepts = int(test_data.iloc[i]['concepts'])
            late_true = int(test_data.iloc[i]['late_trues'])
            if uid not in stu_true_response:
                stu_true_response[uid] = {}
            stu_true_response[uid][concepts] = late_true
    return stu_true_response


def preprocess_stu_rec_ex(stu_rec_ex, Q_matrix, stu_true_response):
    stu_rec_weak_kc = {}
    for uid in stu_rec_ex:
        had_done_kc = stu_true_response[uid]
        rec_ex = stu_rec_ex[uid]
        rec_kc, rec_weak_kc = [], []
        for qid in rec_ex:
            kc = np.where(Q_matrix[qid, :] == 1)[0].tolist()
            rec_kc.extend(kc)
        for kc in rec_kc:
            if kc not in rec_weak_kc and kc in had_done_kc:
                rec_weak_kc.append(kc)
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


def calculate_metrics(stu_true_response, stu_rec_weak_kc, k=1):
    hit, ndcg_list, f1 ,recall = 0, [], 0 , 0
    mrr_list, ap_list = [], []
    hits = []
    valid_test_stu_num = 0
    for uid in stu_true_response:
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

        t = [1] * len(hit_list)
        f1 += f1_score(t, hit_list)
        recall += recall_score(t, hit_list)
        
    hit = hit / valid_test_stu_num
    ndcg = np.mean(ndcg_list)
    mrr = np.mean(mrr_list)
    map_score = np.mean(ap_list)
    f1 = f1 / valid_test_stu_num
    recall = recall / valid_test_stu_num
    print(f'k = {k}, ndcg len = {len(ndcg_list)}')
    return hit, ndcg, map_score, mrr, f1, recall, valid_test_stu_num


if __name__ == '__main__':
    dataset = 'xes3g5m'
    print(f"dataset: {dataset}")

    test_file = f'../PYKT/data/{dataset}/qid_test_question_predictions1.txt'
    REL_generator_file = f'./{dataset}/new_ex.txt'
    Q_matrix = np.load(f'./{dataset}/Q.npy')
    test_data = pd.read_csv(test_file,sep='\t')

    stu_true_response = preprocess_test_data(test_data)
    stu_rec_ex = {}
    with open(REL_generator_file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            stu_id, rec_ex = line.strip().split('\t')
            rec_ex = [int(qid) for qid in rec_ex.split(',')]
            stu_rec_ex[i] = rec_ex
    stu_rec_weak_kc = preprocess_stu_rec_ex(stu_rec_ex, Q_matrix, stu_true_response)
    for k in [1, 3, 5, 10]:
        hit, ndcg, map_score, mrr, f1, recall, valid_test_stu_num = calculate_metrics(stu_true_response, stu_rec_weak_kc, k=k)
        print(f'k = {k:2d} | ndcg: {ndcg:.3f}, map: {map_score:.3f}, mrr: {mrr:.3f}, f1: {f1:.3f}, hit: {hit:.3f}, recall: {recall:.3f}, valid_stu: {valid_test_stu_num}')