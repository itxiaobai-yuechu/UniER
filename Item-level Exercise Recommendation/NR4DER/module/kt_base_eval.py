import os
import random
import pandas as pd
import numpy as np
import torch
import ast
from sklearn.metrics import f1_score, recall_score


def preprocess_test_data(test_data):
    stu_true_response = {}
    test_data['uid'] = [i for i in range(test_data.shape[0])]
    for i in range(test_data.shape[0]):
        kcs = [int(kc) for kc in test_data.iloc[i]['concepts'].split(',')]
        responses = [int(r) for r in test_data.iloc[i]['responses'].split(',')]

        try:
            index = kcs.index(-1)
        except:
            index = len(kcs)
        kcs = kcs[:index]
        responses = responses[:index]
        kc_last_response = {}
        for kc, r in zip(kcs, responses):
            kc_last_response[kc] = r
        stu_true_response[i] = kc_last_response

    return stu_true_response


def ndcg_at_k(hits, k):
    if len(hits) < k:
        hits.extend([0] * (k - len(hits)))

    dcg = np.sum((2 ** np.array(hits[:k]) - 1) / np.log2(np.arange(2, k + 2)))

    sorted_hits = sorted(hits, reverse=True)
    idcg = np.sum((2 ** np.array(sorted_hits[:k]) - 1) / np.log2(np.arange(2, k + 2)))

    if idcg == 0:
        return 0.0

    ndcg = dcg / idcg
    return ndcg


def calculate_metrics(stu_true_response, stu_ks, k=1):
    hit, ndcg_list, f1, recall, mrr = 0, [], 0, 0, 0

    test_stu_num = len(stu_ks)
    valid_stu_num = 0

    for i in range(test_stu_num):
        stu_kc_level = stu_ks[i]
        kc_true_score = {kc: 1 - stu_true_response[i][kc] for kc in stu_true_response[i]}
        if sum(kc_true_score.values()) == 0:
            continue
        rank_kc = [kc for kc, _ in sorted(stu_kc_level.items(), key=lambda item: item[1])
                   if kc in kc_true_score][:k]
        if not rank_kc:
            continue
        valid_stu_num += 1

        hit_list = [kc_true_score[kc] for kc in rank_kc]
        if len(hit_list) < k:
            hit_list.extend([0] * (k - len(hit_list)))

        hit += sum(hit_list) / len(hit_list)

        temp_ndcg = ndcg_at_k(hit_list, k)
        if temp_ndcg != -1:
            ndcg_list.append(temp_ndcg)

        t = [1] * len(hit_list)
        f1 += f1_score(t, hit_list, zero_division=0)
        recall += recall_score(t, hit_list, zero_division=0)

        for rank, hit_num in enumerate(hit_list):
            if hit_num == 1:
                mrr += 1 / (rank + 1)
                break


    if valid_stu_num == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    hit = hit / valid_stu_num
    ndcg = np.mean(ndcg_list) if ndcg_list else 0.0
    f1 = f1 / valid_stu_num
    recall = recall / valid_stu_num
    mrr = mrr / valid_stu_num

    print(f'k = {k}, ndcg len = {len(ndcg_list)}, valid students = {valid_stu_num}/{test_stu_num}')

    return hit, ndcg, f1, recall, mrr


if __name__ == '__main__':
    dataset = os.environ.get('UNIER_DATASET', 'assist2017')
    model = 'dkt'

    test_file = f'./dataset/{dataset}/test_sequences.csv'
    stu_kc_file = f'./model/{model}/{dataset}/qid_test_predictions.txt'

    stu_ks = {}
    error_index = []
    with open(stu_kc_file, 'r') as f:
        for i, line in enumerate(f):
            try:
                stu_info = ast.literal_eval(line.strip())
            except:
                error_index.append(i)
            kcs = stu_info[2]
            kcs_predict = stu_info[4]
            kc_last_pre = {}
            for kc, pre in zip(kcs, kcs_predict):
                kc_last_pre[kc] = pre
            stu_ks[i] = kc_last_pre

    test_data = pd.read_csv(test_file)

    if error_index != []:
        for i in error_index:
            del stu_ks[i]
        test_data = test_data.drop(error_index)

        stu_ks_new = {}
        for i, key in enumerate(stu_ks):
            stu_ks_new[i] = stu_ks[key]
        stu_ks = stu_ks_new


    stu_true_response = preprocess_test_data(test_data)


    print(error_index)
    print(f'Test model is {model}, dataset is {dataset}!!!')
    for k in [10]:
        hit, ndcg, f1, recall, mrr = calculate_metrics(stu_true_response, stu_ks, k=k)
        print(f'k = {k}, ndcg: {ndcg:.3f}, f1: {f1:.3f}, recall: {recall:.3f}, mrr: {mrr:.3f}')
