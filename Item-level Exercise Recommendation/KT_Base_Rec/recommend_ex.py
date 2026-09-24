import os
import random
import pandas as pd
import numpy as np
import torch
import ast
import argparse
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
    hit, ndcg_list = 0, []
    mrr_list, ap_list = [], []
    f1_sum, recall_sum = 0.0, 0.0

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
            ap = prec_sum / min(k, total_weak)
            ap_list.append(ap)
        else:
            ap_list.append(0.0)

        t = [1] * len(hit_list)
        f1_sum += f1_score(t, hit_list, zero_division=0)
        recall_sum += recall_score(t, hit_list, zero_division=0)

    if valid_stu_num == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    hit = hit / valid_stu_num
    ndcg = np.mean(ndcg_list) if ndcg_list else 0.0
    f1 = f1_sum / valid_stu_num
    recall = recall_sum / valid_stu_num
    mrr = np.mean(mrr_list) if len(mrr_list) > 0 else 0.0
    map_score = np.mean(ap_list) if len(ap_list) > 0 else 0.0

    print(f'k = {k}, ndcg len = {len(ndcg_list)}, valid students = {valid_stu_num}/{test_stu_num}')

    return hit, ndcg, f1, recall, mrr, map_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='assist2017')
    parser.add_argument('--model', default='akt', choices=['akt', 'simplekt'])
    parser.add_argument('--test_file', default=None)
    parser.add_argument('--prediction_file', default=None)
    args = parser.parse_args()

    dataset = os.environ.get('UNIER_DATASET', 'assist2017')
    model = {'AKT': 'akt', 'SimpleKT': 'simplekt'}.get(os.environ.get('UNIER_MODEL', 'AKT'), 'akt')

    test_file = args.test_file or f'./dataset/{dataset}/test_sequences.csv'
    stu_kc_file = args.prediction_file or f'./model/{model}/{dataset}/qid_test_predictions.txt'

    predictions_by_row = {}
    error_index = []
    with open(stu_kc_file, 'r') as f:
        for i, line in enumerate(f):
            try:
                stu_info = ast.literal_eval(line.strip())
            except (ValueError, SyntaxError):
                error_index.append(i)
                continue
            kcs = stu_info[2]
            kcs_predict = stu_info[4]
            kc_last_pre = {}
            for kc, pre in zip(kcs, kcs_predict):
                kc_last_pre[kc] = pre
            predictions_by_row[i] = kc_last_pre

    test_data = pd.read_csv(test_file)
    print(len(test_data))

    # Prediction line i belongs to CSV row i.  Keep that source-row identity
    # when malformed lines are skipped; compacting predictions independently
    # would silently assign every following prediction to the wrong student.
    valid_rows = sorted(i for i in predictions_by_row if i < len(test_data))
    extra_rows = sorted(i for i in predictions_by_row if i >= len(test_data))
    if not valid_rows:
        raise ValueError('No prediction lines could be aligned with the test CSV.')
    if error_index:
        print(f'Warning: skipped {len(error_index)} malformed prediction line(s): {error_index[:10]}')
    if extra_rows:
        print(f'Warning: ignored {len(extra_rows)} prediction line(s) beyond the test CSV.')
    if len(valid_rows) < len(test_data):
        print(f'Warning: evaluating {len(valid_rows)}/{len(test_data)} aligned test rows.')

    test_data = test_data.iloc[valid_rows].reset_index(drop=True)
    stu_ks = {new_i: predictions_by_row[source_i]
              for new_i, source_i in enumerate(valid_rows)}

    stu_true_response = preprocess_test_data(test_data)

    print(f'Test model is {model}, dataset is {dataset}!!!')
    for k in [1,3,5,10]:
        hit, ndcg, f1, recall, mrr, map_score = calculate_metrics(stu_true_response, stu_ks, k=k)
        print(f'k = {k}, ndcg: {ndcg:.3f}, f1: {f1:.3f}, recall: {recall:.3f}, mrr: {mrr:.3f}, map: {map_score:.3f}')

