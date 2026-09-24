import pickle

import pandas as pd
import ast
import os
import numpy as np
import torch
from scipy import spatial
from datetime import datetime

from multiprocessing import Pool
import multiprocessing

dataset = os.environ.get('UNIER_DATASET', 'assist2017')

path = f'../data/{dataset}/{dataset}.csv'
df = pd.read_csv(path)

all_uid_mlkc = torch.load(f'../data/{dataset}/pkm.pth').cpu().numpy()

student_kc_pre_list = []
for row in all_uid_mlkc:
    row_str = ','.join([str(round(item, 2)) for item in row])
    student_kc_pre_list.append(row_str)

df['kc_pre'] = student_kc_pre_list
df.to_csv(path, index=False)
print('add kc_pre col done!')


df = pd.read_csv(path)
df_len = len(df)
que_set = set()
kc_set = set()
for i in range(df_len):
    que = [int(x) for x in df['questions'][i].split(',')]
    kc = [int(x) for x in df['concepts'][i].split(',')]
    que_set.update(que)
    kc_set.update(kc)
print("uid_num",df_len)
print("kc_nums",len(kc_set))
print("ex_len",len(que_set))
print(max(que_set)+1)
print(max(kc_set)+1)


uid_len = df_len
if len(kc_set) != max(kc_set):
    kc_nums = max(kc_set)+1
else:
    kc_nums = len(kc_set)
ex_len = len(que_set)



Q = [[0 for _ in range(max(kc_set)+1)] for _ in range(max(que_set)+1)]
df_ = pd.read_csv(path)
df = df_[['questions', 'concepts']]
del df_

df_len = len(df)
error = []
for i in range(df_len):
    kc_list = [int(x) for x in df['concepts'][i].split(',')]
    questions_list = [int(x) for x in df['questions'][i].split(',')]
    for que, kc in zip(questions_list, kc_list):
        if que == -1:
            break
        else:
            Q[que][kc] = 1

with open(f'../data/{dataset}/Q.txt', 'w') as file:
    for i in range(len(Q)):
        line = [str(x) for x in Q[i]]
        line = ','.join(line)
        file.write(line + '\n')
Q_array = np.array(Q, dtype=np.uint8)
np.save(f'../data/{dataset}/Q.npy', Q_array)


all_uid_pkc = torch.load(f'../data/{dataset}/pkc.pth').cpu().numpy()
df = pd.read_csv(path)
student_next_kc_pre_list = []
for row in all_uid_pkc:
    row_str = ','.join([str(round(item, 2)) for item in row])
    student_next_kc_pre_list.append(row_str)

df['next_kc_pre'] = student_next_kc_pre_list
df.to_csv(path, index=False)
print('add next_kc_pre col done!')

df = pd.read_csv(path)
df_len = len(df)

if dataset == 'assist2009':
    timestamps = []
    for i in range(df_len):
        time_len = len([int(x) for x in df['concepts'][i].split(',')])
        timestamps.append(','.join(str(i) for i in range(time_len)))
    df['timestamps'] = timestamps

df['time_interval'] = 0
times_list = []
hours = 600000.
for i in range(df_len):
    kc_dic = {}
    time_list = []
    kc_list = [int(x) for x in df['concepts'][i].split(',')]
    timestamps_list = [int(x) for x in df['timestamps'][i].split(',')]
    for kc, timestamp in zip(kc_list, timestamps_list):
        if kc not in kc_dic.keys():
            kc_dic[kc] = timestamp
            time_list.append(0)
        else:
            if dataset == 'assist2009':
                sub = timestamp - kc_dic[kc]
                sub = float(sub) if sub != 1 else 0
            else:
                sub = float(timestamp - kc_dic[kc]) / hours
            sub = round(sub, 2)
            time_list.append(sub)
            kc_dic[kc] = timestamp
    time_str = [str(s) for s in time_list]
    time_str = ','.join(time_str)
    times_list.append(time_str)
df['time_interval'] = times_list
df.to_csv(path, index=False)
print('add time_interval col done!')


df = pd.read_csv(path)
df_len = len(df)
frkc = []
for i in range(df_len):
    time_interval = [float(x) for x in df['time_interval'][i].split(',')]
    frkc_list = [str(round(1-np.exp(-x), 2)) for x in time_interval]
    frkc.append(','.join(frkc_list))
df['frkc'] = frkc
df.to_csv(path, index=False)
print('add frkc col done!')




df = pd.read_csv(path)
total_rows = len(df)

split_ratio = 0.8
split_index = int(total_rows * split_ratio)
data = df.sample(frac=1, random_state=42).reset_index(drop=True)
data_80 = data[:split_index]
data_20 = data[split_index:]
sorted_data_80 = data_80.sort_values(by='uid')
sorted_data_20 = data_20.sort_values(by='uid')
output_file_80 = f'../data/{dataset}/Train80.csv'
output_file_20 = f'../data/{dataset}/Test20.csv'
sorted_data_80.to_csv(output_file_80, index=False)
sorted_data_20.to_csv(output_file_20, index=False)


Q = np.load(f'../data/{dataset}/Q.npy')


for train_or_test in ['Train80', 'Test20']:
    path = f'../data/{dataset}/{train_or_test}.csv'
    df = pd.read_csv(path)
    df_len = len(df)

    uid_mlkc_dict = dict()
    for i in range(df_len):
        uid = df['uid'][i]
        if uid not in uid_mlkc_dict.keys():
            uid_mlkc_dict['uid' + str(uid)] = dict()
        kc_pre_pt_list = [round(float(x), 2) for x in df['kc_pre'][i].split(',')]
        for j in range(len(kc_pre_pt_list)):
            uid_mlkc_dict['uid' + str(uid)]['kc' + str(j)] = 'mlkc' + str(kc_pre_pt_list[j])

    with open(f'../data/{dataset}/{train_or_test}/mlkc.txt', 'w') as file:
        for uid in uid_mlkc_dict.keys():
            file.write(uid + '\n')
            for kc in uid_mlkc_dict[uid].keys():
                line = kc + '\t' + uid_mlkc_dict[uid][kc]
                file.write(line + '\n')
    uid_frkc_dict = dict()
    uid_exfr_dict = dict()
    for i in range(df_len):
        uid = df['uid'][i]
        kc_is_full = [0] * kc_nums
        if uid not in uid_frkc_dict.keys():
            uid_frkc_dict['uid' + str(uid)] = dict()
        timestamp = [int(x) for x in df['timestamps'][i].split(',')]
        kc_list = [int(x) for x in df['concepts'][i].split(',')]
        frkc = [x for x in df['frkc'][i].split(',')]
        for j in range(len(kc_list)):
            kc = 'kc' + str(kc_list[j])
            kc_is_full[kc_list[j]] = 1
            if kc not in uid_frkc_dict['uid' + str(uid)].keys() or uid_frkc_dict['uid' + str(uid)][kc][1] < timestamp[j]:
                uid_frkc_dict['uid' + str(uid)][kc] = ('frkc' + str(frkc[j]), timestamp[j])
        pos_kc = [i for i in range(kc_nums) if kc_is_full[i] == 0]
        if len(pos_kc) != 0:
            for pos in pos_kc:
                kc = 'kc' + str(pos)
                uid_frkc_dict['uid' + str(uid)][kc] = ('frkc1.0', -1)


        uid_key = 'uid' + str(uid)
        if uid_key not in uid_exfr_dict:
            uid_exfr_dict[uid_key] = dict()
        for j in range(len(Q)):
            ex_kc_pos = np.where(Q[j] == 1)[0]
            sum_frkc = 0
            count = 0
            for pos in ex_kc_pos:
                kc = 'kc' + str(pos)
                try:
                    frkc_val = float(uid_frkc_dict[uid_key][kc][0][4:])
                    sum_frkc += frkc_val
                    count += 1
                except KeyError:
                    continue
            avg_frkc = sum_frkc / count if count > 0 else 0
            uid_exfr_dict[uid_key]['ex' + str(j)] = 'exfr' + str(round(avg_frkc, 2))

    with open(f'../data/{dataset}/{train_or_test}/exfr.txt', 'w') as file:
        for uid in uid_exfr_dict.keys():
            file.write(uid + '\n')
            for kc in uid_exfr_dict[uid].keys():
                line = kc + '\t' + uid_exfr_dict[uid][kc]
                file.write(line + '\n')

    uid_pkc_dict = dict()
    for i in range(df_len):
        uid = df['uid'][i]
        if uid not in uid_pkc_dict.keys():
            uid_pkc_dict['uid' + str(uid)] = {}
        next_kc_pre = df['next_kc_pre'][i].split(',')
        for kc in range(len(next_kc_pre)):
            uid_pkc_dict['uid' + str(uid)]['kc' + str(kc)] = 'pkc' + str(next_kc_pre[kc])

    with open(f'../data/{dataset}/{train_or_test}/pkc.txt', 'w') as file:
        for uid in uid_pkc_dict.keys():
            file.write(uid + '\n')
            for kc in uid_pkc_dict[uid].keys():
                line = kc + '\t' + uid_pkc_dict[uid][kc]
                file.write(line + '\n')
    print(f'save {train_or_test} mlkc, pkc, exfr!')


    if train_or_test == 'Train80':

        uid_mlkc_dict = dict()
        uid_pkc_dict = dict()
        uid_exfr_dict = dict()

        with open(f'../data/{dataset}/{train_or_test}/mlkc.txt', 'r') as file:
            for line in file:
                line = line.strip()
                if line[0] == 'u' and line not in uid_mlkc_dict.keys():
                    uid_mlkc_dict[line] = dict()
                    uid = line
                else:
                    line = line.split('\t')
                    kc = line[0]
                    mlkc = line[1]
                    uid_mlkc_dict[uid][kc] = mlkc

        with open(f'../data/{dataset}/{train_or_test}/pkc.txt', 'r') as file:
            for line in file:
                line = line.strip()
                if line[0] == 'u' and line not in uid_pkc_dict.keys():
                    uid_pkc_dict[line] = dict()
                    uid = line
                else:
                    line = line.split('\t')
                    uid_pkc_dict[uid][line[0]] = line[1]

        with open(f'../data/{dataset}/{train_or_test}/exfr.txt', 'r') as file:
            for line in file:
                line = line.strip()
                if line[0] == 'u' and line not in uid_exfr_dict.keys():
                    uid_exfr_dict[line] = dict()
                    uid = line
                else:
                    line = line.split('\t')
                    uid_exfr_dict[uid][line[0]] = line[1]

        uid_rec_ex = dict()
        r1 = 0.1
        r2 = 0.1
        n = 10

        for i in range(df_len):
            uid = df['uid'][i]
            if uid not in uid_rec_ex.keys():
                uid_rec_ex['uid' + str(uid)] = 0
            uid_rec_ex_list = []

            uid_pkc_info = uid_pkc_dict['uid' + str(uid)]
            sorted_uid_pkc_vec = [
                float(uid_pkc_info.get('kc' + str(k), 'pkc0.0')[3:]) for k in range(kc_nums)
            ]

            for j in range(len(Q)):
                Q_pos = [k for k in range(kc_nums) if Q[j][k] == 1]

                mlkc = 1.0
                for kc in Q_pos:
                    mlkc = mlkc * float(uid_mlkc_dict['uid' + str(uid)]['kc' + str(kc)][4:])

                Q_vec = Q[j]
                cossim = 1 - spatial.distance.cosine(Q_vec, sorted_uid_pkc_vec)

                exfr = float(uid_exfr_dict['uid' + str(uid)]['ex' + str(j)][4:])

                W = np.sqrt((r1 - mlkc) ** 2 + cossim ** 2 + (r2 - exfr) ** 2)

                uid_rec_ex_list.append((j, W))

            sorted_uid_rec_ex_list = sorted(uid_rec_ex_list, key=lambda x: x[1])

            smallest_top_n_exid = [item[0] for item in sorted_uid_rec_ex_list[:n]]

            uid_rec_ex['uid' + str(uid)] = smallest_top_n_exid
            print('uid' + str(uid) + ':\t' + str(smallest_top_n_exid))



        with open(f'../data/{dataset}/{train_or_test}/rec_ex.txt', 'w') as file:
            for uid in uid_rec_ex.keys():
                file.write(uid + '\t')
                for exid in uid_rec_ex[uid][:n-1]:
                    file.write(str(exid) + '\t')
                    line = 0
                file.write(str(uid_rec_ex[uid][-1]) + '\n')

        print(f"save {train_or_test} rex_ex done!")


    test_dict = []
    with open(f'../data/{dataset}/{train_or_test}/mlkc.txt', 'r') as file:
        uid = 0
        for line in file:
            line = line.strip()
            if line[0] == 'u':
                uid = line
            else:
                line = line.split('\t')
                test_dict.append((line[0], line[1], uid))

    with open(f'../data/{dataset}/{train_or_test}/pkc.txt', 'r') as file:
        uid = 0
        for line in file:
            line = line.strip()
            if line[0] == 'u':
                uid = line
            else:
                line = line.split('\t')
                test_dict.append((line[0], line[1], uid))

    with open(f'../data/{dataset}/{train_or_test}/exfr.txt', 'r') as file:
        uid = 0
        for line in file:
            line = line.strip()
            if line[0] == 'u':
                uid = line
            else:
                line = line.split('\t')
                test_dict.append((line[0], line[1], uid))

    if train_or_test == 'Test20':
        with open(f'../data/{dataset}/{train_or_test}/kg_test.txt', 'w') as file:
            for item in test_dict:
                line = item[0] + '\t' + item[1] + '\t' + item[2]
                file.write(line + '\n')
        print(f"save {dataset} kg_test done!")
    else:
        with open(f'../data/{dataset}/{train_or_test}/rec_ex.txt', 'r') as file:
            for line in file:
                line = line.strip().split('\t')
                uid = line[0]
                rec_ex_list = line[1:]
                for rec_ex in rec_ex_list:
                    test_dict.append((uid, 'rec', 'ex' + str(rec_ex)))

        with open(f'../data/{dataset}/{train_or_test}/kg_train.txt', 'w') as file:
            for item in test_dict:
                line = item[0] + '\t' + item[1] + '\t' + item[2]
                file.write(line + '\n')
        print(f"save {dataset} kg_train done!")




mlkc_relation_len = 101
pkc_relation_len = 101
exfr_relation_len = 101
rec_ex_relation_len = 1
mlkc_relation = [round(float(x) * 0.01, 2) for x in range(mlkc_relation_len)]
pkc_relation = [round(float(x) * 0.01, 2) for x in range(pkc_relation_len)]
exfr_relation = [round(float(x) * 0.01, 2) for x in range(exfr_relation_len)]
rec_ex_relation = 'rec'
with open(f'../data/{dataset}/relations.dict', 'w') as file:
    k = 0
    for i in range(mlkc_relation_len):
        line = str(i) + '\t' + 'mlkc' + str(mlkc_relation[k])
        file.write(line + '\n')
        k += 1

    k = 0
    for i in range(mlkc_relation_len, pkc_relation_len + mlkc_relation_len):
        line = str(i) + '\t' + 'pkc' + str(pkc_relation[k])
        file.write(line + '\n')
        k += 1
    k = 0
    for i in range(pkc_relation_len + mlkc_relation_len, pkc_relation_len + mlkc_relation_len + exfr_relation_len):
        line = str(i) + '\t' + 'exfr' + str(exfr_relation[k])
        file.write(line + '\n')
        k += 1
    line = str(pkc_relation_len + mlkc_relation_len + exfr_relation_len) + '\t' + 'rec'
    file.write(line)
print("save relations done!")


sum_len = uid_len + kc_nums + ex_len
with open(f'../data/{dataset}/entities.dict', 'w') as file:
    k = 0
    s = 0
    for i in range(sum_len):
        if i < uid_len:
            line = str(i) + '\t' + 'uid' + str(i)
            file.write(line + '\n')
        elif i >= uid_len and i <uid_len + kc_nums:
            line = str(i) + '\t' + 'kc' + str(k)
            file.write(line + '\n')
            k += 1
        else:
            line = str(i) + '\t' + 'ex' + str(s)
            file.write(line + '\n')
            s += 1
print("save entities done!")
