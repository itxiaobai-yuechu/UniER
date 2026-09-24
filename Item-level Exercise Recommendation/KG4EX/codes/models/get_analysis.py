import numpy as np
import os

dataset = os.environ.get('UNIER_DATASET', 'assist2017')
dict_path = f"../../data/{dataset}/Test20"
ex_rec_uid_path = f"./{dataset}/TransE_adv"

ex_rec_uid = {}
with open(f"{dict_path}/ex_rec_uid.txt", 'r') as f:
    for line in f:
        uid, exs = line.strip().split('\t')
        exs = [int(ex) for ex in exs.split(',')]
        ex_rec_uid[uid] = exs


Q = np.load(f"{dict_path}/Q.npy")

uid_mlkc_dict = {}
uid_pkc_dict = {}
uid_exfr_dict = {}
with open(f"{dict_path}/kg_test.txt", 'r', encoding="UTF-8") as load_file:
    for line in load_file:
        item1, item2, uid = line.strip().split('\t')
        if item2[0] == 'm':
            kc, mlkc, uid = item1, item2, uid
            if uid not in uid_mlkc_dict.keys():
                uid_mlkc_dict[uid] = {}
            uid_mlkc_dict[uid][kc] = 'mlkc' + str(round(float(mlkc[4:]), 2))
        elif item2[0] == 'e':
            ex, exfr, uid = item1, item2, uid
            if uid not in uid_exfr_dict.keys():
                uid_exfr_dict[uid] = {}
            uid_exfr_dict[uid][ex] = 'exfr' + str(round(float(exfr[4:]), 2))
        else:
            kc, pkc, uid = item1, item2, uid
            if uid not in uid_pkc_dict.keys():
                uid_pkc_dict[uid] = {}
            uid_pkc_dict[uid][kc] = 'pkc' + str(round(float(pkc[3:]), 2))

ex_rec_uid_details = {}
for uid in ex_rec_uid.keys():
    ex_rec_uid_details[uid] = {}
    for ex in ex_rec_uid[uid]:
        exfr = uid_exfr_dict[uid]['ex' + str(ex)]
        ex_kcs = np.where(Q[ex] == 1)[0]
        mlkc_list = []
        pkc_list = []
        for kc in ex_kcs:
            mlkc_list.append(('kc' + str(kc), uid_mlkc_dict[uid]['kc' + str(kc)]))
            pkc_list.append(('kc' + str(kc), uid_pkc_dict[uid]['kc' + str(kc)]))
        ex_rec_uid_details[uid]['ex_id' + str(ex)] = {'exfr': exfr, 'mlkc': mlkc_list, 'pkc': pkc_list}
        print(uid, ex, ex_rec_uid_details[uid]['ex_id' + str(ex)])

