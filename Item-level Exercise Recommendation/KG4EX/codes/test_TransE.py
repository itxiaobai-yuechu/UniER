import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils as eval
import pandas as pd
import json

dataset = "assist2017"
dict_path = f"../data/{dataset}"
test_data_path = f"{dict_path}/Test20.csv"
kg_test_path = f"{dict_path}/Test20/kg_test.txt"
embedding_path = f"./models/{dataset}/TransE_adv"

relation_embedding = np.load(f"{embedding_path}/relation_embedding.npy")
entity_embedding = np.load(f"{embedding_path}/entity_embedding.npy")
test_data = pd.read_csv(test_data_path)


Q = np.load(f'{dict_path}/Q.npy')

with open(f"{dict_path}/entities.dict", 'r') as fin:
    entity2id = dict()
    for line in fin:
        eid, entity = line.strip().split('\t')
        entity2id[entity] = int(eid)

with open(f"{dict_path}/relations.dict", 'r') as fin:
    relation2id = dict()
    for line in fin:
        rid, relation = line.strip().split('\t')
        relation2id[relation] = int(rid)


dict_entity_embedding = {}
dict_relation_embedding = {}

for (k, v) in entity2id.items():
    dict_entity_embedding[k] = entity_embedding[v, :]

for (k, v) in relation2id.items():
    dict_relation_embedding[k] = relation_embedding[v, :]

def TransE(head, relation, tail, gamma=12.0):
    score = (head + relation) - tail
    score = gamma - np.linalg.norm(score, ord=2)
    return score


uid_mlkc_dict = {}
uid_pkc_dict = {}
uid_exfr_dict = {}
uid_rec_ex_dict = {}
with open(kg_test_path, 'r', encoding="UTF-8") as load_file:
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



uid_ex_scores = []
user_num = 0
rec_embedding = torch.from_numpy(dict_relation_embedding['rec'])
uid_mlkc_dict_keys_list = [key for key in uid_mlkc_dict.keys()]
print("start!!!")


for uid in uid_mlkc_dict_keys_list:
    user_num += 1
    uid_mlkc_keys = list(uid_mlkc_dict[uid].keys())
    uid_exfr_keys = list(uid_exfr_dict[uid].keys())
    uid_pkc_keys = list(uid_pkc_dict[uid].keys())
    print(f"************************start: {user_num} -- {uid}************************")
    scores = []

    s_mlkc_list = []
    s_pkc_list = []
    s_efr_list = []

    for i in range(len(uid_mlkc_keys)):
        key = uid_mlkc_keys[i]
        if key not in dict_entity_embedding:
            continue
        kc_embedding = torch.from_numpy(dict_entity_embedding[key])

        mlkc_embedding = torch.from_numpy(dict_relation_embedding[uid_mlkc_dict[uid][uid_mlkc_keys[i]]])
        pkc_embedding = torch.from_numpy(dict_relation_embedding[uid_pkc_dict[uid][uid_pkc_keys[i]]])

        s_mlkc_list.append(kc_embedding + mlkc_embedding)
        s_pkc_list.append(kc_embedding + pkc_embedding)

    for qid in range(len(Q)):
        ex_key = 'ex' + str(qid)
        if ex_key not in dict_entity_embedding:
            continue
        e = torch.from_numpy(dict_entity_embedding[ex_key])

        fr1 = 0.0
        fr2 = 0.0

        for s_mlkc in s_mlkc_list:
            fr1 += TransE(s_mlkc, rec_embedding, e)

        for s_pkc in s_pkc_list:
            fr1 += TransE(s_pkc, rec_embedding, e)

        ej_embedding = torch.from_numpy(dict_entity_embedding[uid_exfr_keys[qid]])
        efr_embedding = torch.from_numpy(dict_relation_embedding[uid_exfr_dict[uid][uid_exfr_keys[qid]]])
        s_efr = ej_embedding + efr_embedding
        fr2 = TransE(s_efr, rec_embedding, e)

        O_sel = fr1 / len(uid_mlkc_keys) + fr2
        scores.append(O_sel)

    uid_ex_scores.append((uid, scores))
    print(f"************************finish: {user_num} -- {uid}************************")


with open(f"{dict_path}/ex_rec_uid.txt", 'w') as f:
    for item in uid_ex_scores:
        uid, scores = item[0], item[1]
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        uid_ex_score = [str(item[0]) for item in sorted_scores][:150]
        uid_ex_score_str = ','.join(uid_ex_score)
        f.write(uid + '\t' + uid_ex_score_str + '\n')

print('done!')

uid_ex_scores_serializable = [{"uid": uid, "scores": scores} for uid, scores in uid_ex_scores]
with open(f"{dict_path}/uid_ex_scores.json", "w", encoding="utf-8") as json_file:
    json.dump(uid_ex_scores_serializable, json_file, ensure_ascii=False, indent=4)

