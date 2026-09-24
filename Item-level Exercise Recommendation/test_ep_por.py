import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_seed(int(os.environ.get("UNIER_SEED", "42")))

def _normalize_uid(uid):
    value = str(uid)
    if value.startswith('uid'):
        value = value[3:]
    try:
        return str(int(float(value)))
    except ValueError:
        return str(uid)


def _load_recommendations(path):
    by_uid, ordered = {}, []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if not parts or not parts[-1]:
                continue
            recs = [int(value) for value in parts[-1].split(',') if value]
            ordered.append(recs)
            if len(parts) > 1:
                by_uid[_normalize_uid(parts[0])] = recs
    return by_uid, ordered


def _recommendations_for_row(row, row_index, by_uid, ordered):
    if 'uid' in row.index:
        recs = by_uid.get(_normalize_uid(row['uid']))
        if recs is not None:
            return recs
    return ordered[row_index] if row_index < len(ordered) else []

class KTnet(nn.Module):
    def __init__(self, dkt_para_dict):
        super().__init__()
        input_size = dkt_para_dict['input_size']
        emb_dim = dkt_para_dict['emb_dim']
        hidden_size = dkt_para_dict['hidden_size']
        num_skills = dkt_para_dict['num_skills']
        nlayers = dkt_para_dict['nlayers']
        dropout = dkt_para_dict['dropout']

        self.name = 'DKT'
        self.nhid = hidden_size
        self.nlayers = nlayers
        self.dropout = dropout

        self.embedding_layer = nn.Linear(input_size, emb_dim)
        torch.nn.init.normal_(self.embedding_layer.weight)
        torch.nn.init.zeros_(self.embedding_layer.bias)

        self.rnn = nn.LSTM(emb_dim, hidden_size, nlayers)
        self.fc_out = nn.Linear(hidden_size, num_skills)

        self.dropout = nn.Dropout(p=self.dropout)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        h_0, c_0 = self.init_hidden_state(x.shape[1])

        embed = self.embedding_layer(x)
        output, _ = self.rnn(embed, (h_0, c_0))
        out = self.fc_out(output)
        out = self.dropout(out)
        return out

    def init_hidden_state(self, batch_size):
        device = self.embedding_layer.weight.device
        h_0 = torch.rand((self.nlayers, batch_size, self.nhid), device=device)
        c_0 = torch.rand((self.nlayers, batch_size, self.nhid), device=device)
        return h_0, c_0
    
def load_kt_net(dataset, num_concepts, device):
    feature_dim = 2 * num_concepts
    embed_dim = 128
    hidden_size = 256
    dkt_para_dict = {
        'input_size': feature_dim,
        'emb_dim': embed_dim,
        'hidden_size': hidden_size,
        'num_skills': num_concepts,
        'nlayers': 2,
        'dropout': 0.01,
    }
    kt_net = KTnet(dkt_para_dict).to(device)
    directory = f"pretrianed_kt_models/{dataset}"
    dkt_file = f'{directory}/env_weights/ValBest.ckpt'
    if os.path.exists(dkt_file):
        param_dict = torch.load(dkt_file, map_location=device)
        kt_net.load_state_dict(param_dict)
        kt_net.eval()
    else:
        raise ValueError('dkt net not trained yet!')
    return kt_net, feature_dim

def get_kt_mastery(kt_net, concepts, answers, num_concepts, feature_dim, device):
    with torch.no_grad():
        kt_input = torch.zeros((1, len(concepts), feature_dim)).to(device)
        for i, (c, a) in enumerate(zip(concepts, answers)):
            kt_input[0, i, int(c) + (num_concepts if a == 1 else 0)] = 1
        if len(concepts) > 0:
            kt_output = kt_net(kt_input)[-1, 0, :]
            kt_output = torch.sigmoid(kt_output)
        else:
            kt_output = torch.zeros(num_concepts).to(device)
        return kt_output

def calculate_portion_Ep(dataset, num_concepts, Q_matrix_path, test_csv_path, rerank_txt_path, device):

    Q_matrix = np.load(Q_matrix_path)
    test_df = pd.read_csv(test_csv_path)
    rec_by_uid, rec_ordered = _load_recommendations(rerank_txt_path)
    kt_net, feature_dim = load_kt_net(dataset, num_concepts, device)

    test_ks = [5, 10, 20]
    result_dict = {k: {"ep": 0.0, "valid_num": 0} for k in test_ks}

    for i in range(len(test_df)):
        row = test_df.iloc[i]
        concepts = [int(float(c)) for c in str(row["concepts"]).split(",") if float(c) != -1]
        responses = [int(float(r)) for r in str(row["responses"]).split(",") if float(r) != -1]
        
        total_len = len(concepts)
        if total_len < 5:
            continue
        start_idx = int(total_len * 0.8)
        target_concepts = list(set(concepts[start_idx:]))
        target_num = len(target_concepts)
        if target_num == 0:
            continue
        
        rec_questions = _recommendations_for_row(row, i, rec_by_uid, rec_ordered)
        total_rec_len = len(rec_questions)

        initial_state = get_kt_mastery(kt_net, concepts, responses, num_concepts, feature_dim, device)
        initial_score = sum(1 for c in target_concepts if initial_state[c].item() > 0.5)

        for k in test_ks:
            if total_rec_len < k:
                continue
            
            curr_rec_questions = rec_questions[:k]
            new_concepts = []
            new_responses = []
            for q in curr_rec_questions:
                if q < 0 or q >= len(Q_matrix):
                    continue
                q_concepts = np.where(Q_matrix[q] == 1)[0]
                if len(q_concepts) == 0:
                    continue
                c = q_concepts[0]
                new_concepts.append(c)
                new_responses.append(1 if initial_state[c].item() > 0.5 else 0)
            
            final_concepts = concepts + new_concepts
            final_responses = responses + new_responses
            final_state = get_kt_mastery(kt_net, final_concepts, final_responses, num_concepts, feature_dim, device)
            final_score = sum(1 for c in target_concepts if final_state[c].item() > 0.5)

            denominator = target_num - initial_score
            if denominator == 0:
                continue
            Ep = (final_score - initial_score) / denominator
            print(f"k:{k}, Student {i}: Target Concepts={target_num}, Initial={initial_score}, Final={final_score}, Ep = {Ep:.4f}")

            result_dict[k]["ep"] += Ep
            result_dict[k]["valid_num"] += 1

    print("\n" + "="*60)
    print(f"Dataset: {dataset} | Ep calculated by [Last 20% knowledge points of the student] | Recommendation lengths: 5/10/20")
    print("="*60)
    for k in test_ks:
        valid_num = result_dict[k]["valid_num"]
        mean_ep = result_dict[k]["ep"] / valid_num if valid_num > 0 else 0.0
        print(f"Recommendation Length = {k:2d} | Valid Students = {valid_num:4d} | Average Ep = {mean_ep:.4f}")
    print("="*60 + "\n")

    return result_dict

def calculate_portion_Ep_KTmodel(dataset, num_concepts, test_csv_path, rerank_txt_path, device):

    test_df = pd.read_csv(test_csv_path)
    rec_by_uid, rec_ordered = _load_recommendations(rerank_txt_path)
    kt_net, feature_dim = load_kt_net(dataset, num_concepts, device)

    test_ks = [5, 10, 20]
    result_dict = {k: {"ep": 0.0, "valid_num": 0} for k in test_ks}

    for i in range(len(test_df)):
        row = test_df.iloc[i]
        concepts = [int(float(c)) for c in str(row["concepts"]).split(",") if float(c) != -1]
        responses = [int(float(r)) for r in str(row["responses"]).split(",") if float(r) != -1]
        
        total_len = len(concepts)
        if total_len < 5:
            continue
        start_idx = int(total_len * 0.8)
        target_concepts = list(set(concepts[start_idx:]))
        target_num = len(target_concepts)
        if target_num == 0:
            continue
        
        rec_concepts = _recommendations_for_row(row, i, rec_by_uid, rec_ordered)
        total_rec_len = len(rec_concepts)

        initial_state = get_kt_mastery(kt_net, concepts, responses, num_concepts, feature_dim, device)
        initial_score = sum(1 for c in target_concepts if initial_state[c].item() > 0.5)

        for k in test_ks:
            if total_rec_len < k:
                continue
            
            curr_rec_concepts = rec_concepts[:k]
            new_concepts = []
            new_responses = []
            for c in curr_rec_concepts:
                if c < 0 or c >= num_concepts:
                    continue
                new_concepts.append(c)
                new_responses.append(1 if initial_state[c].item() > 0.5 else 0)
            
            final_concepts = concepts + new_concepts
            final_responses = responses + new_responses
            final_state = get_kt_mastery(kt_net, final_concepts, final_responses, num_concepts, feature_dim, device)
            final_score = sum(1 for c in target_concepts if final_state[c].item() > 0.5)

            denominator = target_num - initial_score
            if denominator == 0:
                continue
            Ep = (final_score - initial_score) / denominator
            print(f"k:{k}, Student {i}: Target Concepts={target_num}, Initial={initial_score}, Final={final_score}, Ep = {Ep:.4f}")

            result_dict[k]["ep"] += Ep
            result_dict[k]["valid_num"] += 1

    print("\n" + "="*60)
    print(f"Dataset: {dataset} | Ep calculated by [Last 20% knowledge points of the student] | Recommendation lengths: 5/10/20")
    print("="*60)
    for k in test_ks:
        valid_num = result_dict[k]["valid_num"]
        mean_ep = result_dict[k]["ep"] / valid_num if valid_num > 0 else 0.0
        print(f"Recommendation Length = {k:2d} | Valid Students = {valid_num:4d} | Average Ep = {mean_ep:.3f}")
    print("="*60 + "\n")

    return result_dict

if __name__ == "__main__":
    requested_device = os.environ.get("UNIER_DEVICE", "cuda:0")
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    dataset = os.environ.get("UNIER_DATASET", "assist2017")
    model = {
        "KCP-ER": "KCPER",
        "AKT": "akt",
        "SimpleKT": "simplekt",
    }.get(os.environ.get("UNIER_MODEL", "DRE"), os.environ.get("UNIER_MODEL", "DRE"))

    print(f"model:{model}, dataset:{dataset}")
    
    if dataset == "assist2009":
        num_concepts = 123
    elif dataset == "assist2012":
        num_concepts = 265
    elif dataset == "assist2017":
        num_concepts = 102
    elif dataset == "algebra2005":
        num_concepts = 112
    elif dataset == "bridge2006":
        num_concepts = 493
    elif dataset == "ednet":
        num_concepts = 188
    elif dataset == "junyi":
        num_concepts = 39
    elif dataset == "nips34":
        num_concepts = 57
    elif dataset == "xes3g5m":
        num_concepts = 865

    if model == 'NR4DER':
        Q_matrix_path = f"NR4DER/dataset/data_200/{dataset}/Q.npy"
        test_csv_path = f"NR4DER/dataset/data_200/{dataset}/test_sequences.csv"
        rerank_txt_path = f"NR4DER/dataset/data_200/{dataset}/Reranking_result_melt_20.txt"
        calculate_portion_Ep(dataset, num_concepts, Q_matrix_path, test_csv_path, rerank_txt_path, device)
    elif model == 'DRE':
        Q_matrix_path = f"./DRE/data/{dataset}/Q.npy"
        test_csv_path = f"./DRE/data/{dataset}/filtered_full_data.csv"
        rerank_txt_path = f"./DRE/data//{dataset}/recommended_problems_final.txt"
        calculate_portion_Ep(dataset, num_concepts, Q_matrix_path, test_csv_path, rerank_txt_path, device)
    elif model == 'KG4EX':
        Q_matrix_path = f"./KG4EX/data/{dataset}/Q.npy"
        test_csv_path = f"./KG4EX/data/{dataset}/Test20_unique_last.csv"
        rerank_txt_path = f"KG4EX/data/{dataset}/Test20/recommend_20.txt"  
        calculate_portion_Ep(dataset, num_concepts, Q_matrix_path, test_csv_path, rerank_txt_path, device)
    elif model == 'ER-TGA':
        Q_matrix_path = f"./ER-TGA/datasets/{dataset}/Q.npy"
        test_csv_path = f"./ER-TGA/datasets/{dataset}/{dataset}.csv"
        rerank_txt_path = f"./ER-TGA/fin_res/{dataset}/recommend.txt" 
        calculate_portion_Ep(dataset, num_concepts, Q_matrix_path, test_csv_path, rerank_txt_path, device)
    elif model == 'KCPER':
        Q_matrix_path = f"./kcp_er/datasets/{dataset}/select/selected/Q.npy"
        test_csv_path = f"./kcp_er/datasets/{dataset}/select/selected/test_sequences.csv"
        rerank_txt_path = f"./kcp_er/datasets/{dataset}/select/selected/recommendation_20.txt"  
        calculate_portion_Ep(dataset, num_concepts, Q_matrix_path, test_csv_path, rerank_txt_path, device)
    elif model == "MulOER-SAN":
        Q_matrix_path = f"./MulOER-SAN/select/{dataset}/Q.npy"
        test_csv_path = f"./MulOER-SAN/PYKT/data/{dataset}/test_gt4.csv"
        rerank_txt_path = f"./MulOER-SAN/select/{dataset}/new_ex_original.txt"  
        calculate_portion_Ep(dataset, num_concepts, Q_matrix_path, test_csv_path, rerank_txt_path, device)
    elif model == "akt":
        test_csv_path = f"./KT_Base_Rec/dataset/{dataset}/test_sequences_filter.csv"
        rerank_txt_path = f"./KT_Base_Rec/model/{model}/{dataset}/recommend.txt"  
        calculate_portion_Ep_KTmodel(dataset, num_concepts, test_csv_path, rerank_txt_path, device)
    elif model == "simplekt":
        test_csv_path = f"./KT_Base_Rec/dataset/{dataset}/test_sequences_filter.csv"
        rerank_txt_path = f"./KT_Base_Rec/model/{model}/{dataset}/recommend.txt"  
        calculate_portion_Ep_KTmodel(dataset, num_concepts, test_csv_path, rerank_txt_path, device)

    print(f"model:{model}")
