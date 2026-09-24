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
    model_dir = os.path.dirname(os.path.abspath(__file__))
    item_level_dir = os.path.dirname(model_dir)
    directory = os.path.join(item_level_dir, "pretrianed_kt_models", dataset)
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

def calculate_all_Ep(dataset, num_concepts, Q_matrix_path, test_csv_path, rerank_txt_path, device):

    Q_matrix = np.load(Q_matrix_path)
    test_df = pd.read_csv(test_csv_path)
    with open(rerank_txt_path, "r") as f:
        rerank_lines = f.readlines()
    kt_net, feature_dim = load_kt_net(dataset, num_concepts, device)

    test_ks = [5, 10, 20]
    result_dict = {k: {"ep": 0.0, "valid_num": 0} for k in test_ks}

    uid_to_rows = {}
    if 'uid' in test_df.columns:
        for idx, v in test_df['uid'].astype(str).items():
            uid_to_rows.setdefault(v, []).append(idx)

    for line in rerank_lines:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 0:
            continue
        uid_str = parts[0].strip()
        target_rows = uid_to_rows.get(uid_str, [])
        if not target_rows:
            try:
                n = int(uid_str)
                if 0 <= n < len(test_df):
                    target_rows = [n]
            except Exception:
                continue
        rec_str = parts[1] if len(parts) > 1 else ''
        if rec_str == '':
            continue
        rec_questions_all = [q for q in rec_str.split(',') if q]

        for row_idx in target_rows:
            row = test_df.iloc[row_idx]
            concepts = [int(float(c)) for c in str(row["concepts"]).split(",") if float(c) != -1]
            responses = [int(float(r)) for r in str(row["responses"]).split(",") if float(r) != -1]
            
            if Q_matrix is None:
                rec_concepts = [int(q) for q in rec_questions_all]
            else:
                rec_concepts = []
                for q in rec_questions_all:
                    try:
                        qid = int(q)
                    except Exception:
                        continue
                    if qid < 0 or qid >= Q_matrix.shape[0]:
                        continue
                    q_concepts = np.where(Q_matrix[qid] == 1)[0]
                    if q_concepts.size == 0:
                        continue
                    rec_concepts.append(int(q_concepts[0]))

            total_rec_len = len(rec_concepts)

            initial_state = get_kt_mastery(kt_net, concepts, responses, num_concepts, feature_dim, device)
            initial_score = (initial_state > 0.5).sum().item()
            full_score = num_concepts

            for k in test_ks:
                if total_rec_len < k:
                    continue
                
                curr_rec_concepts = rec_concepts[:k]

                new_concepts = []
                new_responses = []
                for c in curr_rec_concepts:
                    new_concepts.append(c)
                    new_responses.append(1 if initial_state[c].item() > 0.5 else 0)
                
                final_concepts = concepts + new_concepts
                final_responses = responses + new_responses
                final_state = get_kt_mastery(kt_net, final_concepts, final_responses, num_concepts, feature_dim, device)
                final_score = (final_state > 0.5).sum().item()

                if (full_score - initial_score) != 0:
                    Ep = (final_score - initial_score) / (full_score - initial_score)
                else:
                    Ep = 0.0
                print(f"k:{k}, Student {row_idx}: Initial Score = {initial_score}, Final Score = {final_score}, Ep = {Ep:.4f}")

                result_dict[k]["ep"] += Ep
                result_dict[k]["valid_num"] += 1

    print("\n" + "="*50)
    print(f"Dataset: {dataset} | One-time test recommendation lengths: 5/10/20")
    print("="*50)
    for k in test_ks:
        valid_num = result_dict[k]["valid_num"]
        mean_ep = result_dict[k]["ep"] / valid_num if valid_num > 0 else 0.0
        print(f"Recommendation Length = {k:2d} | Valid Students = {valid_num:4d} | Avg Ep = {mean_ep:.4f}")
    print("="*50 + "\n")

    return result_dict

def calculate_all_Ep_KTmodel(dataset, num_concepts, test_csv_path, rerank_txt_path, device):

    test_df = pd.read_csv(test_csv_path)
    with open(rerank_txt_path, "r") as f:
        rerank_lines = f.readlines()
    kt_net, feature_dim = load_kt_net(dataset, num_concepts, device)

    test_ks = [5, 10, 20]
    result_dict = {k: {"ep": 0.0, "valid_num": 0} for k in test_ks}
    test_df = pd.read_csv(test_csv_path)
    q_dir = os.path.dirname(test_csv_path)
    q_path = os.path.join(q_dir, 'Q.npy')
    Q_matrix = None
    if os.path.exists(q_path):
        Q_matrix = np.load(q_path)
    for i in range(len(test_df)):
        if i >= len(rerank_lines):
            break
            
        row = test_df.iloc[i]
        concepts = [int(float(c)) for c in str(row["concepts"]).split(",") if float(c) != -1]
        responses = [int(float(r)) for r in str(row["responses"]).split(",") if float(r) != -1]
        
        parts = rerank_lines[i].strip().split(maxsplit=1)
        rec_str = parts[1] if len(parts) > 1 else parts[0]
        rec_raw = [q for q in rec_str.split(",") if q]
        rec_concepts = []
        if Q_matrix is None:
            rec_concepts = [int(q) for q in rec_raw]
        else:
            for q in rec_raw:
                try:
                    qid = int(q)
                except Exception:
                    continue
                if qid < 0 or qid >= Q_matrix.shape[0]:
                    continue
                q_concepts = np.where(Q_matrix[qid] == 1)[0]
                if q_concepts.size == 0:
                    continue
                rec_concepts.append(int(q_concepts[0]))
        total_rec_len = len(rec_concepts)

        initial_state = get_kt_mastery(kt_net, concepts, responses, num_concepts, feature_dim, device)
        initial_score = (initial_state > 0.5).sum().item()
        full_score = num_concepts

        for k in test_ks:
            if total_rec_len < k:
                continue
            
            curr_rec_concepts = rec_concepts[:k]

            new_concepts = []
            new_responses = []

            for c in curr_rec_concepts:
                new_concepts.append(c)
                new_responses.append(1 if initial_state[c].item() > 0.5 else 0)

            
            final_concepts = concepts + new_concepts
            final_responses = responses + new_responses
            final_state = get_kt_mastery(kt_net, final_concepts, final_responses, num_concepts, feature_dim, device)
            final_score = (final_state > 0.5).sum().item()

            if (full_score - initial_score) != 0:
                Ep = (final_score - initial_score) / (full_score - initial_score)
            else:
                Ep = 0.0
            print(f"k:{k}, Student {i}: Initial Score = {initial_score}, Final Score = {final_score}, Ep = {Ep:.3f}")

            result_dict[k]["ep"] += Ep
            result_dict[k]["valid_num"] += 1

    print("\n" + "="*50)
    print(f"Dataset: {dataset} | Full Knowledge Test Recommendation Lengths 5/10/20")
    print("="*50)
    for k in test_ks:
        valid_num = result_dict[k]["valid_num"]
        mean_ep = result_dict[k]["ep"] / valid_num if valid_num > 0 else 0.0
        print(f"Recommendation Length = {k:2d} | Valid Students = {valid_num:4d} | Avg Ep = {mean_ep:.3f}")
    print("="*50 + "\n")

    return result_dict

if __name__ == "__main__":
    requested_device = os.environ.get("UNIER_DEVICE", "cuda:0")
    if requested_device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(requested_device)
        print(f"Successfully loaded GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("No GPU detected, running on CPU")
    dataset = os.environ.get("UNIER_DATASET", "assist2017")
    model = 'MMER'
    
    
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
    if model == "MMER":
        model_dir = os.path.dirname(os.path.abspath(__file__))
        test_csv_path = os.path.join(model_dir, "datasets", dataset, "test.csv")
        rerank_txt_path = os.path.join(model_dir, "datasets", dataset, "score_rec_exer_topk_stu_all.txt")
        calculate_all_Ep_KTmodel(dataset, num_concepts, test_csv_path, rerank_txt_path, device)
