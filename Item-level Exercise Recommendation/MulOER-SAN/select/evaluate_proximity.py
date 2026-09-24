import pandas as pd
import torch
import statistics
import os

class ProximityCalculator:
    def __init__(self):
        requested_device = os.environ.get('UNIER_DEVICE', 'cuda:0')
        if requested_device.startswith('cuda') and not torch.cuda.is_available():
            requested_device = 'cpu'
        self.device = torch.device(requested_device)

    def load_assist_data(self, csv_path):

        df = pd.read_csv(csv_path)
        self.total_csv_users = len(df)
        
        user_list = []
        problem_list = []
        correct_list = []

        for row_idx, row in df.iterrows():
            uid = row_idx
            qs = list(map(int, str(row['questions']).strip().split(',')))
            res = list(map(int, str(row['responses']).strip().split(',')))
            for q, r in zip(qs, res):
                user_list.append(uid)
                problem_list.append(q)
                correct_list.append(r)

        data = pd.DataFrame({
            'user_id': user_list, 
            'problem_id': problem_list, 
            'correct': correct_list
        })
        return data

    def load_recommend_dict(self, rec_path):
        rec_dict = {}
        with open(rec_path, 'r', encoding='utf-8') as f:
            for row_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                prob_str = parts[1]
                prob_list = list(map(int, prob_str.split(',')))
                rec_dict[row_idx] = prob_list
        return rec_dict

    def compute_de_ds(self, data, max_problem_id, total_csv_users):

        device = self.device
        E_num = max_problem_id + 1
        S_num = total_csv_users

        user_id = torch.tensor(data['user_id'].values, device=device, dtype=torch.long)
        problem_id = torch.tensor(data['problem_id'].values, device=device, dtype=torch.long)
        correct = torch.tensor(data['correct'].values, device=device, dtype=torch.float32)

        SF = torch.zeros(E_num, device=device)
        num = torch.zeros(E_num, device=device)
        st = torch.zeros((S_num, E_num), device=device, dtype=torch.bool)
        ET = [set() for _ in range(S_num)]

        for si, ej, ans in zip(user_id, problem_id, correct):
            if not st[si, ej]:
                num[ej] += 1
                st[si, ej] = True
                if ans == 0:
                    SF[ej] += 1
            if ans == 1:
                ET[si.item()].add(ej.item())

        de = torch.where(num != 0, SF / num, torch.zeros_like(SF))
        ds = torch.zeros(S_num, device=device)
        for i in range(S_num):
            et = list(ET[i])
            if et:
                ds[i] = de[torch.tensor(et, device=device)].mean()
        return de, ds

    def calculate_final_proximity(self, csv_path, rec_path):
        data = self.load_assist_data(csv_path)
        max_problem_id = data['problem_id'].max()
        de, ds = self.compute_de_ds(data, max_problem_id, self.total_csv_users)

        rec_dict = self.load_recommend_dict(rec_path)
        user_proximity_list = []
        max_valid_uid = self.total_csv_users - 1
        max_valid_pid = max_problem_id

        for uid in rec_dict:
            if uid > max_valid_uid:
                continue
            
            rec_probs = rec_dict[uid]
            if not rec_probs:
                continue
            
            valid_rec_probs = [pid for pid in rec_probs if 0 <= pid <= max_valid_pid]
            if not valid_rec_probs:
                continue

            sum_de = torch.sum(de[torch.tensor(valid_rec_probs, device=self.device)])
            avg_de = sum_de / len(valid_rec_probs)
            user_prox = torch.abs(avg_de - ds[uid]).item()
            
            user_proximity_list.append(user_prox)


        if len(user_proximity_list) >= 2:
            mean_prox = sum(user_proximity_list) / len(user_proximity_list)
            std_prox = statistics.stdev(user_proximity_list)
        elif len(user_proximity_list) == 1:
            mean_prox = user_proximity_list[0]
            std_prox = 0.0
        else:
            mean_prox = 0.0
            std_prox = 0.0

        return 1 - mean_prox, std_prox

if __name__ == "__main__":
    dataset = os.environ.get('UNIER_DATASET', 'assist2017')
    print(f"dataset:{dataset}")
    calc = ProximityCalculator()
    csv_file = f"../PYKT/data/{dataset}/test_gt4.csv"
    rec_file = f"./{dataset}/new_ex_original.txt"
    
    mean_prox, std_prox = calc.calculate_final_proximity(csv_file, rec_file)
    print(f"Proximity Mean: {mean_prox:.3f}")
    print(f"Proximity Std: {std_prox:.3f}")
