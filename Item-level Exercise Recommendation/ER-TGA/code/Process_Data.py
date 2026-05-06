import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from functools import partial
import torch
import math
from numpy import dot
from numpy.linalg import norm
from typing import List, Tuple
import my_functions

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MyDataset():
    def __init__(self, Epsilon, delta,
                 population_size, generation_num, new_offsprings_num, pc=0.6, pm=0.001, Rec_num=5):
        self.Epsilon = Epsilon
        self.delta = delta
        self.population_size = population_size
        self.generation_num = generation_num
        self.pc = pc
        self.pm = pm
        self.new_offsprings_num = new_offsprings_num
        self.Rec_num = Rec_num


    def encode_list(self, lst: List[str], encoder: LabelEncoder) -> List[int]:
        lst = [x for x in lst if x is not None and x != '']
        if not lst:
            return []
        return encoder.transform(lst).tolist()

    def Algebra2005(self, file_path: str = '../datasets/algebra2005/algebra2005.csv') -> Tuple:

        data = pd.read_csv(file_path)
        
        data['user_id_encoded'], user_id_classes = pd.factorize(data['uid'])
        user_id_map = user_id_classes
        S_num = data['user_id_encoded'].nunique()
        
        data['questions'] = data['questions'].astype(str).str.replace(' ', '', regex=False)
        data['questions'] = data['questions'].replace('', pd.NA)
        data = data.dropna(subset=['questions'])
        data['problems_split'] = data['questions'].str.split(',')
        all_problems = data['problems_split'].explode().dropna()
        
        le_questions = LabelEncoder()
        le_questions.fit(all_problems)
        E_num = len(le_questions.classes_)
        problem_id_map = le_questions.classes_
        
        encode_problem = partial(self.encode_list, encoder=le_questions)
        data['problem_id_encoded'] = data['problems_split'].apply(encode_problem)
        
        data['concepts'] = data['concepts'].astype(str).str.replace(' ', '', regex=False)
        data['concepts'] = data['concepts'].replace('', pd.NA)
        data = data.dropna(subset=['concepts'])
        data['skills_split'] = data['concepts'].str.split(',')
        all_skills = data['skills_split'].explode().dropna()
        
        le_skills = LabelEncoder()
        le_skills.fit(all_skills)
        C_num = len(le_skills.classes_)
        
        encode_skill = partial(self.encode_list, encoder=le_skills)
        data['skill_id_encoded'] = data['skills_split'].apply(encode_skill)
        
        data['responses_encoded'] = data['responses'].apply(
            lambda s: list(map(int, s.split(','))) if pd.notna(s) else []
        )
        data['is_repeat_encoded'] = data['is_repeat'].apply(
            lambda s: list(map(int, s.split(','))) if pd.notna(s) else []
        )
        
        X = [[] for _ in range(S_num)]
        Q = torch.zeros((E_num, C_num), dtype=torch.int32, device=device)
        CS = torch.zeros((S_num, C_num), dtype=torch.int32, device=device)
        
        max_seq_len = max([
            data['problem_id_encoded'].apply(len).max() if not data['problem_id_encoded'].empty else 0,
            data['skill_id_encoded'].apply(len).max() if not data['skill_id_encoded'].empty else 0,
            data['responses_encoded'].apply(len).max() if not data['responses_encoded'].empty else 0,
            data['is_repeat_encoded'].apply(len).max() if not data['is_repeat_encoded'].empty else 0
        ])
        
        def pad_series(series: pd.Series, max_len: int) -> np.ndarray:
            return np.array([lst + [-1]*(max_len - len(lst)) for lst in series])
        
        user_ids = torch.tensor(data['user_id_encoded'].values, device=device)
        ei = torch.tensor(pad_series(data['problem_id_encoded'], max_seq_len), device=device)
        con = torch.tensor(pad_series(data['skill_id_encoded'], max_seq_len), device=device)
        res = torch.tensor(pad_series(data['responses_encoded'], max_seq_len), device=device)
        rep = torch.tensor(pad_series(data['is_repeat_encoded'], max_seq_len), device=device)
        
        mask_first = (rep == 0)
        mask_valid = (ei != -1) & (con != -1) & (res != -1) & (rep != -1)
        mask = mask_first & mask_valid
        
        for row_idx in range(len(data)):
            stu_id = user_ids[row_idx].item()
            valid_idx = mask[row_idx]
            ei_valid = ei[row_idx][valid_idx]
            con_valid = con[row_idx][valid_idx]
            res_valid = res[row_idx][valid_idx]
            
            X[stu_id] = [(ej.item(), r.item()) for ej, r in zip(ei_valid, res_valid)]

            
            correct_mask = (res_valid == 1) 
            if correct_mask.any():
                ei_correct = ei_valid[correct_mask]
                con_correct = con_valid[correct_mask]
                Q[ei_correct, con_correct] = 1
                CS[stu_id, con_correct] = 1
        
        Q_np = Q.cpu().numpy()
        CS_np = CS.cpu().numpy()
        
        de_np, ds_np = my_functions.calc_d_gpu(S_num, E_num, X, device)
        de = torch.tensor(de_np, dtype=torch.float64, device=device)
        ds = torch.tensor(ds_np, dtype=torch.float64, device=device)
        
        WKC = my_functions.calc_WKC_gpu(S_num, C_num, CS_np, self.Epsilon, device)
        WKC_np = WKC.cpu().numpy()
        
        CE = my_functions.calc_CESFA_gpu(S_num, E_num, C_num, Q_np, WKC, de, device)
        
        return (
            S_num, E_num, C_num, Q_np, X, CS_np,
            de_np, ds_np, WKC_np, CE,
            problem_id_map, user_id_map
        )   

    def ASSISTments2009(self, file_path: str = '../datasets/assist2009/assist2009.csv') -> Tuple:
        data = pd.read_csv(file_path)
        
        data['user_id_encoded'], user_id_classes = pd.factorize(data['uid'])
        user_id_map = user_id_classes
        S_num = data['user_id_encoded'].nunique()
        
        data['questions'] = data['questions'].astype(str).str.replace(' ', '', regex=False)
        data['questions'] = data['questions'].replace('', pd.NA)
        data = data.dropna(subset=['questions'])
        data['problems_split'] = data['questions'].str.split(',')
        all_problems = data['problems_split'].explode().dropna()
        
        le_questions = LabelEncoder()
        le_questions.fit(all_problems)
        E_num = len(le_questions.classes_)
        problem_id_map = le_questions.classes_
        
        encode_problem = partial(self.encode_list, encoder=le_questions)
        data['problem_id_encoded'] = data['problems_split'].apply(encode_problem)
        
        data['concepts'] = data['concepts'].astype(str).str.replace(' ', '', regex=False)
        data['concepts'] = data['concepts'].replace('', pd.NA)
        data = data.dropna(subset=['concepts'])
        data['skills_split'] = data['concepts'].str.split(',')
        all_skills = data['skills_split'].explode().dropna()
        
        le_skills = LabelEncoder()
        le_skills.fit(all_skills)
        C_num = len(le_skills.classes_)
        
        encode_skill = partial(self.encode_list, encoder=le_skills)
        data['skill_id_encoded'] = data['skills_split'].apply(encode_skill)
        
        data['responses_encoded'] = data['responses'].apply(
            lambda s: list(map(int, s.split(','))) if pd.notna(s) else []
        )
        
        X = [[] for _ in range(S_num)]
        Q = torch.zeros((E_num, C_num), dtype=torch.int32, device=device)
        CS = torch.zeros((S_num, C_num), dtype=torch.int32, device=device)
        
        max_seq_len = max([
            data['problem_id_encoded'].apply(len).max() if not data['problem_id_encoded'].empty else 0,
            data['skill_id_encoded'].apply(len).max() if not data['skill_id_encoded'].empty else 0,
            data['responses_encoded'].apply(len).max() if not data['responses_encoded'].empty else 0
        ])
        
        def pad_series(series: pd.Series, max_len: int) -> np.ndarray:
            return np.array([lst + [-1]*(max_len - len(lst)) for lst in series])
        
        user_ids = torch.tensor(data['user_id_encoded'].values, device=device)
        ei = torch.tensor(pad_series(data['problem_id_encoded'], max_seq_len), device=device)
        con = torch.tensor(pad_series(data['skill_id_encoded'], max_seq_len), device=device)
        res = torch.tensor(pad_series(data['responses_encoded'], max_seq_len), device=device)
        
        mask_valid = (ei != -1) & (con != -1) & (res != -1)
        
        for row_idx in range(len(data)):
            stu_id = user_ids[row_idx].item()
            valid_idx = mask_valid[row_idx]
            ei_valid = ei[row_idx][valid_idx]
            con_valid = con[row_idx][valid_idx]
            res_valid = res[row_idx][valid_idx]
            
            X[stu_id].extend([(ej.item(), r.item()) for ej, r in zip(ei_valid, res_valid)])

            correct_mask = (res_valid == 1) 
            if correct_mask.any():
                ei_correct = ei_valid[correct_mask]
                con_correct = con_valid[correct_mask]
                Q[ei_correct, con_correct] = 1
                CS[stu_id, con_correct] = 1
        
        Q_np = Q.cpu().numpy()
        CS_np = CS.cpu().numpy()
        
        de_np, ds_np = my_functions.calc_d_gpu(S_num, E_num, X, device)
        de = torch.tensor(de_np, dtype=torch.float64, device=device)
        ds = torch.tensor(ds_np, dtype=torch.float64, device=device)
        
        WKC = my_functions.calc_WKC_gpu(S_num, C_num, CS_np, self.Epsilon, device)
        WKC_np = WKC.cpu().numpy()
        
        CE = my_functions.calc_CESFA_gpu(S_num, E_num, C_num, Q_np, WKC, de, device)
        
        return (
            S_num, E_num, C_num, Q_np, X, CS_np,
            de_np, ds_np, WKC_np, CE,
            problem_id_map, user_id_map
        )
    
    def ASSISTments2017(self, file_path: str = '../datasets/assist2017/assist2017.csv') -> Tuple:
        data = pd.read_csv(file_path)
        
        data['user_id_encoded'], user_id_classes = pd.factorize(data['uid'])
        user_id_map = user_id_classes
        S_num = data['user_id_encoded'].nunique()
        
        data['questions'] = data['questions'].astype(str).str.replace(' ', '', regex=False)
        data['questions'] = data['questions'].replace('', pd.NA)
        data = data.dropna(subset=['questions'])
        data['problems_split'] = data['questions'].str.split(',')
        all_problems = data['problems_split'].explode().dropna()
        
        le_questions = LabelEncoder()
        le_questions.fit(all_problems)
        E_num = len(le_questions.classes_)
        problem_id_map = le_questions.classes_
        
        encode_problem = partial(self.encode_list, encoder=le_questions)
        data['problem_id_encoded'] = data['problems_split'].apply(encode_problem)
        
        data['concepts'] = data['concepts'].astype(str).str.replace(' ', '', regex=False)
        data['concepts'] = data['concepts'].replace('', pd.NA)
        data = data.dropna(subset=['concepts'])
        data['skills_split'] = data['concepts'].str.split(',')
        all_skills = data['skills_split'].explode().dropna()
        
        le_skills = LabelEncoder()
        le_skills.fit(all_skills)
        C_num = len(le_skills.classes_)
        
        encode_skill = partial(self.encode_list, encoder=le_skills)
        data['skill_id_encoded'] = data['skills_split'].apply(encode_skill)
        
        data['responses_encoded'] = data['responses'].apply(
            lambda s: list(map(int, s.split(','))) if pd.notna(s) else []
        )
        
        X = [[] for _ in range(S_num)]
        Q = torch.zeros((E_num, C_num), dtype=torch.int32, device=device)
        CS = torch.zeros((S_num, C_num), dtype=torch.int32, device=device)
        
        max_seq_len = max([
            data['problem_id_encoded'].apply(len).max() if not data['problem_id_encoded'].empty else 0,
            data['skill_id_encoded'].apply(len).max() if not data['skill_id_encoded'].empty else 0,
            data['responses_encoded'].apply(len).max() if not data['responses_encoded'].empty else 0
        ])
        
        def pad_series(series: pd.Series, max_len: int) -> np.ndarray:
            return np.array([lst + [-1]*(max_len - len(lst)) for lst in series])
        
        user_ids = torch.tensor(data['user_id_encoded'].values, device=device)
        ei = torch.tensor(pad_series(data['problem_id_encoded'], max_seq_len), device=device)
        con = torch.tensor(pad_series(data['skill_id_encoded'], max_seq_len), device=device)
        res = torch.tensor(pad_series(data['responses_encoded'], max_seq_len), device=device)
        
        mask_valid = (ei != -1) & (con != -1) & (res != -1)
        
        for row_idx in range(len(data)):
            stu_id = user_ids[row_idx].item()
            valid_idx = mask_valid[row_idx]
            ei_valid = ei[row_idx][valid_idx]
            con_valid = con[row_idx][valid_idx]
            res_valid = res[row_idx][valid_idx]
            
            X[stu_id].extend([(ej.item(), r.item()) for ej, r in zip(ei_valid, res_valid)])

            correct_mask = (res_valid == 1) 
            if correct_mask.any():
                ei_correct = ei_valid[correct_mask]
                con_correct = con_valid[correct_mask]
                Q[ei_correct, con_correct] = 1
                CS[stu_id, con_correct] = 1
        
        Q_np = Q.cpu().numpy()
        CS_np = CS.cpu().numpy()
        
        de_np, ds_np = my_functions.calc_d_gpu(S_num, E_num, X, device)
        de = torch.tensor(de_np, dtype=torch.float64, device=device)
        ds = torch.tensor(ds_np, dtype=torch.float64, device=device)
        
        WKC = my_functions.calc_WKC_gpu(S_num, C_num, CS_np, self.Epsilon, device)
        WKC_np = WKC.cpu().numpy()
        
        CE = my_functions.calc_CESFA_gpu(S_num, E_num, C_num, Q_np, WKC, de, device)
        
        return (
            S_num, E_num, C_num, Q_np, X, CS_np,
            de_np, ds_np, WKC_np, CE,
            problem_id_map, user_id_map
        )
    
    def Bridge2006(self, file_path: str = '../datasets/bridge2006/bridge2006.csv') -> Tuple:
        data = pd.read_csv(file_path)
        
        data['user_id_encoded'], user_id_classes = pd.factorize(data['uid'])
        user_id_map = user_id_classes
        S_num = data['user_id_encoded'].nunique()
        
        data['questions'] = data['questions'].astype(str).str.replace(' ', '', regex=False)
        data['questions'] = data['questions'].replace('', pd.NA)
        data = data.dropna(subset=['questions'])
        data['problems_split'] = data['questions'].str.split(',')
        all_problems = data['problems_split'].explode().dropna()
        
        le_questions = LabelEncoder()
        le_questions.fit(all_problems)
        E_num = len(le_questions.classes_)
        problem_id_map = le_questions.classes_
        
        encode_problem = partial(self.encode_list, encoder=le_questions)
        data['problem_id_encoded'] = data['problems_split'].apply(encode_problem)
        
        data['concepts'] = data['concepts'].astype(str).str.replace(' ', '', regex=False)
        data['concepts'] = data['concepts'].replace('', pd.NA)
        data = data.dropna(subset=['concepts'])
        data['skills_split'] = data['concepts'].str.split(',')
        all_skills = data['skills_split'].explode().dropna()
        
        le_skills = LabelEncoder()
        le_skills.fit(all_skills)
        C_num = len(le_skills.classes_)
        
        encode_skill = partial(self.encode_list, encoder=le_skills)
        data['skill_id_encoded'] = data['skills_split'].apply(encode_skill)
        
        data['responses_encoded'] = data['responses'].apply(
            lambda s: list(map(int, s.split(','))) if pd.notna(s) else []
        )
        
        X = [[] for _ in range(S_num)]
        Q = torch.zeros((E_num, C_num), dtype=torch.int32, device=device)
        CS = torch.zeros((S_num, C_num), dtype=torch.int32, device=device)
        
        max_seq_len = max([
            data['problem_id_encoded'].apply(len).max() if not data['problem_id_encoded'].empty else 0,
            data['skill_id_encoded'].apply(len).max() if not data['skill_id_encoded'].empty else 0,
            data['responses_encoded'].apply(len).max() if not data['responses_encoded'].empty else 0
        ])
        
        def pad_series(series: pd.Series, max_len: int) -> np.ndarray:
            return np.array([lst + [-1]*(max_len - len(lst)) for lst in series])
        
        user_ids = torch.tensor(data['user_id_encoded'].values, device=device)
        ei = torch.tensor(pad_series(data['problem_id_encoded'], max_seq_len), device=device)
        con = torch.tensor(pad_series(data['skill_id_encoded'], max_seq_len), device=device)
        res = torch.tensor(pad_series(data['responses_encoded'], max_seq_len), device=device)
        
        mask_valid = (ei != -1) & (con != -1) & (res != -1)
        
        for row_idx in range(len(data)):
            stu_id = user_ids[row_idx].item()
            valid_idx = mask_valid[row_idx]
            ei_valid = ei[row_idx][valid_idx]
            con_valid = con[row_idx][valid_idx]
            res_valid = res[row_idx][valid_idx]
            
            X[stu_id].extend([(ej.item(), r.item()) for ej, r in zip(ei_valid, res_valid)])

            correct_mask = (res_valid == 1) 
            if correct_mask.any():
                ei_correct = ei_valid[correct_mask]
                con_correct = con_valid[correct_mask]
                Q[ei_correct, con_correct] = 1
                CS[stu_id, con_correct] = 1
        
        Q_np = Q.cpu().numpy()
        CS_np = CS.cpu().numpy()
        
        de_np, ds_np = my_functions.calc_d_gpu(S_num, E_num, X, device)
        de = torch.tensor(de_np, dtype=torch.float64, device=device)
        ds = torch.tensor(ds_np, dtype=torch.float64, device=device)
        
        WKC = my_functions.calc_WKC_gpu(S_num, C_num, CS_np, self.Epsilon, device)
        WKC_np = WKC.cpu().numpy()
        
        CE = my_functions.calc_CESFA_gpu(S_num, E_num, C_num, Q_np, WKC, de, device)
        
        return (
            S_num, E_num, C_num, Q_np, X, CS_np,
            de_np, ds_np, WKC_np, CE,
            problem_id_map, user_id_map
        )

    def Ednet(self, file_path: str = '../datasets/ednet/ednet.csv') -> Tuple:
        data = pd.read_csv(file_path)
        
        data['user_id_encoded'], user_id_classes = pd.factorize(data['uid'])
        user_id_map = user_id_classes
        S_num = data['user_id_encoded'].nunique()
        
        data['questions'] = data['questions'].astype(str).str.replace(' ', '', regex=False)
        data['questions'] = data['questions'].replace('', pd.NA)
        data = data.dropna(subset=['questions'])
        data['problems_split'] = data['questions'].str.split(',')
        all_problems = data['problems_split'].explode().dropna()
        
        le_questions = LabelEncoder()
        le_questions.fit(all_problems)
        E_num = len(le_questions.classes_)
        problem_id_map = le_questions.classes_
        
        encode_problem = partial(self.encode_list, encoder=le_questions)
        data['problem_id_encoded'] = data['problems_split'].apply(encode_problem)
        
        data['concepts'] = data['concepts'].astype(str).str.replace(' ', '', regex=False)
        data['concepts'] = data['concepts'].replace('', pd.NA)
        data = data.dropna(subset=['concepts'])
        data['skills_split'] = data['concepts'].str.split(',')
        all_skills = data['skills_split'].explode().dropna()
        
        le_skills = LabelEncoder()
        le_skills.fit(all_skills)
        C_num = len(le_skills.classes_)
        
        encode_skill = partial(self.encode_list, encoder=le_skills)
        data['skill_id_encoded'] = data['skills_split'].apply(encode_skill)
        
        data['responses_encoded'] = data['responses'].apply(
            lambda s: list(map(int, s.split(','))) if pd.notna(s) else []
        )
        
        X = [[] for _ in range(S_num)]
        Q = torch.zeros((E_num, C_num), dtype=torch.int32, device=device)
        CS = torch.zeros((S_num, C_num), dtype=torch.int32, device=device)
        
        max_seq_len = max([
            data['problem_id_encoded'].apply(len).max() if not data['problem_id_encoded'].empty else 0,
            data['skill_id_encoded'].apply(len).max() if not data['skill_id_encoded'].empty else 0,
            data['responses_encoded'].apply(len).max() if not data['responses_encoded'].empty else 0
        ])
        
        def pad_series(series: pd.Series, max_len: int) -> np.ndarray:
            return np.array([lst + [-1]*(max_len - len(lst)) for lst in series])
        
        user_ids = torch.tensor(data['user_id_encoded'].values, device=device)
        ei = torch.tensor(pad_series(data['problem_id_encoded'], max_seq_len), device=device)
        con = torch.tensor(pad_series(data['skill_id_encoded'], max_seq_len), device=device)
        res = torch.tensor(pad_series(data['responses_encoded'], max_seq_len), device=device)
        
        mask_valid = (ei != -1) & (con != -1) & (res != -1)
        
        for row_idx in range(len(data)):
            stu_id = user_ids[row_idx].item()
            valid_idx = mask_valid[row_idx]
            ei_valid = ei[row_idx][valid_idx]
            con_valid = con[row_idx][valid_idx]
            res_valid = res[row_idx][valid_idx]
            
            X[stu_id].extend([(ej.item(), r.item()) for ej, r in zip(ei_valid, res_valid)])

            correct_mask = (res_valid == 1) 
            if correct_mask.any():
                ei_correct = ei_valid[correct_mask]
                con_correct = con_valid[correct_mask]
                Q[ei_correct, con_correct] = 1
                CS[stu_id, con_correct] = 1
        
        Q_np = Q.cpu().numpy()
        CS_np = CS.cpu().numpy()
        
        de_np, ds_np = my_functions.calc_d_gpu(S_num, E_num, X, device)
        de = torch.tensor(de_np, dtype=torch.float64, device=device)
        ds = torch.tensor(ds_np, dtype=torch.float64, device=device)
        
        WKC = my_functions.calc_WKC_gpu(S_num, C_num, CS_np, self.Epsilon, device)
        WKC_np = WKC.cpu().numpy()
        
        CE = my_functions.calc_CESFA_gpu(S_num, E_num, C_num, Q_np, WKC, de, device)
        
        return (
            S_num, E_num, C_num, Q_np, X, CS_np,
            de_np, ds_np, WKC_np, CE,
            problem_id_map, user_id_map
        )
    
    
    def Nips34(self, file_path: str = '../datasets/nips34/nips34.csv') -> Tuple:
        data = pd.read_csv(file_path)
        
        data['user_id_encoded'], user_id_classes = pd.factorize(data['uid'])
        user_id_map = user_id_classes
        S_num = data['user_id_encoded'].nunique()
        
        data['questions'] = data['questions'].astype(str).str.replace(' ', '', regex=False)
        data['questions'] = data['questions'].replace('', pd.NA)
        data = data.dropna(subset=['questions'])
        data['problems_split'] = data['questions'].str.split(',')
        all_problems = data['problems_split'].explode().dropna()
        
        le_questions = LabelEncoder()
        le_questions.fit(all_problems)
        E_num = len(le_questions.classes_)
        problem_id_map = le_questions.classes_
        
        encode_problem = partial(self.encode_list, encoder=le_questions)
        data['problem_id_encoded'] = data['problems_split'].apply(encode_problem)
        
        data['concepts'] = data['concepts'].astype(str).str.replace(' ', '', regex=False)
        data['concepts'] = data['concepts'].replace('', pd.NA)
        data = data.dropna(subset=['concepts'])
        data['skills_split'] = data['concepts'].str.split(',')
        all_skills = data['skills_split'].explode().dropna()
        
        le_skills = LabelEncoder()
        le_skills.fit(all_skills)
        C_num = len(le_skills.classes_)
        
        encode_skill = partial(self.encode_list, encoder=le_skills)
        data['skill_id_encoded'] = data['skills_split'].apply(encode_skill)
        
        data['responses_encoded'] = data['responses'].apply(
            lambda s: list(map(int, s.split(','))) if pd.notna(s) else []
        )
        
        X = [[] for _ in range(S_num)]
        Q = torch.zeros((E_num, C_num), dtype=torch.int32, device=device)
        CS = torch.zeros((S_num, C_num), dtype=torch.int32, device=device)
        
        max_seq_len = max([
            data['problem_id_encoded'].apply(len).max() if not data['problem_id_encoded'].empty else 0,
            data['skill_id_encoded'].apply(len).max() if not data['skill_id_encoded'].empty else 0,
            data['responses_encoded'].apply(len).max() if not data['responses_encoded'].empty else 0
        ])
        
        def pad_series(series: pd.Series, max_len: int) -> np.ndarray:
            return np.array([lst + [-1]*(max_len - len(lst)) for lst in series])
        
        user_ids = torch.tensor(data['user_id_encoded'].values, device=device)
        ei = torch.tensor(pad_series(data['problem_id_encoded'], max_seq_len), device=device)
        con = torch.tensor(pad_series(data['skill_id_encoded'], max_seq_len), device=device)
        res = torch.tensor(pad_series(data['responses_encoded'], max_seq_len), device=device)
        
        mask_valid = (ei != -1) & (con != -1) & (res != -1)
        
        for row_idx in range(len(data)):
            stu_id = user_ids[row_idx].item()
            valid_idx = mask_valid[row_idx]
            ei_valid = ei[row_idx][valid_idx]
            con_valid = con[row_idx][valid_idx]
            res_valid = res[row_idx][valid_idx]
            
            X[stu_id].extend([(ej.item(), r.item()) for ej, r in zip(ei_valid, res_valid)])

            correct_mask = (res_valid == 1) 
            if correct_mask.any():
                ei_correct = ei_valid[correct_mask]
                con_correct = con_valid[correct_mask]
                Q[ei_correct, con_correct] = 1
                CS[stu_id, con_correct] = 1
        
        Q_np = Q.cpu().numpy()
        CS_np = CS.cpu().numpy()
        
        de_np, ds_np = my_functions.calc_d_gpu(S_num, E_num, X, device)
        de = torch.tensor(de_np, dtype=torch.float64, device=device)
        ds = torch.tensor(ds_np, dtype=torch.float64, device=device)
        
        WKC = my_functions.calc_WKC_gpu(S_num, C_num, CS_np, self.Epsilon, device)
        WKC_np = WKC.cpu().numpy()
        
        CE = my_functions.calc_CESFA_gpu(S_num, E_num, C_num, Q_np, WKC, de, device)
        
        return (
            S_num, E_num, C_num, Q_np, X, CS_np,
            de_np, ds_np, WKC_np, CE,
            problem_id_map, user_id_map
        )
    
    

    def Xes3g5m(self, file_path: str = '../datasets/xes3g5m/xes3g5m.csv') -> Tuple:
        data = pd.read_csv(file_path)
        
        data['user_id_encoded'], user_id_classes = pd.factorize(data['uid'])
        user_id_map = user_id_classes
        S_num = data['user_id_encoded'].nunique()
        
        data['questions'] = data['questions'].astype(str).str.replace(' ', '', regex=False)
        data['questions'] = data['questions'].replace('', pd.NA)
        data = data.dropna(subset=['questions'])
        data['problems_split'] = data['questions'].str.split(',')
        all_problems = data['problems_split'].explode().dropna()
        
        le_questions = LabelEncoder()
        le_questions.fit(all_problems)
        E_num = len(le_questions.classes_)
        problem_id_map = le_questions.classes_
        
        encode_problem = partial(self.encode_list, encoder=le_questions)
        data['problem_id_encoded'] = data['problems_split'].apply(encode_problem)
        
        data['concepts'] = data['concepts'].astype(str).str.replace(' ', '', regex=False)
        data['concepts'] = data['concepts'].replace('', pd.NA)
        data = data.dropna(subset=['concepts'])
        data['skills_split'] = data['concepts'].str.split(',')
        all_skills = data['skills_split'].explode().dropna()
        
        le_skills = LabelEncoder()
        le_skills.fit(all_skills)
        C_num = len(le_skills.classes_)
        
        encode_skill = partial(self.encode_list, encoder=le_skills)
        data['skill_id_encoded'] = data['skills_split'].apply(encode_skill)
        
        data['responses_encoded'] = data['responses'].apply(
            lambda s: list(map(int, s.split(','))) if pd.notna(s) else []
        )
        
        X = [[] for _ in range(S_num)]
        Q = torch.zeros((E_num, C_num), dtype=torch.int32, device=device)
        CS = torch.zeros((S_num, C_num), dtype=torch.int32, device=device)
        
        max_seq_len = max([
            data['problem_id_encoded'].apply(len).max() if not data['problem_id_encoded'].empty else 0,
            data['skill_id_encoded'].apply(len).max() if not data['skill_id_encoded'].empty else 0,
            data['responses_encoded'].apply(len).max() if not data['responses_encoded'].empty else 0
        ])
        
        def pad_series(series: pd.Series, max_len: int) -> np.ndarray:
            return np.array([lst + [-1]*(max_len - len(lst)) for lst in series])
        
        user_ids = torch.tensor(data['user_id_encoded'].values, device=device)
        ei = torch.tensor(pad_series(data['problem_id_encoded'], max_seq_len), device=device)
        con = torch.tensor(pad_series(data['skill_id_encoded'], max_seq_len), device=device)
        res = torch.tensor(pad_series(data['responses_encoded'], max_seq_len), device=device)
        
        mask_valid = (ei != -1) & (con != -1) & (res != -1)
        
        for row_idx in range(len(data)):
            stu_id = user_ids[row_idx].item()
            valid_idx = mask_valid[row_idx]
            ei_valid = ei[row_idx][valid_idx]
            con_valid = con[row_idx][valid_idx]
            res_valid = res[row_idx][valid_idx]
            
            X[stu_id].extend([(ej.item(), r.item()) for ej, r in zip(ei_valid, res_valid)])

            correct_mask = (res_valid == 1) 
            if correct_mask.any():
                ei_correct = ei_valid[correct_mask]
                con_correct = con_valid[correct_mask]
                Q[ei_correct, con_correct] = 1
                CS[stu_id, con_correct] = 1
        
        Q_np = Q.cpu().numpy()
        CS_np = CS.cpu().numpy()
        
        de_np, ds_np = my_functions.calc_d_gpu(S_num, E_num, X, device)
        de = torch.tensor(de_np, dtype=torch.float64, device=device)
        ds = torch.tensor(ds_np, dtype=torch.float64, device=device)
        
        WKC = my_functions.calc_WKC_gpu(S_num, C_num, CS_np, self.Epsilon, device)
        WKC_np = WKC.cpu().numpy()
        
        CE = my_functions.calc_CESFA_gpu(S_num, E_num, C_num, Q_np, WKC, de, device)
        
        return (
            S_num, E_num, C_num, Q_np, X, CS_np,
            de_np, ds_np, WKC_np, CE,
            problem_id_map, user_id_map
        )
    
    def ASSISTments2012(self, file_path: str = '../datasets/assist2012/assist2012.csv') -> Tuple:
        data = pd.read_csv(file_path)
        
        data['user_id_encoded'], user_id_classes = pd.factorize(data['uid'])
        user_id_map = user_id_classes
        S_num = data['user_id_encoded'].nunique()
        
        data['questions'] = data['questions'].astype(str).str.replace(' ', '', regex=False)
        data['questions'] = data['questions'].replace('', pd.NA)
        data = data.dropna(subset=['questions'])
        data['problems_split'] = data['questions'].str.split(',')
        all_problems = data['problems_split'].explode().dropna()
        
        le_questions = LabelEncoder()
        le_questions.fit(all_problems)
        E_num = len(le_questions.classes_)
        problem_id_map = le_questions.classes_
        
        encode_problem = partial(self.encode_list, encoder=le_questions)
        data['problem_id_encoded'] = data['problems_split'].apply(encode_problem)
        
        data['concepts'] = data['concepts'].astype(str).str.replace(' ', '', regex=False)
        data['concepts'] = data['concepts'].replace('', pd.NA)
        data = data.dropna(subset=['concepts'])
        data['skills_split'] = data['concepts'].str.split(',')
        all_skills = data['skills_split'].explode().dropna()
        
        le_skills = LabelEncoder()
        le_skills.fit(all_skills)
        C_num = len(le_skills.classes_)
        
        encode_skill = partial(self.encode_list, encoder=le_skills)
        data['skill_id_encoded'] = data['skills_split'].apply(encode_skill)
        
        data['responses_encoded'] = data['responses'].apply(
            lambda s: list(map(int, s.split(','))) if pd.notna(s) else []
        )
        
        X = [[] for _ in range(S_num)]
        Q = torch.zeros((E_num, C_num), dtype=torch.int32, device=device)
        CS = torch.zeros((S_num, C_num), dtype=torch.int32, device=device)
        
        max_seq_len = max([
            data['problem_id_encoded'].apply(len).max() if not data['problem_id_encoded'].empty else 0,
            data['skill_id_encoded'].apply(len).max() if not data['skill_id_encoded'].empty else 0,
            data['responses_encoded'].apply(len).max() if not data['responses_encoded'].empty else 0
        ])
        
        def pad_series(series: pd.Series, max_len: int) -> np.ndarray:
            return np.array([lst + [-1]*(max_len - len(lst)) for lst in series])
        
        user_ids = torch.tensor(data['user_id_encoded'].values, device=device)
        ei = torch.tensor(pad_series(data['problem_id_encoded'], max_seq_len), device=device)
        con = torch.tensor(pad_series(data['skill_id_encoded'], max_seq_len), device=device)
        res = torch.tensor(pad_series(data['responses_encoded'], max_seq_len), device=device)
        
        mask_valid = (ei != -1) & (con != -1) & (res != -1)
        
        for row_idx in range(len(data)):
            stu_id = user_ids[row_idx].item()
            valid_idx = mask_valid[row_idx]
            ei_valid = ei[row_idx][valid_idx]
            con_valid = con[row_idx][valid_idx]
            res_valid = res[row_idx][valid_idx]
            
            X[stu_id].extend([(ej.item(), r.item()) for ej, r in zip(ei_valid, res_valid)])

            correct_mask = (res_valid == 1) 
            if correct_mask.any():
                ei_correct = ei_valid[correct_mask]
                con_correct = con_valid[correct_mask]
                Q[ei_correct, con_correct] = 1
                CS[stu_id, con_correct] = 1
        
        Q_np = Q.cpu().numpy()
        CS_np = CS.cpu().numpy()
        
        de_np, ds_np = my_functions.calc_d_gpu(S_num, E_num, X, device)
        de = torch.tensor(de_np, dtype=torch.float64, device=device)
        ds = torch.tensor(ds_np, dtype=torch.float64, device=device)
        
        WKC = my_functions.calc_WKC_gpu(S_num, C_num, CS_np, self.Epsilon, device)
        WKC_np = WKC.cpu().numpy()
        
        CE = my_functions.calc_CESFA_gpu(S_num, E_num, C_num, Q_np, WKC, de, device)
        
        return (
            S_num, E_num, C_num, Q_np, X, CS_np,
            de_np, ds_np, WKC_np, CE,
            problem_id_map, user_id_map
        )
    
    def Junyi(self, file_path: str = '../datasets/junyi/junyi.csv') -> Tuple:
        data = pd.read_csv(file_path)
        
        data['user_id_encoded'], user_id_classes = pd.factorize(data['uid'])
        user_id_map = user_id_classes
        S_num = data['user_id_encoded'].nunique()
        
        data['questions'] = data['questions'].astype(str).str.replace(' ', '', regex=False)
        data['questions'] = data['questions'].replace('', pd.NA)
        data = data.dropna(subset=['questions'])
        data['problems_split'] = data['questions'].str.split(',')
        all_problems = data['problems_split'].explode().dropna()
        
        le_questions = LabelEncoder()
        le_questions.fit(all_problems)
        E_num = len(le_questions.classes_)
        problem_id_map = le_questions.classes_
        
        encode_problem = partial(self.encode_list, encoder=le_questions)
        data['problem_id_encoded'] = data['problems_split'].apply(encode_problem)
        
        data['concepts'] = data['concepts'].astype(str).str.replace(' ', '', regex=False)
        data['concepts'] = data['concepts'].replace('', pd.NA)
        data = data.dropna(subset=['concepts'])
        data['skills_split'] = data['concepts'].str.split(',')
        all_skills = data['skills_split'].explode().dropna()
        
        le_skills = LabelEncoder()
        le_skills.fit(all_skills)
        C_num = len(le_skills.classes_)
        
        encode_skill = partial(self.encode_list, encoder=le_skills)
        data['skill_id_encoded'] = data['skills_split'].apply(encode_skill)
        
        data['responses_encoded'] = data['responses'].apply(
            lambda s: list(map(int, s.split(','))) if pd.notna(s) else []
        )
        
        X = [[] for _ in range(S_num)]
        Q = torch.zeros((E_num, C_num), dtype=torch.int32, device=device)
        CS = torch.zeros((S_num, C_num), dtype=torch.int32, device=device)
        
        max_seq_len = max([
            data['problem_id_encoded'].apply(len).max() if not data['problem_id_encoded'].empty else 0,
            data['skill_id_encoded'].apply(len).max() if not data['skill_id_encoded'].empty else 0,
            data['responses_encoded'].apply(len).max() if not data['responses_encoded'].empty else 0
        ])
        
        def pad_series(series: pd.Series, max_len: int) -> np.ndarray:
            return np.array([lst + [-1]*(max_len - len(lst)) for lst in series])
        
        user_ids = torch.tensor(data['user_id_encoded'].values, device=device)
        ei = torch.tensor(pad_series(data['problem_id_encoded'], max_seq_len), device=device)
        con = torch.tensor(pad_series(data['skill_id_encoded'], max_seq_len), device=device)
        res = torch.tensor(pad_series(data['responses_encoded'], max_seq_len), device=device)
        
        mask_valid = (ei != -1) & (con != -1) & (res != -1)
        
        for row_idx in range(len(data)):
            stu_id = user_ids[row_idx].item()
            valid_idx = mask_valid[row_idx]
            ei_valid = ei[row_idx][valid_idx]
            con_valid = con[row_idx][valid_idx]
            res_valid = res[row_idx][valid_idx]
            
            X[stu_id].extend([(ej.item(), r.item()) for ej, r in zip(ei_valid, res_valid)])

            correct_mask = (res_valid == 1) 
            if correct_mask.any():
                ei_correct = ei_valid[correct_mask]
                con_correct = con_valid[correct_mask]
                Q[ei_correct, con_correct] = 1
                CS[stu_id, con_correct] = 1
        
        Q_np = Q.cpu().numpy()
        CS_np = CS.cpu().numpy()
        
        de_np, ds_np = my_functions.calc_d_gpu(S_num, E_num, X, device)
        de = torch.tensor(de_np, dtype=torch.float64, device=device)
        ds = torch.tensor(ds_np, dtype=torch.float64, device=device)
        
        WKC = my_functions.calc_WKC_gpu(S_num, C_num, CS_np, self.Epsilon, device)
        WKC_np = WKC.cpu().numpy()
        
        CE = my_functions.calc_CESFA_gpu(S_num, E_num, C_num, Q_np, WKC, de, device)
        
        return (
            S_num, E_num, C_num, Q_np, X, CS_np,
            de_np, ds_np, WKC_np, CE,
            problem_id_map, user_id_map
        )
    
    
    
    

if __name__ == "__main__":
    processor = MyDataset(Epsilon=0.5, delta=0.5,
                          population_size=10, generation_num=3,
                          new_offsprings_num=0.3, pc=0.6, pm=0.001, Rec_num=5)

    try:
        result = processor.ASSISTments2012()
        
        S_num, E_num, C_num, Q, X, CS, de, ds, WKC, CE, problem_id_map, user_id_map = result
    except Exception as e:
        print(f"Execution error: {e}")
        import traceback
        traceback.print_exc()