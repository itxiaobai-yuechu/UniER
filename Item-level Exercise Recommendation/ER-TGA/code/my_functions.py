import torch
import numpy as np
from numpy import dot
from numpy.linalg import norm
import numpy as np
import torch

def calc_d_gpu(S_num: int, E_num: int, X: list, device: torch.device) -> tuple[np.ndarray, np.ndarray]:

    SF = [0] * E_num
    num = [0] * E_num
    ET = [set() for _ in range(S_num)]

    for i in range(S_num):
        st = [0] * E_num
        for (j, p) in X[i]:
            j = int(j)
            p = int(p)
            
            num[j] += 1
            
            if p == 1:
                ET[i].add(j)
            else:
                if st[j] == 0:
                    SF[j] += 1
                    st[j] = 1

    SF_tensor = torch.tensor(SF, dtype=torch.float64, device=device)
    num_tensor = torch.tensor(num, dtype=torch.float64, device=device)
    de_tensor = torch.zeros(E_num, dtype=torch.float64, device=device)
    
    non_zero_mask = num_tensor != 0
    de_tensor[non_zero_mask] = SF_tensor[non_zero_mask] / num_tensor[non_zero_mask]

    ds_tensor = torch.zeros(S_num, dtype=torch.float64, device=device)
    for i in range(S_num):
        et_len = len(ET[i])
        if et_len != 0:
            et_j = torch.tensor(list(ET[i]), dtype=torch.long, device=device)
            ds_tensor[i] = torch.sum(de_tensor[et_j]) / et_len

    de_np = de_tensor.cpu().numpy()
    ds_np = ds_tensor.cpu().numpy()

    return de_np, ds_np

def calc_WKC_gpu(S_num: int, C_num: int, CS_np: np.ndarray, Epsilon: float, device: torch.device) -> torch.Tensor:

    CS = torch.tensor(CS_np, dtype=torch.float64, device=device)
    WKC = torch.where(CS < Epsilon, 
                      torch.tensor(1.0, dtype=torch.float64, device=device),
                      torch.tensor(0.0, dtype=torch.float64, device=device))
    return WKC

def cos_sim_gpu(mat1: torch.Tensor, mat2: torch.Tensor, device: torch.device) -> torch.Tensor:

    mat1 = mat1.to(dtype=torch.float64, device=device)
    mat2 = mat2.to(dtype=torch.float64, device=device)
    
    norm1 = torch.norm(mat1, dim=1, keepdim=True)
    norm2 = torch.norm(mat2, dim=1, keepdim=True)
    
    norm1 = torch.clamp(norm1, min=1e-8)
    norm2 = torch.clamp(norm2, min=1e-8)
    
    dot_prod = torch.matmul(mat1, mat2.T)
    sim = dot_prod / (norm1 * norm2.T)
    
    sim = torch.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)
    return sim


def calc_CESFA_gpu(S_num: int, E_num: int, C_num: int, Q_np: np.ndarray, WKC: torch.Tensor, de: torch.Tensor, device: torch.device, batch_size: int = 32) -> torch.Tensor:

    Q = torch.tensor(Q_np, dtype=torch.float32, device=device)
    de = de.to(dtype=torch.float32, device=device)
    WKC = WKC.to(dtype=torch.float32, device=device)
    
    N = min(25, E_num)
    delta = torch.mean(de)
    CE_list = []

    norm_Q = torch.norm(Q, dim=1, keepdim=True)
    norm_Q = torch.clamp(norm_Q, min=1e-8)

    for i in range(0, S_num, batch_size):
        batch_wkc = WKC[i:i+batch_size]
        
        norm_w = torch.norm(batch_wkc, dim=1, keepdim=True)
        norm_w = torch.clamp(norm_w, min=1e-8)
        sim = torch.matmul(Q, batch_wkc.T) / (norm_Q * norm_w.T)
        sim = torch.nan_to_num(sim, 0.0)
        
        a = 1 - sim
        b = torch.abs(de.unsqueeze(1) - delta)
        dist = torch.sqrt(a**2 + b**2)
        
        top_n = torch.argsort(dist, dim=0)[:N].T
        CE_list.append(top_n)
        
        del batch_wkc, sim, a, b, dist
        torch.cuda.empty_cache()

    CE = torch.cat(CE_list, dim=0)
    CE_padded = torch.full((S_num, N), -1, dtype=torch.int32, device=device)
    CE_padded[:CE.shape[0], :CE.shape[1]] = CE.to(dtype=torch.int32)
    
    return CE_padded


def Avd(S_num, C_num, E_num, Rel, de):
    sum_d = 0
    num_Rel = 0
    for j in range(E_num):
        if Rel[j] == 1:
            num_Rel += 1
            sum_d += de[j]
    return sum_d / num_Rel if num_Rel != 0 else 0

def Rdm(S_num, C_num, E_num, Rel, si, de, ds):
    ans = Avd(S_num, C_num, E_num, Rel, de)
    return abs(ans - ds[si])

def Rkc(S_num, C_num, E_num, Q, WKC, Rel, si):
    num_WKC = sum(WKC[si])
    num_kc = 0
    KC_Rel = [0 for ck in range(C_num)]
    
    for ej in range(E_num):
        if Rel[ej] == 1:
            for ck in range(C_num):
                if Q[ej][ck] == 1:
                    KC_Rel[ck] = 1
    num_kc = sum(KC_Rel)
    
    ans = num_WKC - num_kc
    if ans < 0:
        print(si, 'The recommendation is perfect! Is the deviation 0??')
    return ans / num_WKC if num_WKC != 0 else 0

def Prc(S_num, C_num, E_num, Rel, CE, si):
    return sum(Rel[si]) / sum(CE[si]) if sum(CE[si]) != 0 else 0

def Sim(qi, qj):
    a = norm(qi)
    b = norm(qj)
    return dot(qi, qj) / (a * b) if a * b != 0 else 0

def Div(S_num, C_num, E_num, Q, Rel, si, Epsilon):
    num_Rel = sum(Rel[si])
    ans = 0
    for ei in range(E_num):
        for ej in range(E_num):
            if ei == ej:
                continue
            ans += 1 - Sim(Q[ei], Q[ej])
    return ans / (num_Rel * (num_Rel - 1)) if num_Rel > 1 else 0

def Smt(S_num, C_num, E_num, Rel, si, de):
    num_Rel = sum(Rel[si])
    ans = 0
    for ej in range(E_num - 1):
        ans += abs(de[ej] - de[ej + 1])
    return ans / num_Rel if num_Rel != 0 else 0

def Rct(S_num, C_num, E_num, Q, WKC, de, ds, CE, Rel, si, Epsilon):
    f1 = Rdm(S_num, C_num, E_num, Rel, si, de, ds)
    f2 = Rkc(S_num, C_num, E_num, Q, WKC, Rel, si)
    f3 = Prc(S_num, C_num, E_num, Rel, CE, si)
    f4 = Div(S_num, C_num, E_num, Q, Rel, si, Epsilon)
    f5 = Smt(S_num, C_num, E_num, Rel, si, de)
    return f1 * f2 * f3 * f4 * f5

def Print_Rel(S_num, C_num, E_num, Rel):
    for si in range(S_num):
        print(si, end=':')
        ls = []
        for ej in range(E_num):
            if Rel[si][ej] == 1:
                ls.append(ej)
        print(ls)