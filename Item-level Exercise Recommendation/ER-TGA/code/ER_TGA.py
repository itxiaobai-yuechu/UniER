import random
import torch
import statistics
import queue
import sys
import numpy as np
from numpy import dot
from numpy.linalg import norm

sys.path.append("../")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Genetic_Algorithm():
    def __init__(self, CE, S_num, C_num, X, Q, de, ds, WKC, delta,
                 population_size, generation_num, new_offsprings_num, pc=0.6, pm=0.001, pa=0.3,
                 Rec_num=5, N=50, problem_id_map = None, user_id_map = None, final_output_K=None):
        self.E_num = [len(CE[i]) for i in range(S_num)]
        self.CE = CE
        self.S_num = S_num
        self.C_num = C_num
        self.Q = Q
        self.de = de
        self.ds = ds
        self.WKC = WKC
        self.delta = delta
        self.N = N
        self.population_size = population_size
        self.generation_num = generation_num
        self.pc = pc
        self.pm = pm
        self.pa = pa
        self.new_offsprings_num = new_offsprings_num
        self.Rec_num = Rec_num
        self.X = X
        self.problem_id_map = problem_id_map
        self.user_id_map = user_id_map
        self.final_output_K = final_output_K


        self.all_correct_knowledge = [set() for i in range(self.S_num)]
        for si in range(self.S_num):
            for (ej, correct) in self.X[si]:
                if correct == 0:
                    continue
                for ck in range(self.C_num):
                    if self.Q[ej][ck] == 1:
                        self.all_correct_knowledge[si].add(ck)
        
        self.Q_gpu = torch.tensor(self.Q, dtype=torch.float32, device=device)
        self.de_gpu = torch.tensor(self.de, dtype=torch.float32, device=device)
        self.ds_gpu = torch.tensor(self.ds, dtype=torch.float32, device=device)
        self.WKC_gpu = torch.tensor(self.WKC, dtype=torch.float32, device=device)
        
        self.correct_knowledge_gpu = torch.zeros((self.S_num, self.C_num), dtype=torch.float32, device=device)
        for si in range(self.S_num):
            for ck in self.all_correct_knowledge[si]:
                self.correct_knowledge_gpu[si, ck] = 1
        
        self.Q_norm_gpu = torch.norm(self.Q_gpu, dim=1, keepdim=True)
        self.Q_norm_gpu = torch.nan_to_num(self.Q_norm_gpu)

        self.vis = [False for i in range(self.N + 1)]
        self.Edge_Weight = [[-1, -1, 0, 0] for i in range((int)(self.N * (self.N - 1) / 2))]
        self.Edge_Index = {}
        cnt = 0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                self.Edge_Weight[cnt] = [i, j, 0, 0]
                self.Edge_Index[(i, j)] = cnt
                cnt += 1

    def evaluate(self):

        Rel = self.forward()
        Acc = [0 for i in range(self.S_num)]
        Nov = [0 for i in range(self.S_num)]
        Div = [0 for i in range(self.S_num)]
        Proximity = [0 for i in range(self.S_num)]
        Coverage = [0 for i in range(self.S_num)]
        Quantity = [0 for i in range(self.S_num)]
        Volatility = [0 for i in range(self.S_num)]

        for si in range(self.S_num):
            if self.E_num[si] <= 0:
                continue
            if si >= self.S_num // 10:
                break
            for j in range(self.Rec_num):
                if j >= len(Rel[si]):
                    continue
                Rel_list = Rel[si][j]
                acc, nov, div, Pro, Cov, Qua, Vol = self.Fitness_Acc_Nov_Div_Pro_Cov_Qua_Vol(si, Rel_list)
                Acc[si] = max(Acc[si], acc)
                Nov[si] = max(Nov[si], nov)
                Div[si] = max(Div[si], div)
                Proximity[si] = max(Proximity[si], Pro)
                Coverage[si] = max(Coverage[si], Cov)
                Quantity[si] = max(Quantity[si], Qua)
                Volatility[si] = max(Volatility[si], Vol)

        def calc_stats(arr):
            arr_tensor = torch.tensor(arr, dtype=torch.float32, device=device)
            if len(arr) == 0:
                return 0, 0, 0, 0
            max_val = torch.max(arr_tensor).item()
            mean_val = torch.mean(arr_tensor).item()
            if len(arr) >= 2:
                std_val = torch.std(arr_tensor, unbiased=True).item()
            else:
                std_val = 0
            min_val = torch.min(arr_tensor).item()
            return max_val, mean_val, std_val, min_val

        acc_max, acc_mean, acc_std, acc_min = calc_stats(Acc)
        nov_max, nov_mean, nov_std, nov_min = calc_stats(Nov)
        div_max, div_mean, div_std, div_min = calc_stats(Div)
        pro_max, pro_mean, pro_std, pro_min = calc_stats(Proximity)
        cov_max, cov_mean, cov_std, cov_min = calc_stats(Coverage)
        qua_max, qua_mean, qua_std, qua_min = calc_stats(Quantity)
        vol_max, vol_mean, vol_std, vol_min = calc_stats(Volatility)

        print(f'Acc_max={acc_max}, Acc_mean={acc_mean}, Acc_std={acc_std}, Acc_min={acc_min}')
        print(f'Nov_max={nov_max}, Nov_mean={nov_mean}, Nov_std={nov_std}, Nov_min={nov_min}')
        print(f'Div_max={div_max}, Div_mean={div_mean}, Div_std={div_std}, Div_min={div_min}')
        print(f'Proximity_max={pro_max}, Proximity_mean={pro_mean}, Proximity_std={pro_std}, Proximity_min={pro_min}')
        print(f'Coverage_max={cov_max}, Coverage_mean={cov_mean}, Coverage_std={cov_std}, Coverage_min={cov_min}')
        print(f'Quantity_max={qua_max}, Quantity_mean={qua_mean}, Quantity_std={qua_std}, Quantity_min={qua_min}')
        print(f'Volatility_max={vol_max}, Volatility_mean={vol_mean}, Volatility_std={vol_std}, Volatility_min={vol_min}')

        Acc_filtered = [x for x in Acc if x > 0]
        Nov_filtered = [x for x in Nov if x > 0]
        Div_filtered = [x for x in Div if x > 0]
        Proximity_filtered = [x for x in Proximity if x > 0]
        Coverage_filtered = [x for x in Coverage if x > 0]
        Quantity_filtered = [x for x in Quantity if x > 0]
        Volatility_filtered = [x for x in Volatility if x > 0]

        print('After deleting 0 elements in the evaluation metric list:')

        acc_max_f, acc_mean_f, acc_std_f, acc_min_f = calc_stats(Acc_filtered)
        nov_max_f, nov_mean_f, nov_std_f, nov_min_f = calc_stats(Nov_filtered)
        div_max_f, div_mean_f, div_std_f, div_min_f = calc_stats(Div_filtered)
        pro_max_f, pro_mean_f, pro_std_f, pro_min_f = calc_stats(Proximity_filtered)
        cov_max_f, cov_mean_f, cov_std_f, cov_min_f = calc_stats(Coverage_filtered)
        qua_max_f, qua_mean_f, qua_std_f, qua_min_f = calc_stats(Quantity_filtered)
        vol_max_f, vol_mean_f, vol_std_f, vol_min_f = calc_stats(Volatility_filtered)

        print(f'Acc_max={acc_max_f}, Acc_mean={acc_mean_f}, Acc_std={acc_std_f}, Acc_min={acc_min_f}')
        print(f'Nov_max={nov_max_f}, Nov_mean={nov_mean_f}, Nov_std={nov_std_f}, Nov_min={nov_min_f}')
        print(f'Div_max={div_max_f}, Div_mean={div_mean_f}, Div_std={div_std_f}, Div_min={div_min_f}')
        print(f'Proximity_max={pro_max_f}, Proximity_mean={pro_mean_f}, Proximity_std={pro_std_f}, Proximity_min={pro_min_f}')
        print(f'Coverage_max={cov_max_f}, Coverage_mean={cov_mean_f}, Coverage_std={cov_std_f}, Coverage_min={cov_min_f}')
        print(f'Quantity_max={qua_max_f}, Quantity_mean={qua_mean_f}, Quantity_std={qua_std_f}, Quantity_min={qua_min_f}')
        print(f'Volatility_max={vol_max_f}, Volatility_mean={vol_mean_f}, Volatility_std={vol_std_f}, Volatility_min={vol_min_f}')

    def forward(self):

        Rel = [[] for i in range(self.S_num)]
        for si in range(self.S_num):
            si_original = self.user_id_map[si]
            if self.E_num[si] <= 0:
                continue
            
            Union = self.Init_Population(si)  

            for age in range(self.generation_num): 
                for i in range(self.N):
                    population = Union[i]
                    population = self.Crossover(si, population)
                    population = self.Mutation(si, population)
                    Union[i] = self.Selection(si, population)
                
                for i in range(0, self.N - 1, 2):
                    pop1 = Union[i]
                    pop2 = Union[i + 1]
                    newoffsprings = self.Alliance(pop1, pop2, u_idx=i, v_idx=i+1)
                    if newoffsprings != []:
                        Union[i] = Union[i] + newoffsprings
                
                for i in range(self.N):
                    population = Union[i]
                    Union[i] = self.Selection(si, population)
            
            encoded_lists = []
            for population in Union:
                for c in range(min(self.Rec_num, len(population))):
                    ls = []
                    chromosome = population[c]
                    for k in range(len(chromosome)):
                        if chromosome[k] == 0:
                            continue
                        encoded_problem_id = self.CE[si][k]
                        ls.append(encoded_problem_id)
                    Rel[si].append(ls)
                    encoded_lists.append(ls)

            if len(encoded_lists) > 0:
                best_idx = 0
                best_score = None
                for idx, el in enumerate(encoded_lists):
                    acc, nov, div, Pro, Cov, Qua, Vol = self.Fitness_Acc_Nov_Div_Pro_Cov_Qua_Vol(si, el)
                    score = acc + nov + div + Pro + Cov + Qua + Vol
                    if best_score is None or score > best_score:
                        best_score = score
                        best_idx = idx
                chosen = encoded_lists[best_idx]
                
                if isinstance(self.final_output_K, int) and self.final_output_K > 0:
                    K = self.final_output_K
                    if len(chosen) >= K:
                        chosen = chosen[:K]
                    else:
                        available = [e for e in self.CE[si] if e not in chosen]
                        need = K - len(chosen)
                        chosen = chosen + available[:need]
                
                ls_original = []
                for eid in chosen:
                    if self.problem_id_map is not None:
                        try:
                            orig = int(self.problem_id_map[eid])
                        except Exception:
                            orig = eid
                    else:
                        orig = eid
                    ls_original.append(orig)
                
                ls_str = ",".join(map(str, ls_original))
                print(f"{si_original}\t{ls_str}")

        return Rel

    def Alliance(self, pop1, pop2, u_idx=None, v_idx=None):

        self.vis = [False for _ in range(self.N)]
        
        class alliance(object):
            def __init__(self, Id, u, v, Historical_Marriage_Iterations, Distance_Last_Marriage_Iterations):
                self.Id = Id
                self.u = u
                self.v = v
                self.Historical_Marriage_Iterations = Historical_Marriage_Iterations
                self.Distance_Last_Marriage_Iterations = Distance_Last_Marriage_Iterations
            
            def __lt__(self, other):
                if self.Historical_Marriage_Iterations == other.Historical_Marriage_Iterations:
                    return self.Distance_Last_Marriage_Iterations > other.Distance_Last_Marriage_Iterations
                return self.Historical_Marriage_Iterations < other.Historical_Marriage_Iterations
        
        for idx, (u, v, hist, dist) in enumerate(self.Edge_Weight):
            self.Edge_Weight[idx][3] += 1
        
        que = queue.PriorityQueue()
        for num in range(len(self.Edge_Weight)):
            u, v, hist, dist = self.Edge_Weight[num]
            que.put(alliance(Id=num, u=u, v=v, Historical_Marriage_Iterations=hist, Distance_Last_Marriage_Iterations=dist))
        
        newoffsprings = []
        Len = min(len(pop1), len(pop2))
        pop1 = pop1[:Len]
        pop2 = pop2[:Len]
        selected_pairs = []
        
        if u_idx is not None and v_idx is not None:
            a, b = min(u_idx, v_idx), max(u_idx, v_idx)
            pair_index = self.Edge_Index.get((a, b))
            if pair_index is not None and a < Len and b < Len:
                self.Edge_Weight[pair_index][2] += 1
                self.Edge_Weight[pair_index][3] = 0
                selected_pairs.append((a, b))
        else:
            while not que.empty() and len(selected_pairs) < Len:
                Top = que.get()
                u = Top.u
                v = Top.v
                num = Top.Id
                if u < Len and v < Len and not self.vis[u] and not self.vis[v]:
                    self.vis[u] = self.vis[v] = True
                    self.Edge_Weight[num][2] += 1
                    self.Edge_Weight[num][3] = 0
                    selected_pairs.append((u, v))
        
        if len(selected_pairs) > 0 and len(pop1) > 0 and len(pop2) > 0:
            pop1_tensor = torch.tensor(pop1, dtype=torch.int32, device=device)
            pop2_tensor = torch.tensor(pop2, dtype=torch.int32, device=device)
            E = len(pop1[0]) if pop1 else 0
            
            for (u_idx, v_idx) in selected_pairs:
                if random.random() > self.pa:
                    continue
                if E <= 1:
                    continue
                point = torch.randint(1, E, (1,), device=device).item()
                child1 = pop1_tensor[u_idx][:point].cpu().tolist() + pop2_tensor[v_idx][point:].cpu().tolist()
                child2 = pop2_tensor[v_idx][:point].cpu().tolist() + pop1_tensor[u_idx][point:].cpu().tolist()
                newoffsprings.append(child1)
                newoffsprings.append(child2)
        
        if not newoffsprings:
            if len(pop1) > 0 and len(pop2) > 0:
                c1 = random.choice(pop1)
                c2 = random.choice(pop2)
                if len(c1) > 1:
                    pt = random.randint(1, len(c1)-1)
                    newoffsprings.append(c1[:pt] + c2[pt:])
        
        return newoffsprings

    def Init_Population(self, si):

        population_gpu = torch.randint(
            low=0, high=2,
            size=(self.N, self.population_size, self.E_num[si]),
            dtype=torch.int32,
            device=device
        )
        population = population_gpu.cpu().tolist()
        return population

    def Fitness_Acc_Nov_Div_Pro_Cov_Qua_Vol(self, si, Rel_list):

        n = len(Rel_list)
        if n == 0:
            return 0, 0, 0, 0, 0, 0, 0

        rel_ids_tensor = torch.tensor(Rel_list, dtype=torch.int32, device=device)
        de_rel = self.de_gpu[rel_ids_tensor]
        ds_si = self.ds_gpu[si].unsqueeze(0).repeat(n)
        acc_sum = (1 - torch.abs(ds_si - de_rel)).sum()
        Acc = acc_sum.item() / n

        nov_sum = 0.0
        correct_si = self.correct_knowledge_gpu[si]
        for ej in Rel_list:
            rel_knowledge = self.Q_gpu[ej]
            intersection = (rel_knowledge * correct_si).sum().item()
            union = (rel_knowledge + correct_si).clamp(0, 1).sum().item()
            jaccard = intersection / float(union) if union != 0 else 0
            nov_sum += 1 - jaccard
        Nov = nov_sum / n

        if n > 1:
            k = len(rel_ids_tensor)
            q_sub = self.Q_gpu[rel_ids_tensor]
            q_norm = self.Q_norm_gpu[rel_ids_tensor]
            sim_mat = (q_sub @ q_sub.T) / (q_norm @ q_norm.T)
            sim_mat = torch.nan_to_num(sim_mat)
            upper_tri = sim_mat[torch.triu_indices(k, k, offset=1)]
            avg_sim = upper_tri.mean().item()
            Div = 1 - avg_sim
        else:
            Div = 1.0

        avg_de = de_rel.mean().item()
        proximity_diff = abs(avg_de - self.ds[si])
        Proximity = 1 - proximity_diff

        WKC_tmp = self.WKC_gpu[si].clone()
        total_weak = WKC_tmp.sum().item()
        if total_weak > 0:
            for ej in Rel_list:
                q_ej = self.Q_gpu[ej]
                mask = (q_ej == 1) & (WKC_tmp == 1)
                WKC_tmp[mask] = 0
            covered_weak = total_weak - WKC_tmp.sum().item()
            Coverage = covered_weak / total_weak
        else:
            Coverage = 0.0

        total_candidate = len(self.CE[si])
        Quantity = len(Rel_list) / total_candidate if total_candidate > 0 else 0

        if n > 1:
            de_rel_diff = torch.abs(torch.diff(de_rel)).sum().item()
            avg_vol = de_rel_diff / (n - 1)
            Volatility = 1 - avg_vol
        else:
            Volatility = 1.0

        return Acc, Nov, Div, Proximity, Coverage, Quantity, Volatility

    def Fitness(self, si, population):

        ans = [[0, 0, 0, 0, 0, 0, 0] for i in range(len(population))]
        pop_tensor = torch.tensor(population, dtype=torch.int32, device=device)
        ce_si = self.CE[si].detach().clone().to(dtype=torch.int32, device=device)

        for c in range(len(population)):
            chromosome = pop_tensor[c]
            valid_idx = (chromosome == 1).nonzero().squeeze(-1)
            Rel_list = ce_si[valid_idx].cpu().tolist()
            ans[c][0], ans[c][1], ans[c][2], ans[c][3], ans[c][4], ans[c][5], ans[c][6] = self.Fitness_Acc_Nov_Div_Pro_Cov_Qua_Vol(si, Rel_list)
        
        fitness = [sum(ans[c]) for c in range(len(population))]
        return fitness

    def Selection(self, si, population):

        fitness = self.Fitness(si, population)
        fitness_tensor = torch.tensor(fitness, dtype=torch.float32, device=device)
        sorted_fitness, indices = torch.sort(fitness_tensor, descending=True)
        population_tensor = torch.tensor(population, dtype=torch.int32, device=device)
        sorted_population = population_tensor[indices].cpu().tolist()
        offsprings = sorted_population[:self.population_size]
        return offsprings

    def Crossover(self, si, population):

        offsprings = population.copy()
        cnt = 0
        num = len(population)
        target_cnt = int(self.E_num[si] * self.new_offsprings_num)
        
        if num == 0 or self.E_num[si] <= 1:
            return offsprings

        need_cnt = target_cnt - cnt
        if need_cnt <= 0:
            return offsprings

        c1_c2 = torch.randint(0, num, (need_cnt, 2), dtype=torch.int32, device=device)
        partials = torch.randint(1, self.E_num[si], (need_cnt,), dtype=torch.int32, device=device)
        
        pop_tensor = torch.tensor(population, dtype=torch.int32, device=device)
        
        for i in range(need_cnt):
            c1 = c1_c2[i, 0].item()
            c2 = c1_c2[i, 1].item()
            partial = partials[i].item()
            child = pop_tensor[c1][:partial].cpu().tolist() + pop_tensor[c2][partial:].cpu().tolist()
            offsprings.append(child)
            cnt += 1

        return offsprings

    def Mutation(self, si, population):

        offsprings = population.copy()
        cnt = 0
        num = len(population)
        target_cnt = int(self.E_num[si] * self.new_offsprings_num)
        
        if num == 0 or self.E_num[si] <= 0:
            return offsprings

        need_cnt = target_cnt - cnt
        if need_cnt <= 0:
            return offsprings

        c_ids = torch.randint(0, num, (need_cnt,), dtype=torch.int32, device=device)
        partials = torch.randint(0, self.E_num[si], (need_cnt,), dtype=torch.int32, device=device)
        
        pop_tensor = torch.tensor(population, dtype=torch.int32, device=device)
        
        for i in range(need_cnt):
            c = c_ids[i].item()
            partial = partials[i].item()
            chromosome = pop_tensor[c].clone()
            chromosome[partial] ^= 1
            child = chromosome.cpu().tolist()
            offsprings.append(child)
            cnt += 1

        return offsprings

    def Sim(self, qi, qj):

        a = norm(qi)
        b = norm(qj)
        return dot(qi, qj) / (a * b) if a * b != 0 else 0

if __name__ == "__main__":
    CE = [[1,2,3,4,5] for _ in range(10)]
    S_num = 10
    C_num = 5
    X = [[(0,1), (1,0), (2,1)] for _ in range(10)]
    Q = np.random.randint(0,2,(10,5))
    de = np.random.rand(10)
    ds = np.random.rand(10)
    WKC = np.random.randint(0,2,(10,5))
    delta = 0.5

    ga = Genetic_Algorithm(
        CE=CE, S_num=S_num, C_num=C_num, X=X, Q=Q, de=de, ds=ds, WKC=WKC, delta=delta,
        population_size=20, generation_num=10, new_offsprings_num=0.1,
        Rec_num=5, N=4, problem_id_map=np.arange(10), user_id_map=np.arange(10), final_output_K=5
    )
    
    ga.evaluate()