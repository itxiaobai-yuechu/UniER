import os
import torch
import torch.nn as nn
import pandas as pd
import random
import math
import numpy
import numpy as np
from numpy.linalg import norm
from joblib import Parallel, delayed
import copy

dataset="assist2017"

if dataset == "assist2009":
    kc_num = 123
    group_size = 13392
elif dataset == "assist2012":
    kc_num = 261
    group_size = 44545
elif dataset == "assist2017":
    kc_num = 102
    group_size = 2594
elif dataset == "algebra2005":
    kc_num = 111
    group_size = 35694
elif dataset == "bridge2006":
    kc_num = 493
    group_size = 74276
elif dataset == "ednet":
    kc_num = 189
    group_size = 11311
elif dataset == "nips34":
    kc_num = 57
    group_size = 919
elif dataset == "mooccubex":
    kc_num = 191
    group_size = 199
elif dataset == "xes3g5m":
    kc_num = 865
    group_size = 7652

base_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(base_dir,f'{dataset}/new_data1.txt')


def R_q(path):
    data = pd.read_csv(path, sep='\t')
    grouped = data.groupby('orirow').agg({
        'questions': list,
        'concepts': list,
        'concept_preds': list,
        'merged_data': list
    }).reset_index()
    df = pd.DataFrame(grouped)
    def process_row(row):
        unique_concepts = []
        unique_min_preds = []
        unique_merged = []
        seen_concepts = set()
        concept_data = zip(row['concepts'], row['concept_preds'], row['merged_data'])
        
        for concept, pred, merged in concept_data:
            if concept not in seen_concepts:
                seen_concepts.add(concept)
                unique_concepts.append(concept)
                unique_min_preds.append(pred)
                unique_merged.append(merged)
            else:
                index = unique_concepts.index(concept)
                if pred < unique_min_preds[index]:
                  unique_min_preds[index] = pred
                  unique_merged[index] = merged
        
        return pd.Series({
            'orirow': row['orirow'],
            'questions': row['questions'],
            'unique_concepts': unique_concepts,
            'unique_preds': unique_min_preds,
            'pkc': unique_merged
        })

    processed_df = df.apply(process_row, axis=1)
    return processed_df
def Que_cp(path):
    df = pd.read_csv(path, delimiter='\t')
    df_sorted = df.sort_values(by='questions')
    grouped = df_sorted.groupby('questions').agg({
        'concepts': lambda x: list(set(x))
    }).reset_index()

    return grouped
def iwkc(path):
    data = pd.read_csv(path, sep='\t')
    grouped = data.groupby('orirow').agg({
        'concepts': list,
        'late_trues': list
    }).reset_index()
    results = []
    for index, row in grouped.iterrows():
        orirow = row['orirow']
        concepts = row['concepts']
        late_trues = row['late_trues']
        concept_dict = {}
        for concept, late_true in zip(concepts, late_trues):
            if concept not in concept_dict:
                concept_dict[concept] = {'count': 0, 'late_trues_1_count': 0}
            concept_dict[concept]['count'] += 1
            if late_true == 1:
                concept_dict[concept]['late_trues_1_count'] += 1
        concept_list = list(concept_dict.keys())
        count_list = [v['count'] for v in concept_dict.values()]
        late_trues_1_count_list = [v['late_trues_1_count'] for v in concept_dict.values()]
        iwkc_list = [1-(late_trues_1_count / count )for late_trues_1_count, count in zip(late_trues_1_count_list, count_list)] 
        orirow_results = {
            'orirow': orirow,
            'concept': concept_list,
            'kat': count_list,
            'krt': late_trues_1_count_list,
            'iwkc': iwkc_list
        }
        results.append(orirow_results)
    result_df = pd.DataFrame(results)
    return result_df

r_q = R_q(path)
que_cp = Que_cp(path)
group_size = len(que_cp)

iw_kc = iwkc(path)


print("The student's status:")
print(r_q)
print('The concept of knowledge contained in each question:')
print(que_cp)
print('The student correct rate of each knowledge point')
print(iw_kc)

orirow_data = r_q['orirow']
u_concepts = r_q['unique_concepts']
u_preds = r_q['unique_preds']
u_pkc = r_q['pkc']
q_concepts = que_cp['concepts']
i_concepts = iw_kc['concept']
iw_kc = iw_kc['iwkc']
q_kc = []
u_kc = []
i_kc = []
kc = []
all_kc = []
all_scores = []
all_dis = []
all_diss = []

for kc in q_concepts:
  concepts_list = [0] * kc_num
  for concept in kc:
    concepts_list[concept] = 1


  q_kc.append(concepts_list)
for index,concepts in enumerate(u_concepts):
   concepts_list = [0] * kc_num
   j = 0
   for i in concepts:
     concepts_list[i] = u_pkc[index][j]
     j+=1
   u_kc.append(concepts_list)
for index,concepts in enumerate(i_concepts):
   concepts_list = [1] * kc_num
   j = 0
   for i in concepts:
     concepts_list[i] = iw_kc[index][j]
     j+=1
   i_kc.append(concepts_list)
if not isinstance(u_kc, np.ndarray):
    u_kc = np.array(u_kc)

if not isinstance(i_kc, np.ndarray):
    i_kc = np.array(i_kc)

pkc = u_kc * i_kc

def calculate_cosine_similarity_matrix(pkc, q_kc):
    u_kc_matrix = np.array(pkc)
    q_kc_matrix = np.array(q_kc)
    dot_product_matrix = np.dot(u_kc_matrix, q_kc_matrix.T)
    u_norms = np.linalg.norm(u_kc_matrix, axis=1, keepdims=True)
    q_norms = np.linalg.norm(q_kc_matrix, axis=1, keepdims=True)
    u_norms[u_norms == 0] = 1
    q_norms[q_norms == 0] = 1
    similarity_matrix = dot_product_matrix / (u_norms @ q_norms.T)

    similarity_matrix = np.nan_to_num(similarity_matrix)
    
    return similarity_matrix
sim_kc = calculate_cosine_similarity_matrix(pkc,q_kc)
sim_kc_2 = np.square(sim_kc)
for use_id in range(len(orirow_data)):
    for q in q_concepts:
        q_r = 1
        for i in q:
            if i in u_concepts[use_id]:
                q_r *= u_preds[use_id][u_concepts[use_id].index(i)]
            else:
                q_r *= 0
                dr = 1
                dis = (0.7 - dr)**2
                d = abs(0.7-dr)
                break
            dr = 1 - q_r
            dis = (0.7 - dr)**2
            d = abs(0.7-dr)
        all_scores.append(dr)
        all_dis.append(dis)
        all_diss.append(d)

def split_list(data, group_size):
    grouped_data = [data[i:i + group_size] for i in range(0, len(data), group_size)]
    return grouped_data

D_q = split_list(all_scores,group_size)
dis = split_list(all_dis,group_size)
diss = split_list(all_diss,group_size)
ac_1 = np.sqrt(dis)
sum = sim_kc_2 + dis
ou = sum

    
    

def get_top_200_min_indices(data):
    indexed_data = list(enumerate(data))
    top = sorted(indexed_data, key=lambda x: x[1])
    top_200 = []
    n = 0
    m = 0
    a = 0 
    
    if dataset == "mooccubex":
        max_candidate_num = len(top)
    else:
        max_candidate_num = min(200, len(top))
        
    for i in top:
        if n < 120:
            top_200.append(i)
            if i[0] < len(q_concepts):
                a = q_concepts[i[0]]
            n += 1
            m += 1
            if m == max_candidate_num:
                break
        if n >= 120 and i[0] < len(q_concepts) and a != q_concepts[i[0]]:
            top_200.append(i)
            n = 0
            m += 1
            if m == max_candidate_num:
                break
                
    top_200_values = [x[1] for x in top_200]
    top_200_indices = [x[0] for x in top_200]
    return top_200_values, top_200_indices

ex = []
ac = []
for a in ou:
    top_200_values, top_200_indices = get_top_200_min_indices(a)
    ex.append(top_200_indices)

students = []
exercises = []
for i,exercise in enumerate(ex):
    students.append(i)
    exercises.append(exercise)
df_ex = pd.DataFrame({
    'students': students,
    'exercise': exercises
})
print('CS_EXERCISE:')
print(df_ex)


def CaculateFitness(X,fun,q_kc,array):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :],q_kc,array)
    return fitness

def fitness_set(xx,q_kc,array):
    d = 0
    for i in range(len(xx)):
      for j in range(i+1,len(xx)):
          index_i = int(xx[i] - 1)
          index_j = int(xx[j] - 1)
          if index_i >= len(array) or index_j >= len(array):

              continue
              
          real_qid_i = array[index_i]
          real_qid_j = array[index_j]
          
          if real_qid_i >= len(q_kc) or real_qid_j >= len(q_kc):
              
              continue
          distance = math.sqrt(np.sum([(m - n)**2 for m, n in zip(q_kc[array[index_i]], q_kc[array[index_j]])]))
          d += distance
    return d

def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)[::-1]
    index = np.argsort(Fit, axis=0)[::-1]
    return fitness,index
def SortPosition(X,index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i,:] = X[index[i],:]
    return Xnew

def PDUpdate(X, PDNumber, ST, Max_iter, dim):
    X_new = copy.copy(X)
    R2 = random.random()
    for p in range(PDNumber):
            if R2 < ST:
                X_new[p, :] = X[p, :] * np.exp(-p / (random.random() * Max_iter))
            else:
               X_new[p, :] = X[p, :] + np.random.randn(20) 
    return X_new
def JDUpdate(X, JDNumber, pop, dim, PDNumber):
    X_new = copy.copy(X)
    for i in range(JDNumber):
        a = PDNumber + i 
        if i > (JDNumber) / 2:
                X_new[a, :] = np.random.randn(1,20) * np.exp((X[-1, :] - X[a, :]) / i ** 2)
        else:
                A = np.random.choice([1, -1], size=(1, dim))
                A_T = np.transpose(A)
                A =  np.dot(A_T,(np.dot(A, A_T))^(-1))
                AA = np.dot(np.abs(X[a, :] - X[PDNumber-1, :]),A)*np.ones([1,dim])
                X_new[a, :] = X[PDNumber-1, :] + AA
    return X_new
def SDUpdate(X, pop, SDNumber, fitness, BestF):
    X_new = copy.copy(X)
    dim = X.shape[1]
    Temp = range(pop)
    RandIndex = random.sample(Temp, pop)
    SDchooseIndex = RandIndex[0:SDNumber]
    for i in range(SDNumber):
            if fitness[SDchooseIndex[i]] > BestF:
                X_new[SDchooseIndex[i], :] = X[0, j] + np.random.randn(1,20) * np.abs(X[SDchooseIndex[i], :] - X[0, :])
            elif fitness[SDchooseIndex[i]] == BestF:
                K = 2 * random.random() - 1
                X_new[SDchooseIndex[i], :] = X[SDchooseIndex[i], :] + K * (
                        np.abs(X[SDchooseIndex[i], :] - X[-1, :]) / (fitness[SDchooseIndex[i]] - fitness[-1] + 10E-8))
    return X_new
def BorderCheck(X,ub,lb,pop,dim):
    for i in range(pop):
        for j in range(dim):
            if X[i,j]>ub[j]:
              while True:
                a = np.random.rand() * (ub[j] - lb[j]) + lb[j]
                if int(a) not in X[i,:].astype(int):
                    X[i, j] = a
                    break
            elif X[i,j]<lb[j]:
                while True:
                  a = np.random.rand() * (ub[j] - lb[j]) + lb[j]
                  if int(a) not in X[i,:].astype(int):
                      X[i, j] = a
                      break
    return X



cishu = 0
new_ex = []
for array in ex[0:200]:
    new_ex_list = []
    pop = 50
    dim = 20
    lb = np.ones(dim, dtype=int)
    
    if dataset == "mooccubex":
        ub = np.ones(dim, dtype=int) * len(array)
    else:
        ub = np.ones(dim, dtype=int) * 200
        
    Max_iter = 200
    ST = 0.8
    PD = 0.3
    JD = 0.4
    SD = 0.3
    PDNumber = int(pop * PD)
    JDNumber = int(pop * JD)
    SDNumber = int(pop * SD)

    X = np.zeros([pop, dim], dtype=int)
    for i in range(pop):
        b = []
        for j in range(dim):
            while True:
                if dataset == "mooccubex":
                    upper_bound = len(array)
                else:
                    upper_bound = 200
                    
                a = np.random.randint(1, upper_bound + 1)
                if a not in b:
                    X[i, j] = a
                    b.append(a)
                    break

    fitness = CaculateFitness(X,fitness_set,q_kc,array)
    fitness,sortIndex = SortFitness(fitness)
    X = SortPosition(X,sortIndex)
    GbestScore = copy.copy(fitness[0])
    GbestPositon = np.zeros([1,dim])
    GbestPositon[0,:] = copy.copy(X[0,:])
    Curve = np.zeros([Max_iter,1])
    for i in range(Max_iter):
        BestF = fitness[0]
        
        X = PDUpdate(X,PDNumber,ST,Max_iter,dim)

        X = JDUpdate(X,PDNumber,pop,dim,PDNumber)

        X = SDUpdate(X,pop,SDNumber,fitness,BestF)

        X = BorderCheck(X,ub,lb,pop,dim)

        fitness = CaculateFitness(X,fitness_set,q_kc,array)

        fitness,sortIndex = SortFitness(fitness)
        X = SortPosition(X,sortIndex)
        if(fitness[0]<=GbestScore): 
            GbestScore = copy.copy(fitness[0])
            GbestPositon[0,:] = copy.copy(X[0,:])
        Curve[i] = GbestScore
        
    print('studente:',cishu)
    cishu+=1
    for x in GbestPositon:
      for i in x:
        index = int(i - 1)
        new_ex_list.append(array[index])
    print('ex:',new_ex_list)
    new_ex.append(new_ex_list)
students = []
exercises = []
for i,exercise in enumerate(new_ex):
    students.append(i)
    exercises.append(exercise)
df_ex = pd.DataFrame({
    'students': students,
    'exercise': exercises
})
print('REC_EXERCISE:')
print(df_ex)

with open(f'{dataset}/new_ex.txt', 'w') as f:
    for student, exercise in enumerate(new_ex):
        exercise_str = ','.join(map(str, exercise))
        f.write(f"{student}\t{exercise_str}\n")
