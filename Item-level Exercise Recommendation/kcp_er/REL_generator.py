import numpy as np
import pandas as pd
import torch
import random
from tqdm import tqdm


def distance(L):
    norm = np.linalg.norm(L, axis=1, keepdims=True)
    L_norm = L / norm
    similarity_matrix = np.dot(L_norm, L_norm.T)
    np.fill_diagonal(similarity_matrix, 1)
    dl = 1 - similarity_matrix
    return np.mean(dl)


def REL_generator(Q, es, REL_n=10, tao=100, kB=1, c=0.095, inner_loop=100):
    L = es[:REL_n]
    while tao > 10:
        i = 0
        L_new = None
        while i < inner_loop:
            e = random.choice(es)
            i += 1
            L_temp = L.copy()
            random_index = random.randint(0, len(L) - 1)
            L_temp[random_index] = e
            dist_L_temp = distance(Q[L_temp])
            dist_L = distance(Q[L])
            if dist_L_temp >= dist_L:
                L_new = L_temp
            else:
                p = np.exp(-(dist_L_temp - dist_L) / (kB * tao))
                gama = random.uniform(0, 1)
                if gama >= p:
                    L_new = L_temp
                else:
                    L_new = L
        L = L_new
        tao = c * tao
    return L


if __name__ == '__main__':
    dataset = 'assist2017'
    N = 150
    Q = np.load(f'datasets/{dataset}/select/Q.npy')
    ES = []
    with open(f'datasets/{dataset}/select/EB_filter_{N}.txt', 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            ES.append([int(q) for q in line[1].split(',')])
    REL = []
    REL_n = 100
    random.seed(2025)
    for es in tqdm(ES):
        rel = REL_generator(Q, es, REL_n=REL_n)
        REL.append(rel)
    test_raw = pd.read_csv(f'datasets/{dataset}/select/test_sequences.csv')
    uid = test_raw['uid'].tolist()
    with open(f'datasets/{dataset}/select/REL_generator_result_{REL_n}.txt', 'w') as f:
        for i in range(len(REL)):
            rel = REL[i]
            filter_q = ','.join([str(q) for q in rel])
            f.write(f"{uid[i]}\t{filter_q}\n")
