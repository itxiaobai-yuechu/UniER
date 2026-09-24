import os
import sys
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_path[:cur_path.find('PKSD')] + 'PKSD')
import codecs
import numpy as np
import copy
import time
import random
import matplotlib.pyplot as plt
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import operator
from EduSim.Envs.agent_utils import get_proj_path, get_raw_data_path
import math
import networkx as nx
import random
import sys
import os
import json

sys.path.append(os.path.dirname(sys.path[0]))
from EduSim.Envs.KSS import KSSEnv
from EduSim.Envs.KES import KESEnv
from EduSim.Envs.KES_ASSIST15 import KESASSISTEnv
import gym
import warnings

warnings.filterwarnings('ignore')

relation_tph = {}
relation_hpt = {}


def dataloader(graph):
    print("load file...")

    entity = list(graph.nodes())
    relation = [0]

    triple_list = []
    relation_head = {}
    relation_tail = {}

    for edge in graph.edges():
        h_, t_ = edge[0], edge[1]
        r_ = 0

        triple_list.append([h_, r_, t_])

        if r_ in relation_head:
            if h_ in relation_head[r_]:
                relation_head[r_][h_] += 1
            else:
                relation_head[r_][h_] = 1
        else:
            relation_head[r_] = {}
            relation_head[r_][h_] = 1

        if r_ in relation_tail:
            if t_ in relation_tail[r_]:
                relation_tail[r_][t_] += 1
            else:
                relation_tail[r_][t_] = 1
        else:
            relation_tail[r_] = {}
            relation_tail[r_][t_] = 1

    for r_ in relation_head:
        sum1, sum2 = 0, 0
        for head in relation_head[r_]:
            sum1 += 1
            sum2 += relation_head[r_][head]
        tph = sum2 / sum1
        relation_tph[r_] = tph
    for r_ in relation_tail:
        sum1, sum2 = 0, 0
        for tail in relation_tail[r_]:
            sum1 += 1
            sum2 += relation_tail[r_][tail]
        hpt = sum2 / sum1
        relation_hpt[r_] = hpt

    valid_triple_list = random.sample(triple_list, int(len(triple_list) / 5.0))
    final_triple_list = [x for x in triple_list if x not in valid_triple_list]
    triple_list = final_triple_list

    print("Complete load. entity : %d , relation : %d , train triple : %d, , valid triple : %d" % (
        len(entity), len(relation), len(triple_list), len(valid_triple_list)))

    return entity, relation, triple_list, valid_triple_list


def norm_l1(h, r, t):
    return np.sum(np.fabs(h + r - t))


def norm_l2(h, r, t):
    return np.sum(np.square(h + r - t))


class E(nn.Module):
    def __init__(self, entity_num, relation_num, dim, margin, norm, C, device=None):
        super(E, self).__init__()
        self.entity_num = entity_num
        self.relation_num = relation_num
        self.dim = dim
        self.margin = margin
        self.norm = norm
        self.C = C
        self.device = torch.device(device) if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.ent_embedding = torch.nn.Embedding(num_embeddings=self.entity_num,
                                                embedding_dim=self.dim)
        self.rel_embedding = torch.nn.Embedding(num_embeddings=self.relation_num,
                                                embedding_dim=self.dim)

        self.loss_F = nn.MarginRankingLoss(self.margin, reduction="mean")

        self.__data_init()

    def __data_init(self):
        nn.init.xavier_uniform_(self.ent_embedding.weight.data)
        nn.init.xavier_uniform_(self.rel_embedding.weight.data)
        self.normalization_rel_embedding()
        self.normalization_ent_embedding()

    def normalization_ent_embedding(self):
        norm = self.ent_embedding.weight.detach().cpu().numpy()
        norm_weight = np.sqrt(np.sum(np.square(norm), axis=1, keepdims=True))
        norm = np.divide(norm, norm_weight, out=np.zeros_like(norm), where=norm_weight != 0)

        self.ent_embedding.weight.data.copy_(torch.from_numpy(norm))

    def normalization_rel_embedding(self):
        norm = self.rel_embedding.weight.detach().cpu().numpy()
        norm_weight = np.sqrt(np.sum(np.square(norm), axis=1, keepdims=True))
        norm = np.divide(norm, norm_weight, out=np.zeros_like(norm), where=norm_weight != 0)
        self.rel_embedding.weight.data.copy_(torch.from_numpy(norm))

    def input_pre_transe(self, ent_vector, rel_vector):
        for i in range(self.entity_num):
            self.ent_embedding.weight.data[i] = torch.as_tensor(
                ent_vector[i], device=self.ent_embedding.weight.device
            )
        for i in range(self.relation_num):
            self.rel_embedding.weight.data[i] = torch.as_tensor(
                rel_vector[i], device=self.rel_embedding.weight.device
            )

    def distance(self, h, r, t):
        head = self.ent_embedding(h)
        rel = self.rel_embedding(r)
        tail = self.ent_embedding(t)

        distance = head + rel - tail
        score = torch.norm(distance, p=self.norm, dim=1)
        return score

    def test_distance(self, h, r, t):

        head = self.ent_embedding(h.to(self.device))
        rel = self.rel_embedding(r.to(self.device))
        tail = self.ent_embedding(t.to(self.device))

        distance = head + rel - tail

        score = torch.norm(distance, p=self.norm, dim=1)
        return score.cpu().detach().numpy()

    def scale_loss(self, embedding):
        return torch.clamp(torch.sum(embedding ** 2, dim=1, keepdim=True) - 1.0, min=0.0).sum()

    def forward(self, current_triples, corrupted_triples):
        h, r, t = torch.chunk(current_triples, 3, dim=1)
        h_c, r_c, t_c = torch.chunk(corrupted_triples, 3, dim=1)

        h = torch.squeeze(h, dim=1).to(self.device)
        r = torch.squeeze(r, dim=1).to(self.device)
        t = torch.squeeze(t, dim=1).to(self.device)
        h_c = torch.squeeze(h_c, dim=1).to(self.device)
        r_c = torch.squeeze(r_c, dim=1).to(self.device)
        t_c = torch.squeeze(t_c, dim=1).to(self.device)

        pos = self.distance(h, r, t)
        neg = self.distance(h_c, r_c, t_c)

        entity_embedding = self.ent_embedding(torch.cat([h, t, h_c, t_c]))
        relation_embedding = self.rel_embedding(torch.cat([r, r_c]))

        y = torch.full_like(pos, -1.0, device=self.device)
        loss = self.loss_F(pos, neg, y)

        ent_scale_loss = self.scale_loss(entity_embedding)
        rel_scale_loss = self.scale_loss(relation_embedding)
        return loss + self.C * (ent_scale_loss / len(entity_embedding) + rel_scale_loss / len(relation_embedding))


class TransE:
    def __init__(self, entity, relation, triple_list, embedding_dim=50, lr=0.01, margin=1.0, norm=1, C=1.0,
                 valid_triple_list=None, device=None):
        self.entities = entity
        self.relations = relation
        self.triples = triple_list
        self.dimension = embedding_dim
        self.learning_rate = lr
        self.margin = margin
        self.norm = norm
        self.loss = 0.0
        self.valid_loss = 0.0
        self.valid_triples = valid_triple_list
        self.train_loss = []
        self.validation_loss = []

        self.test_triples = []

        self.C = C
        self.device = torch.device(device) if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

    def data_initialise(self):
        self.model = E(
            len(self.entities), len(self.relations), self.dimension,
            self.margin, self.norm, self.C, self.device
        ).to(self.device)
        self.optim = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def training_run(self, epochs=20, batch_size=10, np_save_path=''):

        n_batches = int(len(self.triples) / batch_size)
        valid_batch = int(len(self.valid_triples) / batch_size) + 1
        print("batch size: ", n_batches, "valid_batch: ", valid_batch)
        for epoch in range(epochs):

            start = time.time()
            self.loss = 0.0
            self.valid_loss = 0.0

            for batch in range(n_batches):

                batch_samples = random.sample(self.triples, batch_size)

                current = []
                corrupted = []
                for sample in batch_samples:
                    corrupted_sample = copy.deepcopy(sample)
                    pr = np.random.random(1)[0]
                    p = relation_tph[int(corrupted_sample[1])] / (
                            relation_tph[int(corrupted_sample[1])] + relation_hpt[int(corrupted_sample[1])])

                    if pr < p:
                        corrupted_sample[0] = random.sample(self.entities, 1)[0]
                        while corrupted_sample[0] == sample[0]:
                            corrupted_sample[0] = random.sample(self.entities, 1)[0]
                    else:
                        corrupted_sample[2] = random.sample(self.entities, 1)[0]
                        while corrupted_sample[2] == sample[2]:
                            corrupted_sample[2] = random.sample(self.entities, 1)[0]

                    current.append(sample)
                    corrupted.append(corrupted_sample)

                current = torch.from_numpy(np.array(current)).long()
                corrupted = torch.from_numpy(np.array(corrupted)).long()
                self.update_triple_embedding(current, corrupted)

            for batch in range(valid_batch):

                batch_samples = random.sample(self.valid_triples, batch_size)

                current = []
                corrupted = []
                for sample in batch_samples:
                    corrupted_sample = copy.deepcopy(sample)
                    pr = np.random.random(1)[0]
                    p = relation_tph[int(corrupted_sample[1])] / (
                            relation_tph[int(corrupted_sample[1])] + relation_hpt[int(corrupted_sample[1])])

                    if pr > p:
                        corrupted_sample[0] = random.sample(self.entities, 1)[0]
                        while corrupted_sample[0] == sample[0]:
                            corrupted_sample[0] = random.sample(self.entities, 1)[0]
                    else:
                        corrupted_sample[2] = random.sample(self.entities, 1)[0]
                        while corrupted_sample[2] == sample[2]:
                            corrupted_sample[2] = random.sample(self.entities, 1)[0]

                    current.append(sample)
                    corrupted.append(corrupted_sample)

                current = torch.from_numpy(np.array(current)).long()
                corrupted = torch.from_numpy(np.array(corrupted)).long()
                self.calculate_valid_loss(current, corrupted)

            end = time.time()
            mean_train_loss = self.loss / n_batches
            mean_valid_loss = self.valid_loss / valid_batch
            print("epoch: ", epoch, "cost time: %s" % (round((end - start), 3)))
            print("Train loss: ", mean_train_loss, "Valid loss: ", mean_valid_loss)

            self.train_loss.append(float(mean_train_loss))
            self.validation_loss.append(float(mean_valid_loss))

        fig = plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(self.train_loss) + 1), self.train_loss, label='Train Loss')
        plt.plot(range(1, len(self.validation_loss) + 1), self.validation_loss, label='Validation Loss')

        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.xlim(0, len(self.train_loss) + 1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.title("TransE Training loss")
        plt.show()

        result = []
        for i, e in enumerate(self.model.ent_embedding.weight):
            result.append(e.cpu().detach().numpy().tolist())
        result = np.array(result)
        np.save(np_save_path, result)
        print('numpy saved')

    def update_triple_embedding(self, correct_sample, corrupted_sample):
        self.optim.zero_grad()
        loss = self.model(correct_sample, corrupted_sample)
        self.loss += loss
        loss.backward()
        self.optim.step()

    def calculate_valid_loss(self, correct_sample, corrupted_sample):
        loss = self.model(correct_sample, corrupted_sample)
        self.valid_loss += loss

    def test_run(self, filter=False):

        self.filter = filter
        hits = 0
        rank_sum = 0
        num = 0

        for triple in self.test_triples:
            start = time.time()
            num += 1
            print(num, triple)
            rank_head_dict = {}
            rank_tail_dict = {}
            head_embedding = []
            tail_embedding = []
            norm_relation = []
            hyper_relation = []
            tamp = []

            head_filter = []
            tail_filter = []
            if self.filter:

                for tr in self.triples:
                    if tr[1] == triple[1] and tr[2] == triple[2] and tr[0] != triple[0]:
                        head_filter.append(tr)
                    if tr[0] == triple[0] and tr[1] == triple[1] and tr[2] != triple[2]:
                        tail_filter.append(tr)
                for tr in self.test_triples:
                    if tr[1] == triple[1] and tr[2] == triple[2] and tr[0] != triple[0]:
                        head_filter.append(tr)
                    if tr[0] == triple[0] and tr[1] == triple[1] and tr[2] != triple[2]:
                        tail_filter.append(tr)
                for tr in self.valid_triples:
                    if tr[1] == triple[1] and tr[2] == triple[2] and tr[0] != triple[0]:
                        head_filter.append(tr)
                    if tr[0] == triple[0] and tr[1] == triple[1] and tr[2] != triple[2]:
                        tail_filter.append(tr)

            for i, entity in enumerate(self.entities):

                head_triple = [entity, triple[1], triple[2]]
                if self.filter:
                    if head_triple in head_filter:
                        continue
                head_embedding.append(head_triple[0])
                norm_relation.append(head_triple[1])
                tail_embedding.append(head_triple[2])

                tamp.append(tuple(head_triple))

            head_embedding = torch.from_numpy(np.array(head_embedding)).long()
            norm_relation = torch.from_numpy(np.array(norm_relation)).long()
            tail_embedding = torch.from_numpy(np.array(tail_embedding)).long()
            distance = self.model.test_distance(head_embedding, norm_relation, tail_embedding)

            for i in range(len(tamp)):
                rank_head_dict[tamp[i]] = distance[i]

            head_embedding = []
            tail_embedding = []
            norm_relation = []
            hyper_relation = []
            tamp = []

            for i, tail in enumerate(self.entities):

                tail_triple = [triple[0], triple[1], tail]
                if self.filter:
                    if tail_triple in tail_filter:
                        continue
                head_embedding.append(tail_triple[0])
                norm_relation.append(tail_triple[1])
                tail_embedding.append(tail_triple[2])
                tamp.append(tuple(tail_triple))

            head_embedding = torch.from_numpy(np.array(head_embedding)).long()
            norm_relation = torch.from_numpy(np.array(norm_relation)).long()
            tail_embedding = torch.from_numpy(np.array(tail_embedding)).long()

            distance = self.model.test_distance(head_embedding, norm_relation, tail_embedding)
            for i in range(len(tamp)):
                rank_tail_dict[tamp[i]] = distance[i]



            rank_head_sorted = sorted(rank_head_dict.items(), key=operator.itemgetter(1), reverse=False)
            rank_tail_sorted = sorted(rank_tail_dict.items(), key=operator.itemgetter(1), reverse=False)

            i = 0
            for i in range(len(rank_head_sorted)):
                if triple[0] == rank_head_sorted[i][0][0]:
                    if i < 10:
                        hits += 1
                    rank_sum = rank_sum + i + 1
                    break

            i = 0
            for i in range(len(rank_tail_sorted)):
                if triple[2] == rank_tail_sorted[i][0][2]:
                    if i < 10:
                        hits += 1
                    rank_sum = rank_sum + i + 1
                    break
            end = time.time()
            print("epoch: ", num, "cost time: %s" % (round((end - start), 3)), str(hits / (2 * num)),
                  str(rank_sum / (2 * num)))
        self.hit_10 = hits / (2 * len(self.test_triples))
        self.mean_rank = rank_sum / (2 * len(self.test_triples))
        print("hits@10: ", self.hit_10)
        print("meanrank: ", self.mean_rank)
        f = open("./result.txt", 'w')
        f.write("hits@10: " + str(self.hit_10) + '\n')
        f.write("meanrank: " + str(self.mean_rank) + '\n')
        f.close()
        return self.hit_10, self.mean_rank


if __name__ == '__main__':
    args = sys.argv[1]
    print('cuda_ava:' + str(torch.cuda.is_available()))
    envKSS: KSSEnv = gym.make('KSS-v2')
    dataRecPath = f'{get_proj_path()}/data/dataProcess/junyi/data/dataRec'
    envKES: KESEnv = gym.make('KES-v1', dataRec_path=dataRecPath)
    dataRecPath = f'{get_raw_data_path()}/ASSISTments2015/processed/'
    envKESASSIST: KESASSISTEnv = gym.make('KESASSIST-v1', dataRec_path=dataRecPath)


    is_directed = True
    test_flag = False
    if args == '1':
        graph = envKSS.knowledge_structure
    elif args == '2':
        graph = envKES.knowledge_structure
    elif args == '3':
        graph = envKESASSIST.knowledge_structure
    else:
        graph = envKESASSIST.knowledge_structure

    if not is_directed:
        graph = graph.to_undirected()
    for edge in graph.edges():
        u, v = edge[0], edge[1]
        graph[u][v]['weight'] = 1.0

    if len(graph.nodes) == 10:
        batch_size = 1
        np_save_path = f'{get_proj_path()}/EduSim/Envs/meta_data/TransEKSSGraphEmbedding.npy'
    elif len(graph.nodes) == 100:
        batch_size = 16
        np_save_path = f'{get_proj_path()}/EduSim/Envs/meta_data/TransEASSISTGraphEmbedding.npy'
    elif len(graph.nodes) == 835:
        batch_size = 30
        np_save_path = f'{get_proj_path()}/EduSim/Envs/meta_data/TransEjunyiGraphEmbedding.npy'
    else:
        raise ValueError('Wrong Graph')


    entity_set, relation_set, triple_list, valid_triple_list = dataloader(graph)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transE = TransE(entity_set, relation_set, triple_list, embedding_dim=50, lr=0.001, margin=4.0, norm=1, C=0.25,
                    valid_triple_list=valid_triple_list, device=device)
    transE.data_initialise()

    transE.training_run(epochs=500, batch_size=batch_size, np_save_path=np_save_path)
