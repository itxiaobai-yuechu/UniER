import os

import numpy as np
from elasticsearch import Elasticsearch


class KTDataset:
    def __init__(self, data_folder, max_len=200):
        folder_name = os.path.basename(data_folder)
        self.dataset_name = folder_name
        with np.load(os.path.join(data_folder, folder_name + '_dataOff.npz'), allow_pickle=True) as data:
            self.data = [data[k] for k in ['skill', 'y', 'real_len']]
        self.data[1] = [_.astype(np.float32) for _ in self.data[1]]
        try:
            self.feats_num = np.max(self.data[0]).item() + 1
        except ValueError:
            self.feats_num = np.max(np.concatenate(self.data[0])).item() + 1
            
        if self.dataset_name == 'assist09':
            self.feats_num = max(self.feats_num, 123)
        elif self.dataset_name == 'assist12':
            self.feats_num = max(self.feats_num, 265)
        elif self.dataset_name == 'assist15':
            self.feats_num = max(self.feats_num, 100)
        elif self.dataset_name == 'assist17':
            self.feats_num = max(self.feats_num, 102)
        elif self.dataset_name == 'algebra2005':
            self.feats_num = max(self.feats_num, 112)
        elif self.dataset_name == 'bridge2006':
            self.feats_num = max(self.feats_num, 493)
        elif self.dataset_name == 'ednet':
            self.feats_num = max(self.feats_num, 188)
        elif self.dataset_name == 'junyi':
            self.feats_num = max(self.feats_num, 39)
        elif self.dataset_name == 'nips34':
            self.feats_num = max(self.feats_num, 57)
        elif self.dataset_name == 'xes3g5m':
            self.feats_num = max(self.feats_num, 865)
        elif self.dataset_name == 'mooccubex':
            self.feats_num = max(self.feats_num, 438)

        self.data = list(zip(*self.data))
        self.users_num = len(self.data)
        self.max_len = max_len
        self.mask = np.zeros(self.max_len, dtype=bool)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        skill, y, real_len = self.data[item]
        skill, y = skill[:self.max_len], y[:self.max_len]
        if len(skill) < self.max_len:
            skill, y = skill.copy(), y.copy()
            skill.resize((self.max_len,))
            y.resize((self.max_len,))
        mask = self.mask.copy()
        mask[:real_len] = True
        return skill, y, mask


class RecDataset(KTDataset):
    def __getitem__(self, item):
        skill, _, real_len = self.data[item]
        skill = skill[:self.max_len + 1]
        if len(skill) < self.max_len + 1:
            skill = skill.copy()
            skill.resize((self.max_len + 1,))
        mask = self.mask.copy()
        mask[:real_len - 1] = True
        return skill[:-1], skill[1:], mask


class RetrievalDataset(KTDataset):
    def __init__(self, data_folder, r=5, train_test_split=0.8, max_len=200):
        super(RetrievalDataset, self).__init__(data_folder, max_len)
        self.es = Elasticsearch(hosts=['http://localhost:9200/']).options(
            request_timeout=20,
            retry_on_timeout=True,
            ignore_status=[400, 404]
        )
        self.safe_users = np.arange(int(len(self.data) * train_test_split))
        self.R = r
        self.index = f'{self.dataset_name}_train'
        self.safe_query = self.get_safe_query()

    def get_safe_query(self):
        safe_query = [[self.data[i][0][0], self.data[i][1][0]] for i in self.safe_users]
        safe_query = np.asarray(safe_query, dtype=np.int32)
        return safe_query

    def get_query(self, user, skills, index_range):
        safe_user = np.random.choice(self.safe_users, self.R + 1, replace=False)
        safe_user = safe_user[safe_user != user][:self.R]
        safe_query = self.safe_query[safe_user]
        query_s = []
        skills_str = ' '
        for _ in index_range:
            skills_str += f' {skills[_]}'
            query = [{'index': self.index},
                     {'size': self.R,
                      'query': {'bool': {'must': [{'term': {'skill': skills[_]}}, {'match': {'history': skills_str}}],
                                         'must_not': {'term': {'user': user}}}}}]
            query_s += query
        result = self.es.msearch(searches=query_s)['responses']
        r_his, r_skill_y, r_len = [], [], []
        for rs in result:
            skill_y, real_len = [], []
            rs = rs['hits']['hits']
            for r in rs:
                r = r['_source']
                his = np.fromstring(r['history'], dtype=np.int32, sep=' ')
                his = np.stack((his, np.asarray(r['y'], dtype=np.int32)), axis=-1)
                if his.ndim == 1:
                    his = np.expand_dims(his, 0)
                his.resize((self.max_len, 2))
                r_his.append(his)
                skill_y.append(his[-1])
                real_len.append(len(skill_y))
            for _ in range(self.R - len(rs)):
                his = safe_query[_:_ + 1].copy()
                his.resize((self.max_len, 2))
                r_his.append(his)
                skill_y.append(his[_])
                real_len.append(len(skill_y))
            r_skill_y.append(skill_y)
            r_len.append(real_len)
        r_his, r_skill_y, r_len = np.asarray(r_his, dtype=np.int32), np.asarray(r_skill_y, dtype=np.int32), np.asarray(
            r_len, dtype=np.int32)
        r_his.resize((self.max_len * self.R, self.max_len, 2))
        r_skill_y.resize((self.max_len, self.R, 2))
        r_len.resize((self.max_len, self.R))
        r_len[r_len < 1] = 1
        return r_his, r_skill_y, r_len

    def __getitem__(self, item):
        skill, y, real_len = self.data[item]
        skill, y = skill[:self.max_len], y[:self.max_len]
        r_his, r_skill_y, r_len = self.get_query(item, skill, range(real_len))
        if len(skill) < self.max_len:
            skill, y = skill.copy(), y.copy()
            skill.resize((self.max_len,))
            y.resize((self.max_len,))
        mask = self.mask.copy()
        mask[:real_len] = True
        return skill, r_his, r_skill_y, y, mask, r_len
