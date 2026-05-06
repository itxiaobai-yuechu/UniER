from torch.utils import data
import csv
from math import *
import numpy as np
import scipy.special as spe
from pad_seq_util import PadSeqUtil
import json
import random
import torch
import os


def data_preprocess(args):
    train_dataset_all = []
    validation_dataset_all = []
    finetune_dataset_all = []
    test_dataset_all = []
    for file_name in (args.train_name):
        train_dataset, validation_dataset = read_json_stu(args, args.file_dir + "/" + file_name, True)
        train_dataset_all.append(train_dataset)
        validation_dataset_all.append(validation_dataset)
    for file_name in (args.test_name):
        tune_dataset, test_dataset = read_json_stu(args, args.file_dir + "/" + file_name, False)
    train_loader_all = []
    valid_loader_all = []
    task_num = len(args.train_name)
    max_batch_count = 0

    for id_task in range(len(train_dataset_all)):
        train_ERset = ER_Dataset(train_dataset_all[id_task])
        valid_ERset = ER_Dataset(validation_dataset_all[id_task])
        train_loader_all.append(data.DataLoader(train_ERset, batch_size=args.batch_size, shuffle=False, drop_last=True))
        valid_loader_all.append(data.DataLoader(valid_ERset, batch_size=args.batch_size, shuffle=False, drop_last=True))
        max_batch_count = max(max_batch_count, len(train_loader_all[id_task]))

    iterator_all_train = []
    iterator_all_valid = []
    for task_n in range(len(train_dataset_all)):
        iterator_all_train.append(train_loader_all[task_n])
        iterator_all_valid.append(valid_loader_all[task_n])
    return max_batch_count, task_num, iterator_all_train, iterator_all_valid, tune_dataset, test_dataset

def read_json_stu(args, json_name, is_train):
    with open(json_name, "r") as f:
        data_raw = f.read()
        data = json.loads(data_raw)
        for i in range(len(data) - 1, -1, -1):
            user_exe = data[i]["logs"]
            exer_id = []
            for j in range(len(user_exe)):
                exer_id.append(int(user_exe[j]["exer_id"]))
            if len(exer_id) < args.min_log:
                data.pop(i)
        if is_train:
            if args.meta_training_flag == 1:
                training_ratio = args.training_ratio
            else:
                training_ratio = 1
        else:
            training_ratio = args.fine_tuning_ratio

        if is_train:
            data = data[0:min(len(data), args.train_stu_num)]
        else:
            data = data[0:min(len(data), args.test_stu_num)]
        train_set, test_set = {}, {}
        train_exer, train_score, train_kc, train_student_id = [], [], [], []
        test_exer, test_score, test_kc, test_student_id = [], [], [], []
        max_train = 0
        for i in range(int(len(data) * training_ratio)):
            user_exe = data[i]["logs"]
            exer_id = []
            score = []
            knowledge = []
            for j in range(len(user_exe)):
                exer_id.append(int(user_exe[j]["exer_id"]))
                score.append(int(user_exe[j]["score"]))
                kcs = np.array(int(user_exe[j]["knowledge_code"]))
                knowledge.append(kcs.tolist())
            max_train = max(max_train, len(exer_id))
            train_exer.append(exer_id)
            train_score.append(score)
            train_kc.append(knowledge)
            train_student_id.append(data[i].get("student_id", i))

        max_test = 0
        for i in range(int(len(data) * training_ratio), len(data)):
            user_exe = data[i]["logs"]
            exer_id = []
            score = []
            knowledge = []
            for j in range(len(user_exe)):
                exer_id.append(int(user_exe[j]["exer_id"]))
                score.append(int(user_exe[j]["score"]))
                kcs = np.array(int(user_exe[j]["knowledge_code"]))
                knowledge.append(kcs.tolist())
            max_test = max(max_test, len(exer_id))
            test_exer.append(exer_id)
            test_score.append(score)
            test_kc.append(knowledge)
            test_student_id.append(data[i].get("student_id", i))
        train_set['exercise'] = train_exer
        train_set['score'] = train_score
        train_set['kc'] = train_kc
        train_set['student_id'] = train_student_id
        train_set['max_length'] = max_train

        test_set['exercise'] = test_exer
        test_set['score'] = test_score
        test_set['kc'] = test_kc
        test_set['student_id'] = test_student_id
        test_set['max_length'] = max_test

        import logging
        logging.getLogger().setLevel(logging.INFO)
        if is_train:

            logging.info('training student_num ' + str(len(train_exer)))
            logging.info('validataion student_num ' + str(len(test_exer)))
        else:
            logging.info('fine-tuning student_num ' + str(len(train_exer)))
            logging.info('testing student_num ' + str(len(test_exer)))
    return train_set, test_set

class ER_Dataset(data.Dataset):
    def __init__(self, data_set):
        self.data_set = data_set
        maxlen = 200

        exer_seq, idx, mask_seq = PadSeqUtil.pad_sequence(data_set['exercise'], return_idx=True, return_mask=True, maxlen=maxlen)
        score_seq, _, _ = PadSeqUtil.pad_sequence(
            data_set['score'], dtype=np.float32,
            maxlen=maxlen
        )
        kc_seq, _, _ = PadSeqUtil.pad_sequence(
            data_set['kc'], dtype=np.int64,
            maxlen=maxlen
        )
        self.exer = exer_seq
        self.score = score_seq
        self.kc = kc_seq
        self.mask = mask_seq

        self.maxlen = maxlen
        self.origin_user_idx = idx
        student_ids = data_set.get('student_id', list(range(len(data_set['exercise']))))
        self.student_id = [student_ids[int(user_idx)] for user_idx in idx]

        segment_counter = {}
        segment_start = []
        for user_idx in idx:
            user_idx_int = int(user_idx)
            seg_no = segment_counter.get(user_idx_int, 0)
            segment_start.append(seg_no * maxlen)
            segment_counter[user_idx_int] = seg_no + 1
        self.segment_start = np.array(segment_start, dtype=np.int64)


    def __getitem__(self, index):
        exer = self.exer[index]
        score = self.score[index]
        mask = self.mask[index]
        kc = self.kc[index]
        data_set = {
            'exercise': exer,
            'score': score,
            'kc': kc,
            'mask': mask,
            'student_id': self.student_id[index],
            'origin_user_idx': self.origin_user_idx[index],
            'segment_start': self.segment_start[index]
        }
        return data_set

    def __len__(self):
        return len(self.exer)