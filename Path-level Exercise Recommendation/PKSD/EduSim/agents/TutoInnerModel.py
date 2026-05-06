import os
import sys
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_path[:cur_path.find('PKSD')] + 'PKSD')
import numpy as np
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from logging import getLogger
from sklearn.preprocessing import OneHotEncoder
from gym import spaces
from collections import defaultdict
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import random
from sklearn import metrics
import json
import re
from EduSim.Envs.agent_utils import get_proj_path, get_raw_data_path

logger = getLogger(__name__)


def get_DAS_input_from_log(history, item_id, n_items, n_skills, n_wins=5):
    X = {}
    X["users"] = np.array([0]).reshape(-1, 1)
    X["items"] = np.empty((0, n_items))
    X["skills"] = np.empty((0, n_skills))
    X["corrects"] = np.empty((0, n_skills * n_wins))
    X["attempts"] = np.empty((0, n_skills * n_wins))
    now_lag = np.zeros(1)

    item_vec = np.zeros(n_items)
    skill_vec = np.zeros(n_skills)
    skill_ids = [item_id]
    item_vec[item_id] = 1
    skill_vec[skill_ids] = 1
    X["items"] = np.vstack((X["items"], item_vec))
    X["skills"] = np.vstack((X["skills"], skill_vec))

    correct_vec = np.zeros(n_skills * n_wins)
    attempt_vec = np.zeros(n_skills * n_wins)
    last_time = -5
    for skill_id in skill_ids:
        correct_nums = 0
        attempt_nums = 0
        for j, inter in enumerate(history[max(len(history) - n_wins, 0):len(history)]):
            if inter[0] == item_id:
                last_time = j
            if inter[0] == skill_id:
                attempt_nums = attempt_nums + 1
                if inter[1] == 1:
                    correct_nums = correct_nums + 1

        correct_vec[skill_id * n_wins: (skill_id + 1) * n_wins] = np.log(1 + np.array(correct_nums))
        attempt_vec[skill_id * n_wins: (skill_id + 1) * n_wins] = np.log(1 + np.array(attempt_nums))
    now_lag[0] = len(history) - last_time
    X["corrects"] = np.vstack((X["corrects"], correct_vec))
    X["attempts"] = np.vstack((X["attempts"], attempt_vec))
    encoded_log_data = np.hstack((X["users"], X["items"], X["skills"], X["corrects"], X["attempts"]))
    encoded_los_data = torch.tensor(encoded_log_data.astype(np.float32)).view(-1)
    lag = torch.tensor(now_lag.astype(np.float32)).view(-1)
    return encoded_los_data, lag


class DASADataset(Dataset):
    def __init__(self, data_path, num_skills, feature_dim, max_sequence_length, n_wins=5, datasetType='ASSIST15'):
        self.data_path = data_path
        self.feature_dim = feature_dim
        self.num_skills = num_skills
        self.max_sequence_length = max_sequence_length
        self.n_wins = n_wins
        self.datasetType = datasetType
        if datasetType not in ['ASSIST15', 'junyi', 'KSS', 'ASSIST09']:
            raise ValueError('DASDataset Type wrong!')
        if datasetType == 'junyi':
            with open(self.data_path, 'r', encoding="utf-8") as f:
                self.datatxt = f.readlines()
            print(f'Total training {len(self.datatxt)} sessions ')
        elif datasetType == 'KSS':
            with open(data_path, 'r') as f:
                datatxt = f.readlines()
            self.KSS_all_sessions_data = []
            for i, line in enumerate(datatxt):
                if line[0] == '[':
                    items_answers = re.findall(r"\d+\.?\d*", line)
                    item_answers = [int(item_answer) for item_answer in items_answers]
                    items = item_answers[:int(len(item_answers) / 2)]
                    answers = item_answers[int(len(item_answers) / 2):]
                    self.KSS_all_sessions_data.append([[item, answers[i]] for i, item in enumerate(items)])
        elif datasetType == 'ASSIST09':
            data = np.load(data_path)
            problem, skill_num, problem_num, y, skill, real_len = data['problem'], data['skill_num'], data['problem_num'], data['y'], data['skill'], data['real_len']
            self.data_all = []
            for i in range(skill.shape[0]):
                self.data_all.append([])
                for j in range(real_len[i]):
                    self.data_all[i].append([int(skill[i, j]), int(y[i, j])])

    def __len__(self):
        if self.datasetType == 'junyi':
            return len(self.datatxt)
        elif self.datasetType == 'ASSIST15':
            return len(os.listdir(self.data_path))
        elif self.datasetType == 'KSS':
            return len(self.KSS_all_sessions_data)
        elif self.datasetType == 'ASSIST09':
            return len(self.data_all)

    def __getitem__(self, index):
        one_session_data = []
        id = index
        while len(one_session_data) == 0:
            if self.datasetType == 'junyi':
                line = self.datatxt[id]
                one_session_data = json.loads(line)
                if len(one_session_data) < 2:
                    one_session_data = []
                    id = random.randint(0, len(self.datatxt) - 1)
            elif self.datasetType == 'ASSIST15':
                with open(self.data_path + str(id) + '.csv', 'r') as f:
                    data = f.readlines()[1:]
                    data = data[:self.max_sequence_length]
                one_session_data = [[int(line.rstrip().split(',')[0]) - 1, int(line.rstrip().split(',')[1])] for i, line in enumerate(data)]
                if len(one_session_data) < 2:
                    one_session_data = []
                    id = random.randint(0, len(os.listdir(self.data_path)) - 1)
            elif self.datasetType == 'KSS':
                one_session_data = self.KSS_all_sessions_data[id]
                if len(one_session_data) < 2:
                    one_session_data = []
                    id = random.randint(0, len(self.KSS_all_sessions_data) - 1)
            elif self.datasetType == 'ASSIST09':
                one_session_data = self.data_all[index]
                if len(one_session_data) < 2:
                    one_session_data = []
                    id = random.randint(0, len(self.datatxt) - 1)

        rd = np.random.uniform()
        if rd > 0.9:
            one_session_data = one_session_data[:max(int(0.3 * len(one_session_data)), 2)]

        label = one_session_data[-1][1]
        encoded_los_data, lag = get_DAS_input_from_log(one_session_data[:-1], one_session_data[-1][0], n_items=self.num_skills, n_skills=self.num_skills, n_wins=self.n_wins)
        label = torch.tensor(label, dtype=torch.float32).view(-1)
        return encoded_los_data, lag, label


class MyLoss(torch.nn.Module):
    def __init__(self, coef):
        super().__init__()
        self.coef = coef

    def _gaussian_loss(self, float_m, float_std, tensor_input):
        float_m = torch.tensor(float_m, dtype=torch.float32).to(tensor_input.device)
        float_std = torch.tensor(float_std, dtype=torch.float32).to(tensor_input.device)
        ret = (1 - Normal(float_m, float_std).log_prob(tensor_input).exp() /
               Normal(float_m, float_std).log_prob(float_m).exp())
        return ret

    def forward(self, pred, target, named_parameters, prev_model_weight_dict):
        model_weight_dict = {}
        for name, param in named_parameters:
            model_weight_dict[name] = param

        fitting_loss = torch.nn.functional.mse_loss(input=pred, target=target, reduction="mean")
        alpha_gaussian_loss = self._gaussian_loss(0.0, 0.6, model_weight_dict["alpha"])
        delta_gaussian_loss = self._gaussian_loss(0.0, 0.95, model_weight_dict["delta"])
        beta_gaussian_loss = self._gaussian_loss(0.05, 0.4, model_weight_dict["beta"])
        attempt_gaussian_loss = self._gaussian_loss(0.008468969901277483, 0.1710594116668055, model_weight_dict["attempt"])
        correct_gaussian_loss = self._gaussian_loss(0.21114390920786338, 0.3781300105771393, model_weight_dict["correct"])
        alpha_fix_loss = torch.nn.functional.l1_loss(
            input=model_weight_dict["alpha"],
            target=prev_model_weight_dict["alpha"],
            reduction="mean",
        )
        delta_fix_loss = torch.nn.functional.l1_loss(
            input=model_weight_dict["delta"],
            target=prev_model_weight_dict["delta"],
            reduction="mean",
        )
        beta_fix_loss = torch.nn.functional.l1_loss(
            input=model_weight_dict["beta"],
            target=prev_model_weight_dict["beta"],
            reduction="mean",
        )
        attempt_fix_loss = torch.nn.functional.l1_loss(
            input=model_weight_dict["attempt"],
            target=prev_model_weight_dict["attempt"],
            reduction="mean",
        )
        correct_fix_loss = torch.nn.functional.l1_loss(
            input=model_weight_dict["correct"],
            target=prev_model_weight_dict["correct"],
            reduction="mean",
        )
        alpha_loss = (1 - self.coef[0]) * alpha_gaussian_loss.mean() + self.coef[0] * alpha_fix_loss
        delta_loss = (1 - self.coef[0]) * delta_gaussian_loss.mean() + self.coef[0] * delta_fix_loss
        beta_loss = (1 - self.coef[0]) * beta_gaussian_loss.mean() + self.coef[0] * beta_fix_loss
        attempt_loss = (1 - self.coef[0]) * attempt_gaussian_loss.mean() + self.coef[0] * attempt_fix_loss
        correct_loss = (1 - self.coef[0]) * correct_gaussian_loss.mean() + self.coef[0] * correct_fix_loss

        loss = (
                self.coef[1] * fitting_loss
                + self.coef[2] * alpha_loss
                + self.coef[3] * delta_loss
                + self.coef[4] * beta_loss
                + self.coef[5] * attempt_loss
                + self.coef[6] * correct_loss
        )
        return loss


class DAS3HInner(torch.nn.Module):
    def __init__(self, n_items, n_skills):
        super().__init__()
        init_attempt = [0.10642711, 0.05139345, -0.05963878, -0.06978262, 0.01394569]
        init_correct = [0.39765859, 0.05107533, 0.1765534, 0.23617729, 0.19425494]
        self.alpha = torch.nn.Parameter(-1 * torch.ones(1).view(-1, 1))
        self.delta = torch.nn.Parameter(torch.zeros(n_items).view(-1, 1))
        self.beta = torch.nn.Parameter(torch.zeros(n_skills).view(-1, 1))
        self.attempt = torch.nn.Parameter(
            torch.tensor(np.tile(init_attempt, n_skills).astype(np.float32)).view(-1, 1)
        )
        self.correct = torch.nn.Parameter(
            torch.tensor(np.tile(init_correct, n_skills).astype(np.float32)).view(-1, 1)
        )
        self.h = torch.ones(1).view(-1, 1) * 0.3
        self.d = torch.ones(1).view(-1, 1) * 0.8

    def forward(self, input_ts, lag_ts):
        self.h = self.h.to(input_ts.device)
        self.d = self.d.to(input_ts.device)
        weight = torch.cat([self.alpha, self.delta, self.beta, self.correct, self.attempt], dim=0)
        memory = torch.sigmoid(torch.mm(input_ts, weight))
        y_pred = (1 - memory) * (1 + self.h * lag_ts) ** (-self.d) + memory

        return y_pred


class InnerModel:
    def __init__(self, n_items, n_skills, n_wins=5, dataPath=None, lr=0.01, read_pretrain_weight=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_items = n_items
        self.n_wins = n_wins
        self.n_skills = n_skills
        self.learning_rate = lr
        self.KT_model = DAS3HInner(n_items, n_skills).to(self.device)
        self.optimizer = torch.optim.Adam(self.KT_model.parameters(), lr=self.learning_rate)
        coef_for_loss_fn = torch.tensor([0.5, 0.8, 0.2 / 6, 0.2 / 6, 0.2 / 6, 0.2 / 6, 0.2 / 6], dtype=torch.float32).to(self.device)
        self.loss_fn = MyLoss(coef_for_loss_fn)
        self.prev_weights_dict = None
        self.val_max_auc = 0.0
        self.test_max_auc = 0.0

        self.batch_size = 64
        self.epoch_num = 20
        self.dataPath = dataPath
        if dataPath == f'{get_raw_data_path()}/ASSISTments2015/processed/':
            self.weight_path = f'{get_proj_path()}/EduSim/Envs/KES_ASSIST15/meta_data/DASKTweight/'
            self.train_dataset = DASADataset(f'{self.dataPath}1/train/', self.n_items, feature_dim=2 * self.n_items, max_sequence_length=100, n_wins=n_wins, datasetType='ASSIST15')
            self.val_dataset = DASADataset(f'{self.dataPath}1/val/', self.n_items, feature_dim=2 * self.n_items, max_sequence_length=100, n_wins=n_wins, datasetType='ASSIST15')
            self.test_dataset = DASADataset(f'{self.dataPath}1/test/', self.n_items, feature_dim=2 * self.n_items, max_sequence_length=100, n_wins=n_wins, datasetType='ASSIST15')
            self.train_size = int(len(self.train_dataset))
            self.val_size = int(len(self.val_dataset))
            self.test_size = int(len(self.test_dataset))

        elif dataPath == f'{get_raw_data_path()}/junyi/data/student_log_kt_None':
            self.weight_path = f'{get_proj_path()}/EduSim/Envs/KES/meta_data/DASKTweight/'
            self.dataset = DASADataset(self.dataPath, self.n_items, feature_dim=2 * self.n_items, max_sequence_length=100, n_wins=n_wins, datasetType='junyi')
            self.train_size = int(0.8 * len(self.dataset))
            self.val_size = int(0.1 * len(self.dataset))
            self.test_size = len(self.dataset) - self.train_size - self.val_size
            self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(self.dataset, [self.train_size, self.val_size, self.test_size])
        elif dataPath == f'{get_proj_path()}/EduSim/Envs/KSS/meta_data/exLog.txt':
            self.weight_path = f'{get_proj_path()}/EduSim/Envs/KSS/meta_data/DASKTweight/'
            self.dataset = DASADataset(self.dataPath, self.n_items, feature_dim=2 * self.n_items, max_sequence_length=20, n_wins=n_wins, datasetType='KSS')
            self.train_size = int(0.8 * len(self.dataset))
            self.val_size = int(0.1 * len(self.dataset))
            self.test_size = len(self.dataset) - self.train_size - self.val_size
            self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(self.dataset, [self.train_size, self.val_size, self.test_size])
        elif dataPath == f'{get_raw_data_path()}/ASSISTments2009/assist09.npz':
            self.weight_path = f'{get_proj_path()}/EduSim/Envs/KES_ASSIST09/meta_data/DASKTweight/'
            self.dataset = DASADataset(self.dataPath, self.n_items, feature_dim=2 * self.n_items, max_sequence_length=200, n_wins=n_wins, datasetType='ASSIST09')
            self.train_size = int(0.8 * len(self.dataset))
            self.val_size = int(0.1 * len(self.dataset))
            self.test_size = len(self.dataset) - self.train_size - self.val_size
            self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(self.dataset, [self.train_size, self.val_size, self.test_size])

        if read_pretrain_weight:
            self.KT_model.load_state_dict(torch.load(f'{self.weight_path}ValBest.pt', map_location=self.device))

        os.makedirs(self.weight_path, exist_ok=True)
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size)
        self.val_loader = DataLoader(dataset=self.val_dataset, batch_size=self.batch_size)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size)

    def _make_model_weight_dict(self):
        ret_dict = {}
        with torch.no_grad():
            ret_dict["alpha"] = self.KT_model.state_dict()["alpha"].clone().detach()
            ret_dict["delta"] = self.KT_model.state_dict()["delta"].clone().detach()
            ret_dict["beta"] = self.KT_model.state_dict()["beta"].clone().detach()
            ret_dict["attempt"] = self.KT_model.state_dict()["attempt"].clone().detach()
            ret_dict["correct"] = self.KT_model.state_dict()["correct"].clone().detach()
            ret_dict["h"] = self.KT_model.h
            ret_dict["d"] = self.KT_model.d
        return ret_dict

    def train_with_dataset(self):
        print(f'Train: # of users: {len(self.train_dataset)}')
        print(f'Validation: # of users: {len(self.val_dataset)}')
        print(f'Test: # of users: {len(self.test_dataset)}')
        train_loss_list = []
        correct = 0
        true_labels = []
        pred_probs = []

        if self.prev_weights_dict is None:
            self.prev_weights_dict = self._make_model_weight_dict()
        for epoch in range(self.epoch_num):
            for i, batch_data in enumerate(self.train_loader):
                encoded_los_data, lag, label = batch_data
                encoded_los_data = encoded_los_data.to(self.device)
                lag = lag.to(self.device)
                label = label.to(self.device)
                y_pred = self.KT_model(encoded_los_data, lag)

                loss = self.loss_fn(y_pred, label, self.KT_model.named_parameters(), self.prev_weights_dict)
                self.prev_weights_dict = self._make_model_weight_dict()

                train_loss_list.append(loss.cpu().item())
                if i % 50 == 0:
                    print(f"epcoh:{epoch + 1}  iteration:{i}   loss:{np.mean(train_loss_list):.6f}")

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_pred_labels = (y_pred >= 0.5).long()
                for j in range(y_pred.size(0)):
                    correct += batch_pred_labels.eq(label).sum().item()
                    for p in y_pred.view(-1):
                        pred_probs.append(p.item())
                    for t in label.view(-1):
                        true_labels.append(t.item())

            acc = correct / len(true_labels)
            auc = metrics.roc_auc_score(true_labels, pred_probs)
            print(f'train acc:{acc}')
            print(f'train auc:{auc}')
            self.test('Val', self.val_loader)

    def train_with_data(self, data, device=None):
        if device is None:
            device = self.device
        label = data[-1][1]
        encoded_los_data, lag = get_DAS_input_from_log(data[:-1], data[-1][0], n_items=self.n_items, n_skills=self.n_skills, n_wins=self.n_wins)
        encoded_los_data = encoded_los_data.view(1, -1).to(device)
        lag = lag.view(1, -1).to(device)
        label = torch.tensor(label, dtype=torch.float32).view(1, -1).to(device)
        y_pred = self.KT_model(encoded_los_data, lag)

        self.prev_weights_dict = self._make_model_weight_dict()
        loss = self.loss_fn(y_pred, label, self.KT_model.named_parameters(), self.prev_weights_dict)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predicit(self, history, item_id, device=None):
        if device is None:
            device = self.device
        encoded_log_data, lag = get_DAS_input_from_log(history, item_id, n_items=self.n_items, n_skills=self.n_skills, n_wins=self.n_wins)
        encoded_log_data = encoded_log_data.view(1, -1).to(device)
        lag = lag.view(1, -1).to(device)
        y_pred = self.KT_model(encoded_log_data, lag)
        label_pred = (y_pred >= 0.5).long().view(-1)
        return label_pred

    def test(self, name, dataloader):
        print(f'...testing on {name} ...')
        self.KT_model.eval()
        test_loss = 0
        correct = 0
        true_labels = []
        pred_probs = []
        with torch.no_grad():
            for i, batch_data in enumerate(dataloader):
                encoded_los_data, lag, label = batch_data
                encoded_los_data = encoded_los_data.to(self.device)
                lag = lag.to(self.device)
                label = label.to(self.device)
                y_pred = self.KT_model(encoded_los_data, lag)
                loss = self.loss_fn(y_pred, label, self.KT_model.named_parameters(), self.prev_weights_dict)
                test_loss = test_loss + loss.item()
                batch_pred_labels = (y_pred >= 0.5).long()
                for j in range(y_pred.size(0)):
                    correct += batch_pred_labels.eq(label).sum().item()
                    for p in y_pred.view(-1):
                        pred_probs.append(p.item())
                    for t in label.view(-1):
                        true_labels.append(t.item())

            test_loss /= (i + 1)

        acc = correct / len(pred_probs)
        auc = metrics.roc_auc_score(true_labels, pred_probs)
        print(f'Test result: Average loss: {test_loss:.6f}  '
              f'acc: {correct}/{len(pred_probs)}={acc}  '
              f'auc: {auc}')
        self.KT_model.train()

        if name == 'Val':
            if self.val_max_auc < auc:
                self.val_max_auc = auc
                cur_weight = self.KT_model.state_dict()
                torch.save(cur_weight, f'{self.weight_path}ValBest.pt')
        else:
            if self.test_max_auc < auc:
                self.test_max_auc = auc
                cur_weight = self.KT_model.state_dict()
                torch.save(cur_weight, f'{self.weight_path}ValBest.pt')


if __name__ == '__main__':
    print('cuda_ava:' + str(torch.cuda.is_available()))
    test_inner = InnerModel(n_items=100, n_skills=100, dataPath=f'{get_raw_data_path()}/ASSISTments2015/processed/', lr=0.01, read_pretrain_weight=False)
    test_inner.train_with_dataset()
    test_inner = InnerModel(n_items=835, n_skills=835, dataPath=f'{get_raw_data_path()}/junyi/data/student_log_kt_None', lr=0.01, read_pretrain_weight=False)
    test_inner.train_with_dataset()
    test_inner = InnerModel(n_items=10, n_skills=10, dataPath=f'{get_proj_path()}/EduSim/Envs/KSS/meta_data/exLog.txt', lr=0.001, read_pretrain_weight=False)
    test_inner.train_with_dataset()
    test_inner = InnerModel(n_items=167, n_skills=167, dataPath=f'{get_raw_data_path()}/ASSISTments2009/assist09.npz', lr=0.01, read_pretrain_weight=False)
    test_inner.train_with_dataset()
