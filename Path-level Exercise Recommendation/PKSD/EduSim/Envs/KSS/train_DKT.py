import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import random

from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import metrics
from sklearn.metrics import r2_score
from EduSim.Envs.deep_model import DKTnet
import os
from EduSim.Envs.agent_utils import get_proj_path


class MyDataset(Dataset):
    def __init__(self, data_path, feature_dim, max_sequence_length):
        self.data_path = data_path
        self.feature_dim = feature_dim
        self.action_dim = 10
        self.max_sequence_length = max_sequence_length
        self.data = []
        with open(self.data_path, 'r', encoding="utf-8") as f:
            datatxt = f.readlines()
        for i, line in enumerate(datatxt):
            if ',,,' in line:
                items = [int(item) for item in datatxt[i + 1].split(',')]
                answers = [int(answer) for answer in datatxt[i + 2].split(',')]
                sample = self.get_feature_matrix(items, answers)
                self.data.append(sample)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_feature_matrix(self, items, answers):
        input_data = torch.FloatTensor(self.max_sequence_length, self.feature_dim)
        input_data.zero_()
        for j in range(len(items)):
            problem_id = items[j]
            if answers[j] == 0:
                input_data[j][problem_id] = 1.0
            elif answers[j] == 1:
                input_data[j][problem_id + self.action_dim] = 1.0
        return input_data


class DKThandler(object):
    def __init__(self):
        self.use_gpu = True
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        print('cuda_ava:' + str(torch.cuda.is_available()))
        self.data_path = f'{get_proj_path()}/EduSim/Envs/KSS/meta_data/DKT_USE_DATA/dataAll.txt'
        self.max_sequence_length = 20
        self.epoch_num = 50
        self.actual_labels = []
        self.pred_labels = []
        self.learning_rate = 0.001
        self.feature_dim = 20
        self.embed_dim = 15
        self.hidden_size = 20
        self.batch_size = 16
        self.num_skills = 10
        self.dataset = MyDataset(self.data_path, self.feature_dim, self.max_sequence_length)
        self.train_size = int(0.8 * len(self.dataset))
        self.val_size = int(0.1 * len(self.dataset))
        self.test_size = len(self.dataset) - self.train_size - self.val_size
        self.train_dataset, self.val_dataset, self.test_dataset = \
            torch.utils.data.random_split(self.dataset, [self.train_size, self.val_size, self.test_size])
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size)
        self.val_loader = DataLoader(dataset=self.val_dataset, batch_size=self.batch_size)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size)
        self.DKTnet = DKTnet(input_size=2 * self.num_skills,
                             emb_dim=self.embed_dim,
                             hidden_size=self.hidden_size,
                             num_skills=self.num_skills,
                             nlayers=2).to(self.device)
        self.optimizer = torch.optim.Adam(self.DKTnet.parameters(), lr=self.learning_rate)
        self.loss_f = nn.BCEWithLogitsLoss()
        if True:
            print('testing')
            self.DKTnet.load_state_dict(torch.load(f'{get_proj_path()}/EduSim/Envs/KSS/meta_data/DKT_USE_DATA/all_data_trained_DKT_model.pth', map_location=self.device)['model'])
            self.test()
            assert 0

    def train(self):
        loss_list = []
        state = {}
        for epoch in range(self.epoch_num):
            for i, batch_data in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                batch_data = batch_data.to(self.device)
                output = self.DKTnet(batch_data)
                _, _, loss = self.compute_loss(output, batch_data)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.DKTnet.parameters(), 20)
                self.optimizer.step()
                loss_list.append(loss.cpu().item())
                if i % 50 == 0:
                    print("epcoh:{}  iteration:{}   loss:{:.6f}".format(epoch + 1, i, np.mean(loss_list)))
            rmse = sqrt(mean_squared_error(self.actual_labels, self.pred_labels))
            fpr, tpr, thresholds = metrics.roc_curve(self.actual_labels, self.pred_labels, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            r2 = r2_score(self.actual_labels, self.pred_labels)
            print('rmse:' + str(rmse))
            print('auc:' + str(auc))
            print('r2:' + str(r2))
            state = {'model': self.DKTnet.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, f'{get_proj_path()}/EduSim/Envs/KSS/meta_data/DKT_USE_DATA/all_data_trained_DKT_model.pth')
            self.test()
            self.DKTnet.train()

    def test(self):
        self.DKTnet.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for i, batch_data in enumerate(self.test_loader):
                batch_data = batch_data.to(self.device)
                y_hat = self.DKTnet(batch_data)
                batch_pred_labels, batch_true_labels, batch_loss = self.compute_loss(y_hat, batch_data)
                test_loss += batch_loss.item()
                ones = torch.ones([batch_data.size(0), self.max_sequence_length - 1, 1]).to(self.device)
                zeors = torch.zeros([batch_data.size(0), self.max_sequence_length - 1, 1]).to(self.device)
                batch_pred_labels = torch.where(batch_pred_labels > 0.5, ones, zeors)
                correct += batch_pred_labels.eq(batch_true_labels).sum().item()
            test_loss /= (i + 1)
            print('Test result: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss,
                correct,
                self.test_size * (self.max_sequence_length - 1),
                100. * correct / (self.test_size * (self.max_sequence_length - 1))))

    def compute_loss(self, output, batch_data):
        target_corrects = torch.tensor([], dtype=torch.int64).to(self.device)
        target_ids = torch.tensor([], dtype=torch.int64).to(self.device)
        output = output.permute(1, 0, 2)
        for episode in range(batch_data.size(0)):
            tmp_target_id = torch.argmax(batch_data[episode, :, :].cpu(), -1)
            ones = torch.ones(tmp_target_id.size())
            zeros = torch.zeros(tmp_target_id.size())
            target_correct = torch.where(tmp_target_id > self.num_skills - 1, ones, zeros).to(self.device).unsqueeze(
                1).unsqueeze(0)
            target_id = torch.where(tmp_target_id > self.num_skills - 1, tmp_target_id - self.num_skills, tmp_target_id) \
                .to(self.device)
            target_id = torch.roll(target_id, -1, 0).unsqueeze(1).unsqueeze(0)
            target_ids = torch.cat((target_ids, target_id), 0)
            target_corrects = torch.cat((target_corrects, target_correct))
        logits = output.gather(2, target_ids)
        preds = torch.sigmoid(logits)
        for p in preds[:, 0:self.max_sequence_length - 1].contiguous().view(-1):
            self.pred_labels.append(p.item())
        for t in target_corrects[:, 1:self.max_sequence_length].contiguous().view(-1):
            self.actual_labels.append(t.item())
        loss = self.loss_f(logits[:, 0:self.max_sequence_length - 1], target_corrects[:, 1:self.max_sequence_length])
        return preds[:, 0:self.max_sequence_length - 1], target_corrects[:, 1:self.max_sequence_length], loss


handler = DKThandler()
handler.train()

