import os
import sys
import json
import numpy as np

from sklearn import metrics
from longling import wf_open
from tqdm import tqdm

import torch
from torch.utils.data import Dataset,DataLoader,random_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
from torch.utils.data import DataLoader, TensorDataset

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.abspath(os.path.join(cur_path, '../..'))
sys.path.insert(0, root_path)

from EduSim.deep_model import KTnet
from EduSim.Envs.agent_utils import get_proj_path, get_raw_data_path, mds_concat





class MyDataset(Dataset):
    def __init__(self, data_path, num_skills, feature_dim, max_sequence_length, dataset='assist17'):
        self.data_path = data_path
        self.feature_dim = feature_dim
        self.num_skills = num_skills
        self.max_sequence_length = max_sequence_length
        self.dataset = dataset
        self.datatxt = []
        with open(self.data_path, 'r', encoding="utf-8") as f:
            self.datatxt = f.readlines()
        print(f'Total training {len(self.datatxt)} sessions ')


    def __getitem__(self, index):
        line = self.datatxt[index]

        session = json.loads(line)
        session = self.get_feature_matrix(session)
        return session

    def __len__(self):
        return len(self.datatxt)

    def get_feature_matrix(self, session):
        input_data = np.zeros(shape=(self.max_sequence_length, self.feature_dim), dtype=np.float32)

        j = 0
        while j < self.max_sequence_length and j < len(session):
            problem_id = session[j][0]
            if session[j][1] == 0:
                input_data[j][problem_id] = 1.0
            elif session[j][1] == 1:
                input_data[j][problem_id + self.num_skills] = 1.0
            j += 1
        return torch.Tensor(input_data)

    def get_data_correct_rate(self):
        answer_cor_num = 0
        all_num = 0
        for _, line in enumerate(self.datatxt):
            session = json.loads(line)
            data = [log[1] for log in session]
            all_num += len(data)
            answer_cor_num += np.array(data).sum().item()
        print(f'1 in data rate:{answer_cor_num / all_num}')

    def get_selected_data(self):
        if self.data_path != f'{get_raw_data_path()}/dataProcess/{self.dataset}/student_log_kt_None':
            raise ValueError('This function can only be used under env KT training situation')

        env_selected_train_data_path = f'{get_raw_data_path()}/dataProcess/{self.dataset}/student_log_kt_None_selected'
        selecte_count = 0
        with wf_open(env_selected_train_data_path) as wf:
            for _, line in tqdm(enumerate(self.datatxt), 'writing in file'):
                session = json.loads(line)
                different_questions = {log[0] for log in session}
                if len(different_questions) > 15:
                    selecte_count += 1
                    print(json.dumps(session), file=wf)
        print(f'final selected count:{selecte_count}')
        print(f'final selected rate:{selecte_count / len(self.datatxt)}')


class EnvKTtrainer:
    def __init__(self, dataset, dataset_KES, train_goal='env_KT', device=None):
        self.dataset = dataset
        self.dataset_KES = dataset_KES
        self.device = device
        with open(f'{get_raw_data_path()}/dataProcess/{self.dataset}/graph_vertex.json') as f:
            ku_dict = json.load(f)
        self.num_skills = len(ku_dict)
        self.feature_dim = 2 * self.num_skills

        self.train_goal = train_goal
        self.test_only = False
        self.env_selected_data_train = False
        print('Current training goal is:' + self.train_goal)

        if self.env_selected_data_train:
            self.env_KT_data_path = f'{get_raw_data_path()}/dataProcess/{self.dataset}/student_log_kt_None_selected'
            print(self.env_KT_data_path)
        else:
            self.env_KT_data_path = f'{get_raw_data_path()}/dataProcess/{self.dataset}/student_log_kt_None'
        self.agent_KT_data_path = f'{get_raw_data_path()}/dataProcess/{self.dataset}/dataOff'
        self.max_sequence_length = 200
        self.epoch_num = 10
        self.test_acc = 0.0
        self.test_auc = 0.0
        self.val_max_auc = 0.0
        self.test_max_auc = 0.0
        if self.train_goal == 'env_KT':
            self.data_path = self.env_KT_data_path
            self.weight_path = f'{get_proj_path()}/EduSim/Envs/{self.dataset_KES}/meta_data/env_weights/'
            os.makedirs(self.weight_path, exist_ok=True)
        else:
            self.data_path = self.agent_KT_data_path
            self.weight_path = f'{get_proj_path()}/EduSim/Envs/{self.dataset_KES}/meta_data/agent_weights/'
            os.makedirs(self.weight_path, exist_ok=True)

        self.dataset_obj = MyDataset(self.data_path, self.num_skills, self.feature_dim, self.max_sequence_length, dataset=self.dataset)

        self.learning_rate = 0.001
        self.embed_dim = 128
        self.hidden_size = 256
        self.batch_size = 64

        self.train_size = int(len(self.dataset_obj)*0.8)
        self.val_size = int(len(self.dataset_obj)*0.1)
        self.test_size = len(self.dataset_obj)-self.train_size-self.val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset_obj, [self.train_size, self.val_size, self.test_size])

        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False)

        KT_input_dict = {
            'input_size': self.feature_dim,
            'emb_dim': self.embed_dim,
            'hidden_size': self.hidden_size,
            'num_skills': self.num_skills,
            'nlayers': 2,
            'dropout': 0.01,
        }

        self.KTnet = KTnet(KT_input_dict).to(device)

        self.optimizer = optim.Adam(self.KTnet.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.BCEWithLogitsLoss()

        if self.test_only:
            self.KTnet = torch.load(f'{self.weight_path}ValBest.pt')
            self.test(name='Test', dataloader=self.test_dataset)
            assert 0


    def forward_fn(self, batch_data):
        output = self.KTnet(batch_data)
        loss, batch_pred_probs, batch_true_labels, sequence_lengths = self.compute_loss(output, batch_data)
        return loss, batch_pred_probs, batch_true_labels, sequence_lengths

    def train(self):
        train_loss_list = []
        self.KTnet.train()
        for epoch in range(self.epoch_num):
            for i, batch_data in enumerate(self.train_dataloader):
                batch_data = batch_data.to(device)
                self.optimizer.zero_grad()
                
                output = self.KTnet(batch_data)
                loss, batch_pred_probs, batch_true_labels, sequence_lengths = self.compute_loss(output, batch_data)
                
                loss.backward()
                self.optimizer.step()

                if i % 20 == 0:
                    print(f"epoch:{epoch + 1}  iteration:{i}  loss:{loss.item()}")

            self.test('Val', self.val_dataloader)

        self.test('test', self.test_dataloader)

    def test(self, name, dataloader):
        print(f'...testing on {name} ...')
        test_loss = 0
        correct = 0
        true_labels = []
        pred_probs = []
        self.KTnet.eval()
        k = 0
        for i, batch_data in enumerate(dataloader):
            batch_data = batch_data.to(device)
            y_hat = self.KTnet(batch_data)

            batch_loss, batch_pred_outs, batch_true_labels, sequence_lengths = self.compute_loss(y_hat, batch_data)
            test_loss += batch_loss.detach().cpu().numpy()

            batch_pred_labels = (batch_pred_outs >= 0.5).float()
            if i % 20 == 0:
                print(f"iteration:{i}   ")

            for j in range(batch_pred_labels.shape[0]):
                if sequence_lengths[j] == 1:
                    continue
                a = batch_pred_labels[j, 0:sequence_lengths[j] - 1]
                b = batch_true_labels[j, 1:sequence_lengths[j]]

                correct = correct + (a == b).sum().item()
                for p in batch_pred_outs[j, 0:sequence_lengths[j] - 1].view(-1):
                    pred_probs.append(p.detach().cpu().numpy())
                for t in batch_true_labels[j, 1:sequence_lengths[j]].view(-1):
                    true_labels.append(t.detach().cpu().numpy())
            k = i

        test_loss /= (k + 1)

        acc = correct / len(pred_probs)
        auc = metrics.roc_auc_score(true_labels, pred_probs)
        print(f'Test result: Average loss: {float(test_loss):.4f}  '
              f'acc: {correct}/{len(pred_probs)}={acc}  '
              f'auc: {auc}')
        self.KTnet.train()

        if name == 'Val':
            if self.val_max_auc < auc:
                self.val_max_auc = auc
                if 'env' in self.train_goal and self.env_selected_data_train:
                    torch.save(self.KTnet, f'{self.weight_path}SelectedValBest.pt')
                else:
                    torch.save(self.KTnet, f'{self.weight_path}ValBest.pt')
                    torch.save(self.KTnet.state_dict(), f'{self.weight_path}ValBest.ckpt')
        else:
            self.test_acc = acc
            self.test_auc = auc

    def compute_loss(self, output, batch_data):
        sequence_lengths = [int(torch.sum(sample)) for sample in batch_data]
        target_corrects = torch.empty(0, dtype=torch.float32).to(device)
        target_ids = torch.empty(0, dtype=torch.float32).to(device)
        output = output.permute(1, 0, 2).to(device)
        for episode in range(batch_data.shape[0]):
            tmp_target_id = torch.argmax(batch_data[episode, :, :],dim = -1)
            ones = torch.ones(tmp_target_id.shape, dtype=torch.float32).to(device)
            zeros = torch.zeros(tmp_target_id.shape, dtype=torch.float32).to(device)
            target_correct = torch.where( tmp_target_id > self.num_skills - 1, ones, zeros).unsqueeze(1).unsqueeze(0).to(device)
            target_id = torch.where(tmp_target_id > self.num_skills - 1, tmp_target_id - self.num_skills, tmp_target_id)
            target_id = torch.roll(target_id, -1, 0).unsqueeze(1).unsqueeze(0).to(device)
            target_ids = mds_concat((target_ids, target_id), 0)
            target_corrects = mds_concat((target_corrects, target_correct), 0)

        logits = output.gather(2, target_ids)
        preds = F.sigmoid(logits)
        loss = torch.Tensor([0.0]).to(device)
        for i, sequence_length in enumerate(sequence_lengths):
            if sequence_length <= 1:
                continue
            a = logits[i, 0:sequence_length - 1]
            b = target_corrects[i, 1:sequence_length]
            loss = loss + self.loss_fn(a, b)

        return loss, preds, target_corrects, sequence_lengths


if __name__ == '__main__':
    np.random.seed(int(os.environ.get('UNIER_SEED', '42')))
    torch.manual_seed(int(os.environ.get('UNIER_SEED', '42')))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(os.environ.get('UNIER_SEED', '42')))

    dataset = os.environ.get('UNIER_PATH_DATASET', 'assist17')
    dataset_KES = f'KES_{dataset}'
    requested_device = os.environ.get('UNIER_DEVICE', 'cuda:0')
    if requested_device.startswith('cuda') and not torch.cuda.is_available():
        requested_device = 'cpu'
    device = torch.device(requested_device)
    handler = EnvKTtrainer( dataset=dataset, dataset_KES=dataset_KES,train_goal='env_KT',device=device)
    handler.train()
    handler_A = EnvKTtrainer(dataset=dataset, dataset_KES=dataset_KES, train_goal='agent_KT',device=device)
    handler_A.train()
    print(sys.path)
