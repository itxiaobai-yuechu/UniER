import torch

from dqn import *
from common import *
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch.optim import Adam
import logging
import random
import numpy as np


class MMAER(torch.nn.Module):
    def __init__(self, args):
        super(MMAER, self).__init__()
        self.gamma = args.gamma
        self.action_num = args.action_num
        self.cpt_num = args.cpt_num
        self.hidden_size = args.hidden_size
        self.init_type = args.init_type
        self.batch_size = args.batch_size
        self.args = args
        self.device = torch.device(args.cuda)

        self.init_states = torch.nn.Parameter(torch.empty(args.batch_size, self.cpt_num))
        self.init_actions = torch.nn.Parameter(torch.empty(args.batch_size, self.cpt_num))
        torch.nn.init.uniform_(self.init_states)
        torch.nn.init.uniform_(self.init_actions)

        self.MF_net = MFNet(args)
        self.rank_net = MLP(
            input_dim=self.cpt_num * 2, output_dim=1,
            dnn_units=[self.hidden_size], dropout_rate=args.dropout_rate
        )
        self.online_net = DQN(2 * self.cpt_num, self.cpt_num, args)

        self._init_params()
        print()
    def MF_single(self, vars, x):
        w, b = vars[0], vars[1]
        x = F.linear(x, w, b)
        w1, b1 = vars[2], vars[3]
        x = F.linear(x, w1, b1)
        x = torch.sigmoid(x)
        return x

    def online_net_single(self, vars, x):
        w, b = vars[0], vars[1]
        x = F.linear(x, w, b)
        w1, b1 = vars[2], vars[3]
        x = F.linear(x, w1, b1)
        x = torch.sigmoid(x)
        return x


    def act_single(self, args, obs, vars):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        q_value = self.online_net_single(vars, obs_tensor)
        v = torch.cat(((1 - q_value).unsqueeze(1), q_value.unsqueeze(1)), dim=1)
        max_values = v.max(dim=1, keepdim=True)[0].squeeze()
        max_q_idx = torch.argmax(input=v, dim=1)
        action = (v[:, 1] > 0.5) * torch.ones(self.args.cpt_num, device=self.device)
        return action, max_values, max_q_idx, v[:, 1]

    def rank_single(self, x, vars):
        w, b = vars[0], vars[1]
        x = F.linear(x, w, b)
        w1, b1 = vars[2], vars[3]
        x = F.linear(x, w1, b1)
        x = torch.sigmoid(x)
        return x
    
    def _init_params(self):
        if self.init_type == 'xavier_normal':
            self.apply(xavier_normal_initialization)
        elif self.init_type == 'xavier_uniform':
            self.apply(xavier_uniform_initialization)
        elif self.init_type == 'kaiming_normal':
            self.apply(kaiming_normal_initialization)
        elif self.init_type == 'kaiming_uniform':
            self.apply(kaiming_uniform_initialization)
        elif self.init_type == 'init_from_pretrained':
            self._load_params_from_pretrained()


    @staticmethod
    def _meta_item(meta, index):

        if meta is None:
            return None
        if torch.is_tensor(meta):
            value = meta[index]
            return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
        if isinstance(meta, (list, tuple)):
            value = meta[index]
            if torch.is_tensor(value):
                return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
            return value
        return meta

    @staticmethod
    def _tensor_to_python_list(tensor):

        if torch.is_tensor(tensor):
            return tensor.detach().cpu().tolist()
        return tensor

    def get_main_loss(self, dataloader, vars, i, is_train, target_net, device):
        exercises, scores, kcs, mask = (dataloader['exercise'], dataloader['score'],
                                        dataloader['kc'], dataloader['mask'])
        student_ids = dataloader.get('student_id', None)
        origin_user_idxs = dataloader.get('origin_user_idx', None)
        segment_starts = dataloader.get('segment_start', None)
        total_step = scores.shape[1]

        kc_encoders = torch.zeros(self.batch_size, total_step, self.cpt_num, device=self.device)
        exercises = exercises.to(self.device)
        kcs = kcs.to(self.device)
        scores = scores.to(self.device)
        mask = mask.to(self.device)
        valid_kcs = kcs.clone()
        invalid_mask = valid_kcs < 0
        valid_kcs[invalid_mask] = 0
        kc_encoders.scatter_(2, valid_kcs.unsqueeze(2), 1)
        kc_encoders = kc_encoders * (~invalid_mask).unsqueeze(2)
        all_target, all_value = [], []
        if vars:
            states, actions = vars[0], vars[1]
        else:
            states, actions = self.init_states.to(self.device), self.init_actions.to(self.device)
        current_step = 0

        score_true_stu, score_rec_stu, score_rec_topk_stu, mask_stu = [], [], [], []
        score_rec_exer_topk_stu = []
        score_rec_kc_topk_stu = []
        score_rec_pred_topk_stu = []
        recommendation_records = []

        mask_batches = []
        true_kcs_topk_stu = []
        rank_true_kcs_topk_stu = []
        MF = self.obtain_MF(actions)
        while current_step < total_step - 1:
            if vars:
                MF = self.MF_single(vars[2:6], MF)
            else:
                MF = self.MF_net(MF)

            actions = self.action_selection(states, MF, vars)
            MF_new = self.obtain_MF(actions)
            end_step = min(current_step + self.args.window_step, total_step)
            kc_encode = kc_encoders[:, current_step:end_step, :]

            score_rec = self.obtain_ranking_score(actions, kc_encode, vars)
            score_true = scores[:, current_step: end_step]
            mask_step = mask[:, current_step: end_step]

            end_step_for_reward = min(current_step + self.args.window_step_for_reward, total_step)
            kc_encode_for_reward = kc_encoders[:, current_step:end_step_for_reward, :]
            score_rec_for_reward = self.obtain_ranking_score(actions, kc_encode_for_reward, vars)
            score_true_for_reward = scores[:, current_step: end_step_for_reward]
            reward = self.obtain_reward(kc_encode_for_reward, score_rec_for_reward, score_true_for_reward)

            for step in range(len(mask_step)):
                tmp_mask = mask_step[step] != 0
                if tmp_mask.sum().item() > 0:
                    valid_true_scores = score_true[step][tmp_mask]
                    valid_pred_scores = score_rec[step][tmp_mask]
                    topk_indices = torch.sort(valid_pred_scores).indices

                    score_true_stu.append(valid_true_scores)
                    score_rec_stu.append(valid_pred_scores)
                    score_rec_topk_stu.append(topk_indices)

                    valid_exercises = exercises[step, current_step:end_step][tmp_mask]
                    valid_kcs_in_window = kcs[step, current_step:end_step][tmp_mask]
                    window_positions = torch.arange(current_step, end_step, device=self.device)[tmp_mask]

                    rec_exercises = valid_exercises[topk_indices]
                    rec_kcs = valid_kcs_in_window[topk_indices]
                    rec_pred_scores = valid_pred_scores[topk_indices]
                    rec_true_scores = valid_true_scores[topk_indices]
                    rec_segment_positions = window_positions[topk_indices]

                    score_rec_exer_topk_stu.append(rec_exercises.detach().cpu())
                    score_rec_kc_topk_stu.append(rec_kcs.detach().cpu())
                    score_rec_pred_topk_stu.append(rec_pred_scores.detach().cpu())

                    student_id = self._meta_item(student_ids, step)
                    origin_user_idx = self._meta_item(origin_user_idxs, step)
                    segment_start = self._meta_item(segment_starts, step)
                    if segment_start is None:
                        ranked_original_positions = self._tensor_to_python_list(rec_segment_positions)
                    else:
                        ranked_original_positions = self._tensor_to_python_list(rec_segment_positions + int(segment_start))

                    recommendation_records.append({
                        'student_id': student_id,
                        'origin_user_idx': origin_user_idx,
                        'segment_start': segment_start,
                        'window_start': int(current_step),
                        'window_end': int(end_step),
                        'ranked_local_indices_in_valid_window': self._tensor_to_python_list(topk_indices),
                        'ranked_segment_positions': self._tensor_to_python_list(rec_segment_positions),
                        'ranked_original_positions': ranked_original_positions,
                        'ranked_exercise_ids': self._tensor_to_python_list(rec_exercises),
                        'ranked_knowledge_codes': self._tensor_to_python_list(rec_kcs),
                        'ranked_true_scores': self._tensor_to_python_list(rec_true_scores),
                        'ranked_pred_scores': self._tensor_to_python_list(rec_pred_scores),
                    })

            states_new = self.update_correct(states, kc_encode, score_true)

            target_q_values = target_net(torch.cat((states_new, MF_new), dim=1))
            target_v = torch.cat(((1 - target_q_values).unsqueeze(2), target_q_values.unsqueeze(2)), dim=2)
            max_target_values = target_v.max(dim=2, keepdim=True)[0].squeeze()
            targets = reward + self.gamma * max_target_values
            all_target.append(targets)

            if vars:
                q_values = self.online_net_single(vars[10:14], torch.cat((states, MF), dim=1))
            else:
                q_values = self.online_net(torch.cat((states, MF), dim=1))

            a = (actions > self.args.threshold).long().unsqueeze(2)
            online_v = torch.cat(((1 - q_values).unsqueeze(2), q_values.unsqueeze(2)), dim=2)
            a_q_value = torch.gather(input=online_v, dim=2, index=a).squeeze()
            all_value.append(a_q_value)

            current_step = end_step
            MF = MF_new
            states = states_new

        label, pred = torch.cat(score_true_stu, dim=0), torch.cat(score_rec_stu, dim=0)
        all_target = torch.cat(all_target, dim=0)
        all_value = torch.cat(all_value, dim=0)
        loss2 = nn.MSELoss(reduction='sum')(all_value, all_target)
        hit_3, NDCG_3, F1_3 = 0, 0, 0
        if is_train == False:
            return (score_true_stu, score_rec_topk_stu, score_rec_exer_topk_stu,
                    score_rec_kc_topk_stu, score_rec_pred_topk_stu, recommendation_records)
        return hit_3, NDCG_3, F1_3, loss2


    def obtain_MF(self, actions):
        MF = torch.empty(self.batch_size, self.cpt_num).to(self.device)
        for bs in range(self.batch_size):
            try:
                MF[bs] = (actions[bs].sum(dim=0) - actions[bs]) / (self.cpt_num - 1)
            except:
                print('\nbs', bs)
                print('actions', actions, type(actions))
                print('MF', MF)
                exit()
        return MF

    def action_selection(self, states, MF, vars):
        EPSILON_DECAY = 100000
        EPSILON_START = 1.0
        EPSILON_END = 0.02
        epsilon = 0.3
        random_sample = torch.rand(self.batch_size, self.cpt_num, device=self.device)
        random_flag = random_sample <= epsilon
        random_act = torch.randint(low=0, high=2, size=(self.batch_size, self.cpt_num), dtype=torch.float32, device=self.device)

        if vars:
            v, max_values, online_act, cot_action = self.act_single(self.args, torch.cat((states, MF), dim=1),
                                                                    vars[10:14])
        else:
            v, max_values, online_act, cot_action = self.online_net.act(self.args, torch.cat((states, MF), dim=1))
        return random_act * random_flag + v * (~random_flag)

    def obtain_ranking_score(self, actions, kc, vars):
        if vars:
            ranking_score = self.rank_single(torch.cat((actions.unsqueeze(1).repeat(1, kc.shape[1], 1), kc), dim=2),
                                             vars[6:10]).squeeze(-1)
        else:
            ranking_score = self.rank_net(torch.cat((actions.unsqueeze(1).repeat(1, kc.shape[1], 1), kc), dim=2)).squeeze(-1)
        return ranking_score

    def obtain_reward(self, kc_encode, score_rec, score_true):
        score_true = score_true.reshape(-1, 1)
        score_rec = score_rec.reshape(-1, 1)
        dist = torch.norm(score_true - score_rec, dim=1, p=2).reshape(self.batch_size, -1, 1)
        dist_repeated = dist.repeat(1, 1, self.cpt_num)
        result = torch.mean(kc_encode * dist_repeated, dim=1)
        return result

    def update_correct(self, states, kc_encode, score_true):
        k = torch.sum(kc_encode, dim=1)
        flag = (k == 0)
        m = torch.where(k == 0, torch.ones_like(k), k)
        z = torch.sum(score_true.unsqueeze(2).repeat(1, 1, self.cpt_num).mul(kc_encode), dim=1)
        return 0.5 * states * (~flag) + 0.5 * z / m + states * flag


    def calculate_matrix(self, score_true_all, score_rec_topk_all, k):
        score = score_true_all
        rank = score_rec_topk_all
        index = np.ones(len(score))
        idx_to_delete = np.where(index == 0)[0]
        
        for idx in sorted(idx_to_delete, reverse=True):
            del score[idx]
            del rank[idx]
        
        score = [1 - each.cpu().numpy() for each in score]
        rank = [each[:k].cpu().numpy() for each in rank]
        
        hits = [score[i][rank[i]] for i in range(len(score))]
        
        ndcg, valid_hits = self.ndcg_at_k_batch(hits, k)
        hits_flatten = [item for sublist in valid_hits for item in sublist]

        try:
            hit = sum(hits_flatten) / len(hits_flatten)
        except:
            hit = 0.0
        t = [1] * len(hits_flatten)
        F1 = f1_score(t, hits_flatten) if len(hits_flatten) > 0 else 0.0

        mrr_list = []
        ap_list = []
        total_true_kc = [sum(s) for s in score]

        for hit_list, true_num in zip(valid_hits, total_true_kc):
            rr = 0.0
            for idx, h in enumerate(hit_list):
                if h == 1:
                    rr = 1.0 / (idx + 1)
                    break
            mrr_list.append(rr)

            if true_num > 0:
                relevant = 0
                prec_sum = 0.0
                for idx, h in enumerate(hit_list):
                    if h == 1:
                        relevant += 1
                        prec_sum += relevant / (idx + 1)
                ap = prec_sum / min(k, true_num)
                ap_list.append(ap)
            else:
                ap_list.append(0.0)

        MRR = np.mean(mrr_list) if len(mrr_list) > 0 else 0.0
        MAP = np.mean(ap_list) if len(ap_list) > 0 else 0.0

        return ndcg, hit, F1, MAP, MRR


    
    def ndcg_at_k_batch(self, hits, k):
        for i in range(len(hits))[::-1]:
            if len(hits[i]) < k:
                del hits[i]
        hits = [np.array(each) for each in hits]
        hits_k = np.array(hits)
        try:
            dcg = np.sum((2 ** hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)
            sorted_hits_k = np.flip(np.sort(hits_k), axis=1)[:, :k]
        except:
            print()
        idcg = np.sum((2 ** sorted_hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)
        idcg[idcg == 0] = np.inf
        ndcg = (dcg / idcg)
        return np.mean(ndcg), hits

    def get_parameters(self):
        params_name_list = []
        for name, param in self.named_parameters():
            params_name_list.append(name)
        return params_name_list