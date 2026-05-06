

from EduSim.Envs.deep_model import *
import networkx as nx
from EduSim.Envs.Transformer import Transformer
from .PPO import PPO
from .AC import ActorCritic
from .Dqn import DQNAgent
from EduSim.Envs.agent_utils import *
import torch
from EduSim.Envs.shared.KSS_KES.KS import influence_control
import copy
import random
from torch_geometric.data import Data, Batch
import math


class PFD(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim1,
                 hidden_dim2,
                 env,
                 RNN_encoder_output_dim,
                 c2=0.001,
                 policy_lr=0.0001,
                 PEC_lr=0.0001,
                 IMP_lr=0.0001,
                 gamma=0.98,
                 device='cuda:0',
                 args={}):
        super(PFD, self).__init__()
        self.name = 'PFD'
        self.policy_mode = 'on_policy'
        self.pre_learn_node = '0'
        self.device = device
        self.gamma = gamma
        self.policy_lr = policy_lr
        self.PEC_lr = PEC_lr
        self.IMP_lr = IMP_lr
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.env = env
        self.args = args
        self.c2 = c2
        self.RNN_encoder_output_dim = RNN_encoder_output_dim
        self.phase_1_episodes = int(args['PFD_phase_1_episdoes_ratio'] * args['max_episode_num'])
        self.num_skills = output_dim
        self.regress_loss_algo = torch.nn.MSELoss()
        self.regress_loss_weight = 1.0

        self.z_size = 2 * self.num_skills
        if self.args['PFD_ImPerEncoder'] == 'DKT':
            self.z_size = self.num_skills
        self.input_dim = self.input_dim + self.z_size

        if self.args['PFD_PerEncoder'] == 'RNN':
            self.perfect_info_encoder = RnnEncoder(input_size=2 * self.num_skills,
                                                   emb_dim=self.num_skills,
                                                   hidden_size=hidden_dim1,
                                                   num_skills=self.z_size,
                                                   nlayers=2).to(self.device)
        elif self.args['PFD_PerEncoder'] == 'Transformer':
            self.perfect_info_encoder = Transformer(max_seq_length=self.args['perfect_log_max_length'],
                                                    input_dim=2 * self.num_skills,
                                                    embedding_size=self.z_size,
                                                    out_dim=self.z_size,
                                                    in_order=True).to(self.device)
        elif self.args['PFD_PerEncoder'] == 'MLP':
            self.perfect_info_encoder = MLPNet(input_dim=self.num_skills,
                                               hidden_dim1=hidden_dim1,
                                               hidden_dim2=hidden_dim2,
                                               output_dim=self.z_size).to(self.device)
        elif self.args['PFD_PerEncoder'] == 'GNN':
            self.env_graph = self.env.knowledge_structure
            self.env_graph_edges = torch.tensor(list(self.env_graph.edges), dtype=torch.int64).t().to(self.device)
            self.initial_graph_embeddings = torch.tensor(self.env.graph_embeddings, dtype=torch.float).to(self.device)
            self.graph_embedding_dim = self.initial_graph_embeddings.size(-1) + 1
            self.perfect_info_encoder = GCNNet(feat_dim=self.graph_embedding_dim,
                                               num_class=self.z_size,
                                               num_node=self.num_skills,
                                               GCN_hidden_dim=self.args['GCN_hidden_dim']).to(self.device)
        elif self.args['PFD_PerEncoder'] == 'No':
            self.perfect_info_encoder = MLPNet(input_dim=self.num_skills,
                                               hidden_dim1=hidden_dim1,
                                               hidden_dim2=hidden_dim2,
                                               output_dim=self.z_size).to(self.device)

        if self.args['PFD_ImPerEncoder'] == 'RNN':
            self.imperfect_info_encoder = RnnEncoder(input_size=2 * self.num_skills,
                                                     emb_dim=self.num_skills,
                                                     hidden_size=hidden_dim1,
                                                     num_skills=self.z_size,
                                                     nlayers=2).to(self.device)
        elif self.args['PFD_ImPerEncoder'] == 'Transformer':
            self.imperfect_info_encoder = Transformer(max_seq_length=self.args['perfect_log_max_length'],
                                                      input_dim=2 * self.num_skills,
                                                      embedding_size=self.z_size,
                                                      out_dim=self.z_size,
                                                      in_order=True).to(self.device)
        elif self.args['PFD_ImPerEncoder'] == 'DKT':
            self.imperfect_info_encoder = nn.Sequential(
                DKTnet(input_size=2 * self.num_skills,
                       emb_dim=self.num_skills,
                       hidden_size=hidden_dim1,
                       num_skills=self.z_size,
                       nlayers=2,
                       dropout=0).to(self.device),
                nn.Sigmoid()
            )
        elif self.args['PFD_ImPerEncoder'] == 'RNN_GNN':
            self.imperfect_info_encoder_R = RnnEncoder(input_size=2 * self.num_skills,
                                                       emb_dim=self.num_skills,
                                                       hidden_size=hidden_dim1,
                                                       num_skills=self.num_skills,
                                                       nlayers=2,
                                                       out_activation='Sigmoid').to(self.device)
            self.imperfect_info_encoder_G = GCNNet(feat_dim=self.graph_embedding_dim,
                                                   num_class=self.z_size,
                                                   num_node=self.num_skills,
                                                   GCN_hidden_dim=self.args['GCN_hidden_dim']).to(self.device)
        elif self.args['PFD_ImPerEncoder'] == 'DKT_GNN':
            self.imperfect_info_encoder_R = nn.Sequential(
                DKTnet(input_size=2 * self.num_skills,
                       emb_dim=self.num_skills,
                       hidden_size=hidden_dim1,
                       num_skills=self.num_skills,
                       nlayers=2,
                       dropout=0).to(self.device),
                nn.Sigmoid(),
            )
            self.imperfect_info_encoder_G = GCNNet(feat_dim=self.graph_embedding_dim,
                                                   num_class=self.z_size,
                                                   num_node=self.num_skills,
                                                   GCN_hidden_dim=self.args['GCN_hidden_dim']).to(self.device)

        if self.args['PFD_base_policy'] == 'PPO':
            self.base_policy = PPO(self.input_dim,
                                   self.output_dim,
                                   hidden_dim1,
                                   hidden_dim2,
                                   self.env,
                                   lr_rate=self.policy_lr,
                                   policy_clip=self.args['ppoclip'],
                                   gamma=self.gamma,
                                   device=self.device,
                                   c2=self.c2,
                                   args=self.args).to(self.device)
        elif self.args['PFD_base_policy'] in ['AC', 'CSEAL']:
            self.base_policy = ActorCritic(input_dim=self.input_dim,
                                           output_dim=self.output_dim,
                                           hidden_dim1=hidden_dim1,
                                           hidden_dim2=hidden_dim2,
                                           env=self.env,
                                           lr_rate=self.policy_lr,
                                           gamma=0.98,
                                           device=self.device,
                                           lr_sche=False,
                                           args=self.args)
        elif self.args['PFD_base_policy'] == 'DQN':
            dqn_advance_types = {'dueling dqn': True,
                                 'multi_step dqn': False,
                                 'n_multi_step': 2,
                                 'double dqn': False,
                                 'prioritized dqn': False,
                                 'noisy dqn': False}
            self.base_policy = DQNAgent(input_dim=self.input_dim,
                                        output_dim=self.output_dim,
                                        hidden_dim1=hidden_dim1,
                                        hidden_dim2=hidden_dim2,
                                        advance_types=dqn_advance_types,
                                        target_update=10,
                                        env=self.env,
                                        lr_rate=0.00005,
                                        gamma=0.98,
                                        device=self.device,
                                        lr_sche=False,
                                        args=self.args)

        self.perfect_info_encoder_optimizer = torch.optim.Adam(self.perfect_info_encoder.parameters(), lr=self.PEC_lr)
        if 'DKT' in self.args['PFD_ImPerEncoder']:
            self.dkt_loss = 0.0
            self.kt_distill_loss = nn.MSELoss()
        if 'GNN' in self.args['PFD_ImPerEncoder']:
            self.imperfect_info_encoder_optimizer_R = torch.optim.Adam(self.imperfect_info_encoder_R.parameters(), lr=self.IMP_lr)
            self.imperfect_info_encoder_optimizer_G = torch.optim.Adam(self.imperfect_info_encoder_G.parameters(), lr=self.IMP_lr)
        else:
            self.imperfect_info_encoder_optimizer = torch.optim.Adam(self.imperfect_info_encoder.parameters(), lr=self.IMP_lr)

    def step(self, states, candidates):


        if self.args['PFD_base_policy'] == 'CSEAL':
            candidates = self.get_CN_candidates()
        current_ks = self.get_learner_ks()
        perfect_logs = [[int(log[0]), log[1]] for log in self.env._learner._logs]
        perfect_logs = perfect_logs[-self.args['perfect_log_max_length']:]
        perfect_log_one_hot = get_feature_matrix(perfect_logs,
                                                 targets=[],
                                                 action_dim=self.num_skills,
                                                 embedding_dim=2 * self.num_skills,
                                                 max_sequence_length=self.args['perfect_log_max_length'],
                                                 device=self.device)
        perfect_log_one_hot = perfect_log_one_hot.unsqueeze(0)
        self.args['steptime_perfect_log_one_hot'] = torch.cat((self.args['steptime_perfect_log_one_hot'], perfect_log_one_hot.detach()), dim=0)
        self.args['steptime_knowledge_state'] = torch.cat((self.args['steptime_knowledge_state'], current_ks), dim=0)

        if self.args['episode_count'] < self.phase_1_episodes or self.args['PFD_phase_1_only']:


            if 'DKT' in self.args['PFD_ImPerEncoder']:
                imperfect_logs = []
                if self.args['step_count'] > 0:
                    imperfect_logs = [[int(log[0]), log[1]] for log in self.env._learner._logs[-self.args['step_count']:]]
                imperfect_log_one_hot = get_feature_matrix(imperfect_logs,
                                                           targets=[],
                                                           action_dim=self.num_skills,
                                                           embedding_dim=2 * self.num_skills,
                                                           max_sequence_length=self.args['perfect_log_max_length'],
                                                           device=self.device)
                imperfect_log_one_hot = imperfect_log_one_hot.unsqueeze(0)
                if 'GNN' in self.args['PFD_ImPerEncoder']:
                    kt_scores = self.imperfect_info_encoder_R(imperfect_log_one_hot)[max(0, len(imperfect_logs) - 1), :, :]
                    self.imperfect_info_encoder_optimizer_R.zero_grad()
                    kt_dist_loss = self.kt_distill_loss(kt_scores, current_ks)
                    kt_dist_loss.backward(retain_graph=True)
                    self.imperfect_info_encoder_optimizer_R.step()
                else:
                    kt_scores = self.imperfect_info_encoder(imperfect_log_one_hot)[max(0, len(imperfect_logs) - 1), :, :]
                    self.imperfect_info_encoder_optimizer.zero_grad()
                    kt_dist_loss = self.kt_distill_loss(kt_scores, current_ks)
                    kt_dist_loss.backward(retain_graph=True)
                    self.imperfect_info_encoder_optimizer.step()

            if self.args['PFD_PerEncoder'] == 'MLP':
                z_t = self.perfect_info_encoder(current_ks)
                self.args['steptime_state_saver'] = torch.cat((self.args['steptime_state_saver'], current_ks), dim=0)
                if self.args['step_count'] > 0:
                    self.args['steptime_next_state_saver'] = torch.cat((self.args['steptime_next_state_saver'], current_ks), dim=0)
            elif self.args['PFD_PerEncoder'] == 'GNN':
                cat_ks = current_ks.view(-1).unsqueeze(1)
                input_graph_embeds = torch.cat((self.initial_graph_embeddings, cat_ks), 1)
                graph_embeddings = self.perfect_info_encoder(input_graph_embeds, self.env_graph_edges)
                z_t = torch.mean(graph_embeddings, dim=0).view(1, -1)

                self.args['steptime_state_saver'] = torch.cat((self.args['steptime_state_saver'], current_ks), dim=0)
                if self.args['step_count'] > 0:
                    self.args['steptime_next_state_saver'] = torch.cat((self.args['steptime_next_state_saver'], current_ks), dim=0)
            elif self.args['PFD_PerEncoder'] == 'No':
                z_t = current_ks
                self.args['steptime_state_saver'] = torch.cat((self.args['steptime_state_saver'], z_t), dim=0)
                if self.args['step_count'] > 0:
                    self.args['steptime_next_state_saver'] = torch.cat((self.args['steptime_next_state_saver'], z_t), dim=0)

            RL_input = torch.cat((states, z_t), dim=1)
            action = self.base_policy.step(RL_input, candidates)
        else:


            imperfect_logs = []
            if self.args['step_count'] > 0:
                imperfect_logs = [[int(log[0]), log[1]] for log in self.env._learner._logs[-self.args['step_count']:]]
            imperfect_log_one_hot = get_feature_matrix(imperfect_logs,
                                                       targets=[],
                                                       action_dim=self.num_skills,
                                                       embedding_dim=2 * self.num_skills,
                                                       max_sequence_length=self.args['perfect_log_max_length'],
                                                       device=self.device)
            imperfect_log_one_hot = imperfect_log_one_hot.unsqueeze(0)

            if self.args['PFD_ImPerEncoder'] == 'RNN':
                imperfect_encoding = self.imperfect_info_encoder(imperfect_log_one_hot)
                z_t_est = imperfect_encoding[max(0, len(imperfect_logs) - 1), :, :]
            elif self.args['PFD_ImPerEncoder'] == 'Transformer':
                imperfect_encoding = self.imperfect_info_encoder(imperfect_log_one_hot)
                z_t_est = torch.mean(imperfect_encoding, dim=0)
            elif self.args['PFD_ImPerEncoder'] == 'DKT':
                imperfect_encoding = self.imperfect_info_encoder(imperfect_log_one_hot)
                z_t_est = imperfect_encoding[max(0, len(imperfect_logs) - 1), :, :]
            elif self.args['PFD_ImPerEncoder'] == 'RNN_GNN':
                imperfect_encoding = self.imperfect_info_encoder_R(imperfect_log_one_hot)
                cat_ks_RNN = imperfect_encoding[max(0, len(imperfect_logs) - 1), :, :].view(-1).unsqueeze(1)
                input_graph_embeds = torch.cat((self.initial_graph_embeddings, cat_ks_RNN), 1)
                graph_embeddings = self.imperfect_info_encoder_G(input_graph_embeds, self.env_graph_edges)
                z_t_est = torch.mean(graph_embeddings, dim=0).view(1, -1)
            elif self.args['PFD_ImPerEncoder'] == 'DKT_GNN':
                imperfect_encoding = self.imperfect_info_encoder_R(imperfect_log_one_hot)
                cat_ks_RNN = imperfect_encoding[max(0, len(imperfect_logs) - 1), :, :].view(-1).unsqueeze(1)
                input_graph_embeds = torch.cat((self.initial_graph_embeddings, cat_ks_RNN), 1)
                graph_embeddings = self.imperfect_info_encoder_G(input_graph_embeds, self.env_graph_edges)
                z_t_est = torch.mean(graph_embeddings, dim=0).view(1, -1)

            if 'DKT' in self.args['PFD_ImPerEncoder']:
                kt_scores = imperfect_encoding[max(0, len(imperfect_logs) - 1), :, :]
                if 'GNN' in self.args['PFD_ImPerEncoder']:
                    self.imperfect_info_encoder_optimizer_R.zero_grad()
                    kt_dist_loss = self.kt_distill_loss(kt_scores, current_ks)
                    kt_dist_loss.backward(retain_graph=True)
                    self.imperfect_info_encoder_optimizer_R.step()
                else:
                    self.imperfect_info_encoder_optimizer.zero_grad()
                    kt_dist_loss = self.kt_distill_loss(kt_scores, current_ks)
                    kt_dist_loss.backward(retain_graph=True)
                    self.imperfect_info_encoder_optimizer.step()

            RL_input = torch.cat((states, z_t_est), dim=1)
            action = self.base_policy.step(RL_input, candidates)
            self.args['steptime_state_saver'] = torch.cat((self.args['steptime_state_saver'], imperfect_log_one_hot.detach()), dim=0)
            if self.args['step_count'] > 0:
                self.args['steptime_next_state_saver'] = torch.cat((self.args['steptime_next_state_saver'], imperfect_log_one_hot.detach()), dim=0)

        return action

    def learn(self, RL_states, actions, RL_next_states, rewards, dones):
        if self.args['episode_count'] < self.phase_1_episodes or self.args['PFD_phase_1_only']:


            perfect_logs = [[int(log[0]), log[1]] for log in self.env._learner._logs]
            perfect_logs = perfect_logs[-self.args['perfect_log_max_length']:]
            perfect_log_one_hot = get_feature_matrix(perfect_logs,
                                                     targets=[],
                                                     action_dim=self.num_skills,
                                                     embedding_dim=2 * self.num_skills,
                                                     max_sequence_length=self.args['perfect_log_max_length'],
                                                     device=self.device)
            perfect_log_one_hot = perfect_log_one_hot.unsqueeze(0)
            if self.args['PFD_PerEncoder'] in ['MLP', 'No', 'GNN']:
                self.args['steptime_next_state_saver'] = torch.cat((self.args['steptime_next_state_saver'],
                                                                    self.get_learner_ks()),
                                                                   dim=0)
            else:
                self.args['steptime_next_state_saver'] = torch.cat((self.args['steptime_next_state_saver'], perfect_log_one_hot.detach()), dim=0)

            if self.args['PFD_PerEncoder'] == 'MLP':
                RL_states_sup_tmp = self.perfect_info_encoder(self.args['steptime_state_saver'])
                RL_next_states_sup_tmp = self.perfect_info_encoder(self.args['steptime_next_state_saver'])
                RL_states_sup = RL_states_sup_tmp
                RL_next_states_sup = RL_next_states_sup_tmp
            elif self.args['PFD_PerEncoder'] == 'GNN':
                init_g_embeds = self.initial_graph_embeddings.repeat(self.args['steptime_state_saver'].size(0), 1)
                batch_ks = self.args['steptime_state_saver'].view(-1).unsqueeze(1)
                input_graph_embeds = torch.cat((init_g_embeds, batch_ks), -1)
                batch_graph_edges = torch.tensor([], dtype=torch.long).to(self.device)
                for i in range(self.args['steptime_state_saver'].size(0)):
                    batch_graph_edges = torch.cat((batch_graph_edges, self.env_graph_edges + i * self.num_skills), dim=1)
                graph_embeddings = self.perfect_info_encoder(input_graph_embeds, batch_graph_edges).view(self.args['steptime_state_saver'].size(0), self.num_skills, self.z_size)
                RL_states_sup = torch.mean(graph_embeddings, dim=1)

                init_g_embeds = self.initial_graph_embeddings.repeat(self.args['steptime_next_state_saver'].size(0), 1)
                batch_ks = self.args['steptime_next_state_saver'].view(-1).unsqueeze(1)
                input_graph_embeds = torch.cat((init_g_embeds, batch_ks), -1)
                graph_embeddings = self.perfect_info_encoder(input_graph_embeds, batch_graph_edges).view(self.args['steptime_state_saver'].size(0), self.num_skills, self.z_size)
                RL_next_states_sup = torch.mean(graph_embeddings, dim=1)
            elif self.args['PFD_PerEncoder'] == 'No':
                RL_states_sup_tmp = self.args['steptime_state_saver']
                RL_next_states_sup_tmp = self.args['steptime_next_state_saver']
                RL_states_sup = RL_states_sup_tmp
                RL_next_states_sup = RL_next_states_sup_tmp

            RL_states = torch.cat((RL_states, RL_states_sup), dim=1)
            RL_next_states = torch.cat((RL_next_states, RL_next_states_sup), dim=1)

            self.perfect_info_encoder_optimizer.zero_grad()
            self.base_policy.learn(RL_states, actions, RL_next_states, rewards, dones)
            self.perfect_info_encoder_optimizer.step()

        else:

            imperfect_logs = []
            for i in range(len(self.args['learner_initial_logs']), len(self.env._learner._logs)):
                imperfect_logs.append([int(self.env._learner._logs[i][0]), int(self.env._learner._logs[i][1])])
            imperfect_log_one_hot = get_feature_matrix(imperfect_logs,
                                                       targets=[],
                                                       action_dim=self.num_skills,
                                                       embedding_dim=2 * self.num_skills,
                                                       max_sequence_length=self.args['perfect_log_max_length'],
                                                       device=self.device)
            imperfect_log_one_hot = imperfect_log_one_hot.unsqueeze(0)
            self.args['steptime_next_state_saver'] = torch.cat((self.args['steptime_next_state_saver'], imperfect_log_one_hot.detach()), dim=0)

            if 'GNN' not in self.args['PFD_ImPerEncoder']:
                RL_states_sup_tmp = self.imperfect_info_encoder(self.args['steptime_state_saver'])
                RL_next_states_sup_tmp = self.imperfect_info_encoder(self.args['steptime_next_state_saver'])
                state_indexs = torch.sum(self.args['steptime_state_saver'], dim=(1, 2)).view(1, -1, 1).expand(1, -1, RL_states_sup_tmp.size(-1)).long() - 1
                zero = torch.zeros_like(state_indexs)
                state_indexs = torch.where(state_indexs < 0, zero, state_indexs)
                RL_states_sup = torch.gather(RL_states_sup_tmp, dim=0, index=state_indexs).squeeze(0)

                next_state_indexs = torch.sum(self.args['steptime_next_state_saver'], dim=(1, 2)).view(1, -1, 1).expand(1, -1, RL_next_states_sup_tmp.size(-1)).long() - 1
                zero = torch.zeros_like(next_state_indexs)
                next_state_indexs = torch.where(next_state_indexs < 0, zero, next_state_indexs)
                RL_next_states_sup = torch.gather(RL_next_states_sup_tmp, dim=0, index=next_state_indexs).squeeze(0)
                if self.args['PFD_ImPerEncoder'] == 'Transformer':
                    RL_states_sup = torch.mean(RL_states_sup_tmp, dim=0)
                    RL_next_states_sup = torch.mean(RL_next_states_sup_tmp, dim=0)
            elif self.args['PFD_ImPerEncoder'] in ['RNN_GNN', 'DKT_GNN']:
                RL_states_sup_tmp = self.imperfect_info_encoder_R(self.args['steptime_state_saver'])
                RL_next_states_sup_tmp = self.imperfect_info_encoder_R(self.args['steptime_next_state_saver'])

                state_indexs = torch.sum(self.args['steptime_state_saver'], dim=(1, 2)).view(1, -1, 1).expand(1, -1, RL_states_sup_tmp.size(-1)).long() - 1
                zero = torch.zeros_like(state_indexs)
                state_indexs = torch.where(state_indexs < 0, zero, state_indexs)
                RL_states_sup = torch.gather(RL_states_sup_tmp, dim=0, index=state_indexs).squeeze(0).view(-1).unsqueeze(1)

                init_g_embeds = self.initial_graph_embeddings.repeat(self.args['steptime_knowledge_state'].size(0), 1)
                input_graph_embeds = torch.cat((init_g_embeds, RL_states_sup), -1)
                batch_graph_edges = torch.tensor([], dtype=torch.long).to(self.device)
                for i in range(self.args['steptime_knowledge_state'].size(0)):
                    batch_graph_edges = torch.cat((batch_graph_edges, self.env_graph_edges + i * self.num_skills), dim=1)
                graph_embeddings = self.imperfect_info_encoder_G(input_graph_embeds, batch_graph_edges).view(self.args['steptime_knowledge_state'].size(0), self.num_skills,
                                                                                                             self.z_size)
                RL_states_sup = torch.mean(graph_embeddings, dim=1)

                next_state_indexs = torch.sum(self.args['steptime_next_state_saver'], dim=(1, 2)).view(1, -1, 1).expand(1, -1, RL_next_states_sup_tmp.size(-1)).long() - 1
                zero = torch.zeros_like(next_state_indexs)
                next_state_indexs = torch.where(next_state_indexs < 0, zero, next_state_indexs)
                RL_next_states_sup = torch.gather(RL_next_states_sup_tmp, dim=0, index=next_state_indexs).squeeze(0).view(-1).unsqueeze(1)

                init_g_embeds = self.initial_graph_embeddings.repeat(self.args['steptime_knowledge_state'].size(0), 1)
                input_graph_embeds = torch.cat((init_g_embeds, RL_next_states_sup), -1)
                batch_graph_edges = torch.tensor([], dtype=torch.long).to(self.device)
                for i in range(self.args['steptime_knowledge_state'].size(0)):
                    batch_graph_edges = torch.cat((batch_graph_edges, self.env_graph_edges + i * self.num_skills), dim=1)
                graph_embeddings = self.imperfect_info_encoder_G(input_graph_embeds, batch_graph_edges).view(self.args['steptime_knowledge_state'].size(0), self.num_skills,
                                                                                                             self.z_size)
                RL_next_states_sup = torch.mean(graph_embeddings, dim=1)

            if 'DKT' not in self.args['PFD_ImPerEncoder']:
                if self.args['PFD_PerEncoder'] == 'No':
                    knowledge_state_encoding = self.args['steptime_knowledge_state']
                elif self.args['PFD_PerEncoder'] == 'MLP':
                    knowledge_state_encoding = self.perfect_info_encoder(self.args['steptime_knowledge_state']).detach()
                elif self.args['PFD_PerEncoder'] == 'GNN':
                    init_g_embeds = self.initial_graph_embeddings.repeat(self.args['steptime_knowledge_state'].size(0), 1)
                    batch_ks = self.args['steptime_knowledge_state'].view(-1).unsqueeze(1)
                    input_graph_embeds = torch.cat((init_g_embeds, batch_ks), -1)
                    batch_graph_edges = torch.tensor([], dtype=torch.long).to(self.device)
                    for i in range(self.args['steptime_knowledge_state'].size(0)):
                        batch_graph_edges = torch.cat((batch_graph_edges, self.env_graph_edges + i * self.num_skills), dim=1)
                    graph_embeddings = self.perfect_info_encoder(input_graph_embeds, batch_graph_edges).view(self.args['steptime_knowledge_state'].size(0), self.num_skills, self.z_size)
                    knowledge_state_encoding = torch.mean(graph_embeddings, dim=1)
                self.regress_loss_weight = min(math.log10(self.args['episode_count'] / 100), 1.0)
                regress_loss = self.regress_loss_weight * self.regress_loss_algo(RL_states_sup, knowledge_state_encoding)

            if self.args['PFD_imper_reg_only']:
                self.imperfect_info_encoder_optimizer.zero_grad()
                regress_loss.backward()
                self.args['PFD_regress_loss'] = regress_loss.detach().cpu().item()
                self.imperfect_info_encoder_optimizer.step()
                RL_states_sup = RL_states_sup.detach()
                RL_next_states_sup = RL_next_states_sup.detach()

            pfd_states = torch.cat((RL_states, RL_states_sup), dim=1)
            pfd_next_states = torch.cat((RL_next_states, RL_next_states_sup), dim=1)

            if not self.args['PFD_imper_reg_only']:
                if 'GNN' in self.args['PFD_ImPerEncoder']:
                    self.imperfect_info_encoder_optimizer_G.zero_grad()
                    self.imperfect_info_encoder_optimizer_R.zero_grad()
                else:
                    self.imperfect_info_encoder_optimizer.zero_grad()

            self.base_policy.learn(pfd_states, actions, pfd_next_states, rewards, dones)

            if self.args['PFD_ImPerEncoder'] == 'DKT_GNN':
                self.imperfect_info_encoder_optimizer_G.step()
            elif self.args['PFD_ImPerEncoder'] != 'DKT' and not self.args['PFD_imper_reg_only']:
                regress_loss.backward(retain_graph=True)
                self.args['PFD_regress_loss'] = regress_loss.detach().cpu().item()
                if self.args['PFD_ImPerEncoder'] == 'RNN_GNN':
                    self.imperfect_info_encoder_optimizer_G.step()
                    self.imperfect_info_encoder_optimizer_R.step()
                else:
                    self.imperfect_info_encoder_optimizer.step()


    def get_CN_candidates(self):
        if int(self.pre_learn_node) != -1:
            learning_item = self.env.learning_item_base[self.pre_learn_node]
            candidates = influence_control(
                self.env.knowledge_structure,
                list(self.get_learner_ks().view(-1).cpu().numpy()),
                learning_item.knowledge,
                allow_shortcut=False,
                target=self.env._learner.target
            )[0]
        else:
            candidates = [i for i in range(self.num_skills)]
        return candidates

    def get_learner_ks(self):
        ks_ = torch.tensor(self.env._learner._state, dtype=torch.float).view(1, -1).to(self.device).detach()
        if self.env.env_name == 'KSS':
            ks_ = torch.sigmoid(ks_)
        return ks_


