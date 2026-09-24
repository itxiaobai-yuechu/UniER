


from EduSim.agents import *
from EduSim.Envs.buffer import *
from EduSim.Envs.agent_utils import *
from EduSim.Envs.shared.KSS_KES.KS import influence_control
import networkx as nx
import copy
import torch
import numpy as np
import random
import time
import os
from EduSim.Envs.agent_utils import get_proj_path, get_raw_data_path
from tqdm import tqdm
from EduSim.Envs.deep_model import *


class AbstractAgent:
    def __init__(self, env, args):
        self.args = args
        self.env = env
        self.initial_learner = []
        self.max_sequence_length = args['max_steps']
        self.action_space = env.action_space
        self.state_t = []
        self.no_target_state_t = torch.tensor([])
        self.rl_state_t = torch.tensor([])
        self.pre_log = None
        self.tmp_logs = []
        self.learner_profile = []
        self.learner_targets = []
        self.DKT_states = []
        self.training_start_time = time.time()
        self.episode_start_time = time.time()
        self.episode_end_time = time.time()
        self.args['cur_goal'] = -1
        self.args['pre_goal'] = -1
        self.args['cur_goal_count'] = 0
        self.pre_step_ks = []
        self.after_step_ks = []

        self.action_dim = len(list(env.action_space))
        self.num_skills = self.action_dim
        self.DKT_input_dim = 2 * self.action_dim

        self.RNN_encoder_input_dim = 2 * self.action_dim
        self.RNN_encoder_output_dim = 4 * self.action_dim
        self.RL_input_dim = self.RNN_encoder_output_dim + self.action_dim

        self.learning_rate = self.args['learning_rate']
        self.batch_size = 20
        self.buffer_size = 1310720
        self.gamma = 0.98
        self.epoch_num = 1
        self.target_update = 10
        self.use_gpu = True
        self.begin_train_length = 10000

        self.device = args['device']
        print(f'self.device:{self.device}')

        self.experiment_idx = str(args['max_steps']) + '_' + args['target_type']
        self.repeat_num = args['repeat_num']
        print(f'Current Experiment Index:{self.experiment_idx}_{self.repeat_num}')
        if self.env.env_name == 'KESassist15':
            self.log_dir_name = f'{get_proj_path()}/EduSim/Envs/KES_ASSIST15/Experiment_logs' \
                                f'/Experiment_{self.experiment_idx}/'
        elif self.env.env_name == 'KESjunyi':
            self.log_dir_name = f'{get_proj_path()}/EduSim/Envs/KES/Experiment_logs' \
                                f'/Experiment_{self.experiment_idx}/'
        elif self.env.env_name == 'KSS':
            self.log_dir_name = f'{get_proj_path()}/EduSim/Envs/KSS/Experiment_logs' \
                                f'/Experiment_{self.experiment_idx}/'
        elif self.env.env_name == 'KESassist09':
            self.log_dir_name = f'{get_proj_path()}/EduSim/Envs/KES_ASSIST09/Experiment_logs' \
                                f'/Experiment_{self.experiment_idx}/'
        elif self.env.env_name == 'KESalgebra2005':
            self.log_dir_name = f'{get_proj_path()}/EduSim/Envs/KES_algebra2005/Experiment_logs' \
                                f'/Experiment_{self.experiment_idx}/'
        elif self.env.env_name == 'KESbridge2006':
            self.log_dir_name = f'{get_proj_path()}/EduSim/Envs/KES_bridge2006/Experiment_logs' \
                                f'/Experiment_{self.experiment_idx}/'
        elif self.env.env_name == 'KESassist12':
            self.log_dir_name = f'{get_proj_path()}/EduSim/Envs/KES_ASSIST12/Experiment_logs' \
                                f'/Experiment_{self.experiment_idx}/'
        elif self.env.env_name == 'KESassist17':
            self.log_dir_name = f'{get_proj_path()}/EduSim/Envs/KES_ASSIST17/Experiment_logs' \
                                f'/Experiment_{self.experiment_idx}/'
        elif self.env.env_name == 'KESnips34':
            self.log_dir_name = f'{get_proj_path()}/EduSim/Envs/KES_nips34/Experiment_logs' \
                                f'/Experiment_{self.experiment_idx}/'
        elif self.env.env_name == 'KESednet':
            self.log_dir_name = f'{get_proj_path()}/EduSim/Envs/KES_ednet/Experiment_logs' \
                                f'/Experiment_{self.experiment_idx}/'
        elif self.env.env_name == 'KESjunyi':
            self.log_dir_name = f'{get_proj_path()}/EduSim/Envs/KES_junyi/Experiment_logs' \
                                f'/Experiment_{self.experiment_idx}/'
        elif self.env.env_name == 'KESmooccube':
            self.log_dir_name = f'{get_proj_path()}/EduSim/Envs/KES_mooccube/Experiment_logs' \
                                f'/Experiment_{self.experiment_idx}/'
        elif self.env.env_name == 'KESxes3g5m':
            self.log_dir_name = f'{get_proj_path()}/EduSim/Envs/KES_xes3g5m/Experiment_logs'\
                                f'/Experiment_{self.experiment_idx}/'

        os.makedirs(self.log_dir_name, exist_ok=True)

        self.has_continuous_action_sapce = False
        self.graph_embedding_input = False
        self.action_bound = 10

        self.intrinsic_reward_flag = False

        self.dktnet_trian = True
        self.goal_orinted_candidate = False
        if 'GON' in args['agent']:
            self.goal_orinted_candidate = True
        self.goal_nav_random = False
        self.CN_candidate = False

        self.mapgo_model_based_mode = False

        self.random_policy = False
        self.use_former = False
        self.lr_sche = False

        self.embed_dim = 128
        self.DKT_hidden_size = 256
        self.hidden_size_1 = 128
        self.hidden_size_2 = 64
        self.hidden_size_3 = 32
        if self.env.env_name == 'KESassist15':
            self.DKT_model_path = f'{get_proj_path()}/EduSim/Envs/KES_ASSIST15/meta_data/agent_weights/ValBest.ckpt'
        elif self.env.env_name == 'KESjunyi':
            self.DKT_model_path = f'{get_proj_path()}/EduSim/Envs/KES_junyi/meta_data/agent_weights/ValBest.ckpt'
        elif self.env.env_name == 'KESassist09':
            self.DKT_model_path = f'{get_proj_path()}/EduSim/Envs/KES_ASSIST09/meta_data/agent_weights/ValBest.ckpt'
        elif self.env.env_name == 'KESassist12':
            self.DKT_model_path = f'{get_proj_path()}/EduSim/Envs/KES_ASSIST12/meta_data/agent_weights/ValBest.ckpt'
        elif self.env.env_name == 'KESassist17':
            self.DKT_model_path = f'{get_proj_path()}/EduSim/Envs/KES_ASSIST17/meta_data/agent_weights/ValBest.ckpt'    
        elif self.env.env_name == 'KESalgebra2005':
            self.DKT_model_path = f'{get_proj_path()}/EduSim/Envs/KES_algebra2005/meta_data/agent_weights/ValBest.ckpt'
        elif self.env.env_name == 'KESbridge2006':
            self.DKT_model_path = f'{get_proj_path()}/EduSim/Envs/KES_bridge2006/meta_data/agent_weights/ValBest.ckpt'
        elif self.env.env_name == 'KESnips34':
            self.DKT_model_path = f'{get_proj_path()}/EduSim/Envs/KES_nips34/meta_data/agent_weights/ValBest.ckpt'
        elif self.env.env_name == 'KESednet':
            self.DKT_model_path = f'{get_proj_path()}/EduSim/Envs/KES_ednet/meta_data/agent_weights/ValBest.ckpt'
        elif self.env.env_name == 'KESjunyi':
            self.DKT_model_path = f'{get_proj_path()}/EduSim/Envs/KES_junyi/meta_data/agent_weights/ValBest.ckpt'
        elif self.env.env_name == 'KESmooccube':
            self.DKT_model_path = f'{get_proj_path()}/EduSim/Envs/KES_mooccube/meta_data/agent_weights/ValBest.ckpt'
        elif self.env.env_name == 'KESxes3g5m':
            self.DKT_model_path = f'{get_proj_path()}/EduSim/Envs/KES_xes3g5m/meta_data/agent_weights/ValBest.ckpt'
            
        self.KTnet = KTnet(input_size=self.DKT_input_dim,
                             emb_dim=self.embed_dim,
                             hidden_size=self.DKT_hidden_size,
                             num_skills=self.action_dim,
                             nlayers=2).to(self.device)
        if self.env.env_name == 'KSS':
            self.KTnet.load_state_dict(torch.load(self.DKT_model_path, map_location=self.device)['model'])
        else:
            state_dict = torch.load(self.DKT_model_path, map_location=self.device)
            if "embedding_layer.bias" in state_dict:
                del state_dict["embedding_layer.bias"]
            self.KTnet.load_state_dict(state_dict)

        self.KTnet_optimizer = torch.optim.Adam(self.KTnet.parameters(), lr=0.0001)

        self.normal_rnnEncoder = RnnEncoder(input_size=self.RNN_encoder_input_dim,
                                            emb_dim=self.embed_dim,
                                            hidden_size=self.hidden_size_1,
                                            num_skills=self.RNN_encoder_output_dim,
                                            nlayers=2).to(self.device)
        if self.args['encoder'] == 'RNN':
            self.rnnEncoder = self.normal_rnnEncoder
        else:
            self.rnnEncoder = self.KTnet

        if self.rnnEncoder.name == 'DKT':
            self.RL_input_dim = 2 * self.action_dim

        if args['agent'] == 'CSEAL':
            self.buffer_size = 1
            self.begin_train_length = 0
            self.epoch_num = 1
            baseline_settings = {
                'Cog_baseline': False,
                'CSEAL_NCN_baseline': False,
                'CN_random': False
            }
            self.agent = CSEAL(self.RL_input_dim,
                               self.action_dim,
                               self.hidden_size_1,
                               self.hidden_size_3,
                               baseline_settings,
                               self.env,
                               lr_rate=self.learning_rate,
                               gamma=self.gamma,
                               device=self.device,
                               args=args)

        self.rnnEncoder_optimizer = torch.optim.Adam(self.rnnEncoder.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.rnn_scheduler = torch.optim.lr_scheduler.StepLR(self.rnnEncoder_optimizer, step_size=5000, gamma=0.95)

        self.train_buffer = ReplayBuffer(self.buffer_size,
                                         action_net=(self.rnnEncoder, self.agent),
                                         episode_length=self.max_sequence_length,
                                         model_based_mode=self.mapgo_model_based_mode,
                                         cudaDevice=args['device'])

        if args['model_based'] == 'DAS3H':
            raise ValueError('RLtutor model based only work with PPO')
            
        if self.args['load_model_test']:
            if os.path.exists(f'{get_proj_path()}/EduSim/SavedAgents/{self.agent.name}_{self.env.env_name}_{self.args["max_steps"]}_{self.args["max_episode_num"]}_{self.args["target_type"]}.pth'):
                checkpoint = torch.load(f'{get_proj_path()}/EduSim/SavedAgents/{self.agent.name}_{self.env.env_name}_{self.args["max_steps"]}_{self.args["max_episode_num"]}_{self.args["target_type"]}.pth')
            else:
                raise ValueError('Checkpoint not exitst!!')

            self.agent.load_state_dict(checkpoint['model_state_dict'])
            self.TestAgent()
            assert 0

    def begin_episode(self, learner_profile):
        self.args['step_count'] = 0
        self.args['steptime_state_saver'] = torch.tensor([], dtype=torch.float).to(self.device)
        self.args['steptime_knowledge_state'] = torch.tensor([], dtype=torch.float).to(self.device)
        self.args['steptime_next_state_saver'] = torch.tensor([], dtype=torch.float).to(self.device)
        self.args['steptime_perfect_log_one_hot'] = torch.tensor([], dtype=torch.float).to(self.device)
        self.args['learner_initial_logs'] = copy.deepcopy(self.env._learner._logs)
        self.args['episode_subgoals'] = []
        self.args['steptime_dkt_ks'] = torch.tensor([], dtype=torch.float).to(self.device)
        self.args['cur_goal'] = -1
        self.args['pre_goal'] = -1
        self.args['cur_goal_count'] = 0
        self.args['current_rec_log'] = []
        self.args['repe_abandon_list'] = []
        self.args['item_count_dict'] = {}

        self.state_t = []
        self.initial_learner = copy.deepcopy(self.env._learner)
        if self.args['know_all_log']:
            self.state_t = copy.deepcopy(self.env._learner._logs[-(self.args['perfect_log_max_length'] - self.args['max_steps']):])
            self.max_sequence_length = self.args['max_steps'] + self.args['perfect_log_max_length'] - self.args['max_steps']
        self.learner_targets = list(learner_profile[0]['target'])
        self.learner_profile = learner_profile[0]['logs']
        if self.use_former:
            self.state_t = learner_profile[0]['logs']
        self.rl_state_t = torch.tensor([])
        self.tmp_logs = []
        self.DKT_states = []
        self.agent.pre_learn_node = '-1'
        self.episode_start_time = time.time()

    def step(self):
        self.pre_step_ks = self.env._learner._state

        one_hot_data = get_feature_matrix(self.state_t, self.learner_targets, self.action_dim, self.RNN_encoder_input_dim,
                                          device=self.device,
                                          max_sequence_length=self.max_sequence_length,
                                          graph_embedding=False,
                                          graph_embeddings=self.env.graph_embeddings)
        one_hot_data = one_hot_data.unsqueeze(0).to(self.device)

        cur_DKT_states = torch.sigmoid(self.KTnet(one_hot_data))[max(0, len(self.state_t) - 1), 0, :]
        self.args['steptime_dkt_ks'] = cur_DKT_states
        if self.goal_orinted_candidate or 'GON' in self.args['agent']:
            candidates = self.goal_oriented_candidate_choose(cur_DKT_states)
            if self.goal_nav_random:
                item = random.sample(candidates, 1)
                return str(item[0])
        elif self.CN_candidate:
            candidates = self.get_CN_candidates()
        else:
            candidates = [i for i in range(self.action_dim)]

        if self.args['cur_goal'] == self.args['pre_goal']:
            self.args['cur_goal_count'] += 1
        else:
            self.args['pre_goal'] = self.args['cur_goal']
            self.args['cur_goal_count'] = 0

        input_data = get_feature_matrix(self.state_t, self.learner_targets, self.action_dim, self.RNN_encoder_input_dim,
                                        device=self.device, max_sequence_length=self.max_sequence_length,
                                        graph_embedding=self.graph_embedding_input,
                                        graph_embeddings=self.env.graph_embeddings)
        input_data = input_data.unsqueeze(0)

        RNNoutput = self.rnnEncoder(input_data)
        if self.rnnEncoder.name == 'DKT':
            RNNoutput = torch.sigmoid(RNNoutput)

        RL_states = batch_cat_targets(RNNoutput[max(0, len(self.state_t) - 1), :, :], [self.learner_targets],
                                      self.num_skills, device=self.device,
                                      graph_embedding=self.graph_embedding_input,
                                      graph_embeddings=self.env.graph_embeddings)
        if self.args['Repe_control']:
            tmp_can = [i for i in candidates if i not in self.args['repe_abandon_list']]
            candidates = tmp_can

        agent_step = self.agent.step(RL_states, candidates)

        if self.train_buffer.size() * self.max_sequence_length < self.begin_train_length or self.random_policy:
            item = np.random.randint(self.action_dim)
        else:
            item = int(agent_step)
        if self.env.env_name == 'KSS':
            item = str(item)
        if self.args['Repe_control']:
            if int(item) in self.args['item_count_dict'].keys():
                self.args['item_count_dict'][int(item)] += 1
                thresh = max(min(self.args['max_steps'] - 2, int(0.4 * self.args['max_steps'])), 3)
                if self.args['item_count_dict'][int(item)] >= thresh:
                    self.args['repe_abandon_list'].append(int(item))
            else:
                self.args['item_count_dict'][int(item)] = 1

        return item

    def observe(self, observation, reward, done, info):
        self.after_step_ks = self.env._learner._state

        self.agent.pre_learn_node = observation[0]
        state = copy.deepcopy(self.state_t)
        self.state_t.append(list(observation))
        self.args['current_rec_log'].append(list(observation))
        next_state = copy.deepcopy(self.state_t)
        self.tmp_logs.append([state, observation[0], reward, next_state, done, self.args['cur_goal']])

        self.args['step_count'] += 1

    def end_episode(self, observation, reward, done, info):
        self.tmp_logs[-1][4] = True
        items = []
        answers = []
        for log in self.state_t:
            items.append(str(log[0]))
            answers.append(str(log[1]))
        num = [str(len(items))]
        self.pre_log = (num, items, answers)

        if not self.random_policy and not self.goal_nav_random:
            episode_reward_reshape(self.tmp_logs, reward, self.learner_targets, episode_dkt_states=self.DKT_states,
                                   targets=self.learner_targets, intrin_sic_reward_flag=self.intrinsic_reward_flag, args=self.args)
            correct_answered_ques = list(set([int(item) for i, item in enumerate(items) if int(answers[i]) == 1]))
            for i, el in enumerate(self.tmp_logs):
                targets = copy.deepcopy(self.learner_targets)
                el[0] = {'state': copy.deepcopy(el[0]), 'targets': targets}
                el[3] = {'next_state': copy.deepcopy(el[3]), 'targets': targets}
                self.train_buffer.add(el[0], el[1], el[2], el[3], el[4], i, correct_answered_ques, items, answers)

        file_name = str(self.repeat_num) + '.txt'
        if self.args.get('episode_count', 0) == 0:
            writing_mode = 'w'
        else:
            writing_mode = 'a'
        with open(self.log_dir_name + file_name, writing_mode) as f:
            lines = []
            if writing_mode == 'w':
                lines.append(f'{self.args}')
            lines.append(f'{items} {answers} \n')
            lines.append(f'reward:{reward}, target:{self.learner_targets}\n')
            if self.goal_orinted_candidate:
                lines.append(f"goals: {self.args['episode_subgoals']} \n {info.get('record', '')} \n")
            else:
                lines.append(f'{info.get("record", "")} \n')
            if self.args.get('PFD_regress_loss', -1) != -1:
                lines.append(f"PFD regress loss: {self.args['PFD_regress_loss']} \n")

            result = ''
            for line in lines:
                result = result + line
            f.write(result)

        if self.train_buffer.size() * self.max_sequence_length > self.begin_train_length and \
                not self.random_policy and \
                not self.goal_nav_random:
            for j in range(self.epoch_num):
                b_s, b_a, b_r, b_ns, b_d = self.train_buffer.sample(self.batch_size)
                transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                self.update(transition_dict)

                if self.mapgo_model_based_mode:
                    b_s, b_a, b_r, b_ns, b_d = self.train_buffer.sample(self.batch_size,
                                                                        multi_step=False,
                                                                        n_multi_step=0,
                                                                        gamma=self.gamma,
                                                                        model_based_sample=True)
                    transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                    self.update(transition_dict)

        if self.args['model_based'] != 'No':
            self.model_based_roll_update()
        self.args['episode_count'] += 1
        self.episode_end_time = time.time()
        
        cumulative_time = self.episode_end_time - getattr(self, 'training_start_time', self.episode_start_time)
        print(f"Episode {self.args['episode_count']} Cumulative Time: {cumulative_time:.2f}s")
        
        if self.args['episode_count'] == self.args['max_episode_num'] and not self.args['load_model_test']:
            torch.save({
                'model_state_dict': self.agent.state_dict(),
                'args': self.args,
            }, f'{get_proj_path()}/EduSim/SavedAgents/{self.agent.name}_{self.env.env_name}_{self.args["max_steps"]}_{self.args["max_episode_num"]}_{self.args["target_type"]}.pth')
            self.TestAgent()

    def update(self, transition_dict):
        actions = torch.tensor([int(item) for item in transition_dict['actions']]).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        states = torch.tensor([], dtype=torch.float).to(self.device)
        next_states = torch.tensor([], dtype=torch.float).to(self.device)
        for i in range(len(transition_dict['states'])):
            states = torch.cat((states, get_feature_matrix(transition_dict['states'][i]['state'],
                                                           transition_dict['states'][i]['targets'],
                                                           self.action_dim,
                                                           self.RNN_encoder_input_dim,
                                                           self.max_sequence_length,
                                                           self.device,
                                                           graph_embedding=self.graph_embedding_input,
                                                           graph_embeddings=self.env.graph_embeddings).unsqueeze(0)), dim=0)
            next_states = torch.cat((next_states, get_feature_matrix(transition_dict['next_states'][i]['next_state'],
                                                                     transition_dict['next_states'][i]['targets'],
                                                                     self.action_dim,
                                                                     self.RNN_encoder_input_dim,
                                                                     self.max_sequence_length,
                                                                     self.device,
                                                                     graph_embedding=self.graph_embedding_input,
                                                                     graph_embeddings=self.env.graph_embeddings
                                                                     ).unsqueeze(0)), dim=0)
        if self.dktnet_trian:
            batch_data_states = torch.tensor([], dtype=torch.float).to(self.device)
            for i in range(len(transition_dict['states'])):
                batch_data_states = torch.cat((batch_data_states, get_feature_matrix(transition_dict['states'][i]['state'],
                                                                                     transition_dict['states'][i]['targets'],
                                                                                     self.action_dim,
                                                                                     self.DKT_input_dim,
                                                                                     self.max_sequence_length,
                                                                                     self.device,
                                                                                     graph_embedding=self.graph_embedding_input,
                                                                                     graph_embeddings=self.env.graph_embeddings
                                                                                     ).unsqueeze(0)), dim=0)
            dktnet_states = self.KTnet(batch_data_states)
            self.KTnet_optimizer.zero_grad()
            batch_data_one_hot = torch.tensor([], dtype=torch.float).to(self.device)
            for i in range(len(transition_dict['states'])):
                batch_data_one_hot = torch.cat((batch_data_one_hot, get_feature_matrix(transition_dict['states'][i]['state'],
                                                                                       transition_dict['states'][i]['targets'],
                                                                                       self.action_dim,
                                                                                       self.DKT_input_dim,
                                                                                       self.max_sequence_length,
                                                                                       self.device,
                                                                                       graph_embedding=False
                                                                                       ).unsqueeze(0).to(self.device)), dim=0)
            dkt_loss = compute_dkt_loss(dktnet_states, batch_data_one_hot, device=self.device)
            dkt_loss.backward()
            self.KTnet_optimizer.step()

        DKT_states = self.rnnEncoder(states)
        DKT_next_states = self.rnnEncoder(next_states)
        states_lengths_ids = torch.tensor([max(0, len(sequence['state']) - 1)
                                           for sequence in transition_dict['states']]).to(self.device)
        next_states_lengths_ids = torch.tensor([max(0, len(sequence['next_state']) - 1)
                                                for sequence in transition_dict['next_states']]).to(self.device)
        if self.rnnEncoder.name == 'DKT':
            DKT_states = torch.sigmoid(DKT_states)
            DKT_next_states = torch.sigmoid(DKT_next_states)
        DKT_states = DKT_states.gather(0, states_lengths_ids.view(1, -1, 1).expand(-1, -1, DKT_states.size(2))).squeeze(0)
        DKT_next_states = DKT_next_states.gather(0, next_states_lengths_ids.view(1, -1, 1).
                                                 expand(-1, -1, DKT_next_states.size(2))).squeeze(0)

        targets = [item['targets'] for item in transition_dict['states']]
        RL_states = batch_cat_targets(DKT_states, targets, self.num_skills, device=self.device,
                                      graph_embedding=self.graph_embedding_input,
                                      graph_embeddings=self.env.graph_embeddings)
        RL_next_states = batch_cat_targets(DKT_next_states, targets, self.num_skills, device=self.device,
                                           graph_embedding=self.graph_embedding_input,
                                           graph_embeddings=self.env.graph_embeddings)

        self.rnnEncoder_optimizer.zero_grad()
        self.agent.learn(RL_states, actions, RL_next_states, rewards, dones)

        if self.rnnEncoder.name != 'DKT':
            self.rnnEncoder_optimizer.step()
            if self.lr_sche:
                self.rnn_scheduler.step()


    def TestAgent(self):
        print('Testing...')
        env = self.env
        test_learner_nums = [1000]
        num_mean_rewards = []
        max_steps = self.args['max_steps']

        for test_num in test_learner_nums:
            rewards = []
            for i in tqdm(range(test_num)):
                learner_profile = env.begin_episode()
                self.begin_episode(learner_profile)
                learning_path = []
                for j in range(max_steps):
                    learning_item = self.step()
                    learning_path.append(learning_item)

                    observation, reward, done, info = env.step(learning_item)
                    self.observe(observation, reward, done, info)
                    if done:
                        break
                observation, reward, done, info = env.end_episode()
                env.reset()
                rewards.append(reward)

            mean_reward = np.mean(np.array(rewards))
            num_mean_rewards.append(mean_reward)
        num_mean_rewards = np.array(num_mean_rewards)
        print(f'Agent test mean rewards on learner nums: {test_learner_nums} rewards: {num_mean_rewards}')

        test_path = f'{get_proj_path()}/EduSim/SavedAgents/{self.agent.name}_{self.env.env_name}_{self.args["max_steps"]}_{self.args["max_episode_num"]}_{self.args["target_type"]}_test.txt'
        if self.args['agent'] == 'PFD':
            base_p_name = self.args['PFD_base_policy']
            test_path = f'{get_proj_path()}/EduSim/SavedAgents/{self.agent.name}_{base_p_name}_{self.env.env_name}_{self.args["max_steps"]}_{self.args["max_episode_num"]}_{self.args["target_type"]}_test.txt'

        with open(test_path, 'w') as f:
            line1 = f'Agent test mean rewards on learner nums: {test_learner_nums} rewards: {num_mean_rewards} \n'
            line2 = 'args: \n' + str(self.args) + '\n'
            result = line1 + line2
            f.write(result)

    def goal_oriented_candidate_choose(self, DKT_states):
        if len(self.state_t) < 1:
            return [i for i in range(self.action_dim)]

        abandon_goals = []
        candidates = []
        while len(candidates) == 0:
            not_achieved_goals = [goal for goal in self.learner_targets if DKT_states[goal] < 0.9]
            goal_state = [[i, DKT_states[i]] for i in not_achieved_goals]
            goal_state = sorted(goal_state, key=lambda x: x[1], reverse=True)
            goal_state_sorted = [i[0] for i in goal_state]
            if len(not_achieved_goals) == 0:
                candidates = [i for i in range(self.action_dim)]

            if self.args['cur_goal'] != -1:
                if 0.5 < DKT_states[self.args['cur_goal']]:
                    self.args['cur_goal'] = -1
                if self.args['cur_goal_count'] >= 5:
                    self.args['cur_goal'] = -1
                if self.args['cur_goal_count'] >= 10 and self.args['cur_goal'] in not_achieved_goals:
                    self.args['cur_goal'] = -1
                    abandon_goals.append(self.args['cur_goal'])

            if self.args['cur_goal'] == -1:
                raise ValueError('Wrong cur goal!')

            if self.args['cur_goal'] != -1:
                subgraph = nx.bfs_tree(self.env.knowledge_structure, self.args['cur_goal'], reverse=True)
                sub_topo_order = list(nx.topological_sort(subgraph))
                candidates = sub_topo_order

            if int(self.agent.pre_learn_node) not in candidates:
                candidates = []
                abandon_goals.append(self.args['cur_goal'])
                self.args['cur_goal'] = -1
                if len(abandon_goals) >= len(not_achieved_goals):
                    candidates = [i for i in range(self.action_dim)]
        self.args['episode_subgoals'].append(self.args['cur_goal'])
        return candidates

    def get_CN_candidates(self):
        if int(self.agent.pre_learn_node) != -1:
            learning_item = self.env.learning_item_base[self.agent.pre_learn_node]
            candidates = influence_control(
                self.env.knowledge_structure,
                self.env._learner._state,
                learning_item.knowledge,
                allow_shortcut=False,
                target=self.env._learner.target
            )[0]
        else:
            candidates = [i for i in range(self.action_dim)]
        return candidates

    def n_step(self, max_steps: int):
        return [self.step() for _ in range(max_steps)]

    def get_learner_ks(self):
        ks_ = torch.tensor(self.env._learner._state, dtype=torch.float).view(1, -1).to(self.device).detach()
        if self.env.env_name == 'KSS':
            ks_ = torch.sigmoid(ks_)
        return ks_
