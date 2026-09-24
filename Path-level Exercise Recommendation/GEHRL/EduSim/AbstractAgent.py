
import copy
import time
import os
import numpy as np

import torch
from torch import nn
from torch.ops import composite as C

from EduSim.agents import HRL
from EduSim.buffer import ReplayBuffer
from EduSim.utils import get_feature_matrix, compute_dkt_loss, episode_reward_reshape, mds_concat
from EduSim.utils import get_proj_path
from EduSim.deep_model import KTnet, RnnEncoder


class AbstractAgent:
    def __init__(self, env, args):
        self.args = args
        self.env = env
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
        self.batch_size = 16
        self.buffer_size = 1310720
        self.gamma = 0.98
        self.epoch_num = 10
        self.target_update = 10
        self.use_gpu = True
        self.begin_train_length = 1000
        self.KTnet_trian = True
        self.random_policy = False

        self.experiment_idx = str(self.args['experiment_idx']) + '_' + args["target_type"]
        self.repeat_num = args['repeat_num']
        print(f'Current Experiment Index:{self.experiment_idx}_{self.repeat_num}')
        self.log_dir_name = (f'{get_proj_path()}/EduSim/Experiment_logs/'
                             f'{self.env.env_name}/Experiment_{self.experiment_idx}/')
        os.makedirs(self.log_dir_name, exist_ok=True)

        self.embed_dim = 128
        self.DKT_hidden_size = 256
        self.hidden_size_1 = 128
        self.hidden_size_2 = 64
        self.hidden_size_3 = 32
        if self.env.env_name == 'KESassist09':
            self.num_skills = 123
            self.DKT_model_path = f'{get_proj_path()}/EduSim/Envs/KES_ASSIST09/meta_data/agent_weights/ValBest.ckpt'
        elif self.env.env_name == 'KESassist12':
            self.num_skills = 265
            self.DKT_model_path = f'{get_proj_path()}/EduSim/Envs/KES_ASSIST12/meta_data/agent_weights/ValBest.ckpt'
        elif self.env.env_name == 'KESassist15':
            self.num_skills = 100
            self.DKT_model_path = f'{get_proj_path()}/EduSim/Envs/KES_ASSIST15/meta_data/agent_weights/ValBest.ckpt'
        elif self.env.env_name == 'KESassist17':
            self.num_skills = 102
            self.DKT_model_path = f'{get_proj_path()}/EduSim/Envs/KES_ASSIST17/meta_data/agent_weights/ValBest.ckpt'
        elif self.env.env_name == 'KESjunyi':
            self.num_skills = 39
            self.DKT_model_path = f'{get_proj_path()}/EduSim/Envs/KES_junyi/meta_data/agent_weights/ValBest.ckpt'
        elif self.env.env_name == 'KESmooccube':
            self.num_skills = 436
            self.DKT_model_path = f'{get_proj_path()}/EduSim/Envs/KES_mooccube/meta_data/agent_weights/ValBest.ckpt'
        elif self.env.env_name == 'KESalgebra2005':
            self.num_skills = 112
            self.DKT_model_path = f'{get_proj_path()}/EduSim/Envs/KES_algebra2005/meta_data/agent_weights/ValBest.ckpt'
        elif self.env.env_name == 'KESbridge2006':
            self.num_skills = 493
            self.DKT_model_path = f'{get_proj_path()}/EduSim/Envs/KES_bridge2006/meta_data/agent_weights/ValBest.ckpt'
        elif self.env.env_name == 'KESednet':
            self.num_skills = 188
            self.DKT_model_path = f'{get_proj_path()}/EduSim/Envs/KES_ednet/meta_data/agent_weights/ValBest.ckpt'
        elif self.env.env_name == 'KESnips34':
            self.num_skills = 57
            self.DKT_model_path = f'{get_proj_path()}/EduSim/Envs/KES_nips34/meta_data/agent_weights/ValBest.ckpt'
        elif self.env.env_name == 'KESxes3g5m':
            self.num_skills = 865
            self.DKT_model_path = f'{get_proj_path()}/EduSim/Envs/KES_xes3g5m/meta_data/agent_weights/ValBest.ckpt'

        self.feature_dim = 2 * self.num_skills
        dkt_para_dict = {
            'input_size': self.DKT_input_dim,
            'emb_dim': self.embed_dim,
            'hidden_size': self.DKT_hidden_size,
            'num_skills': self.action_dim,
            'nlayers': 2,
            'dropout': 0.01,
        }
        self.KTnet = KTnet(dkt_para_dict)
        param_dict = torch.load(self.DKT_model_path, weights_only=False)
        self.KTnet.load_state_dict(param_dict)

        rnn_para_dict = {
            'input_size': self.RNN_encoder_input_dim,
            'emb_dim': self.embed_dim,
            'hidden_size': self.hidden_size_1,
            'num_skills': self.RNN_encoder_output_dim,
            'nlayers': 2,
            'dropout': 0.0,
            'out_activation': 'Tanh'
        }
        self.normal_rnnEncoder = RnnEncoder(rnn_para_dict)
        self.rnnEncoder = self.normal_rnnEncoder

        self.buffer_size = 1
        self.begin_train_length = 0
        self.epoch_num = 1
        c2 = 0.005
        hrl_para_dict = {
            'input_dim': self.RL_input_dim,
            'output_dim': self.action_dim,
            'hidden_dim1': self.hidden_size_1,
            'hidden_dim2': self.hidden_size_3,
            'env': self.env,
            'lr_big': self.learning_rate,
            'lr_lit': self.learning_rate,
            'gamma': self.gamma,
            'c2': c2,
            'RNN_encoder_output_dim': self.RNN_encoder_output_dim,
            'args': self.args,
            'outer_encoder': self.rnnEncoder,
        }

        self.agent = HRL(hrl_para_dict)
        self.KTnet_optimizer = torch.optim.Adam(self.KTnet.parameters(), lr=0.0001)

        self.train_buffer = ReplayBuffer(self.buffer_size, episode_length=self.max_sequence_length)

    def begin_episode(self, learner_profile):
        self.args['step_count'] = 0
        self.args['steptime_state_saver'] = torch.tensor([], dtype=torch.float32)
        self.args['steptime_knowledge_state'] = torch.tensor([], dtype=torch.float32)
        self.args['steptime_next_state_saver'] = torch.tensor([], dtype=torch.float32)
        self.args['steptime_perfect_log_one_hot'] = torch.tensor([], dtype=torch.float32)
        self.args['learner_initial_logs'] = copy.deepcopy(self.env._learner._logs)
        self.args['episode_subgoals'] = []
        self.args['steptime_dkt_ks'] = torch.tensor([], dtype=torch.float32)
        self.args['cur_goal'] = -1
        self.args['pre_goal'] = -1
        self.args['cur_goal_count'] = 0
        self.args['current_rec_log'] = []
        self.args['repe_abandon_list'] = []
        self.args['item_count_dict'] = {}

        self.state_t = []
        self.learner_targets = list(learner_profile[0]['target'])
        self.learner_profile = learner_profile[0]['logs']
        self.rl_state_t = torch.tensor([], dtype=torch.float32)
        self.tmp_logs = []
        self.train_buffer.her_scores = []
        self.DKT_states = []
        self.agent.pre_learn_node = '-1'
        self.episode_start_time = time.time()

    def step(self):
        self.pre_step_ks = self.env._learner._state

        one_hot_data = get_feature_matrix(self.state_t,
                                          self.action_dim,
                                          self.RNN_encoder_input_dim)
        one_hot_data = one_hot_data.unsqueeze(0)

        cur_DKT_states = torch.sigmoid(self.KTnet(one_hot_data))[max(0, len(self.state_t) - 1), 0, :]
        self.args['steptime_dkt_ks'] = cur_DKT_states
        candidates = [i for i in range(self.action_dim)]

        if self.args['cur_goal'] == self.args['pre_goal']:
            self.args['cur_goal_count'] += 1
        else:
            self.args['pre_goal'] = self.args['cur_goal']
            self.args['cur_goal_count'] = 0

        input_data = get_feature_matrix(self.state_t, self.action_dim, self.RNN_encoder_input_dim)
        input_data = input_data.unsqueeze(0)
        state_input_dict = {
            'states': input_data,
            'states_lengths_ids': torch.tensor([max(0, len(self.state_t) - 1)]),
            'targets': [self.learner_targets]
        }
        agent_step = self.agent.step(state_input_dict, candidates)

        if self.train_buffer.size() * self.max_sequence_length < self.begin_train_length or self.random_policy:
            item = np.random.randint(self.action_dim)
        else:
            item = int(agent_step)
        if self.env.env_name == 'KSS':
            item = str(item)
        return item

    def observe(self, observation, reward, done, info):
        self.after_step_ks = self.env._learner._state

        self.agent.pre_learn_node = observation[0]
        state = copy.deepcopy(self.state_t)
        self.state_t.append(list(observation))
        self.args['current_rec_log'].append(list(observation))
        next_state = copy.deepcopy(self.state_t)
        self.tmp_logs.append([state, observation[0], reward, next_state, done, self.args['cur_goal']])
        if self.agent.name == 'HRL':
            input_data = get_feature_matrix(self.state_t,
                                            self.action_dim,
                                            self.RNN_encoder_input_dim)
            input_data = input_data.unsqueeze(0)
            DKT_states = torch.sigmoid(self.KTnet(input_data))[max(0, len(self.state_t) - 1), 0, :]
            self.args['steptime_dkt_ks'] = DKT_states.clone()
            observe_dict = {
                'observation': observation,
                'done': done,
                'input_data': input_data,
                'states_lengths_ids': torch.tensor([max(0, len(self.state_t) - 1)]),
                'targets': [self.learner_targets]
            }
            self.agent.observe(observe_dict)

        self.args['step_count'] += 1

    def end_episode(self, observation, reward, done, info):
        items = []
        answers = []
        for log in self.state_t:
            items.append(str(log[0]))
            answers.append(str(log[1]))
        num = [str(len(items))]
        self.pre_log = (num, items, answers)
        print()
        print(self.pre_log)
        print('reward: ' + str(reward))
        print('info' + str(info))


        episode_reward_reshape(self.tmp_logs, reward)
        for i, el in enumerate(self.tmp_logs):
            targets = copy.deepcopy(self.learner_targets)
            el[0] = {'state': copy.deepcopy(el[0]), 'targets': targets}
            el[3] = {'next_state': copy.deepcopy(el[3]), 'targets': targets}
            self.train_buffer.add(el[0], el[1], el[2], el[3], el[4], i)


        file_name = str(self.repeat_num) + '.txt'
        if self.args['episode_count'] == 1:
            writing_mode = 'w'
        else:
            writing_mode = 'a'
        with open(self.log_dir_name + file_name, writing_mode) as f:
            lines = []
            if writing_mode == 'w':
                lines.append(f'{self.args}')
            if self.args['episode_count'] == 1 and self.agent.name == 'HRL':
                lines.append(f'{self.agent.model_settings} \n {items} {answers} \n')
            else:
                lines.append(f'{items} {answers} \n')
            lines.append(f'reward:{reward}, target:{self.learner_targets}\n')
            if self.agent.name == 'HRL':
                lines.append(f"goals: {self.args['episode_subgoals']} \n episode_reward: {reward} \n")
            else:
                lines.append(f'episode_reward: {reward} \n')

            result = ''
            for line in lines:
                result = result + line
            f.write(result)

        self.args['episode_count'] += 1
        self.episode_end_time = time.time()
        
        cumulative_time = self.episode_end_time - getattr(self, 'training_start_time', self.episode_start_time)
        print(f"Episode {self.args['episode_count']} Cumulative Time: {cumulative_time:.2f}s")

        if (self.train_buffer.size() * self.max_sequence_length > self.begin_train_length and
                not self.random_policy):
            for _ in range(self.epoch_num):
                b_s, b_a, b_r, b_ns, b_d = self.train_buffer.sample(self.batch_size)
                transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                self.update(transition_dict)



    def dkt_forward_fn(self, KTnet_states, batch_data_one_hot):
        dkt_loss = compute_dkt_loss(KTnet_states, batch_data_one_hot)
        return dkt_loss

    def update(self, transition_dict):

        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32).view(-1, 1)


        states = torch.tensor([], dtype=torch.float32)
        next_states = torch.tensor([], dtype=torch.float32)
        for i in range(len(transition_dict['states'])):
            states = mds_concat((states,
                                 get_feature_matrix(transition_dict['states'][i]['state'],
                                                    self.action_dim,
                                                    self.RNN_encoder_input_dim).unsqueeze(0)), 0)
            next_states = mds_concat((next_states,
                                      get_feature_matrix(transition_dict['next_states'][i]['next_state'],
                                                         self.action_dim,
                                                         self.RNN_encoder_input_dim).unsqueeze(0)), 0)
        if self.KTnet_trian:
            self.KTnet_optimizer.zero_grad()
            batch_data_states = torch.tensor([], dtype=torch.float32)
            for i in range(len(transition_dict['states'])):
                batch_data_states = mds_concat((batch_data_states,
                                                get_feature_matrix(transition_dict['states'][i]['state'],
                                                                   self.action_dim,
                                                                   self.DKT_input_dim).unsqueeze(0)), 0)
            KTnet_states = self.KTnet(batch_data_states)
            batch_data_one_hot = torch.tensor([], dtype=torch.float32)
            for i in range(len(transition_dict['states'])):
                batch_data_one_hot = mds_concat((batch_data_one_hot,
                                                 get_feature_matrix(transition_dict['states'][i]['state'],
                                                                    self.action_dim,
                                                                    self.DKT_input_dim
                                                                    ).unsqueeze(0)), 0)

            dkt_loss = self.dkt_forward_fn(KTnet_states, batch_data_one_hot)
            
            if dkt_loss.requires_grad:
                dkt_loss.backward()
                torch.nn.utils.clip_grad_value_(self.KTnet.parameters(), clip_value=self.args['grad_clip'])
                self.KTnet_optimizer.step()

        targets = [item['targets'] for item in transition_dict['states']]
        next_states_lengths_ids = torch.tensor([max(0, len(sequence['next_state']) - 1)
                                                    for sequence in transition_dict['next_states']])

        RL_next_states_dict = {
            'states': next_states,
            'states_lengths_ids': next_states_lengths_ids,
            'targets': targets
        }
        self.agent.learn(RL_next_states_dict, rewards)

    def n_step(self, max_steps: int):
        return [self.step() for _ in range(max_steps)]
