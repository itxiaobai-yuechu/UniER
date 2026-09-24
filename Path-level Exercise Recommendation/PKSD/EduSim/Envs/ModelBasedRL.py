import numpy as np
import torch
from EduSim.Envs.deep_model import *
from EduSim.Envs.agent_utils import get_feature_matrix, batch_cat_targets
import copy
from EduSim.Envs.agent_utils import get_proj_path


def reward_func(score, next_score, targets):
    reward = 0
    for i in targets:
        if score[i] == 0 and next_score[i] == 1:
            reward += 1
    return reward


class ModelEnv(object):
    def __init__(self, model_based_mode, cudaDevice):
        self.model_based_mode = model_based_mode
        self.num_skills = 10
        self.DKT_input_dim = 2 * self.num_skills
        self.RNN_encoder_input_dim = 3 * self.num_skills
        self.max_sequence_length = 20

        self.use_gpu = True
        self.device = cudaDevice
        self.loss_f = nn.BCEWithLogitsLoss()
        self.DKT_model_path = f'{get_proj_path()}/EduSim/Envs/KSS/' \
                              'meta_data/DKT_USE_DATA/all_data_trained_DKT_model.pth'
        self.DKTModel = DKTnet(input_size=self.DKT_input_dim,
                               emb_dim=15,
                               hidden_size=20,
                               num_skills=self.num_skills,
                               nlayers=2).to(self.device)
        self.DKTModel.load_state_dict(torch.load(self.DKT_model_path, map_location=self.device)['model'])
        self.optimizer = torch.optim.Adam(self.DKTModel.parameters(), lr=0.0001)

    def train_model(self, state, action, next_state):
        self.optimizer.zero_grad()
        input_data = get_feature_matrix(state['state'], state['targets'], self.num_skills, self.RNN_encoder_input_dim,
                                        max_sequence_length=self.max_sequence_length)
        input_data = input_data.unsqueeze(0).to(self.device)
        DKT_input_data = input_data[:, :, :-self.num_skills]
        sequence_length = len(state['state'])
        learner_state = self.DKTModel(DKT_input_data)[max(sequence_length - 1, 0), 0, :]
        answer_label = torch.tensor(next_state['next_state'][-1][1], dtype=torch.float).to(self.device)
        loss = self.loss_f(learner_state.gather(0, int(action)), answer_label)
        loss.backward()
        self.optimizer.step()

    def get_roll_out(self, buffer, traj_num, state, action_net, roll_steps):
        learner_state = []
        buffer['fake'][traj_num['fake']] = []
        traj_num['fake'] += 1

        for i in range(roll_steps):
            input_data = get_feature_matrix(state['state'], state['targets'], self.num_skills, self.RNN_encoder_input_dim,
                                            max_sequence_length=self.max_sequence_length + roll_steps)
            input_data = input_data.unsqueeze(0).to(self.device)
            DKT_input_data = input_data[:, :, :-self.num_skills]
            with torch.no_grad():
                sequence_length = len(state['state'])
                learner_state = torch.sigmoid(self.DKTModel(DKT_input_data)[max(sequence_length - 1, 0), 0, :])
                before_learn_scores = [1 if ele > 0.5 else 0 for l, ele in enumerate(learner_state)]

                DKToutput = action_net[0](input_data)
                if action_net[0].name == 'DKT':
                    RL_states = batch_cat_targets(DKToutput[max(0, len(state['state']) - 1), :, :], [state['targets']],
                                                  device=self.device)
                    agent_step = action_net[1].step(RL_states)
                else:
                    agent_step = action_net[1].step(DKToutput[-1, :, :])
                idx = agent_step
                item = str(int(idx))

                item_score = before_learn_scores[int(item)]
                new_logs = copy.deepcopy(state['state'])
                new_logs.append([item, item_score])

                sequence_length += 1
                input_data = get_feature_matrix(new_logs, state['targets'], self.num_skills, self.RNN_encoder_input_dim,
                                                max_sequence_length=self.max_sequence_length + roll_steps)
                input_data = input_data.unsqueeze(0).to(self.device)
                DKT_input_data = input_data[:, :, :-self.num_skills]
                learner_state = torch.sigmoid(self.DKTModel(DKT_input_data)[max(sequence_length - 1, 0), 0, :])
                after_learn_scores = [1 if ele > 0.5 else 0 for l, ele in enumerate(learner_state)]

            rollout_state = copy.deepcopy(state)
            rollout_action = item
            rollout_reward = reward_func(before_learn_scores, after_learn_scores, state['targets'])
            rollout_next_state = {'next_state': new_logs, 'targets': state['targets']}
            rollout_done = True
            for j in state['targets']:
                if after_learn_scores[j] == 0:
                    rollout_done = False
            buffer['fake'][traj_num['fake'] - 1].append((rollout_state, rollout_action, rollout_reward,
                                                         rollout_next_state, rollout_done))

            state = {'state': new_logs, 'targets': state['targets']}

        ones = torch.ones(learner_state.size()).to(learner_state.device)
        zeros = torch.zeros(learner_state.size()).to(learner_state.device)
        final_scores = torch.where(learner_state > 0.5, ones, zeros)
        targets = np.where(final_scores.cpu().numpy() > 0)[0]

        return list(targets)
