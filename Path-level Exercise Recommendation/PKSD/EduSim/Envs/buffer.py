import numpy as np
import random
import collections
import time
from EduSim.Envs.ModelBasedRL import *
import copy
from EduSim.Envs.agent_utils import *


class SumTree(object):

    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, p)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):

        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]


class Memory(object):

    epsilon = 0.01
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.
    count = 0

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def add(self, state, action, reward, next_state, done, index):
        transition = [state, action, reward, next_state, done]
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)
        self.count += 1

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n,), dtype=object), np.empty(
            (n, 1))
        pri_seg = self.tree.total_p / n
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        want_id = -self.tree.capacity + self.count
        if want_id >= 0:
            want_id = None
            self.count = self.tree.capacity
        min_prob = np.min(
            self.tree.tree[-self.tree.capacity:want_id]) / self.tree.total_p

        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], b_memory[i] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def size(self):
        return self.count


class ReplayBuffer(object):
    def __init__(self, capacity, action_net, episode_length=20, model_based_mode=False, cudaDevice=''):
        self.policy_mode = action_net[1].policy_mode
        self.agent_name = action_net[1].name
        self.capacity = capacity
        self.buffer = {
            'env': {},
            'her': {},
            'fake': {}
        }
        self.traj_num = {
            'env': 0,
            'her': 0,
            'fake': 0
        }
        self.delete_traj_id = {
            'env': 0,
            'her': 0,
            'fake': 0
        }

        self.next_idx = 0
        self.episode_length = episode_length
        self.action_net = action_net
        self.roll_steps = 3
        self.modelEnv = ModelEnv(model_based_mode=model_based_mode, cudaDevice=cudaDevice)
        self.model_based_mode = model_based_mode

    def add(self, state, action, reward, next_state, done, index, episdoes_correct_answeres, episode_items, episode_answers):
        if self.model_based_mode and len(state['state']) < self.episode_length - self.roll_steps:
            behav_state = {'state': copy.deepcopy(state['state']), 'targets': copy.deepcopy(state['targets'])}
            _ = self.modelEnv.get_roll_out(self.buffer, self.traj_num,
                                           behav_state, self.action_net, roll_steps=self.roll_steps)
        episdoes_correct_answeres = [int(el) for el in episdoes_correct_answeres]
        episode_items = [int(el) for el in episode_items]
        episode_answers = [int(el) for el in episode_answers]
        if index == 0:
            self.buffer['env'][self.traj_num['env']] = []
            self.traj_num['env'] += 1
        self.buffer['env'][self.traj_num['env'] - 1].append((state, action, reward, next_state, done))

    def sample(self, batch_size, multi_step=False, n_multi_step=0, gamma=0.99, model_based_sample=False):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        pool_type = self.random_choose_pool_type()
        while len(self.buffer[pool_type]) == 0:
            pool_type = self.random_choose_pool_type()

        if self.model_based_mode:
            if model_based_sample:
                pool_type = 'fake'
            else:
                pool_type = 'env'

        if self.policy_mode == 'off_policy':
            for i in range(batch_size):
                if multi_step:
                    traj_id = random.randint(self.delete_traj_id[pool_type], self.delete_traj_id[pool_type] +
                                             len(self.buffer[pool_type]) - 1)
                    while len(self.buffer[pool_type][traj_id]) < n_multi_step:
                        traj_id = random.randint(self.delete_traj_id[pool_type], self.delete_traj_id[pool_type] +
                                                 len(self.buffer[pool_type]) - 1)
                    finish = random.randint(n_multi_step, len(self.buffer[pool_type][traj_id]))
                    begin = finish - n_multi_step
                    sum_reward = 0
                    data = self.buffer[pool_type][traj_id][begin: finish]
                    state = data[0][0]
                    action = data[0][1]
                    for j in range(n_multi_step):
                        sum_reward += (gamma ** j) * data[j][2]
                        states_look_ahead = data[j][3]
                        if data[j][4]:
                            done = True
                            break
                        else:
                            done = False
                    states.append(state)
                    actions.append(action)
                    rewards.append(sum_reward)
                    next_states.append(states_look_ahead)
                    dones.append(done)
                else:
                    traj_id = random.randint(self.delete_traj_id[pool_type], self.delete_traj_id[pool_type] +
                                             len(self.buffer[pool_type]) - 1)
                    transition = random.choice(self.buffer[pool_type][traj_id])
                    states.append(transition[0])
                    actions.append(transition[1])
                    rewards.append(transition[2])
                    next_states.append(transition[3])
                    dones.append(transition[4])
        else:
            traj_id = random.randint(self.delete_traj_id[pool_type], self.delete_traj_id[pool_type] +
                                     len(self.buffer[pool_type]) - 1)
            transitions = self.buffer[pool_type][traj_id]
            for transition in transitions:
                states.append(transition[0])
                actions.append(transition[1])
                rewards.append(transition[2])
                next_states.append(transition[3])
                dones.append(transition[4])

        states = tuple(state for state in states)
        actions = tuple(action for action in actions)
        rewards = tuple(reward for reward in rewards)
        next_states = tuple(next_state for next_state in next_states)
        dones = tuple(done for done in dones)

        while self.size() >= self.capacity:
            pool_type = 'env'
            if len(self.buffer['her']) > len(self.buffer['env']):
                pool_type = 'her'
            elif len(self.buffer['fake']) // self.episode_length > len(self.buffer['env']):
                pool_type = 'fake'

            del self.buffer[pool_type][self.delete_traj_id[pool_type]]
            self.delete_traj_id[pool_type] += 1
        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer['env']) + len(self.buffer['her']) + len(self.buffer['fake'])

    def random_choose_pool_type(self):
        if self.model_based_mode:
            pool_type = random.choice(['env', 'fake'])
        else:
            pool_type = 'env'

        return pool_type

    def clear(self):
        self.buffer = {
            'env': {},
            'fake': {}
        }
        self.traj_num = {
            'env': 0,
            'fake': 0
        }
        self.delete_traj_id = {
            'env': 0,
            'fake': 0
        }
