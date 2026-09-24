import random


class ReplayBuffer:
    def __init__(self, capacity, episode_length=20):
        self.capacity = capacity
        self.buffer = {
            'env': {},
        }
        self.traj_num = {
            'env': 0,
        }
        self.delete_traj_id = {
            'env': 0,
        }
        self.next_idx = 0
        self.episode_length = episode_length
        self.roll_steps = 3
        self.policy_mode = 'on_policy'

    def add(self, state, action, reward, next_state, done, index):
        if index == 0:
            self.buffer['env'][self.traj_num['env']] = []
            self.traj_num['env'] += 1
        self.buffer['env'][self.traj_num['env'] - 1].append((state, action, reward, next_state, done))

    def sample(self, batch_size, multi_step=False, n_multi_step=0, gamma=0.99):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        pool_type = 'env'

        if self.policy_mode == 'off_policy':
            for _ in range(batch_size):
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
            del self.buffer[pool_type][self.delete_traj_id[pool_type]]
            self.delete_traj_id[pool_type] += 1
        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer['env'])

    def clear(self):
        self.buffer = {
            'env': {},
        }
        self.traj_num = {
            'env': 0,
        }
        self.delete_traj_id = {
            'env': 0,
        }
