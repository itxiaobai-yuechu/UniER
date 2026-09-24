


import gym
import networkx as nx
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from KT import Agent_KT
from AC import ActorCritic, Data_P
from PPO_pre import PPO_pre, Data_L
from PPO_sim import PPO_sim
import EduSim
from EduSim.utils.agent_utils import get_raw_data_path
from EduSim.Envs.KES_assist09.Env import KESASSIST09Env
from EduSim.Envs.KES_assist12.Env import KESASSIST12Env
from EduSim.Envs.KES_assist17.Env import KESASSIST17Env
from EduSim.Envs.KES_algebra2005.Env import KESalgebra2005Env
from EduSim.Envs.KES_bridge2006.Env import KESbridge2006Env
from EduSim.Envs.KES_ednet.Env import KESednetEnv
from EduSim.Envs.KES_junyi.Env import KESjunyiEnv
from EduSim.Envs.KES_nips34.Env import KESnips34Env
from EduSim.Envs.KES_xes3g5m.Env import KESxes3g5mEnv
from EduSim.Envs.shared.KSS_KES import episode_reward
from EduSim.deep_model import KTnet

def get_kt_mastery(kt_net, concepts, answers, num_concepts, feature_dim, device):
    with torch.no_grad():
        kt_input = torch.zeros((1, len(concepts), feature_dim)).to(device)
        for i, (c, a) in enumerate(zip(concepts, answers)):
            kt_input[0, i, int(c) + (num_concepts if a == 1 else 0)] = 1
        if len(concepts) > 0:
            kt_output = kt_net(kt_input)[-1, 0, :]
            kt_output = torch.sigmoid(kt_output)
        else:
            kt_output = torch.zeros(num_concepts).to(device)
        return kt_output


def load_kt_net(dataset, num_concepts, device):
    feature_dim = 2 * num_concepts
    kt_net = KTnet({
        'input_size': feature_dim,
        'emb_dim': 128,
        'hidden_size': 256,
        'num_skills': num_concepts,
        'nlayers': 2,
        'dropout': 0.01,
    }).to(device)

    directory = f"../EduSim/Envs/KES_{dataset}/meta_data"
    dkt_file = f'{directory}/env_weights/ValBest.ckpt'
    kt_net.load_state_dict(torch.load(dkt_file, map_location=device))
    kt_net.eval()

    return kt_net, feature_dim


def train(env, pre_agent, sim_agent, dif_agent,
          max_episode_num, batch_size, L_max_steps,
          dataset, num_concepts, num_questions):
    
    L_rewards = []
    KT = Agent_KT(dataset, num_concepts, num_questions,
                  env.learning_item_base, env.concept_difficulty)

    kt_net, feature_dim = load_kt_net(dataset, num_concepts, device)

    th = 0.001
    start_time = time.time()

    for episode in tqdm.tqdm(range(max_episode_num), desc="Episode"):

        env.reset()
        init_profile, _ = env.begin_episode()

        Know_G = nx.DiGraph()
        Know_G.add_edges_from(list(env.knowledge_structure.edges))

        Know_G_sim = nx.Graph()
        Know_G_sim.add_edges_from(list(env.knowledge_structure_sim.edges))

        target = list(init_profile["target"])

        init_logs = init_profile["logs"]
        init_logs_q = init_profile["logs_q"]

        current_concepts = [int(x[0]) for x in init_logs]
        current_answers = [int(x[1]) for x in init_logs]
        current_questions = [int(x[0]) for x in init_logs_q]

        questions_difficulty_H = [env.learning_item_base.index[str(q)].difficulty * 50 for q in current_questions]
        concepts_difficulty_H = [env.concept_difficulty[str(c)] * 50 for c in current_concepts]

        P_state = KT.forward_state(current_questions, current_concepts,
                                   questions_difficulty_H, concepts_difficulty_H,
                                   current_answers)

        L_state = get_kt_mastery(
            kt_net, current_concepts, current_answers,
            num_concepts, feature_dim, device
        ).unsqueeze(0)

        pre_action = 1
        pre_state = L_state.clone().detach()
        isin = 0
        ifcontinue = 0

        last_tor = 5
        last_prac_num = 0
        know = 0
        l_steps = 0

        while True:
            l_steps += 1

            if pre_action == 1:
                know, tolerance, init_diff = pre_agent.take_action(
                    L_state, know, target, Know_G,
                    k_hop=1, threshhold=0.5,
                    last_tor=last_tor, last_prac_num=last_prac_num
                )
                t_know = know
            else:
                know, tolerance, init_diff = sim_agent.take_action(
                    L_state, know, know, Know_G_sim,
                    k_hop=1, threshhold=0.5,
                    last_tor=last_tor, last_prac_num=last_prac_num
                )

            p_steps = 0
            for i in range(int(tolerance)):
                p_steps += 1

                ques = dif_agent.take_action(know, P_state, init_diff)
                ques_id = int(ques)

                next_P_state, obs, p_reward, done, _, \
                current_questions, current_concepts, current_answers = \
                    env.step_p(current_questions, current_concepts, current_answers,
                               ques_id, is_commit=False)

                score = obs[2]

                if score == 1 or i == int(tolerance) - 1:
                    next_P_state, obs, p_reward, done, _, \
                    current_questions, current_concepts, current_answers = \
                        env.step_p(current_questions, current_concepts, current_answers,
                                   ques_id, is_commit=True)

                    P_state = next_P_state

                    if done and know in target:
                        target.remove(know)
                    break

                if i != 0:
                    dif_agent.store_transition(
                        Data_P(P_state.cpu().numpy(),
                               ques_id,
                               p_reward,
                               next_P_state.cpu().numpy(),
                               done)
                    )

            obs, l_reward, done, _ = env.step_l()

            next_L_state = get_kt_mastery(
                kt_net, current_concepts, current_answers,
                num_concepts, feature_dim, device
            ).unsqueeze(0)

            gain = (next_L_state[0][know] - pre_state[0][know]).item()

            if gain <= th:
                if pre_action == 1:
                    pre_action = 0
                else:
                    if isin == 1:
                        pre_action = 1
                        isin = 0
                        ifcontinue = 1
                    else:
                        isin = 1
            else:
                if pre_action == 0:
                    pre_action = 1
                    isin = 0

            if ifcontinue:
                ifcontinue = 0
                continue

            if pre_action == 1:
                pre_agent.store_transition(
                    Data_L(L_state.cpu().numpy(),
                           int(know),
                           l_reward,
                           next_L_state.cpu().numpy(),
                           done)
                )
            else:
                sim_agent.store_transition(
                    Data_L(L_state.cpu().numpy(),
                           int(know),
                           l_reward,
                           next_L_state.cpu().numpy(),
                           done)
                )

            L_state = next_L_state
            pre_state = L_state.clone().detach()

            last_tor = int(tolerance)
            last_prac_num = p_steps

            if dif_agent.memory_counter >= batch_size:
                dif_agent.learn()

            if pre_agent.memory_counter >= batch_size:
                pre_agent.learn()

            if sim_agent.memory_counter >= batch_size:
                sim_agent.learn()

            if l_steps >= L_max_steps or done or len(target) == 0:
                break

        observation, l_reward, done, info = env.end_episode()
        L_rewards.append(l_reward)

        print('Episode: {} | Episode Reward: {:.2f}'.format(episode, l_reward))
        
        window = 100
        if L_rewards:
            window_rewards = L_rewards[max(0, len(L_rewards) - window):]
            avg_reward = sum(window_rewards) / len(window_rewards)
        else:
            avg_reward = 0.0
        print(f'Average Reward (last {min(window, len(L_rewards))}): {avg_reward:.4f}')
        
        elapsed_time = time.time() - start_time
        print(f'Cumulative Time: {elapsed_time:.2f}s')


def test(env, pre_agent, sim_agent, dif_agent,
         dataset, num_concepts, num_questions,
         L_max_steps, test_episodes):

    learning_gains = []

    KT = Agent_KT(dataset, num_concepts, num_questions,
                  env.learning_item_base, env.concept_difficulty)

    kt_net, feature_dim = load_kt_net(dataset, num_concepts, device)

    th = 0.001

    for episode in tqdm.tqdm(range(test_episodes), desc="Test Episode"):

        env.reset()
        init_profile, _ = env.begin_episode()

        Know_G = nx.DiGraph()
        Know_G.add_edges_from(list(env.knowledge_structure.edges))
        Know_G_sim = nx.Graph()
        Know_G_sim.add_edges_from(list(env.knowledge_structure_sim.edges))

        target = list(init_profile["target"])
        initial_target_num = len(target)

        if initial_target_num == 0:
            continue

        init_logs = init_profile["logs"]
        init_logs_q = init_profile["logs_q"]

        current_concepts = [int(log[0]) for log in init_logs]
        current_answers = [int(log[1]) for log in init_logs]
        current_questions = [int(log[0]) for log in init_logs_q]

        init_kt = get_kt_mastery(
            kt_net, current_concepts, current_answers,
            num_concepts, feature_dim, device
        )
        init_mastery = sum([1 for t in target if init_kt[int(t)] > 0.5])

        questions_difficulty_H = [env.learning_item_base.index[str(q)].difficulty * 50 for q in current_questions]
        concepts_difficulty_H = [env.concept_difficulty[str(c)] * 50 for c in current_concepts]

        P_state = KT.forward_state(current_questions, current_concepts,
                                   questions_difficulty_H, concepts_difficulty_H,
                                   current_answers)

        L_state = init_kt.unsqueeze(0)

        pre_action = 1
        pre_state = L_state.clone().detach()
        isin = 0
        ifcontinue = 0

        last_tor = 5
        last_prac_num = 0
        know = 0
        l_steps = 0

        while True:
            l_steps += 1

            if pre_action == 1:
                know, tolerance, init_diff = pre_agent.take_action(
                    L_state, know, target, Know_G,
                    k_hop=1, threshhold=0.5,
                    last_tor=last_tor, last_prac_num=last_prac_num
                )
                t_know = know
            else:
                know, tolerance, init_diff = sim_agent.take_action(
                    L_state, know, know, Know_G_sim,
                    k_hop=1, threshhold=0.5,
                    last_tor=last_tor, last_prac_num=last_prac_num
                )

            p_steps = 0
            for i in range(int(tolerance)):
                p_steps += 1

                ques = dif_agent.take_action(know, P_state, init_diff)
                ques_id = int(ques) if not isinstance(ques, torch.Tensor) else int(ques.item())

                next_P_state, obs, _, done, _, \
                current_questions, current_concepts, current_answers = \
                    env.step_p(current_questions, current_concepts, current_answers,
                               ques_id, is_commit=False)

                score = obs[2]

                if score == 1 or i == int(tolerance) - 1:
                    next_P_state, obs, _, done, _, \
                    current_questions, current_concepts, current_answers = \
                        env.step_p(current_questions, current_concepts, current_answers,
                                   ques_id, is_commit=True)

                    P_state = next_P_state

                    if done and know in target:
                        target.remove(know)
                    break

            _, _, done, _ = env.step_l()

            next_L_state = get_kt_mastery(
                kt_net, current_concepts, current_answers,
                num_concepts, feature_dim, device
            ).unsqueeze(0)

            gain = (next_L_state[0][know] - pre_state[0][know]).item()

            if gain <= th:
                if pre_action == 1:
                    pre_action = 0
                else:
                    if isin == 1:
                        pre_action = 1
                        isin = 0
                        ifcontinue = 1
                    else:
                        isin = 1
            else:
                if pre_action == 0:
                    pre_action = 1
                    isin = 0

            if ifcontinue:
                ifcontinue = 0
                continue

            L_state = next_L_state
            pre_state = L_state.clone().detach()

            last_tor = int(tolerance)
            last_prac_num = p_steps

            if l_steps >= L_max_steps or done or len(target) == 0:
                break

        final_logs = env._learner.profile["logs"]
        final_concepts = [int(log[0]) for log in final_logs]
        final_answers = [int(log[1]) for log in final_logs]

        final_kt = get_kt_mastery(
            kt_net, final_concepts, final_answers,
            num_concepts, feature_dim, device
        )

        final_mastery = sum([
            1 for t in init_profile["target"]
            if final_kt[int(t)] > 0.5
        ])

        env.end_episode()

        reward = episode_reward(init_mastery, final_mastery, initial_target_num)
        learning_gains.append(reward)

        print(f"Episode {episode} Reward: {reward:.4f}")

    avg_gain = sum(learning_gains) / len(learning_gains)
    print(f"Average Learning Gain: {avg_gain:.4f}")

    return learning_gains

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    seed = int(os.environ.get('UNIER_SEED', '42'))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    dataset = os.environ.get('UNIER_PATH_DATASET', 'assist17')
    L_max_steps = int(os.environ.get('UNIER_MAX_STEPS', '10'))
    max_episode_num = int(os.environ.get('UNIER_EPISODES', '5000'))
    target_type = os.environ.get('UNIER_TARGET_TYPE', 'all')

    if dataset == 'assist09':
        num_concepts = 123
        num_questions = 17737
        dataRecPath = f'{get_raw_data_path()}/dataProcess/assist09/dataRec'
        dataRecQPath = f'{get_raw_data_path()}/dataProcess/assist09/dataRec_q'
        env = KESASSIST09Env(dataRec_path=dataRecPath, dataRec_q_path=dataRecQPath, seed=42)
    elif dataset == 'assist12':
        num_concepts = 265
        num_questions = 53070
        dataRecPath = f'{get_raw_data_path()}/dataProcess/assist12/dataRec'
        dataRecQPath = f'{get_raw_data_path()}/dataProcess/assist12/dataRec_q'
        env = KESASSIST12Env(dataRec_path=dataRecPath, dataRec_q_path=dataRecQPath, seed=42)
    elif dataset == 'assist17':
        num_concepts = 102
        num_questions = 3162
        dataRecPath = f'{get_raw_data_path()}/dataProcess/assist17/dataRec'
        dataRecQPath = f'{get_raw_data_path()}/dataProcess/assist17/dataRec_q'
        env = KESASSIST17Env(dataRec_path=dataRecPath, dataRec_q_path=dataRecQPath, seed=42)
    elif dataset == 'algebra2005':
        num_concepts = 112
        num_questions = 173113
        dataRecPath = f'{get_raw_data_path()}/dataProcess/algebra2005/dataRec'
        dataRecQPath = f'{get_raw_data_path()}/dataProcess/algebra2005/dataRec_q'
        env = KESalgebra2005Env(dataRec_path=dataRecPath, dataRec_q_path=dataRecQPath, seed=42)
    elif dataset == 'bridge2006':
        num_concepts = 493
        num_questions = 129263
        dataRecPath = f'{get_raw_data_path()}/dataProcess/bridge2006/dataRec'
        dataRecQPath = f'{get_raw_data_path()}/dataProcess/bridge2006/dataRec_q'
        env = KESbridge2006Env(dataRec_path=dataRecPath, dataRec_q_path=dataRecQPath, seed=42)
    elif dataset == 'ednet':
        num_concepts = 188
        num_questions = 11901
        dataRecPath = f'{get_raw_data_path()}/dataProcess/ednet/dataRec'
        dataRecQPath = f'{get_raw_data_path()}/dataProcess/ednet/dataRec_q'
        env = KESednetEnv(dataRec_path=dataRecPath, dataRec_q_path=dataRecQPath, seed=42)
    elif dataset == 'junyi':
        num_concepts = 39
        num_questions = 721
        dataRecPath = f'{get_raw_data_path()}/dataProcess/junyi/dataRec'
        dataRecQPath = f'{get_raw_data_path()}/dataProcess/junyi/dataRec_q'
        env = KESjunyiEnv(dataRec_path=dataRecPath, dataRec_q_path=dataRecQPath, seed=42)
    elif dataset == 'nips34':
        num_concepts = 57
        num_questions = 948
        dataRecPath = f'{get_raw_data_path()}/dataProcess/nips34/dataRec'
        dataRecQPath = f'{get_raw_data_path()}/dataProcess/nips34/dataRec_q'
        env = KESnips34Env(dataRec_path=dataRecPath, dataRec_q_path=dataRecQPath, seed=42)
    elif dataset == 'xes3g5m':
        num_concepts = 865
        num_questions = 7652
        dataRecPath = f'{get_raw_data_path()}/dataProcess/xes3g5m/dataRec'
        dataRecQPath = f'{get_raw_data_path()}/dataProcess/xes3g5m/dataRec_q'
        env = KESxes3g5mEnv(dataRec_path=dataRecPath, dataRec_q_path=dataRecQPath, seed=42)
    env.target_type = target_type

    state_dim = num_concepts
    action_dim = num_concepts
    hidden_dim = 128
    actor_lr = 0.001
    critic_lr = 0.001
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    batch_size = 16
    requested_device = os.environ.get('UNIER_DEVICE', 'cuda:0')
    if requested_device.startswith('cuda') and not torch.cuda.is_available():
        requested_device = 'cpu'
    device = torch.device(requested_device)

    pre_agent = PPO_pre(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device,batch_size)
    sim_agent = PPO_sim(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device,batch_size)
    dif_agent = ActorCritic(num_questions, hidden_dim, num_questions, actor_lr,
                        critic_lr, gamma, env.learning_item_base.knowledge2item, device, batch_size)

    train(env, pre_agent, sim_agent, dif_agent,max_episode_num,
          batch_size=16,
          L_max_steps=10,
          dataset=dataset,
          num_concepts=num_concepts,
          num_questions=num_questions)
    
    test(env, pre_agent, sim_agent, dif_agent,
     dataset, num_concepts, num_questions,
     L_max_steps, 500)
