import gym
import networkx as nx
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from KT import Agent_KT
from AC import ActorCritic,Data_P
from PPO import PPO,Data_L
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
    embed_dim = 128
    hidden_size = 256
    dkt_para_dict = {
        'input_size': feature_dim,
        'emb_dim': embed_dim,
        'hidden_size': hidden_size,
        'num_skills': num_concepts,
        'nlayers': 2,
        'dropout': 0.01,
    }
    kt_net = KTnet(dkt_para_dict).to(device)
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                f"../EduSim/Envs/KES_{dataset}/meta_data")
    dkt_file = f'{directory}/env_weights/ValBest.ckpt'
    if os.path.exists(dkt_file):
        param_dict = torch.load(dkt_file, map_location=device)
        kt_net.load_state_dict(param_dict)
        kt_net.eval()
    else:
        raise ValueError('dkt net not trained yet!')
    return kt_net, feature_dim

def train(env, L_agent, P_agent, max_episode_num, batch_size, L_max_steps, dataset, num_concepts, num_questions):
    L_rewards = []
    all_logs_episode_L = []
    all_logs_episode_P = []
    KT = Agent_KT(dataset, num_concepts, num_questions, env.learning_item_base, env.concept_difficulty)
    kt_net, feature_dim = load_kt_net(dataset, num_concepts, device)

    import time
    start_time = time.time()
    for episode in tqdm.tqdm(range(max_episode_num), desc="Episode"):
        env.reset()
        logs_episode_L = []
        logs_episode_P = []

        init_profile, _ = env.begin_episode()
        Know_G = nx.DiGraph()
        Know_G.add_edges_from(list(env.knowledge_structure.edges))
        target = list(init_profile["target"])

        logs_episode_L.append({"learning targets": target})
        init_logs = init_profile["logs"]
        init_logs_q = init_profile["logs_q"]

        init_concepts = []
        init_questions = []
        init_answers = []
        for log in init_logs:
            init_concepts.append(int(log[0]))
            init_answers.append(int(log[1]))
        for log in init_logs_q:
            init_questions.append(int(log[0]))

        current_questions = list(init_questions)
        current_concepts = list(init_concepts)
        current_answers = list(init_answers)


        last_tor = 5
        last_prac_num = 0

        logs_episode_L.append({"length of initial logs": len(init_concepts)})
        questions_difficulty_H = [env.learning_item_base.index[str(qid)].difficulty * 50 for qid in init_questions]
        concepts_difficulty_H = [env.concept_difficulty[str(c)] * 50 for c in init_concepts]

        
        P_state = KT.forward_state(init_questions, init_concepts, questions_difficulty_H, concepts_difficulty_H, init_answers)
        next_P_state = P_state
        L_state = get_kt_mastery(kt_net, init_concepts, init_answers, num_concepts, feature_dim, device).unsqueeze(0)
        logs_episode_L.append({"init knowledge state": L_state})

        l_steps = 0
        l_knows = []
        know = 0

        while True:
            l_steps += 1
            know, tolerance, init_diff = L_agent.take_action(
                L_state, know, target, Know_G,
                k_hop=1, threshhold=0.5,
                last_tor=last_tor, last_prac_num=last_prac_num
            )

            l_knows.append(know)
            p_steps = 0
            for i in range(int(tolerance)):
                p_steps += 1
                ques = P_agent.take_action(know, P_state, init_diff)
                ques_id = int(ques.item()) if isinstance(ques, torch.Tensor) else int(ques)

                dimkt_prob = P_state[0, ques_id].item()

                init_diff = None

                next_P_state, observation, p_reward, done, _,current_questions, current_concepts,current_answers = env.step_p(current_questions, current_concepts, current_answers, ques_id, is_commit=False)
                
                score = observation[2]
                
                if score == 1  or i == int(tolerance) - 1:
                    next_P_state, observation, p_reward, done, _,current_questions, current_concepts,current_answers = env.step_p(
                        current_questions, current_concepts, current_answers, ques_id, is_commit=True)
                    
                    P_state = next_P_state
                    
                    if done:
                        if target and know in target:
                            target.remove(know)
                    break
                
                if i != 0:
                    P_agent.store_transition(
                        Data_P(
                            P_state.detach().cpu().numpy(),
                            int(ques),
                            p_reward,   
                            next_P_state.detach().cpu().numpy(),
                            done
                        )
                    )


            logs_episode_P.append("tolerance is {}, practice {} times".format(tolerance, p_steps))


            if P_agent.memory_counter >= batch_size:
                P_agent.learn()

            observation, l_reward, done, info = env.step_l()
            
            final_logs_t = env._learner.profile["logs"]
            current_concepts_t = [int(log[0]) for log in final_logs_t]
            current_answers_t = [int(log[1]) for log in final_logs_t]

            next_L_state = get_kt_mastery(kt_net, current_concepts_t, current_answers_t, num_concepts, feature_dim, device).unsqueeze(0)

            if l_steps >= L_max_steps:
                print("Learning Max steps!"+"{}".format(l_steps))
                logs_episode_L.append("Learning Max steps!"+"{}".format(l_steps))
                break

            if done or len(target)==0:
                print("Learning success!"+"{}".format(l_steps))
                break

            L_agent.store_transition(Data_L(L_state.detach().cpu().numpy(), int(know), l_reward, next_L_state.detach().cpu().numpy(), done))

            L_state = next_L_state
            last_tor = int(tolerance)
            last_prac_num = p_steps



        if L_agent.memory_counter >= batch_size:
            L_agent.learn()

        observation, l_reward, done, info = env.end_episode()
        L_rewards.append(l_reward)

        elapsed_time = time.time() - start_time
        print('Episode: {} | Episode Reward: {:.2f} | Accumulated Time: {:.2f}s'.format(episode, l_reward, elapsed_time))
        
        window = 100
        if L_rewards:
            window_rewards = L_rewards[max(0, len(L_rewards) - window):]
            avg_reward = sum(window_rewards) / len(window_rewards)
        else:
            avg_reward = 0.0
        print(f'Average Reward (last {min(window, len(L_rewards))}): {avg_reward:.4f}')

        logs_episode_L.append("learned knowledges: {}".format(l_knows))
        logs_episode_L.append(current_answers)

        all_logs_episode_L.append(logs_episode_L)


    print("all_logs_episode:",all_logs_episode_L)
    with open("all_logs_episode.txt", "w", encoding="utf-8") as f:
        for log in all_logs_episode_L:
            f.write(str(log) + "\n\n")
    print("all_logs_episode saved!")




def test(env, L_agent, P_agent, dataset, num_concepts, num_questions, L_max_steps, test_episodes):
    learning_gains = []
    KT = Agent_KT(dataset, num_concepts, num_questions, env.learning_item_base, env.concept_difficulty)
    kt_net, feature_dim = load_kt_net(dataset, num_concepts, device)
    
    for episode in tqdm.tqdm(range(test_episodes), desc="Test Episode"):
        env.reset()
        init_profile, _ = env.begin_episode()
        Know_G = nx.DiGraph()
        Know_G.add_edges_from(list(env.knowledge_structure.edges))
        target = list(init_profile["target"])
        initial_target_num = len(target)

        if initial_target_num == 0:
            continue

        init_logs = init_profile["logs"]
        init_concepts = [int(log[0]) for log in init_logs]
        init_answers = [int(log[1]) for log in init_logs]
        init_logs_q = init_profile["logs_q"]
        init_questions = [int(log[0]) for log in init_logs_q]

        init_kt_output = get_kt_mastery(kt_net, init_concepts, init_answers, num_concepts, feature_dim, device)
        init_mastery = sum([1 for t in target if init_kt_output[int(t)] > 0.5])

        last_tor = 5
        last_prac_num = 0

        questions_difficulty_H = [env.learning_item_base.index[str(qid)].difficulty * 50 for qid in init_questions]
        concepts_difficulty_H = [env.concept_difficulty[str(c)] * 50 for c in init_concepts]

        P_state = KT.forward_state(init_questions, init_concepts, questions_difficulty_H, concepts_difficulty_H, init_answers)
        L_state = get_kt_mastery(kt_net, init_concepts, init_answers, num_concepts, feature_dim, device).unsqueeze(0)

        l_steps = 0
        know = 0
        current_questions = list(init_questions)
        current_answers = list(init_answers)
        current_concepts = list(init_concepts)

        while True:
            l_steps += 1
            know, tolerance, init_diff = L_agent.take_action(
                L_state, know, target, Know_G,
                k_hop=1, threshhold=0.5,
                last_tor=last_tor, last_prac_num=last_prac_num
            )
            
            p_steps = 0
            for i in range(int(tolerance)):
                p_steps += 1
                ques = P_agent.take_action(know, P_state, init_diff)
                ques_id = int(ques.item()) if isinstance(ques, torch.Tensor) else int(ques)

                dimkt_prob = P_state[0, ques_id].item()

                init_diff = None

                next_P_state, observation, _, done, _,current_questions,current_concepts,current_answers = env.step_p(
                    current_questions, current_concepts, current_answers, ques_id, is_commit=False)
                
                score = observation[2]
                
                if score == 1 or i == int(tolerance) - 1:
                    next_P_state, observation, _, done, _,current_questions,current_concepts,current_answers = env.step_p(
                        current_questions, current_concepts, current_answers, ques_id, is_commit=True)
                    
                    P_state = next_P_state
                    
                    if done:
                        if target and know in target:
                            target.remove(know)
                    break

            _, _, done, _ = env.step_l()
            
            next_L_state = get_kt_mastery(kt_net, current_concepts, current_answers, num_concepts, feature_dim, device).unsqueeze(0)

            if l_steps >= L_max_steps or done or len(target) == 0:
                break

            L_state = next_L_state
            last_tor = int(tolerance)
            last_prac_num = p_steps

        final_logs = env._learner.profile["logs"]
        final_concepts = [int(log[0]) for log in final_logs]
        final_answers = [int(log[1]) for log in final_logs]

        final_kt_output = get_kt_mastery(kt_net, final_concepts, final_answers, num_concepts, feature_dim, device)
        final_mastery = sum([1 for t in init_profile["target"] if final_kt_output[int(t)] > 0.5])

        env.end_episode()
        
        reward = episode_reward(init_mastery, final_mastery, initial_target_num)
        learning_gains.append(reward)
        print(f"Episode {episode} Reward: {reward:.4f}")

    if learning_gains:
        avg_gain = sum(learning_gains) / len(learning_gains)
        print(f"500 Students Average Learning Reward: {avg_gain:.4f}")
    else:
        print("No valid episodes for calculating learning reward.")
    return learning_gains


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dataset = 'assist17'
    L_max_steps = 10
    max_episode_num = 5000
    target_type = 'all'

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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    L_agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device,batch_size)
    P_agent = ActorCritic(num_questions, hidden_dim, num_questions, actor_lr,
                        critic_lr, gamma, env.learning_item_base.knowledge2item, device, batch_size)
    train(env, L_agent, P_agent, max_episode_num, batch_size, L_max_steps, dataset, num_concepts, num_questions)
    
    test(env, L_agent, P_agent, dataset, num_concepts, num_questions, L_max_steps, 500)
