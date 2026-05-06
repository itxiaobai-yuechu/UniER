import torch
import numpy as np
import torch.nn.functional as F
import EQN_GPU as EQN
from EQN_GPU import  ReplayBuffer
import utils
import csv
import time
import math
import random
import utils
from tqdm import tqdm 

def train(net, train_file_path, batch_size, pattern, istrain, dataset, n_actions):
    print("\ntrain----------------\n")
    net.train()
    nin =  n_states
    nout = n_actions
    limit = math.sqrt(6 / (nin + nout))
    episode_return = 0 
    concept_batch_set = set()
    return_list = []    
    
    with open(train_file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        train_data = [row for row in reader]

    for x in tqdm(range(0, len(train_data), 4), desc="Training Progress", unit="batch"): 
        time_start = time.time()
        if len(train_data[x : x+4][0]) < min_interaction: 
            continue 

        a_student = train_data[x : x+4]
        concept_batch_set.clear()     
        concept_batch = a_student[2]   
        padded_concept_list = []
        performance_list    = []
        exercise_list       = []
        reward_list         = []
        state =  torch.empty(120).uniform_(-limit, limit) 
        action = random.randint(0, n_actions - 1)               
        all_reward = 0                                  

        for index, one_ex_concept in tqdm(enumerate(concept_batch), 
                                           desc=f"Processing student {x//4 + 1}", 
                                           total=len(concept_batch), 
                                           leave=False): 
            split_items = one_ex_concept.split('_')
            for split_item in split_items:          
                concept_batch_set.add(float(split_item))

            concept = torch.tensor(list(map(float, split_items)))  
            concept_set = set(map(float, split_items))  
            padding_length = 4 - concept.size(0)
            padded_concept = F.pad(concept, (0, padding_length), "constant", 0) 
            padded_concept_list.append(padded_concept)
            performance = int(a_student[3][index])  
            performance_list.append(performance)
            exercise = a_student[1][index]        
            exercise_list.append(exercise)
            
            if pattern == "M" or pattern == "m":
                st, max_q_index, _ = net(padded_concept, 
                                         performance, exercise, A, P, 
                                         exercise_module_dim, d_dict, istrain, pattern, dataset)
            elif pattern == "R" or pattern == "r":
                st, max_q_index, _ = net(padded_concept_list, 
                                         performance_list, exercise_list, 
                                         A, P, exercise_module_dim, d_dict, istrain, pattern, dataset) 
                
            Recommended_exercise = P[max_q_index]
            next_state = st.view(-1)                           
            next_action = net.take_action(Recommended_exercise) 
            dt = d_dict.get(exercise, [])[0]               
            dt_rec = d_dict.get(next_action, [])[0]             
            concept_next_set = set(map(float, A[P.index(next_action)]))  
            performance_history = list(map(float, a_student[3][:index]))         
            next_all_reward = reward.merge_reward(performance, concept_set, 
                                                  concept_next_set, dt, dt_rec, 
                                                  performance_history, concept_batch_set)
            reward_list.append(next_all_reward)               
            replay_buffer.add(state, int(action), all_reward, next_state)
            state = next_state  
            action = next_action
            all_reward = next_all_reward

            if replay_buffer.size() >= batch_size:  
                s, a, r, ns = replay_buffer.sample(batch_size)
                transition_dict = {
                    'states': s,
                    'actions': a,
                    'next_states': ns,
                    'rewards': r,
                }  
                net.update(transition_dict, dataset)

        time_end = time.time()
        print("using time:", time_end - time_start, "sec")
        episode_return = sum(reward_list) / len(a_student[0])  
        return_list.append(episode_return)








def test(net, train_file_path, test_file_path, pattern, k):
    print("\ntest-----------------\n")
    net.eval()
    with open(test_file_path, mode='r', newline='', encoding='utf-8') as file1:
        reader = csv.reader(file1)
        test_data = [row for row in reader]
    with open(train_file_path, mode='r', newline='', encoding='utf-8') as file2:
        reader = csv.reader(file2)
        train_data = [row for row in reader]

    NDCG, F1, Recall = [], [], []
    istrain = False
    
    problem_out_path = f'data/{dataset}/recommended_problems.txt'
    concept_out_path = f'data/{dataset}/recommended_concepts.txt'
    open(problem_out_path, 'w', encoding='utf-8').close()
    open(concept_out_path, 'w', encoding='utf-8').close()

    for x in range(0, len(train_data), 4): 
        if len(train_data[x : x+4][0]) < min_interaction: 
            continue 
        a_student     = train_data[x : x+4]
        student_id    = a_student[0][0]
        exercise      = a_student[1][-1]        
        performance   = int(a_student[3][-1])  
        exercise_list    = a_student[1]
        performance_list = list(map(int, a_student[3])) 
        concept_list     = a_student[2]       
        padded_concept_list = []           

        for a_concept in concept_list:
            split_items = a_concept.split('_')
            concept = torch.tensor(list(map(float, split_items)))  
            padding_length = 4 - concept.size(0)
            padded_concept = F.pad(concept, (0, padding_length), "constant", 0) 
            padded_concept_list.append(padded_concept)
        end_padded_concept = padded_concept_list[-1]

        if pattern == "M" or pattern == "m":
            _, _,top_twenty_indices = net(end_padded_concept, 
                                          performance, exercise, A,P, 
                                          exercise_module_dim, d_dict, istrain, pattern,dataset)
        elif pattern == "R" or pattern == "r":
            _, _,top_twenty_indices = net(padded_concept_list, 
                                          performance_list, exercise_list, A, P, 
                                          exercise_module_dim, d_dict, istrain, pattern,dataset)
        
        Recommended_list = [A[i] for i in top_twenty_indices]
        Recommended_problem_ids = [P[i] for i in top_twenty_indices]

        with open(problem_out_path, 'a', encoding='utf-8') as pf, \
            open(concept_out_path, 'a', encoding='utf-8') as cf:
            pf.write(f"{student_id}\t{','.join(map(str, Recommended_problem_ids))}\n")
            concept_lines = ['_'.join(map(str, concept)) for concept in Recommended_list]
            cf.write(f"{student_id}\t{','.join(concept_lines)}\n")

        for i in range(0, len(test_data), 4):      
            a_test_student = test_data[i : i+4]
            real_per_concept = []
            if a_test_student[0][0] == student_id:
                real_per = a_test_student[2]              
                real_per_spl = [list(map(int, item.split('_'))) for item in real_per] 
                for c_index, c in enumerate(real_per_spl):
                    if a_test_student[3][c_index] == "1":
                        real_per_concept.append(c)

                Recommended_list = [list(map(int, sublist)) for sublist in Recommended_list]
                actual_scores = [1 if exercise in real_per_concept else 0 for exercise in Recommended_list]
                ndcg = utils.ndcg_at_k(actual_scores, k)
                recall = utils.recall_at_k(actual_scores, k)
                precision = utils.precision_at_k(actual_scores, k)
                F1.append(utils.f1_at_k(actual_scores, k))
                Recall.append(recall)
                NDCG.append(ndcg)
    return NDCG, F1, Recall

if __name__ == '__main__':
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    istrain=True
    dataset='assist2017'
    print(f"dataset:{dataset}")
    train_file_path = f'data/{dataset}/train_valid_processed.csv'
    test_file_path  = f'data/{dataset}/test_processed.csv'
    A_file_path     = f'data/{dataset}/question_skill_mapping.csv'
    d_file_path     = f'data/{dataset}/difficulty.csv'
    d_dict = EQN.load_file_data(d_file_path) 
    if dataset == 'assist2009':
        n_actions = 17737
    elif dataset == 'nips34':
        n_actions = 948
    elif dataset == 'assist2012':
        n_actions = 53070
    elif dataset == 'assist2017':
        n_actions = 3162
    elif dataset == 'algebra2005':
        n_actions = 173113
    elif dataset == 'bridge2006':
        n_actions = 129263
    elif dataset == 'ednet':
        n_actions = 11901
    elif dataset == 'original':
        n_actions = 2129

    min_interaction = 20
    alpha1, alpha2, alpha3 = 1, 1, 1 
    beta1, beta2 = -1 , 1
    capacity = 500
    lr = 2e-3               
    gamma = 0.9            
    epsilon = 0.9            
    target_update = 100      
    n_states = 120
    n_hidden = 128
    batch_size = 64                
    exercise_module_dim = 10  
    N = 5                     
    g = 0                     
    pattern = "r"    

    reward = utils.MultiObjectiveReward(alpha1, alpha2, alpha3, beta1, beta2, N, g)
    replay_buffer = ReplayBuffer(capacity)
    with open(A_file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        all_concept_data = [] 
        all_problem_data = [] 
        for row in csv_reader:
            all_concept_data.append(row["skill_id"].split('_'))  
            all_problem_data.append(row['problem_id'])           
    A = all_concept_data      
    P = all_problem_data     
    module = EQN.EQN(all_problem_data,
                    n_states=n_states,
                    n_hidden=n_hidden,
                    n_actions=n_actions,
                    learning_rate=lr,
                    gamma=gamma,
                    epsilon=epsilon,
                    target_update=target_update,
                    device=device,
                    dataset=dataset)
    
    train(module,train_file_path, batch_size, pattern, istrain, dataset, n_actions)

    for K in [1,3,5,10]:
        NDCG, F1, Recall = test(module, train_file_path, test_file_path, pattern, K)
        ndcg_sum, f1_sum, recall_sum = 0, 0, 0
        for i in NDCG:
            ndcg_sum += i
        ndcg = ndcg_sum / len(NDCG)
        for i in F1:
            f1_sum += i
        f1 = f1_sum / len(F1)
        for i in Recall:
            recall_sum += i
        recall = recall_sum / len(Recall)
        print(f"K={K}, NDCG={ndcg}, F1={f1}, Recall={recall}")

