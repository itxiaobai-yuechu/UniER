
import os
import sys
import logging
import warnings
from argparse import ArgumentParser


import torch


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from EduSim.utils import get_raw_data_path
from EduSim.Envs.KES_ASSIST09 import KESASSIST09Env, kes_assist_train_eval
from EduSim.Envs.KES_ASSIST12 import KESASSIST12Env
from EduSim.Envs.KES_ASSIST15 import KESASSIST15Env
from EduSim.Envs.KES_ASSIST17 import KESASSIST17Env
from EduSim.Envs.KES_junyi import KESjunyiEnv
from EduSim.Envs.KES_mooccube import KESmooccubeEnv
from EduSim.Envs.KES_algebra2005 import KESalgebra2005Env
from EduSim.Envs.KES_bridge2006 import KESbridge2006Env
from EduSim.Envs.KES_ednet import KESednetEnv
from EduSim.Envs.KES_nips34 import KESnips34Env
from EduSim.Envs.KES_xes3g5m import KESxes3g5mEnv

from EduSim import AbstractAgent

cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_path[:cur_path.find('GEHRL')] + 'GEHRL')

warnings.filterwarnings('ignore')




def main():
    parser = ArgumentParser("learning_path_recommendation")
    simulator = ['KESassist09', 'KESassist12', 'KESassist15','KESassist17', 'KESjunyi','KESmooccube','KESalgebra2005','KESbridge2006','KESednet','KESnips34','KESxes3g5m']
    agent = ['HRL']
    seeds = [1, 5, 10]

    HRL_high_policy = ['PPO', 'AC']
    HRL_reward_high_level = ['dkt', 'test', 'env-only', 'all-dkt', 'all-test', 'rs1', 'rs2', 'rs3']
    HRL_high_goals_encoding = ['No', 'order', 'disorder']
    HRL_subgoals_topo_order_constraint = ['No', 'hard', 'soft']
    HRL_low_policy = ['PPO', 'AC']
    HRL_reward_low_level = ['test_epi', 'dkt', 'test', 'god']
    HRL_candidates_for_low_level = ['embedding', 'No', 'CN', 'goalprerequisites']
    HRL_low_know_all_goals = ['disorder', 'No', 'order', 'disorderTransf']
    HRL_subgoals_continuity = ['No', 'yes']
    target_types = ['all', 'portion']

    parser.add_argument('-s', '--simulator', type=str, choices=simulator, default='KESassist15')
    parser.add_argument('-a', '--agent', type=str, choices=agent, default='HRL')
    parser.add_argument('-e', '--experiment_idx', type=int, default=2, help='Experiment id for one model')
    parser.add_argument('-r', '--repeat_num', type=int, default=1, help='Experiment id for one seed')
    parser.add_argument('-m', '--max_steps', type=int, default=5)
    parser.add_argument('-k', '--know_all_log', type=bool, default=False)
    parser.add_argument('-RC', '--Repe_control',type=int,default=0,choices=[0, 1])
    parser.add_argument('--grad_clip', type=float, default=0.5)
    parser.add_argument('--step_count', type=int, default=0)
    parser.add_argument('--episode_count', type=int, default=0)
    parser.add_argument('--steptime_state_saver', default=torch.tensor([],
                                                                           dtype=torch.float32))
    parser.add_argument('--steptime_next_state_saver', default=torch.tensor([],
                                                                                dtype=torch.float32))
    parser.add_argument('--steptime_perfect_log_one_hot', default=torch.tensor([],
                                                                                   dtype=torch.float32))
    parser.add_argument('--steptime_knowledge_state', default=torch.tensor([],
                                                                               dtype=torch.float32))
    parser.add_argument('--learner_initial_logs', default=[])
    parser.add_argument('--steptime_dkt_ks', default=torch.tensor([],
                                                                      dtype=torch.float32))
    parser.add_argument('--episode_subgoals', default=[])
    parser.add_argument('--pre_goal', type=int, default=-1)
    parser.add_argument('--cur_goal', type=int, default=-1)
    parser.add_argument('--cur_goal_count', type=int, default=0)
    parser.add_argument('--current_rec_log', type=list, default=[])
    parser.add_argument('--repe_abandon_list', type=list, default=[])
    parser.add_argument('--item_count_dict', type=dict, default={})
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.00005)
    parser.add_argument('-ppoclip', '--ppoclip', type=float, default=0.1)
    parser.add_argument('--perfect_log_max_length', type=int, default=50)
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--seed', type=int, choices=seeds)
    parser.add_argument('--max_episode_num', type=int, default=3000)
    parser.add_argument('--dataRecPath', type=str, default='')
    parser.add_argument('-gonat', '--gonathresh', type=float, default=0.9)
    parser.add_argument('--graph_embedding_input', type=bool, default=True)
    parser.add_argument('--graph_embedding_type', type=str, default='node2vec')
    parser.add_argument('--HRL_high_policy', type=str, choices=HRL_high_policy, default='PPO')
    parser.add_argument('--HRL_reward_high_level', type=str, choices=HRL_reward_high_level,
                        default='env-only')
    parser.add_argument('--HRL_candidates_for_high_level', type=str, default='No')
    parser.add_argument('--HRL_high_goals_encoding', type=str,
                        choices=HRL_high_goals_encoding, default='No')
    parser.add_argument('--HRL_subgoals_topo_order_constraint', type=str,
                        choices=HRL_subgoals_topo_order_constraint, default='No')
    parser.add_argument('--HRL_deep_high_with__1', type=str, default='No')
    parser.add_argument('--HRL_subgoals_continuity', type=str,
                        choices=HRL_subgoals_continuity, default='No')

    parser.add_argument('--HRL_low_policy', type=str, choices=HRL_low_policy, default='AC')
    parser.add_argument('--HRL_reward_low_level', type=str,
                        choices=HRL_reward_low_level, default='test_epi')
    parser.add_argument('--HRL_sub_weight', type=float, default=1.0)
    parser.add_argument('--HRL_env_weight', type=float, default=1.0)
    parser.add_argument('--HRL_candidaates_for_low_level', type=str,
                        choices=HRL_candidates_for_low_level, default='embedding')
    parser.add_argument('--HRL_low_know_all_goals', type=str,
                        choices=HRL_low_know_all_goals, default='disorder')

    parser.add_argument('--HRL_random_low_level', type=bool, default=False)
    parser.add_argument('--HRL_asynchronous_train', type=bool, default=True)
    parser.add_argument('--HRL_as_tr_episode', type=int, default=1)
    parser.add_argument('--HRL_embcan_num', type=int, default=60)
    parser.add_argument('--target_type', type=str, choices=target_types, default='all')
    args = parser.parse_args().__dict__
    args['seed'] = seeds[args['repeat_num'] - 1]
    args['ppoclip'] = 0.9

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args['cuda_device'])
    if torch.cuda.is_available():
        print(f"CUDA Available! Using GPU: {torch.cuda.get_device_name(0)} (Physical Device {args['cuda_device']})")
        device = 'cuda:0'
        if hasattr(torch, 'set_default_device'):
            torch.set_default_device(device)
        else:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print("Warning: CUDA not available. Running on CPU.")

    if args['simulator'] == 'KESassist09':
        args['dataRecPath'] = f'{get_raw_data_path()}/data/dataProcess/assist09/dataRec'
        env = KESASSIST09Env(dataRec_path=args['dataRecPath'], seed=args['seed'])
        env.target_type = args['target_type']
    elif args['simulator'] == 'KESassist12':
        args['dataRecPath'] = f'{get_raw_data_path()}/data/dataProcess/assist12/dataRec'
        env = KESASSIST12Env(dataRec_path=args['dataRecPath'], seed=args['seed'])
        env.target_type = args['target_type']
    elif args['simulator'] == 'KESassist15':
        args['dataRecPath'] = f'{get_raw_data_path()}/data/dataProcess/assist15/dataRec'
        env = KESASSIST15Env(dataRec_path=args['dataRecPath'], seed=args['seed'])
        env.target_type = args['target_type']
    elif args['simulator'] == 'KESassist17':
        args['dataRecPath'] = f'{get_raw_data_path()}/data/dataProcess/assist17/dataRec'
        env = KESASSIST17Env(dataRec_path=args['dataRecPath'], seed=args['seed'])
        env.target_type = args['target_type']
    elif args['simulator'] == 'KESjunyi':
        args['dataRecPath'] = f'{get_raw_data_path()}/data/dataProcess/junyi/dataRec'
        env = KESjunyiEnv(dataRec_path=args['dataRecPath'], seed=args['seed'])
        env.target_type = args['target_type']
    elif args['simulator'] == 'KESmooccube':
        args['dataRecPath'] = f'{get_raw_data_path()}/data/dataProcess/mooccube/dataRec'
        env = KESmooccubeEnv(dataRec_path=args['dataRecPath'], seed=args['seed'])
        env.target_type = args['target_type']
    elif args['simulator'] == 'KESalgebra2005':
        args['dataRecPath'] = f'{get_raw_data_path()}/data/dataProcess/algebra2005/dataRec'
        env = KESalgebra2005Env(dataRec_path=args['dataRecPath'], seed=args['seed'])
        env.target_type = args['target_type']
    elif args['simulator'] == 'KESbridge2006':
        args['dataRecPath'] = f'{get_raw_data_path()}/data/dataProcess/bridge2006/dataRec'
        env = KESbridge2006Env(dataRec_path=args['dataRecPath'], seed=args['seed'])
        env.target_type = args['target_type']
    elif args['simulator'] == 'KESednet':
        args['dataRecPath'] = f'{get_raw_data_path()}/data/dataProcess/ednet/dataRec'
        env = KESednetEnv(dataRec_path=args['dataRecPath'], seed=args['seed'])
        env.target_type = args['target_type']
    elif args['simulator'] == 'KESnips34':
        args['dataRecPath'] = f'{get_raw_data_path()}/data/dataProcess/nips34/dataRec'
        env = KESnips34Env(dataRec_path=args['dataRecPath'], seed=args['seed'])
        env.target_type = args['target_type']
    elif args['simulator'] == 'KESxes3g5m':
        args['dataRecPath'] = f'{get_raw_data_path()}/data/dataProcess/xes3g5m/dataRec'
        env = KESxes3g5mEnv(dataRec_path=args['dataRecPath'], seed=args['seed'])
        env.target_type = args['target_type']

    agent = AbstractAgent(env, args)
    env_input_dict = {}
    env_input_dict['agent'] = agent
    env_input_dict['env'] = env
    env_input_dict['max_steps'] = args['max_steps']
    env_input_dict['max_episode_num'] = args['max_episode_num']
    env_input_dict['level'] = "summary"
    env_input_dict['n_step'] = False
    env_input_dict['train'] = True
    env_input_dict['logger'] = logging
    env_input_dict['values'] = None
    env_input_dict['monitor'] = None

    kes_assist_train_eval(env_input_dict)


if __name__ == '__main__':
    main()
