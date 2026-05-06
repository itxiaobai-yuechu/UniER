
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from EduSim.Envs.agent_utils import get_proj_path, get_raw_data_path
from argparse import ArgumentParser
from EduSim.Envs.KES import KESEnv, kes_train_eval
from EduSim.Envs.KES_ASSIST15 import KESASSIST15Env, kes_assist_train_eval
from EduSim.Envs.KES_ASSIST09 import KESASSIST09Env, kes_assist_train_eval
from EduSim.Envs.KES_ASSIST12 import KESASSIST12Env, kes_assist_train_eval
from EduSim.Envs.KES_ASSIST17 import KESASSIST17Env, kes_assist_train_eval
from EduSim.Envs.KES_algebra2005 import KESalgebra2005Env, kes_algebra2005_train_eval
from EduSim.Envs.KES_bridge2006 import KESbridge2006Env, kes_bridge2006_train_eval
from EduSim.Envs.KES_ednet import KESednetEnv, kes_ednet_train_eval
from EduSim.Envs.KES_nips34 import KESnips34Env, kes_nips34_train_eval
from EduSim.Envs.KES_junyi import KESjunyiEnv, kes_junyi_train_eval
from EduSim.Envs.KES_mooccube import KESmooccubeEnv, kes_mooccube_train_eval
from EduSim.Envs.KES_xes3g5m import KESxes3g5mEnv, kes_xes3g5m_train_eval
from EduSim.Envs.KSS import KSSEnv, kss_train_eval
from EduSim.Envs import AbstractAgent
import gym
import warnings
import torch

warnings.filterwarnings('ignore')


def main():
    parser = ArgumentParser("learning_path_recommendation")
    simulator = ['KSS', 'KESjunyi', 'KESassist15', 'KESassist09','KESassist12','KESassist17','KESalgebra2005','KESbridge2006','KESednet','KESnips34','KESmooccube','KESxes3g5m'] 
    seeds = [1, 5, 10]
    encoder = ['RNN', 'DKT']

    parser.add_argument('-s', '--simulator', type=str, choices=simulator, default='KSS')
    parser.add_argument('-a', '--agent', type=str, default='AC')
    parser.add_argument('--encoder', type=str, choices=encoder, default='RNN')
    parser.add_argument('-e', '--experiment_idx', type=int, default=2, help='Experiment id for one model')
    parser.add_argument('-r', '--repeat_num', type=int, default=1, help='Experiment id for one seed')
    parser.add_argument('-m', '--max_steps', type=int, default=20)
    parser.add_argument('-k', '--know_all_log', type=bool, default=False)
    parser.add_argument('-RC', '--Repe_control', type=bool, default=False)
    parser.add_argument('-wid', '--wid', type=bool, default=False)
    parser.add_argument('-lmt', '--load_model_test', default=False)
    
    parser.add_argument('--step_count', type=int, default=0)
    parser.add_argument('--episode_count', type=int, default=0)
    parser.add_argument('--steptime_state_saver', default=None)
    parser.add_argument('--steptime_next_state_saver', default=None)
    parser.add_argument('--steptime_perfect_log_one_hot', default=None)
    parser.add_argument('--steptime_knowledge_state', default=None)
    parser.add_argument('--learner_initial_logs', default=[])
    parser.add_argument('--steptime_dkt_ks', default=None)
    parser.add_argument('--episode_subgoals', default=[])
    parser.add_argument('--pre_goal', type=int, default=-1)
    parser.add_argument('--cur_goal', type=int, default=-1)
    parser.add_argument('--cur_goal_count', type=int, default=0)
    parser.add_argument('--current_rec_log', type=list, default=[])
    parser.add_argument('--repe_abandon_list', type=list, default=[])
    parser.add_argument('--item_count_dict', type=dict, default={})
    parser.add_argument('-GND', '--GCN_hidden_dim', type=int, default=32)
    
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.00005)
    parser.add_argument('-ppoclip', '--ppoclip', type=float, default=0.1)
    parser.add_argument('--perfect_log_max_length', type=int, default=50)
    parser.add_argument('--cudaDevice', type=str, default='cuda')
    parser.add_argument('--seed', type=int, choices=seeds)
    parser.add_argument('--max_episode_num', type=int, default=10000)
    parser.add_argument('--dataRecPath', type=str, default='')
    target_types = ['all', 'portion']
    parser.add_argument('--target_type', type=str, choices=target_types, default='portion')

    args = parser.parse_args().__dict__
    args['seed'] = seeds[args['repeat_num'] - 1]
    
    if args['simulator'] == 'KESjunyi':
        args['dataRecPath'] = f'{get_raw_data_path()}/dataProcess/junyi/dataRec'
    elif args['simulator'] == 'KESassist15':
        args['dataRecPath'] = f'{get_raw_data_path()}/dataProcess/assist15/dataRec'
    elif args['simulator'] == 'KESassist09':
        args['dataRecPath'] = f'{get_raw_data_path()}/dataProcess/assist09/dataRec'
    elif args['simulator'] == 'KESassist12':
        args['dataRecPath'] = f'{get_raw_data_path()}/dataProcess/assist12/dataRec'
    elif args['simulator'] == 'KESassist17':
        args['dataRecPath'] = f'{get_raw_data_path()}/dataProcess/assist17/dataRec'
    elif args['simulator'] == 'KESalgebra2005':
        args['dataRecPath'] = f'{get_raw_data_path()}/dataProcess/algebra2005/dataRec'
    elif args['simulator'] == 'KESbridge2006':
        args['dataRecPath'] = f'{get_raw_data_path()}/dataProcess/bridge2006/dataRec'
    elif args['simulator'] == 'KESednet':
        args['dataRecPath'] = f'{get_raw_data_path()}/dataProcess/ednet/dataRec'
    elif args['simulator'] == 'KESnips34':
        args['dataRecPath'] = f'{get_raw_data_path()}/dataProcess/nips34/dataRec'
    elif args['simulator'] == 'KESmooccube':
        args['dataRecPath'] = f'{get_raw_data_path()}/dataProcess/mooccube/dataRec'
    elif args['simulator'] == 'KESxes3g5m':
        args['dataRecPath'] = f'{get_raw_data_path()}/dataProcess/xes3g5m/dataRec'
    print('seed=' + str(args['seed']))
    args['device'] = torch.device(args['cudaDevice'])
    args['steptime_state_saver'] = torch.tensor([], dtype=torch.float, device=args['device'])
    args['steptime_next_state_saver'] = torch.tensor([], dtype=torch.float, device=args['device'])
    args['steptime_perfect_log_one_hot'] = torch.tensor([], dtype=torch.float, device=args['device'])
    args['steptime_knowledge_state'] = torch.tensor([], dtype=torch.float, device=args['device'])
    args['steptime_dkt_ks'] = torch.tensor([], dtype=torch.float, device=args['device'])
    print(args)
    
    if args['simulator'] == 'KESjunyi':
        env = gym.make('KESjunyi-v1', dataRec_path=args['dataRecPath'], seed=args['seed'], cudaDevice=args['device'])
        env.target_type = args['target_type']  
        agent = AbstractAgent(env, args)
        kes_junyi_train_eval(
            agent,
            env,
            max_steps=args['max_steps'],
            max_episode_num=args['max_episode_num'],
            level="summary",
            board_dir="./kes_logs"
        )
    elif args['simulator'] == 'KESassist15':
        env = gym.make('KESASSIST15-v1', dataRec_path=args['dataRecPath'], seed=args['seed'], cudaDevice=args['device'])
        env.target_type = args['target_type']
        agent = AbstractAgent(env, args)
        kes_assist_train_eval(
            agent,
            env,
            max_steps=args['max_steps'],
            max_episode_num=args['max_episode_num'],
            level="summary",
            board_dir="./kes_logs"
        )
    elif args['simulator'] == 'KSS':
        env = gym.make('KSS-v2', seed=args['seed'], cudaDevice=args['device'])
        env.target_type = args['target_type']
        agent = AbstractAgent(env, args)
        kss_train_eval(
            agent,
            env,
            max_steps=args['max_steps'],
            max_episode_num=args['max_episode_num'],
            level="summary",
            board_dir="./kss_logs"
        )
    elif args['simulator'] == 'KESassist09':
        env = gym.make('KESASSIST09-v1', dataRec_path=args['dataRecPath'], seed=args['seed'], cudaDevice=args['device'])
        env.target_type = args['target_type']
        agent = AbstractAgent(env, args)
        kes_assist_train_eval(
            agent,
            env,
            max_steps=args['max_steps'],
            max_episode_num=args['max_episode_num'],
            level="summary",
            board_dir="./kes_logs"
        )
    elif args['simulator'] == 'KESassist12':
        env = gym.make('KESASSIST12-v1', dataRec_path=args['dataRecPath'], seed=args['seed'], cudaDevice=args['device'])
        env.target_type = args['target_type']
        agent = AbstractAgent(env, args)
        kes_assist_train_eval(
            agent,
            env,
            max_steps=args['max_steps'],
            max_episode_num=args['max_episode_num'],
            level="summary",
            board_dir="./kes_logs"
        )
    elif args['simulator'] == 'KESassist17':
        env = gym.make('KESASSIST17-v1', dataRec_path=args['dataRecPath'], seed=args['seed'], cudaDevice=args['device'])
        env.target_type = args['target_type']
        agent = AbstractAgent(env, args)
        kes_assist_train_eval(
            agent,
            env,
            max_steps=args['max_steps'],
            max_episode_num=args['max_episode_num'],
            level="summary",
            board_dir="./kes_logs"
        )
    elif args['simulator'] == 'KESalgebra2005':
        env = gym.make('KESalgebra2005-v1', dataRec_path=args['dataRecPath'], seed=args['seed'], cudaDevice=args['device'])
        env.target_type = args['target_type']
        agent = AbstractAgent(env, args)
        kes_algebra2005_train_eval(
            agent,
            env,
            max_steps=args['max_steps'],
            max_episode_num=args['max_episode_num'],
            level="summary",
            board_dir="./kes_logs"
        )
    elif args['simulator'] == 'KESbridge2006':
        env = gym.make('KESbridge2006-v1', dataRec_path=args['dataRecPath'], seed=args['seed'], cudaDevice=args['device'])
        env.target_type = args['target_type']
        agent = AbstractAgent(env, args)
        kes_bridge2006_train_eval(
            agent,
            env,
            max_steps=args['max_steps'],
            max_episode_num=args['max_episode_num'],
            level="summary",
            board_dir="./kes_logs"
        )
    elif args['simulator'] == 'KESednet':
        env = gym.make('KESednet-v1', dataRec_path=args['dataRecPath'], seed=args['seed'], cudaDevice=args['device'])
        env.target_type = args['target_type']
        agent = AbstractAgent(env, args)
        kes_ednet_train_eval(
            agent,
            env,
            max_steps=args['max_steps'],
            max_episode_num=args['max_episode_num'],
            level="summary",
            board_dir="./kes_logs"
        )
    elif args['simulator'] == 'KESnips34':
        env = gym.make('KESnips34-v1', dataRec_path=args['dataRecPath'], seed=args['seed'], cudaDevice=args['device'])
        env.target_type = args['target_type']
        agent = AbstractAgent(env, args)
        kes_nips34_train_eval(
            agent,
            env,
            max_steps=args['max_steps'],
            max_episode_num=args['max_episode_num'],
            level="summary",
            board_dir="./kes_logs"
        )
    elif args['simulator'] == 'KESmooccube':
        env = gym.make('KESmooccube-v1', dataRec_path=args['dataRecPath'], seed=args['seed'], cudaDevice=args['device'])
        env.target_type = args['target_type']
        agent = AbstractAgent(env, args)
        kes_mooccube_train_eval(
            agent,
            env,
            max_steps=args['max_steps'],
            max_episode_num=args['max_episode_num'],
            level="summary",
            board_dir="./kes_logs"
        )
    elif args['simulator'] == 'KESxes3g5m':
        env = gym.make('KESxes3g5m-v1', dataRec_path=args['dataRecPath'], seed=args['seed'], cudaDevice=args['device'])
        env.target_type = args['target_type']
        agent = AbstractAgent(env, args)
        kes_xes3g5m_train_eval(
            agent,
            env,
            max_steps=args['max_steps'],
            max_episode_num=args['max_episode_num'],
            level="summary",
            board_dir="./kes_logs"
        )
if __name__ == '__main__':
    main()
