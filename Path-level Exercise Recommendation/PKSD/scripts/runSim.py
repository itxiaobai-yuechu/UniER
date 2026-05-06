
import sys
import os

sys.path.append(os.path.dirname(sys.path[0]))
from EduSim.Envs.agent_utils import get_proj_path, get_raw_data_path
from argparse import ArgumentParser
from EduSim.Envs.KES_ASSIST15 import KESASSIST15Env, kes_assist_train_eval
from EduSim.Envs.KES_ASSIST09 import KESASSIST09Env, kes_assist_train_eval
from EduSim.Envs.KES_ASSIST12 import KESASSIST12Env, kes_assist_train_eval
from EduSim.Envs.KES_ASSIST17 import KESASSIST17Env, kes_assist_train_eval
from EduSim.Envs.KES_ednet import KESednetEnv, kes_ednet_train_eval
from EduSim.Envs.KES_junyi import KESjunyiEnv, kes_junyi_train_eval
from EduSim.Envs.KES_mooccube import KESmooccubeEnv, kes_mooccube_train_eval
from EduSim.Envs.KES_nips34 import KESnips34Env, kes_nips34_train_eval
from EduSim.Envs.KES_algebra2005 import KESalgebra2005Env, kes_algebra2005_train_eval
from EduSim.Envs.KES_bridge2006 import KESbridge2006Env, kes_bridge2006_train_eval
from EduSim.Envs.KES_xes3g5m import KESxes3g5mEnv, kes_exs3g5m_train_eval
from EduSim.Envs.KSS import KSSEnv, kss_train_eval
from EduSim.Envs import AbstractAgent
import gym
import warnings
import torch

warnings.filterwarnings('ignore')


def main():
    parser = ArgumentParser("learning_path_recommendation")
    simulator = ['KSS', 'KESjunyi', 'KESassist15', 'KESassist09','KESassist12','KESassist17','KESednet','KESalgebra2005','KESbridge2006','KESmooccube','KESnips34','KESxes3g5m']
    agent = ['DQN', 'AC', 'PPO', 'CSEAL', 'PFD', 'DKTRec']
    seeds = [1, 5, 10]
    RepeControAgents = ['DQN']
    encoder = ['RNN', 'DKT']

    model_based = ['No', 'DAS3H', 'DKT']

    PFD_PerEncoder = ['MLP', 'No', 'GNN']
    PFD_ImPerEncoder = ['RNN', 'Transformer', 'DKT', 'RNN_GNN', 'DKT_GNN']
    PFD_Perfect_info_type = ['knowledge_state']
    target_types = ['portion', 'all']

    parser.add_argument('-s', '--simulator', type=str, choices=simulator, default='KSS')
    parser.add_argument('-a', '--agent', type=str, choices=agent, default='PFD')
    parser.add_argument('--encoder', type=str, choices=encoder, default='RNN')
    parser.add_argument('-e', '--experiment_idx', type=int, default=2, help='Experiment id for one model')
    parser.add_argument('-r', '--repeat_num', type=int, default=1, help='Experiment id for one seed')
    parser.add_argument('-m', '--max_steps', type=int, default=20)
    parser.add_argument('-mb', '--model_based', type=str, choices=model_based, default='No')
    parser.add_argument('-k', '--know_all_log', type=bool, default=False)
    parser.add_argument('-RC', '--Repe_control', type=bool, default=True)
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
    parser.add_argument('--PFD_regress_loss', type=float, default=-1)
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

    parser.add_argument('--PFD_base_policy', type=str, choices=agent, default='PPO')
    parser.add_argument('--PFD_perfect_info_type', type=str, choices=PFD_Perfect_info_type, default='knowledge_state')
    parser.add_argument('-ppe', '--PFD_PerEncoder', type=str, choices=PFD_PerEncoder, default='GNN')
    parser.add_argument('-pie', '--PFD_ImPerEncoder', type=str, choices=PFD_ImPerEncoder, default='RNN_GNN')
    parser.add_argument('-p1er', '--PFD_phase_1_episdoes_ratio', type=float, default=0.3)
    parser.add_argument('-p1ol', '--PFD_phase_1_only', type=bool, default=False)
    parser.add_argument('-ppr', '--PFD_privileged_reward', type=bool, default=False)
    parser.add_argument('--PFD_imper_reg_only', type=bool, default=False)
    parser.add_argument('--target_type', type=str, choices=target_types, default='portion')

    args = parser.parse_args().__dict__
    args['seed'] = seeds[args['repeat_num'] - 1]
    if args['simulator'] == 'KESassist15':
        args['dataRecPath'] = f'{get_raw_data_path()}/dataProcess/assist15/dataRec'
    elif args['simulator'] == 'KESassist09':
        args['dataRecPath'] = f'{get_raw_data_path()}/dataProcess/assist09/dataRec'
    elif args['simulator'] == 'KESassist12':
        args['dataRecPath'] = f'{get_raw_data_path()}/dataProcess/assist12/dataRec'
    elif args['simulator'] == 'KESassist17':
        args['dataRecPath'] = f'{get_raw_data_path()}/dataProcess/assist17/dataRec'
    elif args['simulator'] == 'KESednet':
        args['dataRecPath'] = f'{get_raw_data_path()}/dataProcess/ednet/dataRec'
    elif args['simulator'] == 'KESjunyi':
        args['dataRecPath'] = f'{get_raw_data_path()}/dataProcess/junyi/dataRec'
    elif args['simulator'] == 'KESalgebra2005':
        args['dataRecPath'] = f'{get_raw_data_path()}/dataProcess/algebra2005/dataRec'
    elif args['simulator'] == 'KESbridge2006':
        args['dataRecPath'] = f'{get_raw_data_path()}/dataProcess/bridge2006/dataRec'
    elif args['simulator'] == 'KESmooccube':
        args['dataRecPath'] = f'{get_raw_data_path()}/dataProcess/mooccube/dataRec'
    elif args['simulator'] == 'KESnips34':
        args['dataRecPath'] = f'{get_raw_data_path()}/dataProcess/nips34/dataRec'
    elif args['simulator'] == 'KESxes3g5m':
        args['dataRecPath'] = f'{get_raw_data_path()}/dataProcess/xes3g5m/dataRec'

    if 'GNN' in args['PFD_ImPerEncoder']:
        args['PFD_PerEncoder'] = 'GNN'
    if args['PFD_ImPerEncoder'] == 'DKT':
        args['PFD_PerEncoder'] = 'No'
    if args['PFD_PerEncoder'] == 'GNN' and args['PFD_ImPerEncoder'] == 'RNN':
        args['PFD_ImPerEncoder'] = 'RNN_GNN'

    if args['agent'] in RepeControAgents or (args['agent'] == 'PFD' and args['PFD_base_policy'] in RepeControAgents):
        args['Repe_control'] = True

    print('seed=' + str(args['seed']))
    args['device'] = torch.device(args['cudaDevice'])
    args['steptime_state_saver'] = torch.tensor([], dtype=torch.float, device=args['device'])
    args['steptime_next_state_saver'] = torch.tensor([], dtype=torch.float, device=args['device'])
    args['steptime_perfect_log_one_hot'] = torch.tensor([], dtype=torch.float, device=args['device'])
    args['steptime_knowledge_state'] = torch.tensor([], dtype=torch.float, device=args['device'])
    args['steptime_dkt_ks'] = torch.tensor([], dtype=torch.float, device=args['device'])
    print(args)
    if args['simulator'] == 'KESassist15':
        env = gym.make('KES-v3', dataRec_path=args['dataRecPath'], seed=args['seed'], cudaDevice=args['device'])
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
    elif args['simulator'] == 'KESassist09':
        env = gym.make('KES-v1', dataRec_path=args['dataRecPath'], seed=args['seed'], cudaDevice=args['device'])
        env.target_type = args['target_type']
        agent = AbstractAgent(env, args)
        kes_assist_train_eval(
            agent,
            env,
            max_steps=args['max_steps'],
            max_episode_num=args['max_episode_num'],
            level="summary",
            board_dir="./kes09_logs"
        )
    elif args['simulator'] == 'KESassist12':
        env = gym.make('KES-v2', dataRec_path=args['dataRecPath'], seed=args['seed'], cudaDevice=args['device'])
        env.target_type = args['target_type']
        agent = AbstractAgent(env, args)
        kes_assist_train_eval(
            agent,
            env,
            max_steps=args['max_steps'],
            max_episode_num=args['max_episode_num'],
            level="summary",
            board_dir="./kes12_logs"
        )
    elif args['simulator'] == 'KESassist17':
        env = gym.make('KES-v4', dataRec_path=args['dataRecPath'], seed=args['seed'], cudaDevice=args['device'])
        env.target_type = args['target_type']
        agent = AbstractAgent(env, args)
        kes_assist_train_eval(
            agent,
            env,
            max_steps=args['max_steps'],
            max_episode_num=args['max_episode_num'],
            level="summary",
            board_dir="./kes17_logs"
        )
    elif args['simulator'] == 'KESalgebra2005':
        env = gym.make('KES-v5', dataRec_path=args['dataRecPath'], seed=args['seed'], cudaDevice=args['device'])
        env.target_type = args['target_type']
        agent = AbstractAgent(env, args)
        kes_algebra2005_train_eval(
            agent,
            env,
            max_steps=args['max_steps'],
            max_episode_num=args['max_episode_num'],
            level="summary",
            board_dir="./algebra2005_logs"
        )
    elif args['simulator'] == 'KESbridge2006':
        env = gym.make('KES-v6', dataRec_path=args['dataRecPath'], seed=args['seed'], cudaDevice=args['device'])
        env.target_type = args['target_type']
        agent = AbstractAgent(env, args)
        kes_bridge2006_train_eval(
            agent,
            env,
            max_steps=args['max_steps'],
            max_episode_num=args['max_episode_num'],
            level="summary",
            board_dir="./bridge2006_logs"
        )
    elif args['simulator'] == 'KESednet':
        env = gym.make('KES-v7', dataRec_path=args['dataRecPath'], seed=args['seed'], cudaDevice=args['device'])
        env.target_type = args['target_type']
        agent = AbstractAgent(env, args)
        kes_ednet_train_eval(
            agent,
            env,
            max_steps=args['max_steps'],
            max_episode_num=args['max_episode_num'],
            level="summary",
            board_dir="./ednet_logs"
        )
    elif args['simulator'] == 'KESjunyi':
        env = gym.make('KES-v8', dataRec_path=args['dataRecPath'], seed=args['seed'], cudaDevice=args['device'])
        env.target_type = args['target_type']
        agent = AbstractAgent(env, args)
        kes_junyi_train_eval(
            agent,
            env,
            max_steps=args['max_steps'],
            max_episode_num=args['max_episode_num'],
            level="summary",
            board_dir="./junyi_logs"
        )
    elif args['simulator'] == 'KESmooccube':
        env = gym.make('KES-v9', dataRec_path=args['dataRecPath'], seed=args['seed'], cudaDevice=args['device'])
        env.target_type = args['target_type']
        agent = AbstractAgent(env, args)
        kes_mooccube_train_eval(
            agent,
            env,
            max_steps=args['max_steps'],
            max_episode_num=args['max_episode_num'],
            level="summary",
            board_dir="./mooccube_logs"
        )
    elif args['simulator'] == 'KESnips34':
        env = gym.make('KES-v10', dataRec_path=args['dataRecPath'], seed=args['seed'], cudaDevice=args['device'])
        env.target_type = args['target_type']
        agent = AbstractAgent(env, args)
        kes_nips34_train_eval(
            agent,
            env,
            max_steps=args['max_steps'],
            max_episode_num=args['max_episode_num'],
            level="summary",
            board_dir="./nips34_logs"
        )
    elif args['simulator'] == 'KESxes3g5m':
        env = gym.make('KES-v11', dataRec_path=args['dataRecPath'], seed=args['seed'], cudaDevice=args['device'])
        env.target_type = args['target_type']
        agent = AbstractAgent(env, args)
        kes_exs3g5m_train_eval(
            agent,
            env,
            max_steps=args['max_steps'],
            max_episode_num=args['max_episode_num'],
            level="summary",
            board_dir="./xes3g5m_logs"
        )


if __name__ == '__main__':
    main()
