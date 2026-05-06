from argparse import ArgumentParser, Namespace
import torch

def get_exp_configure(model):
    configure_list = {
        'embed_size': 128,
        'hidden_size': 128,
        'output_size': 1,
        'batch_size': 128,
        'dropout': 0.5,
        'decay_step': 100,
        'l2_reg': 4e-5,
        'pre_hidden_sizes': [256, 64, 16],
        'retrieval': False,
        'forRec': False
    }
    if model == 'CoKT':
        configure_list.update({'batch_size': 8, 'retrieval': True, 'embed_size': 48,
                               'hidden_size': 64, 'pre_hidden_sizes': [128, 64, 16]})
    if model == 'GRU4Rec':
        configure_list.update({'forRec': True, 'without_label': True})
    return configure_list


def get_options(parser: ArgumentParser, reset_args=None):
    if reset_args is None:
        reset_args = {}
    model = ['DKT', 'Transformer', 'CoKT', 'GRU4Rec']
    dataset = ['assist09', 'assist12', 'assist15','assist17', 'algebra2005', 'bridge2006','ednet','junyi','nips34','xes3g5m','mooccubex']
    parser.add_argument('-m', '--model', type=str, choices=model, default='DKT', help="Model to use")
    parser.add_argument('-d', '--dataset', type=str, choices=dataset, default='assist09', help="Dataset to use")
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./Scripts/Envs/KES/meta_data')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--without_label', action='store_true')
    parser.add_argument('-c', '--cuda', type=int, default=0)
    parser.add_argument('-e', "--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--valid_step', type=int, default=100)
    parser.add_argument("--postfix", type=str, default='', help="a string appended to the file name of the saved model")
    parser.add_argument("--rand_seed", type=int, default=-1, help="random seed for torch and numpy")
    parser.set_defaults(**reset_args)
    args = parser.parse_args().__dict__
    
    exp_configure = get_exp_configure(args['model'])
    args.update(exp_configure)
    args = Namespace(**args)
    
    args.exp_name = '_'.join([args.model, args.dataset])
    if args.without_label:
        args.exp_name += '_without'
    if args.postfix != '':
        args.exp_name += f'_{args.postfix}'
    
    if torch.cuda.is_available() and args.cuda >= 0:
        torch.cuda.set_device(args.cuda)
        args.device = f'cuda:{args.cuda}'
    else:
        args.device = 'cpu'
    
    return args