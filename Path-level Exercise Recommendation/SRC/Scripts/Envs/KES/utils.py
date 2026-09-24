import os
import torch

from longling import path_append, abs_current_dir
from KTScripts.options import get_exp_configure
from KTScripts.utils import load_model


def load_d_agent(model_name, dataset_name, skill_num, with_label=True):
    model_parameters = get_exp_configure(model_name)
    model_parameters.update({
        'feat_nums': skill_num, 
        'model': model_name, 
        'without_label': not with_label
    })
    
    if model_name == 'GRU4Rec':
        model_parameters.update({'output_size': skill_num})
    
    model = load_model(model_parameters)
    
    model_folder = path_append(abs_current_dir(__file__), os.path.join('meta_data'))
    model_path = os.path.join(model_folder, f'{model_name}_{dataset_name}')
    if not with_label:
        model_path += '_without'
    
    ckpt_full_path = f'{model_path}.pth' 
    if not os.path.exists(ckpt_full_path):
        ckpt_full_path = f'{model_path}.ckpt'
        
    state_dict = torch.load(ckpt_full_path, map_location='cpu')
    model.load_state_dict(state_dict)
    
    model.eval()
    
    return model


def episode_reward(initial_score, final_score, full_score):

    delta = final_score - initial_score
    normalize_factor = full_score - initial_score + 1e-9
    return delta / normalize_factor