import torch
import numpy as np
from numpy import int32
from numpy.random import randint

from .Agent import SRC
from .Agent.utils import generate_path


def load_agent(args):
    if args.agent == 'SRC':
        return SRC(
            skill_num=args.skill_num,
            input_size=args.embed_size,
            weight_size=args.hidden_size,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            with_kt=args.withKT
        )
    raise NotImplementedError


def get_data(args, batchSize, data_path, path_type, n):
    data = np.load(data_path)
    skills = data['skill']
    responses = data['y']
    real_lens = data['real_len']
    num_students = skills.shape[0]
    padding = 0
    
    targets_list = []
    initial_logs_list = []
    initial_answers_list = []
    
    count = 0
    while count < batchSize:
        sid = np.random.randint(0, num_students)
        s_seq = skills[sid][:real_lens[sid]]
        r_seq = responses[sid][:real_lens[sid]]
        seq_len = len(s_seq)
        
        if seq_len < 5: 
            continue
            
        split_60 = int(seq_len * 0.6)
        split_80 = int(seq_len * 0.8)
            
        initial_logs = s_seq[:split_60]
        initial_answers = r_seq[:split_60]
        
        raw_targets = s_seq[split_80:] 
        
        seen = set()
        targets = [x for x in raw_targets if not (x in seen or seen.add(x))]
        
        if len(initial_logs) == 0 or len(targets) == 0:
            continue
        
        initial_logs_list.append(initial_logs)
        initial_answers_list.append(initial_answers)
        targets_list.append(targets)
        count += 1

    max_log_len = max(len(x) for x in initial_logs_list)
    max_target_len = max(len(x) for x in targets_list)
    
    padded_logs = [list(x) + [padding] * (max_log_len - len(x)) for x in initial_logs_list]
    padded_answers = [list(x) + [padding] * (max_log_len - len(x)) for x in initial_answers_list]
    padded_targets = [list(x) + [padding] * (max_target_len - len(x)) for x in targets_list]
    
    initial_logs = torch.tensor(padded_logs, dtype=torch.long)
    initial_answers = torch.tensor(padded_answers, dtype=torch.long)
    targets = torch.tensor(padded_targets, dtype=torch.long)

    if args.target_type == "all":
        targets = torch.arange(args.skill_num, dtype=torch.long)
        targets = targets.unsqueeze(0).repeat(batchSize, 1)

    paths = generate_path(batchSize, args.skill_num, path_type, n)
    return targets, initial_logs, paths, initial_answers