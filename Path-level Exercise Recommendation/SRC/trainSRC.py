import os
import time
from argparse import ArgumentParser

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import PolynomialLR
from tqdm import tqdm

from KTScripts.DataLoader import KTDataset
from KTScripts.utils import set_random_seed
from Scripts.Agent.utils import pl_loss
from Scripts.Envs import KESEnv
from Scripts.Optimizer import ModelWithLoss, ModelWithOptimizer
from Scripts.options import get_options
from Scripts.utils import load_agent, get_data
from deep_model import DKTnet


def main(args):
    set_random_seed(args.rand_seed)
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_file_path = os.path.join(args.data_dir, f"{args.dataset}/{args.dataset}_dataRec.npz")
    env = KESEnv(data_file_path, args.model, args.dataset,device = device)
    args.skill_num = env.skill_num
    model = load_agent(args).to(device)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    model_path = os.path.join(args.save_dir, args.exp_name + str(args.path)) + ".ckpt"
    
    if args.load_model:
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Load Model From {model_path}")
        else:
            raise FileNotFoundError(
                f'--load_model was requested, but no SRC checkpoint was found at {model_path}'
            )

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = PolynomialLR(optimizer, total_iters=200, power=0.5)

    criterion = pl_loss
    model_with_loss = ModelWithLoss(model, criterion)
    model_train = ModelWithOptimizer(model_with_loss, optimizer)
    
    all_mean_rewards, all_rewards = [], []
    skill_num, batch_size = args.skill_num, args.batch_size
    best_reward = -1e9

    print('-' * 20 + "Training Start" + '-' * 20)
    
    model_train.train()
    
    for epoch in range(args.num_epochs):
        avg_time = 0
        epoch_mean_rewards = []
        
        for i in tqdm(range(200)):
            t0 = time.perf_counter()
            targets, initial_logs, origin_path, _ = get_data(args, batch_size, data_file_path, args.path, args.steps)
            targets, initial_logs, origin_path = targets.to(device), initial_logs.to(device), origin_path.to(device)
            initial_log_scores = env.begin_episode(targets, initial_logs)
            data = (targets, initial_logs, initial_log_scores, origin_path, args.steps)
            with torch.no_grad():
                result = model(*data)
            env.n_step(result[0], binary=True)
            rewards = env.end_episode()
            loss_tensor = model_train(targets, initial_logs, initial_log_scores, origin_path, result[2].detach(), rewards)
            scheduler.step()
            loss_val = loss_tensor.item()
            mean_reward = torch.mean(rewards).item()
            avg_time += time.perf_counter() - t0
            epoch_mean_rewards.append(mean_reward)
            all_rewards.append(mean_reward)
            
            print(f'Epoch:{epoch}\tbatch:{i}\tavg_time:{avg_time/(i+1):.4f}\tloss:{loss_val:.4f}\treward:{mean_reward:.4f}')
        
        current_epoch_reward = np.mean(epoch_mean_rewards)
        all_mean_rewards.append(current_epoch_reward)
        
        if current_epoch_reward > best_reward:
            best_reward = current_epoch_reward
            torch.save(model.state_dict(), model_path)
            print("New Best Result Saved!")
            
        print(f"Best Reward Now:{best_reward:.4f}")

    if not os.path.exists(args.visual_dir):
        os.makedirs(args.visual_dir)
    np.save(os.path.join(args.visual_dir, f'{args.exp_name}_{args.path}'), np.array(all_rewards))




    kt_test_rewards = []
    dkt_para_dict = {
        'input_size': args.skill_num * 2,
        'emb_dim': 128,
        'hidden_size': 256,
        'num_skills': args.skill_num,
        'nlayers': 2, 
        'dropout': 0.0
    }
    kt_model = DKTnet(dkt_para_dict).to(device)
    kt_weight_path = f"data/{args.dataset}/env_weights/ValBest.ckpt"
    if os.path.exists(kt_weight_path):
        kt_model.load_state_dict(torch.load(kt_weight_path, map_location=device))
        print(f"Loaded Extra KT Model from {kt_weight_path}")
    else:
        raise FileNotFoundError(
            f'Required KT environment checkpoint was not found at {kt_weight_path}'
        )
    kt_model.eval()

    model.eval()
    model.load_state_dict(torch.load(model_path, map_location=device))
    batch_size = 200
    with torch.no_grad():
        targets, initial_logs, origin_path, initial_answers = get_data(args, batch_size, data_file_path, args.path, args.steps)
        targets, initial_logs, origin_path, initial_answers = targets.to(device), initial_logs.to(device), origin_path.to(device), initial_answers.to(device)
        initial_log_scores = env.begin_episode(targets, initial_logs)
        data = (targets, initial_logs, initial_log_scores, origin_path, args.steps)
        result = model(*data)
        env.n_step(result[0], binary=True)
        rewards = env.end_episode()
        
    paths = result[0].long()
    initial_state_preds = []
    final_state_preds = []
    padding = 0
    
    for b in range(batch_size):
        valid_mask = initial_logs[b] != padding
        valid_logs = initial_logs[b][valid_mask].long()
        valid_answers = initial_answers[b][valid_mask].long()
        
        valid_init_features = valid_logs + valid_answers * args.skill_num
        if len(valid_init_features) == 0:
            valid_init_features = torch.tensor([padding], device=device).long()
            
        init_feat = torch.nn.functional.one_hot(valid_init_features, num_classes=args.skill_num * 2).float().unsqueeze(0)
        init_pred = torch.sigmoid(kt_model(init_feat)[-1])
        initial_state_preds.append(init_pred)
        
        path = paths[b]
        path_prob = init_pred[0, path]
        correct_mask = (path_prob > 0.5).long()
        path_simulated = path + correct_mask * args.skill_num

        final_log = torch.cat([valid_init_features, path_simulated], dim=0)
        final_feat = torch.nn.functional.one_hot(final_log, num_classes=args.skill_num * 2).float().unsqueeze(0)
        final_pred = torch.sigmoid(kt_model(final_feat)[-1])
        final_state_preds.append(final_pred)

    initial_state_pred = torch.cat(initial_state_preds, dim=0) 
    final_state_pred = torch.cat(final_state_preds, dim=0)

    initial_scores, final_scores, full_scores = [], [], []
    
    for b in range(batch_size):
        cur_targets = targets[b]
        cur_targets = cur_targets[cur_targets != padding]
        cur_targets = torch.unique(cur_targets).long()
        
        init_p = initial_state_pred[b, cur_targets]
        fin_p = final_state_pred[b, cur_targets]
        initial_scores.append((init_p > 0.5).float().sum().item())
        final_scores.append((fin_p > 0.5).float().sum().item())
        full_scores.append(float(cur_targets.numel()))

    initial_score = torch.tensor(initial_scores, device=device)
    final_score = torch.tensor(final_scores, device=device)
    full_score = torch.tensor(full_scores, device=device)
    unmastered = full_score - initial_score
    raw_kt_reward = (final_score - initial_score) / (unmastered + 1e-9)
    kt_reward = torch.where(unmastered == 0, torch.tensor(0.0, device=device), raw_kt_reward)
    kt_test_rewards.append(torch.mean(kt_reward).item())
    
    print(f"Mean Extra KT Reward for Test: {np.mean(kt_test_rewards)}")


if __name__ == '__main__':
    parser = ArgumentParser("LearningPath-Planing")
    args_ = get_options(parser, {'agent': 'SRC', 'simulator': 'KES'})
    main(args_)
