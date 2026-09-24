import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from module.rapid_self import RAPID
from module.data_load_1 import Rapid_Dataset
import psutil
import time
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='assist2017')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--user_num', type=int, default=1855, help='the number of users')
    parser.add_argument('--ex_num', type=int, default=948, help='the number of exercises')
    parser.add_argument('--kc_num', type=int, default=57, help='the number of concepts')

    parser.add_argument('--user_hidden_size',type=int, default=64, help='the size of user embedding')
    parser.add_argument('--ex_hidden_size', type=int, default=64, help='the size of exercise embedding')
    parser.add_argument('--LSTM_hidden_size', type=int, default=64, help='the size of LSTM hidden state')
    parser.add_argument('--dropout', type=float, default=0.01, help='the dropout rate')
    parser.add_argument('--batch_size', type=int, default=12, help='the batch size')
    parser.add_argument('--epochs', type=int, default=50, help='the number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='the learning rate')
    parser.add_argument('--reg_lambda', type=float, default=1e-5, help='the regularization parameter')
    parser.add_argument('--init_rank_len', type=int, default=150, help='the length of initial ranking list')
    parser.add_argument('--eval_step', type=int, default=1, help='the step of evaluation')
    parser.add_argument('--output_type', type=str, default='det', choices=['det', 'pro'])
    parser.add_argument('--train_data_path', type=str, default=os.path.join(PROJECT_ROOT, 'dataset/data_200/assist2017/train_valid_sequences.csv'))
    parser.add_argument('--Q_matrix_file', type=str, default=os.path.join(PROJECT_ROOT, 'dataset/data_200/assist2017/Q.npy'))
    parser.add_argument('--test_data_path', type=str, default=os.path.join(PROJECT_ROOT, 'dataset/data_200/assist2017/test_sequences.csv'))
    parser.add_argument('--EB_save_path', type=str, default=os.path.join(PROJECT_ROOT, 'dataset/data_200/assist2017/EB_mlstm3_del_0.7.txt'))
    parser.add_argument('--Reranking_result_save_path', type=str, default=os.path.join(PROJECT_ROOT, 'dataset/data_200/assist2017/Reranking_result_melt.txt'))
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()
    return args

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_name = os.environ.get('UNIER_DATASET', args.dataset)
    requested_device = os.environ.get('UNIER_DEVICE', 'cuda:0')
    if requested_device.startswith('cuda') and not torch.cuda.is_available():
        requested_device = 'cpu'
    device = torch.device(requested_device)
    args.device = device

    data_dir = os.path.join(PROJECT_ROOT, 'dataset', 'data_200', dataset_name)
    args.train_data_path = os.path.join(data_dir, 'train_valid_sequences.csv')
    args.Q_matrix_file = os.path.join(data_dir, 'Q.npy')
    args.test_data_path = os.path.join(data_dir, 'test_sequences.csv')
    args.EB_save_path = os.path.join(data_dir, 'EB_mlstm3_del_0.7.txt')
    args.Reranking_result_save_path = os.path.join(data_dir, 'Reranking_result_melt.txt')

    df_train = pd.read_csv(args.train_data_path)
    df_test = pd.read_csv(args.test_data_path)
    merged_df = pd.concat([df_train[['questions', 'concepts']], df_test[['questions', 'concepts']]], ignore_index=True)
    df_len = len(merged_df)
    que_set = set()
    kc_set = set()
    for i in range(df_len):
        que = [int(x) for x in merged_df['questions'][i].split(',')]
        kc = [int(x) for x in merged_df['concepts'][i].split(',')]
        que_set.update(que)
        kc_set.update(kc)
    args.user_num = df_len
    args.ex_num = max(que_set)+1
    args.kc_num = max(kc_set)+1

    dataset = Rapid_Dataset(args)
    test_data = pd.read_csv(args.test_data_path)


    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,drop_last=True)
    test_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = RAPID(args).to(device)
    print(f'train memory allocated: {torch.cuda.memory_allocated()}bytes')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg_lambda)
    Q_matrix = torch.tensor(np.load(args.Q_matrix_file)).to(device)
    Q_matrix = Q_matrix.cpu().numpy()


    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss
    torch.cuda.empty_cache()
    for epoch in range(args.epochs):
        start_time1 = time.time()
        print(f'train memory allocated: {torch.cuda.memory_allocated()}bytes')
        model.train()
        model.istrain = True
        train_loss = 0
        for batch_data in tqdm(train_dataloader):
            batch_data = [data.to(device) for data in batch_data]
            loss, _ = model.forward(batch_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print('Epoch: {}, Loss: {:.4f}'.format(epoch, train_loss / len(train_dataloader)))
        end_time1 = time.time()
        print(f'Epoch time: {end_time1 - start_time1}s')
        memory_after = process.memory_info().rss
        print(f'train memory used: {(memory_after - memory_before) / (1000 * 1000)}MB')
        print(f'train memory allocated: {torch.cuda.memory_allocated() / (1000 * 1000)}MB')

        memory_before1 = process.memory_info().rss
        torch.cuda.empty_cache()
        start_time2 = time.time()
        if epoch % args.eval_step == 0:
            print(f'train memory allocated: {torch.cuda.memory_allocated()}bytes')
            model.eval()
            model.istrain = False
            u_rerank_list = []
            for batch_data in tqdm(test_dataloader):
                batch_data = [data.to(device) for data in batch_data]
                _, u_rerank = model.forward(batch_data)
                u_rerank_list.append(u_rerank.cpu().numpy())

            u_rerank_list = np.concatenate(u_rerank_list, axis=0)
            metrics = model.evaluate(u_rerank_list, test_data, Q_matrix)
            end_time2 = time.time()
            print('hit:{}\nndcg:{}\nmap:{}\nmrr:{}\nprecision:{}\nrecall:{}\nf1:{}\ndiv:{}\nvalid_stu_num:{}'.format(metrics['hit'], metrics['ndcg'], metrics['map'], metrics['mrr'],
              metrics['precision'], metrics['recall'], metrics['f1'], metrics['div'], metrics['valid_stu_num']))
            print(f'Test time: {end_time2 - start_time2}')
            memory_after1 = process.memory_info().rss
            print(f'test memory used: {(memory_after1 - memory_before1) / (1000 * 1000)}MB')
            print(f'test memory allocated: {torch.cuda.memory_allocated() / (1000 * 1000)}MB')

            uid = test_data['uid'].tolist()
            output_lines = []
            for i in range(len(u_rerank_list)):
                rel = u_rerank_list[i]
                filter_q = ','.join([str(q) for q in rel])
                output_lines.append(f"{uid[i]}\t{filter_q}\n")

            output_dir = os.path.dirname(os.path.abspath(args.Reranking_result_save_path))
            os.makedirs(output_dir, exist_ok=True)
            stem, suffix = os.path.splitext(args.Reranking_result_save_path)
            epoch_output_path = f'{stem}_epoch_{epoch + 1}{suffix}'
            for output_path in (epoch_output_path, args.Reranking_result_save_path):
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.writelines(output_lines)
            print(f'Saved epoch recommendations to {epoch_output_path}; latest copy is {args.Reranking_result_save_path}.')


if __name__ == '__main__':
    args = parse_args()
    main(args)
