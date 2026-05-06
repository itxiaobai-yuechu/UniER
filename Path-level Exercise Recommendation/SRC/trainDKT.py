import os
import time
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from argparse import ArgumentParser

from KTScripts.BackModels import nll_loss
from KTScripts.DataLoader import KTDataset, RecDataset, RetrievalDataset
from KTScripts.PredictModel import ModelWithLoss, ModelWithLossMask
from KTScripts.utils import set_random_seed, load_model, evaluate_utils
from KTScripts.options import get_options




def main(args: Namespace):
    print()
    set_random_seed(args.rand_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = RecDataset if args.forRec else (RetrievalDataset if args.retrieval else KTDataset)
    dataset = dataset(os.path.join(args.data_dir, args.dataset))
    args.feat_nums, args.user_nums = dataset.feats_num, dataset.users_num

    if args.forRec:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    elif args.retrieval:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    total_size = len(dataloader.dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataloader.dataset, [train_size, test_size])

    train_data = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_data = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    train_total = len(train_data)
    test_total = len(test_data)

    if args.forRec:
        args.output_size = args.feat_nums

    model = load_model(args).to(device)
    model_path = os.path.join(args.save_dir, args.exp_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.load_model:
        model.load_state_dict(torch.load(f'{model_path}.ckpt'))
        print(f"Load Model From {model_path}.ckpt")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
    scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=train_total//10+1, power=0.5)

    if args.forRec:
        criterion = ModelWithLossMask(model, nll_loss)
    else:
        criterion = ModelWithLoss(model, nn.BCELoss(reduction='mean'))

    best_val_auc = 0
    print('-' * 20 + "Training Start" + '-' * 20)

    for epoch in range(args.num_epochs):
        avg_time = 0
        model.train()

        for i, data in tqdm(enumerate(train_data), total=train_total):
            t0 = time.perf_counter()
            data = [d.to(device) for d in data]
            optimizer.zero_grad()
            loss, output_data = criterion(*data)
            loss.backward()
            optimizer.step()

            acc, auc = evaluate_utils(*output_data)
            avg_time += time.perf_counter() - t0

            print('Epoch:{}\tbatch:{}\tavg_time:{:.4f}\tloss:{:.4f}\tacc:{:.4f}\tauc:{:.4f}'
                  .format(epoch, i, avg_time / (i + 1), loss.item(), acc, auc))

        scheduler.step()
        print('-' * 20 + "Validating Start" + '-' * 20)

        val_eval = [[], []]
        loss_total, data_total = 0, 0
        model.eval()

        with torch.no_grad():
            for data in tqdm(test_data, total=test_total):
                data = [d.to(device) for d in data]
                loss, output_data = criterion.output(*data)
                val_eval[0].append(output_data[0].detach().cpu().numpy())
                val_eval[1].append(output_data[1].detach().cpu().numpy())
                loss_total += loss.item() * len(data[0])
                data_total += len(data[0])

        val_eval = [np.concatenate(_) for _ in val_eval]
        acc, auc = evaluate_utils(*val_eval)
        print(f"Validating loss:{loss_total / data_total:.4f} acc:{acc:.4f} auc:{auc:.4f}")

        if auc >= best_val_auc:
            best_val_auc = auc
            torch.save(model.state_dict(), f'{model_path}.ckpt')
            print("New best result Saved!")
        print(f"Best Auc Now:{best_val_auc:.4f}")

    print('-' * 20 + "Testing Start" + '-' * 20)
    val_eval = [[], []]
    loss_total, data_total = 0, 0
    model.eval()

    with torch.no_grad():
        for data in tqdm(test_data, total=test_total):
            data = [d.to(device) for d in data]
            loss, output_data = criterion.output(*data)
            val_eval[0].append(output_data[0].detach().cpu().numpy())
            val_eval[1].append(output_data[1].detach().cpu().numpy())
            loss_total += loss.item() * len(data[0])
            data_total += len(data[0])

    val_eval = [np.concatenate(_) for _ in val_eval]
    print(val_eval[0], val_eval[0].mean())
    print(val_eval[1], val_eval[1].mean())
    acc, auc = evaluate_utils(*val_eval)
    print(f"Testing loss:{loss_total / data_total:.4f} acc:{acc:.4f} auc:{auc:.4f}")


if __name__ == '__main__':
    parser = ArgumentParser("LearningPath-Planing")
    args_ = get_options(parser)
    main(args_)