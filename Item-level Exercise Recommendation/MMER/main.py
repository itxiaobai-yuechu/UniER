import  argparse
import  torch, os
import  numpy as np
from mmer import MMAER
from dqn import *
from data_preprocess import *
from tqdm import tqdm
from common import *
import logging
import json
import csv
import random

def train_step(model, train_dataset, i, is_train, target_net, device, vars=None):
    hit, ndcg, F1, loss2 = model.get_main_loss(train_dataset, vars, i, is_train, target_net, device)
    return hit, loss2

def validation_step(model, validation_dataset, i, is_train, target_net, device, vars):
    hit, ndcg, F1, loss2 = model.get_main_loss(validation_dataset, vars, i, is_train, target_net, device)
    return hit, loss2

def tune_step(model, tune_dataset, i, device, is_train, target_net, vars):
    hit, ndcg, F1, loss2 = model.get_main_loss(tune_dataset, vars, i, is_train, target_net, device)
    return hit, loss2

def test_step(model, test_dataset, i, device, vars, target_net, is_train):
    return model.get_main_loss(test_dataset, vars, i, is_train, target_net, device)

def _sequence_to_text_line(seq, value_type='int'):
    values = []
    for x in seq:
        if torch.is_tensor(x):
            x = x.detach().cpu().item()
        if value_type == 'float':
            values.append(str(float(x)))
        else:
            values.append(str(int(x)))
    return ",".join(values)


def _write_sequence_file(path, sequences, value_type='int'):
    with open(path, 'w', encoding='utf-8') as f:
        for seq in sequences:
            f.write(_sequence_to_text_line(seq, value_type=value_type) + "\n")


def _write_sequence_file_with_ids(path, sequences, records, value_type='int'):

    with open(path, 'w', encoding='utf-8') as f:
        for seq, rec in zip(sequences, records):
            user_id = None
            if isinstance(rec, dict):
                user_id = rec.get('student_id')
                if user_id is None:
                    user_id = rec.get('origin_user_idx')
            seq_str = _sequence_to_text_line(seq, value_type=value_type)
            if user_id is None or user_id == '':
                f.write(seq_str + "\n")
            else:
                f.write(str(user_id) + " " + seq_str + "\n")


def _write_recommendation_mapping_files(args, score_true_stu_all, score_rec_topk_stu_all,
                                        score_rec_exer_topk_stu_all, score_rec_kc_topk_stu_all,
                                        score_rec_pred_topk_stu_all, recommendation_records_all):

    output_dir = args.file_dir
    os.makedirs(output_dir, exist_ok=True)

    _write_sequence_file(f"{output_dir}/score_true_stu_all.txt", score_true_stu_all, value_type='int')
    _write_sequence_file(f"{output_dir}/score_rec_topk_stu_all.txt", score_rec_topk_stu_all, value_type='int')
    _write_sequence_file_with_ids(f"{output_dir}/score_rec_exer_topk_stu_all.txt", score_rec_exer_topk_stu_all, recommendation_records_all, value_type='int')
    _write_sequence_file(f"{output_dir}/score_rec_kc_topk_stu_all.txt", score_rec_kc_topk_stu_all, value_type='int')
    _write_sequence_file(f"{output_dir}/score_rec_pred_topk_stu_all.txt", score_rec_pred_topk_stu_all, value_type='float')

    jsonl_path = f"{output_dir}/recommendation_mapping.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for record in recommendation_records_all:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    csv_path = f"{output_dir}/recommendation_mapping.csv"
    fieldnames = [
        'record_id', 'student_id', 'origin_user_idx', 'segment_start',
        'window_start', 'window_end', 'rank', 'local_index_in_valid_window',
        'segment_position', 'original_position', 'exercise_id',
        'knowledge_code', 'true_score', 'pred_score'
    ]
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record_id, record in enumerate(recommendation_records_all):
            n = len(record['ranked_exercise_ids'])
            for rank in range(n):
                writer.writerow({
                    'record_id': record_id,
                    'student_id': record.get('student_id'),
                    'origin_user_idx': record.get('origin_user_idx'),
                    'segment_start': record.get('segment_start'),
                    'window_start': record.get('window_start'),
                    'window_end': record.get('window_end'),
                    'rank': rank + 1,
                    'local_index_in_valid_window': record['ranked_local_indices_in_valid_window'][rank],
                    'segment_position': record['ranked_segment_positions'][rank],
                    'original_position': record['ranked_original_positions'][rank],
                    'exercise_id': record['ranked_exercise_ids'][rank],
                    'knowledge_code': record['ranked_knowledge_codes'][rank],
                    'true_score': record['ranked_true_scores'][rank],
                    'pred_score': record['ranked_pred_scores'][rank],
                })
    print(f"Saved recommendation mappings to {jsonl_path} and {csv_path}")


def cal_loss(args, loss2):
    return loss2

def update_target_net(args, model, target_net):
    for name, para in target_net.named_parameters():
        for name_model, para_model in model.named_parameters():
            if name_model == 'online_net.' + name:
                para.data.copy_(para.data * (1 - args.tau) + args.tau * para_model.data)


def test(args, model, tune_dataset, test_dataset, epoch, target_net, opt, device):
    tune_ERset = ER_Dataset(tune_dataset)
    test_ERset = ER_Dataset(test_dataset)
    tune_batch_loader = data.DataLoader(tune_ERset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_batch_loader = data.DataLoader(test_ERset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    print(" fine_tuning")
    for id_batch in tqdm(range(len(tune_batch_loader))):
        tune_batch_data = list(tune_batch_loader)[id_batch]
        if args.fine_tuning_ratio != 0:
            for e in range(args.tune_step):
                hit_tune, loss2 = tune_step(model, tune_batch_data, epoch, device, is_train=True, target_net=target_net, vars=None)
                lossa = cal_loss(args, loss2)
                opt.zero_grad()
                lossa.backward()
                opt.step()
    print(" testing")
    score_true_stu_all, score_rec_topk_stu_all = [], []
    score_rec_exer_topk_stu_all, score_rec_kc_topk_stu_all = [], []
    score_rec_pred_topk_stu_all, recommendation_records_all = [], []
    for id_batch in tqdm(range(len(test_batch_loader))):
        test_batch_data = list(test_batch_loader)[id_batch]
        (score_true_stu, score_rec_topk_stu, score_rec_exer_topk_stu,
         score_rec_kc_topk_stu, score_rec_pred_topk_stu,
         recommendation_records) = test_step(model, test_batch_data, epoch, device, vars=None, target_net=target_net, is_train=False)
        score_true_stu_all.extend(score_true_stu)
        score_rec_topk_stu_all.extend(score_rec_topk_stu)
        score_rec_exer_topk_stu_all.extend(score_rec_exer_topk_stu)
        score_rec_kc_topk_stu_all.extend(score_rec_kc_topk_stu)
        score_rec_pred_topk_stu_all.extend(score_rec_pred_topk_stu)
        recommendation_records_all.extend(recommendation_records)

    NDCG_1, hit_1, F1_1, MAP_1, MRR_1 = model.calculate_matrix(score_true_stu_all, score_rec_topk_stu_all, k=1)
    NDCG_3, hit_3, F1_3, MAP_3, MRR_3 = model.calculate_matrix(score_true_stu_all, score_rec_topk_stu_all, k=3)
    NDCG_5, hit_5, F1_5, MAP_5, MRR_5 = model.calculate_matrix(score_true_stu_all, score_rec_topk_stu_all, k=5)
    print(f'@1	NDCG:{NDCG_1:.3f}	Hit:{hit_1:.3f}	F1:{F1_1:.3f}	MAP:{MAP_1:.3f}	MRR:{MRR_1:.3f}')
    print(f'@3	NDCG:{NDCG_3:.3f}	Hit:{hit_3:.3f}	F1:{F1_3:.3f}	MAP:{MAP_3:.3f}	MRR:{MRR_3:.3f}')
    print(f'@5	NDCG:{NDCG_5:.3f}	Hit:{hit_5:.3f}	F1:{F1_5:.3f}	MAP:{MAP_5:.3f}	MRR:{MRR_5:.3f}')
    return (NDCG_1, score_true_stu_all, score_rec_topk_stu_all,
            score_rec_exer_topk_stu_all, score_rec_kc_topk_stu_all,
            score_rec_pred_topk_stu_all, recommendation_records_all)


def train(args, model, target_net, max_batch_count, task_num, train_dataloader, val_dataloader, opt, tune_dataset, test_dataset, device):
    fw_lr = args.fw_lr
    ndcg_old = -1.0
    for e in range(args.epoch):
        print("epoch:/ ", e)
        fw_lr = fw_lr * (1. / (1. + args.fw_wd * e))
        for id_batch in tqdm(range(max_batch_count)):
            lossb_list = [0]
            lossb_list2 = [0]
            is_train = True
            for task_id, (task_train_loader, task_valid_loader) in enumerate(zip(train_dataloader, val_dataloader)):
                if id_batch > len(list(task_train_loader)) - 1:
                    continue
                traindata = list(task_train_loader)[id_batch]
                validdata = list(task_valid_loader)[id_batch]
                hit_train, loss2 = train_step(model, traindata, e, is_train, target_net, device)
                lossa = cal_loss(args, loss2)
                grad = torch.autograd.grad(lossa, list(model.parameters()))
                fast_weights = list(map(lambda p: p[1] - fw_lr * p[0], zip(grad, list(model.parameters()))))

                for k in range(1, args.update_step):
                    hit_train, loss2 = train_step(model, traindata, e, is_train, target_net, device, fast_weights)
                    lossa = cal_loss(args, loss2)
                    grad = torch.autograd.grad(lossa, fast_weights)
                    fast_weights = list(map(lambda p: p[1] - fw_lr * p[0],
                                            zip(grad, fast_weights)))

                    hit_valid, loss2 = validation_step(model, validdata, e, is_train, target_net, device, fast_weights)
                    if k == args.update_step - 1:
                        lossb = cal_loss(args, loss2)
                        lossb_list[-1] += lossb

            opt.zero_grad()
            loss_res = lossb_list[-1] / len(args.train_name)
            loss_res.backward()
            opt.step()

        (NDCG1, score_true_stu_all, score_rec_topk_stu_all,
         score_rec_exer_topk_stu_all, score_rec_kc_topk_stu_all,
         score_rec_pred_topk_stu_all, recommendation_records_all) = test(
            args, model, tune_dataset, test_dataset, e, target_net, opt, device
        )
        if NDCG1 > ndcg_old:
            ndcg_old = NDCG1
            update_target_net(args, model, target_net)
            torch.save(
                {
                    'epoch': e,
                    'ndcg_at_1': float(NDCG1),
                    'model_state_dict': model.state_dict(),
                    'target_state_dict': target_net.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                },
                os.path.join(args.file_dir, 'best_checkpoint.pt'),
            )
            _write_recommendation_mapping_files(
                args, score_true_stu_all, score_rec_topk_stu_all,
                score_rec_exer_topk_stu_all, score_rec_kc_topk_stu_all,
                score_rec_pred_topk_stu_all, recommendation_records_all
            )


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(args)

    device = torch.device(args.cuda)
    model = MMAER(args).to(device)
    target_net = DQN(2 * args.cpt_num, args.cpt_num, args).to(device)
    max_batch_count, task_num, train_dataloader, val_dataloader, tune_dataset, test_dataset = data_preprocess(args)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )
    max_batch_count = min(len(val_dataloader[0]), len(train_dataloader[0]))
    train(args, model, target_net, max_batch_count, task_num, train_dataloader, val_dataloader, opt, tune_dataset, test_dataset, device)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--file_dir', type=str, help='data_file_dir', default='datasets/assist2017')
    argparser.add_argument('--train_name', type=list, help='list_for_train_data', default=['train_valid.json'])
    argparser.add_argument('--test_name', type=list, help='list_for_test_data', default=['test.json'])
    argparser.add_argument('--cpt_num', type=int, default=112)
    argparser.add_argument('--min_log', type=int, default=5)
    argparser.add_argument('--epoch', type=int, help='epoch number', default=300)
    argparser.add_argument('--init_type', type=str, help='init_type', default='xavier_uniform')
    argparser.add_argument('--batch_size', type=int, help='batch_size', default=256)
    argparser.add_argument('--hidden_size', type=int, help='hidden_size', default=200)
    argparser.add_argument('--dropout_rate', type=float, help='drop_out', default=0.0)
    argparser.add_argument('--lr', type=float, help='lr', default=0.0001)
    argparser.add_argument('--wd', type=float, help='wd', default=0.0)
    argparser.add_argument('--fw_lr', type=float, help='fw_lr', default=0.01)
    argparser.add_argument('--fw_wd', type=float, help='fw_wd', default=0.1)
    argparser.add_argument('--alpha', type=float, help='alpha', default=0)
    argparser.add_argument('--beta', type=float, help='beta', default=1)
    argparser.add_argument('--update_step', type=int, help='update_step', default=5)
    argparser.add_argument('--tune_step', type=int, help='tune_step', default=1)
    argparser.add_argument('--tau', type=float, help='threshold', default=0.0001)
    argparser.add_argument('--gamma', type=float, help='discount factor for target values', default=0.9)
    argparser.add_argument('--action_num', type=int, default=2)
    argparser.add_argument('--window_step', type=int, help='window_step', default=20)
    argparser.add_argument('--window_step_for_reward', type=int, help='window_step', default=20)
    argparser.add_argument('--threshold', type=float, help='threshold', default=0.6)
    argparser.add_argument('--num_param', type=int, help='threshold', default=10)
    argparser.add_argument('--meta_training_flag', type=int, default=1)
    argparser.add_argument('--training_ratio', type=float, default=0.5)
    argparser.add_argument('--fine_tuning_ratio', type=float, default=0.2)
    argparser.add_argument('--train_stu_num', type=int, default=99999999)
    argparser.add_argument('--test_stu_num', type=int, default=99999999)
    argparser.add_argument('--cuda', type=str, default='cuda:0')
    args = argparser.parse_args()
    with open('args.json', 'w') as f:
        json.dump(vars(args), f)
    main(args)
