from mmer import MMAER
import torch
import json
import os
from types import SimpleNamespace


if __name__ == '__main__':
    with open(f'args.json', 'r') as f:
        args = json.load(f)
        args = SimpleNamespace(**args)
        print(args)
    model = MMAER(args)
    
    true_stu_file = f"{args.file_dir}/score_true_stu_all.txt"
    rec_topk_file = f"{args.file_dir}/score_rec_topk_stu_all.txt"
    def load_tensor(file_path):
        tensors = []
        requested_device = os.environ.get('UNIER_DEVICE', 'cuda:0')
        if requested_device.startswith('cuda') and not torch.cuda.is_available():
            requested_device = 'cpu'
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    values = list(map(int, line.strip().split(',')))
                    tensors.append(torch.tensor(values, device=requested_device))
        return tensors
    score_true_stu_all = load_tensor(true_stu_file)
    score_rec_topk_stu_all = load_tensor(rec_topk_file)


    NDCG_1, hit_1, F1_1, MAP_1, MRR_1 = model.calculate_matrix(score_true_stu_all, score_rec_topk_stu_all, k=1)
    NDCG_3, hit_3, F1_3, MAP_3, MRR_3 = model.calculate_matrix(score_true_stu_all, score_rec_topk_stu_all, k=3)
    NDCG_5, hit_5, F1_5, MAP_5, MRR_5 = model.calculate_matrix(score_true_stu_all, score_rec_topk_stu_all, k=5)
    NDCG_10, hit_10, F1_10, MAP_10, MRR_10 = model.calculate_matrix(score_true_stu_all, score_rec_topk_stu_all, k=10)

    print(f'@1\tNDCG:{NDCG_1:.3f}\tHit:{hit_1:.3f}\tF1:{F1_1:.3f}\tMAP:{MAP_1:.3f}\tMRR:{MRR_1:.3f}')
    print(f'@3\tNDCG:{NDCG_3:.3f}\tHit:{hit_3:.3f}\tF1:{F1_3:.3f}\tMAP:{MAP_3:.3f}\tMRR:{MRR_3:.3f}')
    print(f'@5\tNDCG:{NDCG_5:.3f}\tHit:{hit_5:.3f}\tF1:{F1_5:.3f}\tMAP:{MAP_5:.3f}\tMRR:{MRR_5:.3f}')
    print(f'@10\tNDCG:{NDCG_10:.3f}\tHit:{hit_10:.3f}\tF1:{F1_10:.3f}\tMAP:{MAP_10:.3f}\tMRR:{MRR_10:.3f}') 
    
 


