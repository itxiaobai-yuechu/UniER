import pandas as pd
import ast

def generate_recommend_list(stu_ks, csv_uid_map, save_path="recommend.txt", topk=20):
    written_count = 0
    with open(save_path, 'w', encoding='utf-8') as f:
        for idx in range(len(stu_ks)):
            real_uid = csv_uid_map[idx]
            stu_kc_level = stu_ks[idx]
            
            rank_kc = [str(kc) for kc, v in sorted(stu_kc_level.items(), key=lambda item: item[1])[:topk]]
            kc_str = ",".join(rank_kc)
            
            f.write(f"{real_uid}\t{kc_str}\n")
            written_count += 1

    with open(save_path, 'r', encoding='utf-8') as f:
        file_lines = sum(1 for _ in f)
    

if __name__ == '__main__':
    dataset = 'xes3g5m'
    model = 'akt'

    test_file = f'./dataset/{dataset}/test_sequences.csv'
    stu_kc_file = f'./model/{model}/{dataset}/qid_test_predictions.txt'

    stu_ks = {}
    error_index = []
    total_txt_lines = 0

    with open(stu_kc_file, 'r') as f:
        for i, line in enumerate(f):
            total_txt_lines = i + 1
            try:
                stu_info = ast.literal_eval(line.strip())
            except:
                error_index.append(i)
                continue
            kcs = stu_info[2]
            kcs_predict = stu_info[4]
            kc_last_pre = {kc: pre for kc, pre in zip(kcs, kcs_predict)}
            stu_ks[i] = kc_last_pre

    test_data = pd.read_csv(test_file)
    test_data = test_data.reset_index(drop=True)
    
    csv_uid_map = test_data['uid'].to_dict()
    csv_total_lines = len(test_data)

    if error_index:
        valid_indices = [idx for idx in range(total_txt_lines) if idx not in error_index]
        stu_ks_aligned = {}
        for new_idx, old_idx in enumerate(valid_indices):
            stu_ks_aligned[new_idx] = stu_ks[old_idx]
        stu_ks = stu_ks_aligned


    save_file = f"./model/{model}/{dataset}/recommend.txt"
    generate_recommend_list(stu_ks, csv_uid_map, save_file, topk=20)