import os
import pandas as pd
import ast
import argparse

def generate_recommend_list(predictions_by_row, csv_uid_map, save_path="recommend.txt", topk=20):
    """Write weak-KC recommendations without losing the source CSV row id."""
    written_count = 0
    with open(save_path, 'w', encoding='utf-8') as f:
        for source_row in sorted(predictions_by_row):
            if source_row not in csv_uid_map:
                continue
            kc_scores = predictions_by_row[source_row]
            ranked_kcs = [kc for kc, _ in sorted(kc_scores.items(), key=lambda item: item[1])[:topk]]
            if not ranked_kcs:
                continue
            uid = csv_uid_map[source_row]
            f.write(f"{uid}\t{','.join(str(kc) for kc in ranked_kcs)}\n")
            written_count += 1
    print(f'Wrote {written_count} aligned recommendation line(s) to {save_path}.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='assist2017')
    parser.add_argument('--model', default='akt', choices=['akt', 'simplekt'])
    parser.add_argument('--test_file', default=None)
    parser.add_argument('--prediction_file', default=None)
    parser.add_argument('--output_file', default=None)
    args = parser.parse_args()

    dataset = os.environ.get('UNIER_DATASET', 'assist2017')
    model = {'AKT': 'akt', 'SimpleKT': 'simplekt'}.get(os.environ.get('UNIER_MODEL', 'AKT'), 'akt')

    test_file = args.test_file or f'./dataset/{dataset}/test_sequences.csv'
    stu_kc_file = args.prediction_file or f'./model/{model}/{dataset}/qid_test_predictions.txt'

    predictions_by_row = {}
    error_index = []

    with open(stu_kc_file, 'r') as f:
        for i, line in enumerate(f):
            try:
                stu_info = ast.literal_eval(line.strip())
            except (ValueError, SyntaxError):
                error_index.append(i)
                continue
            kcs = stu_info[2]
            kcs_predict = stu_info[4]
            kc_last_pre = {kc: pre for kc, pre in zip(kcs, kcs_predict)}
            predictions_by_row[i] = kc_last_pre

    test_data = pd.read_csv(test_file)
    test_data = test_data.reset_index(drop=True)
    
    csv_uid_map = test_data['uid'].to_dict()
    extra_rows = [i for i in predictions_by_row if i >= len(test_data)]
    if error_index:
        print(f'Warning: skipped {len(error_index)} malformed prediction line(s): {error_index[:10]}')
    if extra_rows:
        print(f'Warning: ignored {len(extra_rows)} prediction line(s) beyond the test CSV.')
        for i in extra_rows:
            predictions_by_row.pop(i)


    save_file = args.output_file or f"./model/{model}/{dataset}/recommend.txt"
    generate_recommend_list(predictions_by_row, csv_uid_map, save_file, topk=20)
