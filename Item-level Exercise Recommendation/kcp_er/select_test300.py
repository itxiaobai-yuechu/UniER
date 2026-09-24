import argparse
import os
import random

import numpy as np
import pandas as pd
import torch


KC_NUM = {
    'assist2009': 123, 'nips34': 57, 'assist2012': 265,
    'assist2017': 102, 'algebra2005': 112, 'bridge2006': 493,
    'ednet': 188, 'junyi': 39, 'xes3g5m': 865,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=os.environ.get('UNIER_DATASET', 'assist2017'))
    parser.add_argument('--seed', type=int, default=int(os.environ.get('UNIER_SEED', '42')))
    parser.add_argument('--sample_size', type=int, default=None,
                        help='number of rows to select; default keeps all rows in shuffled order')
    parser.add_argument('--input_dir', default=None,
                        help='directory containing original test_sequences.csv, pkc.pth and pkm.pth')
    parser.add_argument('--output_dir', default=None,
                        help='separate output directory; defaults to <input_dir>/selected')
    return parser.parse_args()


def main(args):
    if args.dataset not in KC_NUM:
        raise ValueError(f'Unknown dataset {args.dataset!r}; no KC count is configured.')

    input_dir = args.input_dir or os.path.join('datasets', args.dataset, 'select')
    output_dir = args.output_dir or os.path.join(input_dir, 'selected')
    os.makedirs(output_dir, exist_ok=True)

    test_path = os.path.join(input_dir, 'test_sequences.csv')
    pkc_path = os.path.join(input_dir, 'pkc.pth')
    pkm_path = os.path.join(input_dir, 'pkm.pth')
    test_raw = pd.read_csv(test_path)
    pkc = torch.load(pkc_path, map_location='cpu')
    pkm = torch.load(pkm_path, map_location='cpu')

    if len(test_raw) != len(pkc) or len(test_raw) != len(pkm):
        raise ValueError(
            f'Input row mismatch: CSV={len(test_raw)}, pkc={len(pkc)}, pkm={len(pkm)}.'
        )

    sample_size = len(test_raw) if args.sample_size is None else args.sample_size
    if sample_size < 1 or sample_size > len(test_raw):
        raise ValueError(f'sample_size must be in [1, {len(test_raw)}], got {sample_size}.')

    rng = random.Random(args.seed)
    select_index = np.asarray(rng.sample(range(len(test_raw)), sample_size), dtype=np.int64)
    selected_test = test_raw.iloc[select_index].reset_index(drop=True)
    selected_pkc = pkc[select_index.tolist()]
    selected_pkm = pkm[select_index.tolist()]

    selected_test.to_csv(os.path.join(output_dir, 'test_sequences.csv'), index=False)
    torch.save(selected_pkc, os.path.join(output_dir, 'pkc.pth'))
    torch.save(selected_pkm, os.path.join(output_dir, 'pkm.pth'))
    np.save(os.path.join(output_dir, 'selected_indices.npy'), select_index)

    q_data = selected_test[['questions', 'concepts']].copy()
    q_data['questions'] = q_data['questions'].apply(
        lambda value: ','.join(map(str.strip, str(value).split(',')))
    )
    q_data['concepts'] = q_data['concepts'].apply(
        lambda value: ','.join(map(str.strip, str(value).split(',')))
    )
    all_questions = [
        int(q) for q_list in q_data['questions'].values
        for q in q_list.split(',') if int(q) != -1
    ]
    if not all_questions:
        raise ValueError('No non-padding question ids were found in the selected data.')

    Q = np.zeros((max(all_questions) + 1, KC_NUM[args.dataset]))
    for _, row in q_data.iterrows():
        questions = row['questions'].split(',')
        concepts = row['concepts'].split(',')
        for question, concept in zip(questions, concepts):
            qid, kid = int(question), int(concept)
            if qid == -1 or kid == -1:
                continue
            Q[qid, kid] = 1
    np.save(os.path.join(output_dir, 'Q.npy'), Q)
    print(f'Selected {sample_size} aligned rows into {output_dir}; original inputs were not overwritten.')


if __name__ == '__main__':
    main(parse_args())
