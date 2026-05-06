import pandas as pd
import os

dataset = 'assist2012'

txt_path = f'../../fin_res/{dataset}/recommend.txt'
csv_path = f'../../datasets/{dataset}/{dataset}.csv'
output_path = f'../../datasets/{dataset}/{dataset}_filter.csv'

os.makedirs(f'./{dataset}', exist_ok=True)

target_uids = []
with open(txt_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        uid = line.split(' ')[0]
        target_uids.append(uid)

target_uids = set(target_uids)


df = pd.read_csv(csv_path, dtype={'uid': str})

filtered_df = df[df['uid'].isin(target_uids)]

filtered_df.to_csv(output_path, index=False, encoding='utf-8')

