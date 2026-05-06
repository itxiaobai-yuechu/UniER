import pandas as pd
import os

dataset = 'nips34'

txt_path = f'./{dataset}/recommended_problems_final.txt'
csv_path = f'./{dataset}/full_data.csv'
output_path = f'./{dataset}/filtered_full_data.csv'

os.makedirs(f'./{dataset}', exist_ok=True)

target_uids = []
with open(txt_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        uid = line.split('\t')[0]
        target_uids.append(uid)

target_uids = set(target_uids)
print(f"Extracted number of users from recommendation list: {len(target_uids)}")

df = pd.read_csv(csv_path, dtype={'uid': str})

filtered_df = df[df['uid'].isin(target_uids)]

filtered_df.to_csv(output_path, index=False, encoding='utf-8')

print(f"Original full data user count: {df['uid'].nunique()}")
print(f"Filtered data user count: {filtered_df['uid'].nunique()}")
print(f"Original data total rows: {len(df)}")
print(f"Filtered data total rows: {len(filtered_df)}")
print(f"Filtering completed! File saved to: {output_path}")