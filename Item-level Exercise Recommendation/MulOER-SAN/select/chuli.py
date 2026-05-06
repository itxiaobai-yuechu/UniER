import os
import sys
import pandas as pd
import json

dataset="assist2017"
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir,  f'{dataset}/new_data0.txt')
map_save_path = os.path.join(base_dir,  f'{dataset}/question_map.json')
df = pd.read_csv(file_path, sep='\t')
orirow_counts = df['orirow'].value_counts()
valid_orirows = orirow_counts[orirow_counts > 3].index
df_filtered = df[df['orirow'].isin(valid_orirows)].copy()
df_filtered['orirow'] = pd.factorize(df_filtered['orirow'])[0]
df_sorted = df_filtered.sort_values(by='questions').reset_index(drop=True)
q_encoded, q_uniques = pd.factorize(df_sorted['questions'])
df_sorted['questions'] = pd.factorize(df_sorted['questions'])[0]

question_map = {int(old_id): int(new_id) for old_id, new_id in zip(q_uniques, range(len(q_uniques)))}
with open(map_save_path, 'w', encoding='utf-8') as f:
    json.dump(question_map, f, indent=4, ensure_ascii=False)

df_c = df_sorted.sort_values(by='concepts').reset_index(drop=True)
df_c['concepts'] = pd.factorize(df_c['concepts'])[0]
df = df_c.sort_values(by='orirow').reset_index(drop=True)
output_file_path = os.path.join(base_dir, f'{dataset}/new_data1.txt')
df.to_csv(output_file_path, sep='\t', index=False)