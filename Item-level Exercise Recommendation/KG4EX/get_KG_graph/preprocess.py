import pandas as pd
import os

dataset = os.environ.get('UNIER_DATASET', 'assist2017')

train_path = f'../data/{dataset}/train_valid_sequences.csv'
test_path = f'../data/{dataset}/test_sequences.csv'
output_path = f'../data/{dataset}/{dataset}.csv'

os.makedirs(f'../data/{dataset}', exist_ok=True)

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

df_combined = pd.concat([df_train, df_test], ignore_index=True)

df_combined.to_csv(output_path, index=False, encoding='utf-8')

