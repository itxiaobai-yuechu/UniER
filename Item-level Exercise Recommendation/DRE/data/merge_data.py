import pandas as pd
import os

dataset = 'nips34'

train_path = f'./{dataset}/train_valid.csv'
test_path = f'./{dataset}/test.csv'
output_path = f'./{dataset}/full_data.csv'

os.makedirs(f'./{dataset}', exist_ok=True)

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

df_combined = pd.concat([df_train, df_test], ignore_index=True)

df_combined.to_csv(output_path, index=False, encoding='utf-8')

