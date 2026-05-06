import pandas as pd

dataset = 'xes3g5m'
test20_path = f"../../data/{dataset}/Test20.csv"  
output_path = f"../../data/{dataset}/Test20_unique_last.csv" 

df = pd.read_csv(test20_path)

df_unique_last = df.groupby('uid', as_index=False).tail(1)

df_unique_last = df_unique_last.reset_index(drop=True)

df_unique_last.to_csv(output_path, index=False, encoding="utf-8")

