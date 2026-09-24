import pandas as pd

data = "assist2017"
print(f"Processing dataset: {data}")
data = pd.read_csv(f"./{data}/new_data1.txt", sep='\t')

kc_num = data['concepts'].nunique()
print(f"{kc_num}")