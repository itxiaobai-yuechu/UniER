import os
import sys
import json
import random
import pandas as pd
from longling import wf_open
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from EduSim.utils import get_proj_path

dataset = 'statics2011'
base_dir = f'{get_proj_path()}/dataProcess/{dataset}'

train_file = os.path.join(base_dir, 'train_valid.csv')
test_file = os.path.join(base_dir, 'test.csv')

df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)
data = pd.concat([df_train, df_test], ignore_index=True)

concepts_list = data['concepts'].apply(lambda x: x.split(","))
responses_list = data['responses'].apply(lambda x: x.split(","))

output_data = []
for concepts, responses in zip(concepts_list, responses_list):
    combined = [[int(concept), int(response)] for concept, response in zip(concepts, responses)]
    output_data.append(combined)

output_file = os.path.join(base_dir, 'student_log_kt_None')
with open(output_file, 'w', encoding='utf-8') as f:
    for session in output_data:
        f.write(json.dumps(session) + "\n")

random.shuffle(output_data)
split_idx = len(output_data) // 2

dataOff_path = os.path.join(base_dir, 'dataOff')
dataRec_path = os.path.join(base_dir, 'dataRec')

with wf_open(dataOff_path) as wf1, wf_open(dataRec_path) as wf2:
    for i, session in tqdm(enumerate(output_data), 'Splitting data...', total=len(output_data)):
        if i <= split_idx:
            print(json.dumps(session), file=wf1)
        else:
            print(json.dumps(session), file=wf2)

print('Data split completed.')