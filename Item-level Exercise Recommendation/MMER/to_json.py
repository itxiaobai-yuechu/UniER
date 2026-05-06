import json
import pandas as pd
import numpy as np
import argparse

def csv_to_json(dataset, dataset_type):
    df = pd.read_csv(f'datasets/{dataset}/{dataset_type}.csv')
    df_len = len(df)
    data = []

    for i in range(df_len):
        stu_info = {}
        uid = df.iloc[i]['uid']
        questions = df.iloc[i]['questions'].split(',')
        responses = df.iloc[i]['responses'].split(',')
        concepts = df.iloc[i]['concepts'].split(',')

        stu_info['student_id'] = str(uid)
        stu_info['logs'] = []
        for que, res, kc in zip(questions, responses, concepts):
            stu_info['logs'].append({
                'exer_id': str(que),
                'score': str(res),
                'knowledge_code': str(kc)
            })
        data.append(stu_info)

    with open(f'datasets/{dataset}/{dataset_type}.json', 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    dataset="junyi"
    csv_to_json(dataset,"train_valid")
    csv_to_json(dataset,"test")