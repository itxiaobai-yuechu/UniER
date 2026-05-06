import pandas as pd

dataset="xes3g5m"

df = pd.read_csv(f'../PYKT/data/{dataset}/qid_test_question_predictions.txt', sep='\t')

new_rows = []

for _, row in df.iterrows():
    questions = str(row['questions']).split(',')
    concepts = str(row['concepts']).split(',')
    preds = str(row['concept_preds']).split(',')

    if len(questions) == len(concepts) == len(preds) and len(questions) > 1:
        for q, c, p in zip(questions, concepts, preds):
            new_row = row.copy()
            new_row['questions'] = int(q)
            new_row['concepts'] = int(c)
            new_row['concept_preds'] = float(p)
            new_rows.append(new_row)
    else:
        new_rows.append(row)

new_df = pd.DataFrame(new_rows)

new_df.to_csv(f'../PYKT/data/{dataset}/qid_test_question_predictions1.txt', sep='\t', index=False)