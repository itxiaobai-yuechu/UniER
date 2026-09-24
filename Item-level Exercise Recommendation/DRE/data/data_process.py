import pandas as pd
import os

def generate_question_skill_mapping(dataset):
    test_file = f'./{dataset}/test.csv'

    train_file = f'./{dataset}/train_valid.csv'

    test_df = pd.read_csv(test_file)
    train_df = pd.read_csv(train_file)
    data_df = pd.concat([test_df, train_df], ignore_index=True)
    mapping = []
    for _, row in data_df.iterrows():
        questions = row['questions'].split(',')
        concepts = row['concepts'].split(',')
        if len(questions) != len(concepts):
            print(f"Warning: The number of questions and concepts in row {row.name} do not match!")
            continue
        for question, concept in zip(questions, concepts):
            mapping.append([question, concept])
    mapping_df = pd.DataFrame(mapping, columns=['problem_id', 'skill_id'])
    mapping_df = mapping_df.drop_duplicates()
    mapping_df.to_csv(f'./{dataset}/question_skill_mapping.csv', index=False)

    print(f"Mapping relationship saved as question_skill_mapping.csv")


def process_and_split_data(dataset):
    test_file = f'./{dataset}/test.csv'

    train_file = f'./{dataset}/train_valid.csv'
    test_df = pd.read_csv(test_file)
    train_df = pd.read_csv(train_file)
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    test_lines = []
    train_valid_lines = []
    for _, row in combined_df.iterrows():
        uid = str(row['uid'])
        questions = row['questions'].split(',')
        concepts = row['concepts'].split(',')
        responses = row['responses'].split(',')
        total_len = len(questions)
        split_idx = int(total_len * 0.7)
        test_q = questions[split_idx:]
        test_c = concepts[split_idx:]
        test_r = responses[split_idx:]
        test_uid = [uid] * len(test_q)
        test_lines.append(','.join(test_uid))
        test_lines.append(','.join(test_q))
        test_lines.append(','.join(test_c))
        test_lines.append(','.join(test_r))
        train_q = questions[:split_idx]
        train_c = concepts[:split_idx]
        train_r = responses[:split_idx]
        train_uid = [uid] * len(train_q)
        if len(train_q) > 0:
            train_valid_lines.append(','.join(train_uid))
            train_valid_lines.append(','.join(train_q))
            train_valid_lines.append(','.join(train_c))
            train_valid_lines.append(','.join(train_r))

    os.makedirs(f'./{dataset}', exist_ok=True)
    test_file_output = f'./{dataset}/test_processed.csv'
    train_file_output = f'./{dataset}/train_valid_processed.csv'
    with open(test_file_output, 'w') as f:
        for line in test_lines:
            f.write(line + '\n')
    with open(train_file_output, 'w') as f:
        for line in train_valid_lines:
            f.write(line + '\n')


def process_and_calculate_difficulty(dataset):
    test_file = f'./{dataset}/test.csv'

    train_file = f'./{dataset}/train_valid.csv'
    test_df = pd.read_csv(test_file)
    train_df = pd.read_csv(train_file)
    combined_df = pd.concat([test_df, train_df], ignore_index=True)
    question_stats = {}
    for _, row in combined_df.iterrows():
        questions = row['questions'].split(',')
        responses = row['responses'].split(',')
        for qid, response in zip(questions, responses):
            qid = int(qid)
            if qid not in question_stats:
                question_stats[qid] = {'total': 0, 'incorrect': 0}
            question_stats[qid]['total'] += 1
            if response == '0':
                question_stats[qid]['incorrect'] += 1
    difficulty_data = []
    for qid, stats in question_stats.items():
        error_rate = stats['incorrect'] / stats['total']
        difficulty_data.append([qid, error_rate])
    difficulty_df = pd.DataFrame(difficulty_data)
    difficulty_df = difficulty_df.sort_values(by=0)
    difficulty_file = f'./{dataset}/difficulty.csv'
    difficulty_df.to_csv(difficulty_file, index=False)
    print(f"Question difficulty data saved as {difficulty_file}")

def main():
    dataset = os.environ.get('UNIER_DATASET', 'assist2017')
    generate_question_skill_mapping(dataset)
    process_and_split_data(dataset)
    process_and_calculate_difficulty(dataset)

if __name__ == "__main__":
    main()
