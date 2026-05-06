import numpy as np
import ast

def process(dataset,padding, data_type):
    max_len = 200

    skills = []
    ys = []
    real_lens = []

    with open(f'./data/{dataset}/{data_type}', 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            student = ast.literal_eval(line)

            student = [item for item in student if isinstance(item, list) and len(item) == 2]

            concept_seq = [item[0] for item in student]
            response_seq = [item[1] for item in student]

            real_len = len(concept_seq)

            concept_seq = concept_seq + [padding] * (max_len - real_len)
            response_seq = response_seq + [padding] * (max_len - real_len)

            concept_seq = concept_seq[:max_len]
            response_seq = response_seq[:max_len]
            real_len = min(real_len, max_len)

            skills.append(concept_seq)
            ys.append(response_seq)
            real_lens.append(real_len)

    skills = np.array(skills, dtype=np.int32)
    ys = np.array(ys, dtype=np.int32)
    real_lens = np.array(real_lens, dtype=np.int32)

    np.savez(f'./data/{dataset}/{dataset}_{data_type}.npz', skill=skills, y=ys, real_len=real_lens)

if __name__ == '__main__':
    dataset = 'xes3g5m'
    padding = 0
    process(dataset, padding, 'dataRec')
    process(dataset, padding, 'dataOff')