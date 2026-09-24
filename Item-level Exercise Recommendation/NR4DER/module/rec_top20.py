import os

dataset = os.environ.get('UNIER_DATASET', 'assist2017')
input_file = f'../dataset/data_200/{dataset}/Reranking_result_melt.txt'
output_file = f'../dataset/data_200/{dataset}/Reranking_result_melt_20.txt'
with open(input_file, 'r') as file:
    output_data = []
    for line in file:
        parts = line.strip().split('\t')
        uid = parts[0]
        numbers = parts[1].split(',')
        numbers = numbers[:20]
        numbers_str = ','.join(numbers)
        output_data.append(f"{uid}\t{numbers_str}")
with open(output_file, 'w') as f:
    for line in output_data:
        f.write(line + '\n')
