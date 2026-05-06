import math
import os

dataset = 'nips34'

input_file = f'./{dataset}/recommended_problems.txt'
output_file = f'./{dataset}/recommended_problems_final.txt'

with open(input_file, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

total_lines = len(lines)
top_quarter_lines = math.floor(total_lines / 4)

selected_lines = lines[:top_quarter_lines]

with open(output_file, 'w', encoding='utf-8') as f:
    for line in selected_lines:
        f.write(line + '\n')
