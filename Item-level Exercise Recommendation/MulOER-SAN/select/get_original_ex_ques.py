import os
import json

dataset = os.environ.get("UNIER_DATASET", "assist2017")
base_dir = os.path.dirname(os.path.abspath(__file__))
map_path = os.path.join(base_dir, f"{dataset}/question_map.json")
input_ex_path = os.path.join(base_dir, f"{dataset}/new_ex.txt")
output_ex_path = os.path.join(base_dir, f"{dataset}/new_ex_original.txt")

with open(map_path, 'r', encoding='utf-8') as f:
    question_map = json.load(f)
reverse_map = {str(new_id): str(old_id) for old_id, new_id in question_map.items()}

result = []
with open(input_ex_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        student_id, ex_list_str = line.split('\t')
        ex_encoded = ex_list_str.split(',')
        ex_original = [reverse_map[ex_id] for ex_id in ex_encoded if ex_id in reverse_map]
        result.append(f"{student_id}\t{','.join(ex_original)}")

with open(output_ex_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(result))
