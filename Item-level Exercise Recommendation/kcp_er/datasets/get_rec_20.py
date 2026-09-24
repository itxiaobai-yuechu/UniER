import os

dataset = os.environ.get("UNIER_DATASET", "assist2017")

input_file = f"./{dataset}/select/selected/REL_generator_result_100.txt"
output_file = f"./{dataset}/select/selected/recommendation_20.txt"
KEEP_NUM = 20

with open(input_file, 'r', encoding='utf-8') as f_read, \
     open(output_file, 'w', encoding='utf-8') as f_write:

    for line in f_read:
        line = line.strip()
        if not line:
            continue
        
        user_id, exercise_str = line.split('\t')
        
        exercise_list = exercise_str.split(',')
        
        top20_exercise = exercise_list[:KEEP_NUM]
        
        new_line = f"{user_id}\t{','.join(top20_exercise)}"
        
        f_write.write(new_line + '\n')
