dataset = "xes3g5m"

input_file = f"../../data/{dataset}/ex_rec_uid.txt"
output_file = f"../../data/{dataset}/Test20/recommend_20.txt"
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

