import os

dataset = 'assist2009'

file_path = f'../../fin_res/{dataset}/recommend.txt'



def format_output_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            new_lines.append("")
            continue
        new_line = line.replace(' ', '\t', 1)
        new_lines.append(new_line)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))
    

format_output_file(file_path)
