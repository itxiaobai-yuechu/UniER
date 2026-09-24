import csv
import sys
import os

def concat_csv_fields(file1_path, file2_path, output_path):
    csv.field_size_limit(sys.maxsize)
    keep_fields = ["uid", "questions", "concepts", "responses"]
    all_rows = []

    for file_path in [file1_path, file2_path]:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filtered_row = {field: row[field].strip() if row[field] else "" for field in keep_fields}
                all_rows.append(filtered_row)

    with open(output_path, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=keep_fields)
        writer.writeheader()
        writer.writerows(all_rows)

if __name__ == "__main__":
    dataset = os.environ.get('UNIER_DATASET', 'assist2017')
    FILE1 = f"../datasets/{dataset}/train_valid.csv"
    FILE2 = f"../datasets/{dataset}/test.csv"
    OUTPUT = f"../datasets/{dataset}/{dataset}.csv"
    concat_csv_fields(FILE1, FILE2, OUTPUT)
