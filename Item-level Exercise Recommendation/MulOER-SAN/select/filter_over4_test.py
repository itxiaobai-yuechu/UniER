import pandas as pd
import os

dataset = os.environ.get("UNIER_DATASET", "assist2017")
INPUT_FILE_PATH = f"../PYKT/data/{dataset}/test.csv"
OUTPUT_FILE_PATH = f"../PYKT/data/{dataset}/test_gt4.csv"
ANSWER_SEQ_COL = "questions"
STUDENT_ID_COL = "uid"

def filter_students_by_answer_length(input_path, output_path, answer_seq_col, student_id_col):
    df = pd.read_csv(input_path)

    df["answer_length"] = df[answer_seq_col].apply(
        lambda x: len(str(x).split(",")) if pd.notna(x) and str(x).strip() != "" else 0
    )

    df_filtered = df[df["answer_length"] > 4].copy()

    df_filtered = df_filtered.drop(columns=["answer_length"])

    df_filtered.to_csv(output_path, index=False)
        
if __name__ == "__main__":
    filter_students_by_answer_length(
        input_path=INPUT_FILE_PATH,
        output_path=OUTPUT_FILE_PATH,
        answer_seq_col=ANSWER_SEQ_COL,
        student_id_col=STUDENT_ID_COL
    )
