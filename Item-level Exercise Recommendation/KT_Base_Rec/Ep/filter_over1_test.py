import pandas as pd
import os

dataset = os.environ.get("UNIER_DATASET", "assist2017")
INPUT_CSV = f"../dataset/{dataset}/test_sequences.csv"
OUTPUT_CSV = f"../dataset/{dataset}/test_sequences_filter.csv"

QUESTION_COL = "questions"
USER_ID_COL = "uid"

def filter_real_sequences():
    df = pd.read_csv(INPUT_CSV)

    def calculate_real_length(seq):
        if pd.isna(seq) or str(seq).strip() == "":
            return 0
        seq_list = str(seq).split(",")
        real_items = [item.strip() for item in seq_list if item.strip() != "-1"]
        return len(real_items)

    df["real_length"] = df[QUESTION_COL].apply(calculate_real_length)

    df_filtered = df[df["real_length"] > 1].copy()

    df_filtered = df_filtered.drop(columns=["real_length"])
    df_filtered.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

if __name__ == "__main__":
    filter_real_sequences()
