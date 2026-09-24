from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


def prepare_src(dataset: str, seed: int, project_root: Optional[Path] = None) -> None:
    """Create SRC's dataOff/dataRec files using the original path-model split logic."""

    base_dir = (project_root or Path.cwd()) / "data" / dataset
    train = pd.read_csv(base_dir / "train_valid.csv")
    test = pd.read_csv(base_dir / "test.csv")
    data = pd.concat([train, test], ignore_index=True)

    output_data = []
    for concepts, responses in zip(data["concepts"], data["responses"]):
        pairs = [[int(concept), int(response)] for concept, response in zip(concepts.split(","), responses.split(","))]
        output_data.append(pairs)

    # The source implementations use random.shuffle; seeding keeps unified runs reproducible.
    random.Random(seed).shuffle(output_data)
    split_idx = len(output_data) // 2
    data_off = base_dir / "dataOff"
    data_rec = base_dir / "dataRec"
    for output_path, rows in ((data_off, output_data[: split_idx + 1]), (data_rec, output_data[split_idx + 1 :])):
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    print(f"SRC data split written to {base_dir}")


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args != ["src"]:
        raise SystemExit("usage: python -m Unier_cli.path_preprocess src")
    dataset = os.environ.get("UNIER_PATH_DATASET", "assist17")
    seed = int(os.environ.get("UNIER_SEED", "42"))
    prepare_src(dataset, seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
