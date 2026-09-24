from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict

import pandas as pd


def dre_postprocess(repo_root: Path, dataset: str) -> Dict[str, Path]:
    """Run the three former DRE post-processing scripts in their original order."""
    data_dir = repo_root / "Item-level Exercise Recommendation" / "DRE" / "data" / dataset
    recommendation = data_dir / "recommended_problems.txt"
    final_recommendation = data_dir / "recommended_problems_final.txt"
    train_path = data_dir / "train_valid.csv"
    test_path = data_dir / "test.csv"
    full_data = data_dir / "full_data.csv"
    filtered_data = data_dir / "filtered_full_data.csv"

    missing = [path for path in (recommendation, train_path, test_path) if not path.exists()]
    if missing:
        rendered = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"DRE post-processing inputs are missing:\n{rendered}")

    lines = [line.strip() for line in recommendation.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = lines[: math.floor(len(lines) / 4)]
    final_recommendation.write_text(
        "".join(f"{line}\n" for line in selected),
        encoding="utf-8",
    )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    combined = pd.concat([train, test], ignore_index=True)
    combined.to_csv(full_data, index=False, encoding="utf-8")

    target_uids = {line.split("\t", 1)[0] for line in selected}
    combined_as_text = pd.read_csv(full_data, dtype={"uid": str})
    filtered = combined_as_text[combined_as_text["uid"].isin(target_uids)]
    filtered.to_csv(filtered_data, index=False, encoding="utf-8")

    print(f"DRE selected recommendation users: {len(target_uids)}")
    print(f"DRE combined rows: {len(combined_as_text)}")
    print(f"DRE filtered rows: {len(filtered)}")
    return {
        "recommended_problems_final": final_recommendation,
        "full_data": full_data,
        "filtered_full_data": filtered_data,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UniER model-specific post-processing")
    subparsers = parser.add_subparsers(dest="command", required=True)
    dre = subparsers.add_parser("dre", help="Run DRE recommendation/data post-processing")
    dre.add_argument("--dataset", default="assist2017")
    dre.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "dre":
        dre_postprocess(args.repo_root.resolve(), args.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
