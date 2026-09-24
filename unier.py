from __future__ import annotations

import argparse
import sys
from pathlib import Path

from Unier_cli.registry import MODEL_SPECS, get_model_spec, model_names
from Unier_cli.runner import BenchmarkRunner


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Model name, alias, or 'all'")
    parser.add_argument("--dataset", default="assist2017", help="Dataset name (aliases such as assist17 are accepted)")
    parser.add_argument(
        "--data-root",
        default="Datasets",
        help="Unified dataset root; relative paths are resolved from the UniER repository",
    )
    parser.add_argument(
        "--data-mode",
        choices=("auto", "hardlink", "copy"),
        default="auto",
        help="How canonical inputs are exposed at legacy model paths",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Replace a conflicting model-local compatibility input with the unified source",
    )
    parser.add_argument("--device", default="cuda:0", help="Device such as cuda:0 or cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--hidden-dim", type=int, default=1000)
    parser.add_argument("--concept-count", type=int)
    parser.add_argument("--target-type", choices=("all", "portion"), default="all")
    parser.add_argument(
        "--dkt-sort",
        action="store_true",
        help=(
            "For item-level models, dynamically reorder each recommendation list "
            "with the pretrained DKT mastery state during Ep evaluation"
        ),
    )
    parser.add_argument("--top-k", default="1,3,5,10,20")
    parser.add_argument("--ertga-generations", type=int, default=10)
    parser.add_argument("--ertga-population", type=int, default=12)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--run-id", default="latest")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true", help="Print resolved commands without executing them")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip required-input checks")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed or optional command")
    parser.add_argument("--fresh", action="store_true", help="Ignore metrics and records from an existing run directory")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified runner for all UniER item-level and path-level models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action, help_text in (
        ("prepare", "Run model-local preprocessing"),
        ("train", "Train/infer and generate recommendations"),
        ("evaluate", "Run every applicable metric and print a report"),
        ("report", "Print previously collected metrics"),
        ("run", "Run prepare, train, evaluate, and report"),
    ):
        subparser = subparsers.add_parser(action, help=help_text, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        add_common_arguments(subparser)
    subparsers.add_parser("list", help="List supported models")
    return parser


def list_models() -> None:
    print("Item-level models:")
    for spec in MODEL_SPECS:
        if spec.level == "item":
            print(f"  {spec.name}")
    print("Path-level models:")
    for spec in MODEL_SPECS:
        if spec.level == "path":
            print(f"  {spec.name}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "list":
        list_models()
        return 0

    repo_root = Path(__file__).resolve().parent
    if args.model.lower() == "all":
        specs = MODEL_SPECS
    else:
        try:
            specs = (get_model_spec(args.model),)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    failures = []
    for spec in specs:
        try:
            BenchmarkRunner(repo_root, args, spec).execute(args.action)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            if args.keep_going or len(specs) > 1:
                failures.append((spec.name, str(exc)))
                print(f"ERROR [{spec.name}]: {exc}", file=sys.stderr)
                continue
            raise SystemExit(str(exc)) from exc
    if failures:
        print("\nFailed models:", file=sys.stderr)
        for model, message in failures:
            print(f"  {model}: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
