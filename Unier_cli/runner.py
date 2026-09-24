from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from .data import DatasetMaterializer, resolve_data_root
from .metrics import load_metrics, parse_metrics, print_report, save_metrics
from .registry import CommandSpec, ModelSpec


CONCEPT_COUNTS = {
    "assist2009": 123,
    "assist2012": 265,
    "assist2015": 100,
    "assist2017": 102,
    "algebra2005": 112,
    "bridge2006": 493,
    "ednet": 188,
    "junyi": 39,
    "nips34": 57,
    "xes3g5m": 865,
}

DATASET_ALIASES = {
    "assist09": "assist2009",
    "assist2009": "assist2009",
    "assist12": "assist2012",
    "assist2012": "assist2012",
    "assist15": "assist2015",
    "assist2015": "assist2015",
    "assist17": "assist2017",
    "assist2017": "assist2017",
    "algebra2005": "algebra2005",
    "bridge2006": "bridge2006",
    "ednet": "ednet",
    "junyi": "junyi",
    "nips34": "nips34",
    "xes3g5m": "xes3g5m",
}

PATH_DATASETS = {
    "assist2009": "assist09",
    "assist2012": "assist12",
    "assist2015": "assist15",
    "assist2017": "assist17",
}

PLER_ENV_DIRS = {
    "assist09": "KES_ASSIST09",
    "assist12": "KES_ASSIST12",
    "assist15": "KES_ASSIST15",
    "assist17": "KES_ASSIST17",
}

PLER_COMMON_PREPARE_LABELS = {
    "data_process",
    "transition_graph",
    "env_dkt",
    "graph_embedding",
}

ERTGA_DATASETS = {
    "assist2009": "ASSISTments2009",
    "assist2012": "ASSISTments2012",
    "assist2017": "ASSISTments2017",
    "algebra2005": "Algebra2005",
    "bridge2006": "Bridge2006",
    "ednet": "Ednet",
    "junyi": "Junyi",
    "nips34": "Nips34",
    "xes3g5m": "Xes3g5m",
}

DKT_SORT_SCRIPTS = {
    "ep_all": "test_ep_all_update.py",
    "ep_portion": "test_ep_por_update.py",
}


def stage_commands(spec: ModelSpec, stage: str, dkt_sort: bool = False) -> tuple[CommandSpec, ...]:
    """Select original or DKT-reordered Ep commands for one runner stage."""
    commands = tuple(getattr(spec, stage))
    if not dkt_sort or spec.level != "item" or stage != "evaluate":
        return commands

    selected: List[CommandSpec] = []
    for command in commands:
        script = DKT_SORT_SCRIPTS.get(command.label)
        if script is None:
            selected.append(command)
            continue
        selected.append(
            CommandSpec(
                label=command.label,
                cwd="Item-level Exercise Recommendation",
                argv=(
                    "{python}",
                    script,
                    "--dataset",
                    "{dataset}",
                    "--model",
                    spec.name,
                    "--device",
                    "{device}",
                    "--overwrite",
                ),
                description="Dynamically reorder recommendations using DKT mastery",
                optional=command.optional,
            )
        )
    return tuple(selected)

def normalize_dataset(name: str) -> str:
    key = name.lower().replace("-", "")
    if key not in DATASET_ALIASES:
        supported = ", ".join(sorted(set(DATASET_ALIASES.values())))
        raise ValueError(f"Unknown dataset {name!r}. Supported datasets: {supported}")
    return DATASET_ALIASES[key]


def _cuda_index(device: str) -> str:
    if device.lower() == "cpu":
        return "0"
    return device.split(":", 1)[1] if ":" in device else "0"


def render_values(args, spec: ModelSpec) -> Dict[str, str]:
    dataset = normalize_dataset(args.dataset)
    path_dataset = PATH_DATASETS.get(dataset, dataset)
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 1024 if spec.name == "KG4EX" else 256
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 0.001 if spec.name == "KG4EX" else 0.0001
    simulator = "KES" + path_dataset
    kt_model = "akt" if spec.name == "AKT" else "simplekt" if spec.name == "SimpleKT" else spec.name.lower()
    return {
        "python": str(Path(args.python).resolve()) if Path(args.python).exists() else args.python,
        "dataset": dataset,
        "path_dataset": path_dataset,
        "pler_env_dir": PLER_ENV_DIRS.get(path_dataset, f"KES_{path_dataset}"),
        "graph_embedding_ckpt": f"{path_dataset}GraphEmbedding.ckpt",
        "graph_embedding_npy": f"{path_dataset.upper()}GraphEmbedding.npy",
        "simulator": simulator,
        "kt_model": kt_model,
        "device": args.device,
        "device_value": args.device,
        "cuda_index": _cuda_index(args.device),
        "cuda_flag": "--cuda" if args.device.lower() != "cpu" else "",
        "seed": str(args.seed),
        "epochs": str(args.epochs),
        "episodes": str(args.episodes),
        "max_steps": str(args.max_steps),
        "batch_size": str(batch_size),
        "learning_rate": str(learning_rate),
        "hidden_dim": str(args.hidden_dim),
        "target_type": args.target_type,
        "top_k": args.top_k,
        "concept_count": str(CONCEPT_COUNTS.get(dataset, args.concept_count or 0)),
        "ertga_dataset": ERTGA_DATASETS.get(dataset, dataset),
        "ertga_generations": str(args.ertga_generations),
        "ertga_population": str(args.ertga_population),
    }


def render_command(command: CommandSpec, values: Mapping[str, str]) -> List[str]:
    rendered: List[str] = []
    for token in command.argv:
        value = token.format_map(values)
        if value:
            rendered.append(value)
    return rendered


def _command_text(argv: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(argv))


class BenchmarkRunner:
    def __init__(self, repo_root: Path, args, spec: ModelSpec):
        self.repo_root = repo_root.resolve()
        self.args = args
        self.spec = spec
        self.values = render_values(args, spec)
        self.dataset = self.values["dataset"]
        self.data_root = resolve_data_root(self.repo_root, args.data_root)
        self.data = DatasetMaterializer(
            self.repo_root,
            self.data_root,
            self.values,
            spec,
            mode=args.data_mode,
            refresh=args.refresh_data,
        )
        self.run_dir = (
            Path(args.output_dir).resolve()
            if Path(args.output_dir).is_absolute()
            else self.repo_root / args.output_dir
        ) / spec.level / spec.name / self.dataset / args.run_id
        self.metrics_path = self.run_dir / "metrics.json"
        self.records_path = self.run_dir / "command_records.json"
        self.log_path = self.run_dir / "run.log"
        self.metrics = {} if args.fresh else load_metrics(self.metrics_path)
        self.records: List[dict] = []
        if not args.fresh and self.records_path.exists():
            try:
                self.records = json.loads(self.records_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self.records = []

    def _environment(self) -> Dict[str, str]:
        env = os.environ.copy()
        previous_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(self.repo_root) + (os.pathsep + previous_pythonpath if previous_pythonpath else "")
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env.update(
            {
                "UNIER_MODEL": self.spec.name,
                "UNIER_DATASET": self.dataset,
                "UNIER_PATH_DATASET": self.values["path_dataset"],
                "UNIER_DATA_ROOT": str(self.data.dataset_dir),
                "UNIER_DEVICE": self.args.device,
                "UNIER_SEED": str(self.args.seed),
                "UNIER_EPOCHS": str(self.args.epochs),
                "UNIER_EPISODES": str(self.args.episodes),
                "UNIER_MAX_STEPS": str(self.args.max_steps),
                "UNIER_TARGET_TYPE": self.args.target_type,
                "UNIER_TOP_K": self.args.top_k,
            }
        )
        return env

    def _write_state(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        save_metrics(self.metrics_path, self.metrics)
        self.records_path.write_text(json.dumps(self.records, ensure_ascii=False, indent=2), encoding="utf-8")
        config = {
            "model": self.spec.name,
            "level": self.spec.level,
            "dataset": self.dataset,
            "updated_at": datetime.now().astimezone().isoformat(),
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(self.args).items()},
        }
        (self.run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        with (self.run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("metric", "value"))
            writer.writerows(sorted(self.metrics.items()))

    def preflight(self, stage: str, commands: Iterable[CommandSpec]) -> None:
        missing_scripts: List[Path] = []
        for command in commands:
            argv = render_command(command, self.values)
            cwd = (self.repo_root / command.cwd).resolve()
            if len(argv) >= 2 and argv[1].endswith(".py"):
                script = (cwd / argv[1]).resolve()
                if not script.exists():
                    missing_scripts.append(script)
        if missing_scripts:
            rendered = "\n".join(f"  - {path}" for path in missing_scripts)
            raise FileNotFoundError(f"Registered scripts are missing:\n{rendered}")

    def run_command(self, stage: str, command: CommandSpec) -> bool:
        argv = render_command(command, self.values)
        cwd = (self.repo_root / command.cwd).resolve()
        print(f"[{self.spec.name}:{stage}:{command.label}] cwd={cwd}")
        print(f"  {_command_text(argv)}")
        if self.args.dry_run:
            return True

        self.run_dir.mkdir(parents=True, exist_ok=True)
        started = datetime.now().astimezone()
        process = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=self._environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        output_lines: List[str] = []
        assert process.stdout is not None
        with self.log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{started.isoformat()}] {self.spec.name}:{stage}:{command.label}\n")
            log.write(f"cwd={cwd}\ncommand={_command_text(argv)}\n")
            for line in process.stdout:
                print(line, end="")
                log.write(line)
                output_lines.append(line)
        return_code = process.wait()
        output = "".join(output_lines)
        parsed = parse_metrics(output, command.label)
        self.metrics.update(parsed)
        record = {
            "stage": stage,
            "label": command.label,
            "cwd": str(cwd),
            "argv": argv,
            "started_at": started.isoformat(),
            "finished_at": datetime.now().astimezone().isoformat(),
            "return_code": return_code,
            "metrics": parsed,
        }
        self.records.append(record)
        self._write_state()
        if return_code != 0:
            message = f"Command failed with exit code {return_code}: {command.label}"
            if command.optional or self.args.keep_going:
                print(f"WARNING: {message}")
                return False
            raise RuntimeError(message)
        return True

    def run_stage(self, stage: str) -> None:
        commands = stage_commands(
            self.spec,
            stage,
            dkt_sort=getattr(self.args, "dkt_sort", False),
        )
        self.data.materialize(
            stage,
            dry_run=self.args.dry_run,
            allow_missing=self.args.skip_preflight,
        )
        if self.spec.level == "path" and stage == "prepare":
            skipped = tuple(command for command in commands if command.label in PLER_COMMON_PREPARE_LABELS)
            commands = tuple(command for command in commands if command.label not in PLER_COMMON_PREPARE_LABELS)
            if skipped:
                labels = ", ".join(command.label for command in skipped)
                print(f"[{self.spec.name}:prepare] reuse unified PLER artifacts; skipped: {labels}")
        if not commands:
            print(f"[{self.spec.name}:{stage}] no model-specific commands remain for this stage.")
            return
        self.preflight(stage, commands)
        for command in commands:
            self.run_command(stage, command)

    def report(self) -> None:
        if not self.args.dry_run:
            self._write_state()
        print_report(self.spec.name, self.dataset, self.metrics, self.run_dir)

    def execute(self, action: str) -> None:
        if self.spec.notes:
            for note in self.spec.notes:
                print(f"NOTE [{self.spec.name}]: {note}")
        if getattr(self.args, "dkt_sort", False) and self.spec.level != "item":
            print(f"NOTE [{self.spec.name}]: --dkt-sort is item-level only and is ignored.")
        if action == "run":
            for stage in ("prepare", "train", "evaluate"):
                self.run_stage(stage)
            self.report()
        elif action == "report":
            self.report()
        else:
            self.run_stage(action)
            if action == "evaluate":
                self.report()
