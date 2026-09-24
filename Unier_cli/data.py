from __future__ import annotations

import filecmp
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

from .registry import ITEM, PATH, ModelSpec


@dataclass(frozen=True)
class DataAsset:
    """A canonical dataset input and the legacy path expected by a model."""

    source: str
    target: str
    stages: Tuple[str, ...]
    mutable: bool = False


def asset(source: str, target: str, *stages: str, mutable: bool = False) -> DataAsset:
    return DataAsset(source, target, tuple(stages), mutable)


MODEL_DATA_ASSETS: Dict[str, Tuple[DataAsset, ...]] = {
    "DRE": (
        asset("train_valid.csv", f"{ITEM}/DRE/data/{{dataset}}/train_valid.csv", "prepare"),
        asset("test.csv", f"{ITEM}/DRE/data/{{dataset}}/test.csv", "prepare"),
        asset("pkm.pth", f"{ITEM}/DRE/data/{{dataset}}/pkm.pth", "evaluate"),
    ),
    "AKT": (
        asset(
            "test_sequences.csv",
            f"{ITEM}/KT_Base_Rec/dataset/{{dataset}}/test_sequences.csv",
            "train",
        ),
        asset(
            "models/akt/qid_test_predictions.txt",
            f"{ITEM}/KT_Base_Rec/model/akt/{{dataset}}/qid_test_predictions.txt",
            "train",
        ),
    ),
    "SimpleKT": (
        asset(
            "test_sequences.csv",
            f"{ITEM}/KT_Base_Rec/dataset/{{dataset}}/test_sequences.csv",
            "train",
        ),
        asset(
            "models/simplekt/qid_test_predictions.txt",
            f"{ITEM}/KT_Base_Rec/model/simplekt/{{dataset}}/qid_test_predictions.txt",
            "train",
        ),
    ),
    "MMER": (
        asset("train_valid.csv", f"{ITEM}/MMER/datasets/{{dataset}}/train_valid.csv", "prepare"),
        asset("test.csv", f"{ITEM}/MMER/datasets/{{dataset}}/test.csv", "prepare"),
        asset("pkm.pth", f"{ITEM}/MMER/datasets/{{dataset}}/pkm.pth", "evaluate"),
    ),
    "KCP-ER": (
        # The modified selector writes into select/selected and leaves canonical inputs unchanged.
        asset(
            "test_sequences.csv",
            f"{ITEM}/kcp_er/datasets/{{dataset}}/select/test_sequences.csv",
            "prepare",
            mutable=True,
        ),
        asset("pkm.pth", f"{ITEM}/kcp_er/datasets/{{dataset}}/select/pkm.pth", "prepare"),
        asset("pkc.pth", f"{ITEM}/kcp_er/datasets/{{dataset}}/select/pkc.pth", "prepare"),
    ),
    "MulOER-SAN": (
        asset("test.csv", f"{ITEM}/MulOER-SAN/PYKT/data/{{dataset}}/test.csv", "evaluate"),
        asset(
            "models/muloer-san/qid_test_question_predictions.txt",
            f"{ITEM}/MulOER-SAN/PYKT/data/{{dataset}}/qid_test_question_predictions.txt",
            "prepare",
        ),
    ),
    "KG4EX": (
        asset(
            "train_valid_sequences.csv",
            f"{ITEM}/KG4EX/data/{{dataset}}/train_valid_sequences.csv",
            "prepare",
        ),
        asset("test_sequences.csv", f"{ITEM}/KG4EX/data/{{dataset}}/test_sequences.csv", "prepare"),
        asset("pkm.pth", f"{ITEM}/KG4EX/data/{{dataset}}/pkm.pth", "prepare"),
        asset("pkc.pth", f"{ITEM}/KG4EX/data/{{dataset}}/pkc.pth", "prepare"),
    ),
    "ER-TGA": (
        asset("train_valid.csv", f"{ITEM}/ER-TGA/datasets/{{dataset}}/train_valid.csv", "prepare"),
        asset("test.csv", f"{ITEM}/ER-TGA/datasets/{{dataset}}/test.csv", "prepare"),
        asset(
            "test_sequences.csv",
            f"{ITEM}/ER-TGA/evaluate/{{dataset}}/test_sequences.csv",
            "evaluate",
        ),
        asset("pkm.pth", f"{ITEM}/ER-TGA/evaluate/{{dataset}}/pkm.pth", "evaluate"),
    ),
    "NR4DER": (
        asset(
            "train_valid_sequences.csv",
            f"{ITEM}/NR4DER/dataset/data_200/{{dataset}}/train_valid_sequences.csv",
            "prepare",
        ),
        asset(
            "test_sequences.csv",
            f"{ITEM}/NR4DER/dataset/data_200/{{dataset}}/test_sequences.csv",
            "prepare",
        ),
    ),
}


def data_assets_for(spec: ModelSpec) -> Tuple[DataAsset, ...]:
    if spec.level == "path":
        if spec.name == "SRC":
            base = f"{PATH}/SRC/data/{{path_dataset}}"
        else:
            base = f"{PATH}/{spec.name}/data/dataProcess/{{path_dataset}}"
        common_files = (
            "train_valid.csv",
            "test.csv",
            "graph_vertex.json",
            "dataOff",
            "dataRec",
            "student_log_kt_None",
            "MyTransitionGraph.npy",
            "nxgraph.pkl",
            "prerequisite.json",
            "{graph_embedding_ckpt}",
            "{graph_embedding_npy}",
        )
        assets = tuple(asset(name, f"{base}/{name}", "prepare", "train") for name in common_files)
        if spec.name == "AC":
            checkpoint_base = f"{PATH}/AC/EduSim/Envs/{{pler_env_dir}}/meta_data"
            assets += (
                asset(
                    "models/ac/env_weights/ValBest.ckpt",
                    f"{checkpoint_base}/env_weights/ValBest.ckpt",
                    "train",
                ),
                asset(
                    "models/ac/env_weights/ValBest.pt",
                    f"{checkpoint_base}/env_weights/ValBest.pt",
                    "train",
                ),
                asset(
                    "models/ac/agent_weights/ValBest.ckpt",
                    f"{checkpoint_base}/agent_weights/ValBest.ckpt",
                    "train",
                ),
                asset(
                    "models/ac/agent_weights/ValBest.pt",
                    f"{checkpoint_base}/agent_weights/ValBest.pt",
                    "train",
                ),
            )
        return assets
    return MODEL_DATA_ASSETS.get(spec.name, ())


def resolve_data_root(repo_root: Path, configured: str) -> Path:
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


class DatasetMaterializer:
    """Expose canonical inputs at legacy model paths without manual copying."""

    def __init__(
        self,
        repo_root: Path,
        data_root: Path,
        values: Mapping[str, str],
        spec: ModelSpec,
        mode: str = "auto",
        refresh: bool = False,
    ):
        if mode not in {"auto", "hardlink", "copy"}:
            raise ValueError(f"Unknown data materialization mode: {mode}")
        self.repo_root = repo_root.resolve()
        self.data_root = data_root.resolve()
        self.values = values
        self.spec = spec
        self.mode = mode
        self.refresh = refresh
        self.dataset_dir = self.data_root / values["dataset"]

    def assets_for_stage(self, stage: str) -> Tuple[DataAsset, ...]:
        return tuple(item for item in data_assets_for(self.spec) if stage in item.stages)

    def _paths(self, item: DataAsset) -> Tuple[Path, Path]:
        source = (self.dataset_dir / item.source.format_map(self.values)).resolve()
        target = (self.repo_root / item.target.format_map(self.values)).resolve()
        if not source.is_relative_to(self.dataset_dir):
            raise ValueError(f"Dataset source escapes the configured data root: {source}")
        if not target.is_relative_to(self.repo_root):
            raise ValueError(f"Model data target escapes the repository: {target}")
        return source, target

    def required_sources(self, stage: str) -> Tuple[Path, ...]:
        return tuple(self._paths(item)[0] for item in self.assets_for_stage(stage))

    def _same_file_or_content(self, source: Path, target: Path) -> bool:
        try:
            if os.path.samefile(source, target):
                return True
        except OSError:
            pass
        try:
            return filecmp.cmp(source, target, shallow=False)
        except OSError:
            return False

    def _place(self, source: Path, target: Path, item: DataAsset) -> str:
        target.parent.mkdir(parents=True, exist_ok=True)

        if item.mutable:
            shutil.copy2(source, target)
            return "copy (writable compatibility input)"

        if target.exists():
            if self._same_file_or_content(source, target):
                return "already mapped"
            if not self.refresh:
                raise FileExistsError(
                    f"Model-local input already exists and differs from the unified source:\n"
                    f"  target: {target}\n"
                    f"  source: {source}\n"
                    "Move the authoritative file into the unified dataset directory or pass --refresh-data."
                )
            target.unlink()

        if self.mode in {"auto", "hardlink"}:
            try:
                os.link(source, target)
                return "hardlink"
            except OSError:
                if self.mode == "hardlink":
                    raise

        shutil.copy2(source, target)
        return "copy"

    def materialize(self, stage: str, dry_run: bool = False, allow_missing: bool = False) -> None:
        stage_assets = self.assets_for_stage(stage)
        if not stage_assets:
            return

        resolved = [(item, *self._paths(item)) for item in stage_assets]
        missing = [source for _, source, _ in resolved if not source.is_file()]
        if missing and not dry_run and not allow_missing:
            rendered = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(
                f"Required unified dataset inputs for {self.spec.name} {stage} are missing:\n"
                f"{rendered}\n"
                f"Place them under {self.dataset_dir} or select another directory with --data-root."
            )

        for item, source, target in resolved:
            if dry_run:
                suffix = " [writable copy]" if item.mutable else ""
                print(f"[{self.spec.name}:data:{stage}] {source} -> {target}{suffix}")
                continue
            if not source.is_file():
                print(f"[{self.spec.name}:data:{stage}] skipped missing optional source: {source}")
                continue
            method = self._place(source, target, item)
            print(f"[{self.spec.name}:data:{stage}] {method}: {source} -> {target}")


def canonical_layout(specs: Iterable[ModelSpec]) -> Tuple[str, ...]:
    """Return the unique canonical relative inputs used by a collection of models."""

    return tuple(sorted({item.source for spec in specs for item in data_assets_for(spec)}))
