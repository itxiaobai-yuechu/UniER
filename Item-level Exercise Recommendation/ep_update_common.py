#!/usr/bin/env python3
"""Shared implementation for dynamic Ep evaluation and recommendation reordering."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# The shared implementation lives directly under the item-level model root.
PROJECT_ROOT = Path(__file__).resolve().parent
STEPS = (1, 5, 10, 20)
THRESHOLD = 0.5
INFERENCE_SEED = 42

DATASET_ALIASES = {
    "assist09": "assist2009",
    "assist9": "assist2009",
    "assist12": "assist2012",
    "assist17": "assist2017",
}
NUM_CONCEPTS = {
    "algebra2005": 112,
    "assist2009": 123,
    "assist2012": 265,
    "assist2017": 102,
    "bridge2006": 493,
    "ednet": 188,
    "junyi": 39,
    "nips34": 57,
    "xes3g5m": 865,
}
MODEL_ALIASES = {
    "nr4der": "NR4DER",
    "melt-lstm": "NR4DER",
    "dre": "DRE",
    "kg4ex": "KG4EX",
    "er-tga": "ER-TGA",
    "ertga": "ER-TGA",
    "kcper": "KCPER",
    "kcp-er": "KCPER",
    "kcp_er": "KCPER",
    "muloer-san": "MulOER-SAN",
    "muloersan": "MulOER-SAN",
    "akt": "akt",
    "simplekt": "simpleKT",
    "simple_kt": "simpleKT",
    "mmer": "MMER",
}
MODEL_SPECS = {
    "NR4DER": {
        "q_path": "NR4DER/dataset/data_200/{dataset}/Q.npy",
        "test_path": "NR4DER/dataset/data_200/{dataset}/test_sequences.csv",
        "recommendation_path": (
            "NR4DER/dataset/data_200/{dataset}/Reranking_result_melt_20.txt"
        ),
        "recommendation_kind": "question",
    },
    "DRE": {
        "q_path": "DRE/data/{dataset}/Q.npy",
        "test_path": "DRE/data/{dataset}/filtered_full_data.csv",
        "recommendation_path": "DRE/data/{dataset}/recommended_problems_final.txt",
        "recommendation_kind": "question",
    },
    "KG4EX": {
        "q_path": "KG4EX/data/{dataset}/Q.npy",
        "test_path": "KG4EX/data/{dataset}/Test20_unique_last.csv",
        "recommendation_path": "KG4EX/data/{dataset}/Test20/recommend_20.txt",
        "recommendation_kind": "question",
    },
    "ER-TGA": {
        "q_path": "ER-TGA/datasets/{dataset}/Q.npy",
        "test_path": "ER-TGA/datasets/{dataset}/{dataset}.csv",
        "recommendation_path": "ER-TGA/fin_res/{dataset}/recommend.txt",
        "recommendation_kind": "question",
    },
    "KCPER": {
        "q_path": "kcp_er/datasets/{dataset}/select/selected/Q.npy",
        "test_path": "kcp_er/datasets/{dataset}/select/selected/test_sequences.csv",
        "recommendation_path": (
            "kcp_er/datasets/{dataset}/select/selected/recommendation_20.txt"
        ),
        "recommendation_kind": "question",
    },
    "MulOER-SAN": {
        "q_path": "MulOER-SAN/select/{dataset}/Q.npy",
        "test_path": "MulOER-SAN/PYKT/data/{dataset}/test_gt4.csv",
        "recommendation_path": "MulOER-SAN/select/{dataset}/new_ex_original.txt",
        "recommendation_kind": "question",
    },
    "akt": {
        "q_path": None,
        "test_path": "KT_Base_Rec/dataset/{dataset}/test_sequences.csv",
        "recommendation_path": "KT_Base_Rec/model/akt/{dataset}/recommend.txt",
        "recommendation_kind": "concept",
    },
    "simpleKT": {
        "q_path": None,
        "test_path": "KT_Base_Rec/dataset/{dataset}/test_sequences.csv",
        "recommendation_path": (
            "KT_Base_Rec/model/simplekt/{dataset}/recommend.txt"
        ),
        "recommendation_kind": "concept",
    },
    "MMER": {
        "q_path": "MMER/datasets/{dataset}/Q.npy",
        "test_path": "MMER/datasets/{dataset}/test.csv",
        "recommendation_path": (
            "MMER/datasets/{dataset}/score_rec_exer_topk_stu_all.txt"
        ),
        "recommendation_kind": "question",
        "line_format": "uid_space",
        "match_by_uid": True,
    },
}


class KTnet(nn.Module):
    """The DKT architecture used by the existing test_ep scripts."""

    def __init__(self, parameters):
        super().__init__()
        self.nhid = parameters["hidden_size"]
        self.nlayers = parameters["nlayers"]
        self.dropout = parameters["dropout"]

        self.embedding_layer = nn.Linear(parameters["input_size"], parameters["emb_dim"])
        torch.nn.init.normal_(self.embedding_layer.weight)
        torch.nn.init.zeros_(self.embedding_layer.bias)
        self.rnn = nn.LSTM(parameters["emb_dim"], self.nhid, self.nlayers)
        self.fc_out = nn.Linear(self.nhid, parameters["num_skills"])
        self.dropout_layer = nn.Dropout(p=self.dropout)

    def forward(self, value):
        value = value.permute(1, 0, 2)
        hidden, cell = self.init_hidden_state(value.shape[1])
        embedded = self.embedding_layer(value)
        output, _ = self.rnn(embedded, (hidden, cell))
        return self.dropout_layer(self.fc_out(output))

    def init_hidden_state(self, batch_size):
        device = self.embedding_layer.weight.device
        hidden = torch.rand((self.nlayers, batch_size, self.nhid), device=device)
        cell = torch.rand((self.nlayers, batch_size, self.nhid), device=device)
        return hidden, cell


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def normalize_dataset(raw_dataset):
    normalized = raw_dataset.strip().lower()
    normalized = DATASET_ALIASES.get(normalized, normalized)
    if normalized not in NUM_CONCEPTS:
        choices = ", ".join(sorted(NUM_CONCEPTS))
        raise ValueError(f"Unsupported dataset '{raw_dataset}'. Choices: {choices}")
    return normalized


def normalize_model(raw_model):
    normalized = raw_model.strip().lower()
    if normalized not in MODEL_ALIASES:
        choices = ", ".join(MODEL_SPECS)
        raise ValueError(f"Unsupported model '{raw_model}'. Choices: {choices}")
    return MODEL_ALIASES[normalized]


def resolve_paths(dataset, model):
    spec = MODEL_SPECS[model]
    paths = {
        "test": PROJECT_ROOT / spec["test_path"].format(dataset=dataset),
        "recommendation": PROJECT_ROOT
        / spec["recommendation_path"].format(dataset=dataset),
    }
    if spec["q_path"] is not None:
        paths["q"] = PROJECT_ROOT / spec["q_path"].format(dataset=dataset)

    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(f"Required input file is missing:\n{joined}")
    return spec, paths


def display_path(path):
    for base in (PROJECT_ROOT, PROJECT_ROOT.parent):
        try:
            return path.relative_to(base)
        except ValueError:
            continue
    return path


def load_kt_net(dataset, num_concepts, device):
    feature_dim = 2 * num_concepts
    parameters = {
        "input_size": feature_dim,
        "emb_dim": 128,
        "hidden_size": 256,
        "num_skills": num_concepts,
        "nlayers": 2,
        "dropout": 0.01,
    }
    checkpoint_candidates = []
    unified_data_root = os.environ.get("UNIER_DATA_ROOT")
    if unified_data_root:
        checkpoint_candidates.append(
            Path(unified_data_root) / "models" / "dkt" / "env_weights" / "ValBest.ckpt"
        )
    checkpoint_candidates.extend(
        (
            PROJECT_ROOT.parent
            / "Datasets"
            / dataset
            / "models"
            / "dkt"
            / "env_weights"
            / "ValBest.ckpt",
            PROJECT_ROOT
            / "pretrianed_kt_models"
            / dataset
            / "env_weights"
            / "ValBest.ckpt",
        )
    )
    checkpoint = next((path for path in checkpoint_candidates if path.is_file()), None)
    if checkpoint is None:
        expected = "\n".join(str(path) for path in dict.fromkeys(checkpoint_candidates))
        raise FileNotFoundError(f"DKT checkpoint is missing. Checked:\n{expected}")

    network = KTnet(parameters).to(device)
    weights = torch.load(checkpoint, map_location=device, weights_only=True)
    network.load_state_dict(weights)
    network.eval()
    return network, feature_dim, checkpoint


def fixed_seed_forward(network, kt_input, device):
    """Use an identical random LSTM initial state on every inference."""
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    try:
        torch.manual_seed(INFERENCE_SEED)
        if device.type == "cuda":
            torch.cuda.manual_seed(INFERENCE_SEED)
        return network(kt_input)
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, device)


def get_kt_mastery(network, concepts, responses, num_concepts, feature_dim, device):
    if len(concepts) != len(responses):
        raise ValueError(
            f"History length mismatch: concepts={len(concepts)}, responses={len(responses)}"
        )

    with torch.no_grad():
        if not concepts:
            return torch.zeros(num_concepts, device=device)

        kt_input = torch.zeros((1, len(concepts), feature_dim), device=device)
        for index, (concept, response) in enumerate(zip(concepts, responses)):
            if not 0 <= concept < num_concepts:
                raise ValueError(f"Concept ID {concept} is outside [0, {num_concepts - 1}]")
            if response not in (0, 1):
                raise ValueError(f"Response {response} is not binary")
            kt_input[0, index, concept + (num_concepts if response == 1 else 0)] = 1

        output = fixed_seed_forward(network, kt_input, device)[-1, 0, :]
        return torch.sigmoid(output)


def parse_int_sequence(value):
    parsed = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        number = int(float(item))
        if number != -1:
            parsed.append(number)
    return parsed


def parse_history(row):
    concepts = parse_int_sequence(row["concepts"])
    responses = parse_int_sequence(row["responses"])
    if len(concepts) != len(responses):
        raise ValueError(
            f"History has different concept/response lengths: {len(concepts)} != {len(responses)}"
        )
    return concepts, responses


def parse_recommendation_line(raw_line, line_number, line_format="tab"):
    newline = "\n" if raw_line.endswith("\n") else ""
    content = raw_line[:-1] if newline else raw_line
    if line_format == "uid_space":
        parts = content.split(maxsplit=1)
        if not parts:
            raise ValueError(f"Recommendation line {line_number} has no uid")
        prefix = parts[0]
        item_text = parts[1] if len(parts) > 1 else ""
    elif line_format == "tab":
        try:
            prefix, item_text = content.rsplit("\t", 1)
        except ValueError as error:
            raise ValueError(
                f"Recommendation line {line_number} is not tab-separated"
            ) from error
    else:
        raise ValueError(f"Unsupported recommendation line format: {line_format}")

    items = []
    for item in item_text.split(","):
        item = item.strip()
        if item:
            items.append(int(item))
    return prefix, items, newline


def format_recommendation_line(prefix, items, newline, line_format):
    separator = " " if line_format == "uid_space" else "\t"
    return prefix + separator + ",".join(map(str, items)) + newline


def mmer_first_uid_records(test_df, recommendation_lines, max_students):
    """Keep only the first recommendation line for each MMER student uid."""
    if "uid" not in test_df.columns:
        raise ValueError("MMER test.csv must contain a uid column")

    uid_to_row = {}
    duplicate_test_uids = 0
    for row_index, uid in test_df["uid"].astype(str).items():
        if uid in uid_to_row:
            duplicate_test_uids += 1
            continue
        uid_to_row[uid] = row_index

    records = []
    seen_uids = set()
    duplicate_recommendation_uids = 0
    unmatched_uids = 0
    for line_number, raw_line in enumerate(recommendation_lines, start=1):
        uid, _, _ = parse_recommendation_line(
            raw_line, line_number, line_format="uid_space"
        )
        if uid in seen_uids:
            duplicate_recommendation_uids += 1
            continue
        seen_uids.add(uid)
        row_index = uid_to_row.get(uid)
        if row_index is None:
            unmatched_uids += 1
        records.append((line_number, raw_line, row_index))

    unique_recommendation_uids = len(records)
    if max_students is not None:
        records = records[:max_students]

    return records, {
        "unique_recommendation_uids": unique_recommendation_uids,
        "duplicate_recommendation_uids": duplicate_recommendation_uids,
        "duplicate_test_uids": duplicate_test_uids,
        "unmatched_uids": unmatched_uids,
    }


def map_items_to_concepts(items, recommendation_kind, q_matrix, num_concepts, line_number):
    """Return mappable (item, concept) entries and retained unmappable question IDs."""
    candidates = []
    unmapped_items = []
    for position, item in enumerate(items, start=1):
        if recommendation_kind == "concept":
            concept = item
            if not 0 <= concept < num_concepts:
                raise ValueError(
                    f"Line {line_number}, recommendation {position}: concept ID {concept} "
                    f"is outside [0, {num_concepts - 1}]"
                )
        else:
            if item < 0 or item >= q_matrix.shape[0]:
                raise ValueError(
                    f"Line {line_number}, recommendation {position}: question ID {item} "
                    f"is outside Q-matrix rows"
                )
            question_concepts = np.flatnonzero(q_matrix[item] == 1)
            if len(question_concepts) == 0:
                unmapped_items.append(item)
                continue
            concept = int(question_concepts[0])
            if not 0 <= concept < num_concepts:
                raise ValueError(
                    f"Line {line_number}, recommendation {position}: concept ID {concept} "
                    f"is outside [0, {num_concepts - 1}]"
                )
        candidates.append((item, concept))
    return candidates, unmapped_items


def score_state(state, mode, target_concepts):
    if mode == "all":
        return int((state > THRESHOLD).sum().item())
    return sum(1 for concept in target_concepts if state[concept].item() > THRESHOLD)


def dynamic_reorder(
    network,
    concepts,
    responses,
    candidates,
    num_concepts,
    feature_dim,
    device,
    mode,
    target_concepts,
):
    initial_state = get_kt_mastery(
        network, concepts, responses, num_concepts, feature_dim, device
    )
    evaluate_ep = mode == "all" or target_concepts is not None
    if evaluate_ep:
        initial_score = score_state(initial_state, mode, target_concepts)
        maximum_score = num_concepts if mode == "all" else len(target_concepts)
        denominator = maximum_score - initial_score
    else:
        initial_score = 0
        denominator = 0

    current_concepts = list(concepts)
    current_responses = list(responses)
    current_state = initial_state
    remaining = list(candidates)
    ordered_items = []
    ep_by_step = {}

    for step in range(1, len(candidates) + 1):
        selected_index = 0
        for index, (_, concept) in enumerate(remaining):
            if current_state[concept].item() > THRESHOLD:
                selected_index = index
                break

        item, concept = remaining.pop(selected_index)
        response = 1 if current_state[concept].item() > THRESHOLD else 0
        ordered_items.append(item)
        current_concepts.append(concept)
        current_responses.append(response)
        current_state = get_kt_mastery(
            network,
            current_concepts,
            current_responses,
            num_concepts,
            feature_dim,
            device,
        )

        if evaluate_ep and step in STEPS:
            final_score = score_state(current_state, mode, target_concepts)
            ep_by_step[step] = (
                (final_score - initial_score) / denominator if denominator else 0.0
            )

    return ordered_items, ep_by_step


def fixed_hidden_state(network, batch_size, device):
    """Repeat the exact fixed single-student DKT state across a batch."""
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    try:
        torch.manual_seed(INFERENCE_SEED)
        if device.type == "cuda":
            torch.cuda.manual_seed(INFERENCE_SEED)
        hidden, cell = network.init_hidden_state(1)
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, device)
    return (
        hidden.expand(-1, batch_size, -1).contiguous(),
        cell.expand(-1, batch_size, -1).contiguous(),
    )


def get_kt_mastery_batch_fixed(
    network, histories, num_concepts, feature_dim, device
):
    """Batched equivalent of the update script's fixed-seed DKT inference."""
    result = torch.zeros((len(histories), num_concepts), device=device)
    active = [index for index, (concepts, _) in enumerate(histories) if concepts]
    if not active:
        return result

    for _, (concepts, responses) in enumerate(histories):
        if len(concepts) != len(responses):
            raise ValueError(
                f"History length mismatch: concepts={len(concepts)}, responses={len(responses)}"
            )

    active_histories = [histories[index] for index in active]
    max_length = max(len(concepts) for concepts, _ in active_histories)
    kt_input = torch.zeros(
        (len(active_histories), max_length, feature_dim), device=device
    )
    lengths = []
    for batch_index, (concepts, responses) in enumerate(active_histories):
        length = len(concepts)
        concept_ids = torch.as_tensor(concepts, dtype=torch.long, device=device)
        response_ids = torch.as_tensor(responses, dtype=torch.long, device=device)
        if torch.any(concept_ids < 0) or torch.any(concept_ids >= num_concepts):
            raise ValueError("Concept ID is outside the configured concept range")
        if torch.any((response_ids != 0) & (response_ids != 1)):
            raise ValueError("Response is not binary")
        positions = torch.arange(length, device=device)
        feature_ids = concept_ids + response_ids * num_concepts
        kt_input[batch_index, positions, feature_ids] = 1
        lengths.append(length)

    with torch.no_grad():
        hidden, cell = fixed_hidden_state(network, len(active_histories), device)
        embedded = network.embedding_layer(kt_input.permute(1, 0, 2))
        output, _ = network.rnn(embedded, (hidden, cell))
        output = network.dropout_layer(network.fc_out(output))
        time_indices = torch.as_tensor(
            [length - 1 for length in lengths], dtype=torch.long, device=device
        )
        batch_indices = torch.arange(len(active_histories), device=device)
        states = torch.sigmoid(output[time_indices, batch_indices, :])
        result[torch.as_tensor(active, dtype=torch.long, device=device)] = states
    return result


def run_mmer_batched(mode, args, dataset, model, device):
    """Evaluate MMER update lists in batches without changing selection semantics."""
    num_concepts = NUM_CONCEPTS[dataset]
    spec, paths = resolve_paths(dataset, model)
    set_seed(INFERENCE_SEED)
    test_df = pd.read_csv(paths["test"])
    recommendation_lines = paths["recommendation"].read_text(encoding="utf-8").splitlines(
        keepends=True
    )
    q_matrix = np.load(paths["q"])
    network, feature_dim, checkpoint = load_kt_net(dataset, num_concepts, device)
    records, uid_stats = mmer_first_uid_records(
        test_df, recommendation_lines, args.max_students
    )

    print(f"mode={mode} model={model} dataset={dataset} device={device}")
    print(f"test_csv={paths['test'].relative_to(PROJECT_ROOT)}")
    print(f"recommendations={paths['recommendation'].relative_to(PROJECT_ROOT)}")
    print(f"dkt_checkpoint={display_path(checkpoint)}")
    print(f"fixed_inference_seed={INFERENCE_SEED}")
    print(
        "uid_matching=first recommendation per uid; "
        f"source_unique_uids={uid_stats['unique_recommendation_uids']} "
        f"records_to_process={len(records)} "
        f"duplicate_recommendation_uids_ignored="
        f"{uid_stats['duplicate_recommendation_uids']} "
        f"unmatched_uids={uid_stats['unmatched_uids']}"
    )
    if uid_stats["duplicate_test_uids"]:
        print("warning: duplicate test.csv uids detected; the first matching row is used")
    print(f"mmer_batched_update=True batch_size={args.batch_size}")

    totals = {step: {"ep": 0.0, "valid": 0} for step in STEPS}
    output_lines = [None] * len(records)
    participants = []
    skipped_portion_metric = 0
    empty_recommendations = 0
    unmapped_recommendation_items = 0
    missing_test_uid = 0

    for output_index, (line_number, raw_line, row_index) in enumerate(records):
        prefix, items, newline = parse_recommendation_line(
            raw_line, line_number, line_format="uid_space"
        )
        if row_index is None:
            missing_test_uid += 1
            output_lines[output_index] = raw_line
            continue
        if not items:
            empty_recommendations += 1
            output_lines[output_index] = raw_line
            continue

        row = test_df.iloc[row_index]
        concepts, responses = parse_history(row)
        target_concepts = None
        if mode == "portion":
            if len(concepts) >= 5:
                start_index = int(len(concepts) * 0.8)
                target_concepts = list(set(concepts[start_index:]))
            if not target_concepts:
                skipped_portion_metric += 1

        candidates, unmapped_items = map_items_to_concepts(
            items,
            spec["recommendation_kind"],
            q_matrix,
            num_concepts,
            line_number,
        )
        unmapped_recommendation_items += len(unmapped_items)
        if not candidates:
            empty_recommendations += 1
            output_lines[output_index] = raw_line
            continue

        participants.append(
            {
                "output_index": output_index,
                "prefix": prefix,
                "newline": newline,
                "concepts": concepts,
                "responses": responses,
                "target_concepts": target_concepts,
                "remaining": list(candidates),
                "unmapped_items": unmapped_items,
                "ordered_items": [],
            }
        )

    for batch_start in range(0, len(participants), args.batch_size):
        batch = participants[batch_start : batch_start + args.batch_size]
        initial_states = get_kt_mastery_batch_fixed(
            network,
            [(item["concepts"], item["responses"]) for item in batch],
            num_concepts,
            feature_dim,
            device,
        )
        for index, item in enumerate(batch):
            item["current_concepts"] = list(item["concepts"])
            item["current_responses"] = list(item["responses"])
            item["current_state"] = initial_states[index]
            item["evaluate_ep"] = mode == "all" or item["target_concepts"] is not None
            if item["evaluate_ep"]:
                item["initial_score"] = score_state(
                    item["current_state"], mode, item["target_concepts"]
                )
                maximum_score = (
                    num_concepts if mode == "all" else len(item["target_concepts"])
                )
                item["denominator"] = maximum_score - item["initial_score"]

        max_steps = max(len(item["remaining"]) for item in batch)
        for step in range(1, max_steps + 1):
            active = []
            for item in batch:
                if not item["remaining"]:
                    continue
                selected_index = 0
                for index, (_, concept) in enumerate(item["remaining"]):
                    if item["current_state"][concept].item() > THRESHOLD:
                        selected_index = index
                        break
                question, concept = item["remaining"].pop(selected_index)
                response = 1 if item["current_state"][concept].item() > THRESHOLD else 0
                item["ordered_items"].append(question)
                item["current_concepts"].append(concept)
                item["current_responses"].append(response)
                active.append(item)

            if not active:
                break
            updated_states = get_kt_mastery_batch_fixed(
                network,
                [
                    (item["current_concepts"], item["current_responses"])
                    for item in active
                ],
                num_concepts,
                feature_dim,
                device,
            )
            for index, item in enumerate(active):
                item["current_state"] = updated_states[index]
                if step in STEPS and item["evaluate_ep"]:
                    final_score = score_state(
                        item["current_state"], mode, item["target_concepts"]
                    )
                    ep = (
                        (final_score - item["initial_score"]) / item["denominator"]
                        if item["denominator"]
                        else 0.0
                    )
                    totals[step]["ep"] += ep
                    totals[step]["valid"] += 1

        for item in batch:
            reordered_items = item["ordered_items"] + item["unmapped_items"]
            output_lines[item["output_index"]] = format_recommendation_line(
                item["prefix"], reordered_items, item["newline"], "uid_space"
            )
        completed = min(batch_start + len(batch), len(participants))
        print(f"processed={completed}/{len(participants)}", flush=True)

    if any(line is None for line in output_lines):
        raise RuntimeError("MMER update output was not populated for every retained uid")

    print("\nDynamic Ep results")
    for step in STEPS:
        valid = totals[step]["valid"]
        mean_ep = totals[step]["ep"] / valid if valid else 0.0
        print(f"step={step:2d} valid_students={valid:6d} mean_Ep={mean_ep:.6f}")
    print(
        f"skipped_portion_metric={skipped_portion_metric} "
        f"empty_recommendations={empty_recommendations} "
        f"unmapped_recommendation_items={unmapped_recommendation_items} "
        f"missing_test_uid={missing_test_uid}"
    )

    destination = output_path_for(paths["recommendation"])
    backup_content = "".join(output_lines)
    if args.no_write:
        print("backup=not written (--no-write)")
    elif destination.exists() and not args.overwrite:
        if destination.read_text(encoding="utf-8") != backup_content:
            raise FileExistsError(
                f"Backup already exists and differs: {destination}. "
                "Use --overwrite to replace it."
            )
        print(f"backup={destination.relative_to(PROJECT_ROOT)} (existing identical file reused)")
    else:
        destination.write_text(backup_content, encoding="utf-8")
        print(f"backup={destination.relative_to(PROJECT_ROOT)}")
    return totals


def output_path_for(source):
    return source.with_name(f"{source.stem}_update{source.suffix}")


def build_parser(mode):
    parser = argparse.ArgumentParser(
        description=(
            "Dynamically reorder recommendations with DKT and report Ep at "
            "steps 1, 5, 10 and 20."
        )
    )
    parser.add_argument("--dataset", required=True, help="Dataset, for example assist09.")
    parser.add_argument(
        "--model",
        required=True,
        nargs="+",
        help=(
            "One or more models, for example ER-TGA KCPER. "
            "Use 'all' to run every model with available inputs."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device such as auto, cuda:0 or cpu. Default: auto.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing *_update.txt backup.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Evaluate without writing the reordered recommendation backup.",
    )
    parser.add_argument(
        "--max-students",
        type=int,
        default=None,
        help="Evaluate only the first N aligned rows. Requires --no-write.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="MMER-only DKT inference batch size. Default: 32.",
    )
    parser.set_defaults(mode=mode)
    return parser


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def unique_models(raw_models):
    if len(raw_models) == 1 and raw_models[0].strip().lower() == "all":
        return list(MODEL_SPECS)

    if any(raw_model.strip().lower() == "all" for raw_model in raw_models):
        raise ValueError("'all' must be the only value passed to --model")

    selected = []
    for raw_model in raw_models:
        model = normalize_model(raw_model)
        if model not in selected:
            selected.append(model)
    return selected


def run_one(mode, args, dataset, model, device):
    if model == "MMER":
        return run_mmer_batched(mode, args, dataset, model, device)

    num_concepts = NUM_CONCEPTS[dataset]
    spec, paths = resolve_paths(dataset, model)

    set_seed(INFERENCE_SEED)
    test_df = pd.read_csv(paths["test"])
    recommendation_lines = paths["recommendation"].read_text(encoding="utf-8").splitlines(
        keepends=True
    )
    q_matrix = np.load(paths["q"]) if "q" in paths else None
    network, feature_dim, checkpoint = load_kt_net(dataset, num_concepts, device)

    line_format = spec.get("line_format", "tab")
    match_by_uid = spec.get("match_by_uid", False)
    if match_by_uid:
        records, uid_stats = mmer_first_uid_records(
            test_df, recommendation_lines, args.max_students
        )
        reordered_lines = []
    else:
        aligned_rows = min(len(test_df), len(recommendation_lines))
        if args.max_students is not None:
            aligned_rows = min(aligned_rows, args.max_students)
        records = [
            (index + 1, recommendation_lines[index], index)
            for index in range(aligned_rows)
        ]
        uid_stats = None
        reordered_lines = list(recommendation_lines)

    print(f"mode={mode} model={model} dataset={dataset} device={device}")
    print(f"test_csv={paths['test'].relative_to(PROJECT_ROOT)}")
    print(f"recommendations={paths['recommendation'].relative_to(PROJECT_ROOT)}")
    print(f"dkt_checkpoint={display_path(checkpoint)}")
    print(f"fixed_inference_seed={INFERENCE_SEED}")
    if match_by_uid:
        print(
            "uid_matching=first recommendation per uid; "
            f"source_unique_uids={uid_stats['unique_recommendation_uids']} "
            f"records_to_process={len(records)} "
            f"duplicate_recommendation_uids_ignored="
            f"{uid_stats['duplicate_recommendation_uids']} "
            f"unmatched_uids={uid_stats['unmatched_uids']}"
        )
        if uid_stats["duplicate_test_uids"]:
            print(
                "warning: duplicate test.csv uids detected; "
                "the first matching row is used"
            )
    elif len(test_df) != len(recommendation_lines):
        print(
            "warning: test rows and recommendation rows differ; "
            f"processing first {len(records)} aligned rows"
        )

    totals = {step: {"ep": 0.0, "valid": 0} for step in STEPS}
    skipped_portion_metric = 0
    empty_recommendations = 0
    unmapped_recommendation_items = 0
    missing_test_uid = 0

    for processed, (line_number, raw_line, row_index) in enumerate(records, start=1):
        prefix, items, newline = parse_recommendation_line(
            raw_line, line_number, line_format
        )
        if row_index is None:
            missing_test_uid += 1
            if match_by_uid:
                reordered_lines.append(raw_line)
            continue

        row = test_df.iloc[row_index]
        concepts, responses = parse_history(row)

        target_concepts = None
        if mode == "portion":
            if len(concepts) >= 5:
                start_index = int(len(concepts) * 0.8)
                target_concepts = list(set(concepts[start_index:]))
            if not target_concepts:
                skipped_portion_metric += 1

        if not items:
            empty_recommendations += 1
            if match_by_uid:
                reordered_lines.append(raw_line)
            continue

        candidates, unmapped_items = map_items_to_concepts(
            items,
            spec["recommendation_kind"],
            q_matrix,
            num_concepts,
            line_number,
        )
        unmapped_recommendation_items += len(unmapped_items)
        if not candidates:
            empty_recommendations += 1
            if match_by_uid:
                reordered_lines.append(raw_line)
            continue

        ordered_items, ep_by_step = dynamic_reorder(
            network,
            concepts,
            responses,
            candidates,
            num_concepts,
            feature_dim,
            device,
            mode,
            target_concepts,
        )
        reordered_items = ordered_items + unmapped_items
        reordered_line = format_recommendation_line(
            prefix, reordered_items, newline, line_format
        )
        if match_by_uid:
            reordered_lines.append(reordered_line)
        else:
            reordered_lines[row_index] = reordered_line

        for step, ep in ep_by_step.items():
            totals[step]["ep"] += ep
            totals[step]["valid"] += 1

        if processed % 100 == 0 or processed == len(records):
            print(f"processed={processed}/{len(records)}", flush=True)

    print("\nDynamic Ep results")
    for step in STEPS:
        valid = totals[step]["valid"]
        mean_ep = totals[step]["ep"] / valid if valid else 0.0
        print(f"step={step:2d} valid_students={valid:6d} mean_Ep={mean_ep:.6f}")

    print(
        f"skipped_portion_metric={skipped_portion_metric} "
        f"empty_recommendations={empty_recommendations} "
        f"unmapped_recommendation_items={unmapped_recommendation_items} "
        f"missing_test_uid={missing_test_uid}"
    )

    destination = output_path_for(paths["recommendation"])
    backup_content = "".join(reordered_lines)
    if args.no_write:
        print("backup=not written (--no-write)")
    elif destination.exists() and not args.overwrite:
        if destination.read_text(encoding="utf-8") != backup_content:
            raise FileExistsError(
                f"Backup already exists and differs: {destination}. "
                "Use --overwrite to replace it."
            )
        print(f"backup={destination.relative_to(PROJECT_ROOT)} (existing identical file reused)")
    else:
        destination.write_text(backup_content, encoding="utf-8")
        print(f"backup={destination.relative_to(PROJECT_ROOT)}")

    return totals


def run(mode, argv=None):
    parser = build_parser(mode)
    args = parser.parse_args(argv)
    if args.max_students is not None and args.max_students < 1:
        parser.error("--max-students must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_students is not None and not args.no_write:
        parser.error("--max-students requires --no-write to avoid incomplete backups")

    dataset = normalize_dataset(args.dataset)
    try:
        models = unique_models(args.model)
    except ValueError as error:
        parser.error(str(error))
    device = resolve_device(args.device)

    results = {}
    skipped = []
    for index, model in enumerate(models, start=1):
        print("\n" + "=" * 72)
        print(f"[{index}/{len(models)}] model={model}, dataset={dataset}")
        print("=" * 72)
        try:
            results[model] = run_one(mode, args, dataset, model, device)
        except FileNotFoundError as error:
            if len(args.model) == 1 and args.model[0].strip().lower() == "all":
                skipped.append((model, str(error)))
                print(f"SKIP model={model}: required input is missing")
            else:
                raise

    if len(results) > 1:
        print("\n" + "=" * 72)
        print("Batch summary")
        print("model	" + "	".join(f"Ep@{step}" for step in STEPS))
        for model, totals in results.items():
            values = []
            for step in STEPS:
                valid = totals[step]["valid"]
                mean_ep = totals[step]["ep"] / valid if valid else 0.0
                values.append(f"{mean_ep:.6f}")
            print(model + "	" + "	".join(values))
        if skipped:
            print(f"skipped_models={','.join(model for model, _ in skipped)}")

    return results


if __name__ == "__main__":
    raise SystemExit("Run test_ep_all_update.py or test_ep_por_update.py instead.")
