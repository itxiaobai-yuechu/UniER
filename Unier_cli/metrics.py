from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Mapping


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _pairs(line: str) -> Dict[str, float]:
    result: Dict[str, float] = {}
    pattern = re.compile(
        rf"\b(ndcg|map|mrr|precision|recall|f1|hit|div|acc|nov|vol|cov)\b\s*[:=]\s*({NUMBER})",
        re.IGNORECASE,
    )
    for name, value in pattern.findall(line):
        result[name.lower()] = float(value)
    return result


def parse_metrics(output: str, label: str) -> Dict[str, float]:
    """Parse the existing scripts' human-readable output without changing formulas."""
    metrics: Dict[str, float] = {}
    current_ep_scope = "portion" if "portion" in label else "all"

    for raw_line in output.splitlines():
        line = raw_line.strip()
        lower = line.lower()

        k_match = re.search(r"\bk\s*=\s*(\d+)", line, re.IGNORECASE)
        if not k_match:
            k_match = re.search(r"\bK=(\d+)", line)
        if not k_match:
            k_match = re.search(r"^@(\d+)\b", line)
        if not k_match:
            k_match = re.search(r"recommendation list length is n\s*=\s*(\d+)", line, re.IGNORECASE)
        k = k_match.group(1) if k_match else None

        for name, value in _pairs(line).items():
            key = f"{name}@{k}" if k is not None else name
            metrics[key] = value

        summary = re.search(
            rf"\b(acc|nov|div|vol|cov|pro)\b\s*(?:mean)?\s*[:=]\s*({NUMBER}).*?\bstd\s*[:=]\s*({NUMBER})",
            line,
            re.IGNORECASE,
        )
        if summary:
            name = summary.group(1).lower()
            if name == "pro":
                name = "proximity"
            metrics[name] = float(summary.group(2))
            metrics[f"{name}_std"] = float(summary.group(3))

        prox_mean = re.search(rf"proximity\s+mean\s*[:=：]\s*({NUMBER})", line, re.IGNORECASE)
        if prox_mean:
            metrics["proximity"] = float(prox_mean.group(1))
        prox_std = re.search(rf"proximity\s+std\s*[:=：]\s*({NUMBER})", line, re.IGNORECASE)
        if prox_std:
            metrics["proximity_std"] = float(prox_std.group(1))

        ep_match = re.search(
            rf"Recommendation Length\s*=\s*(\d+).*?(?:Average|Avg) Ep\s*=\s*({NUMBER})",
            line,
            re.IGNORECASE,
        )
        if ep_match:
            metrics[f"ep_{current_ep_scope}@{ep_match.group(1)}"] = float(ep_match.group(2))

        dynamic_ep_match = re.search(
            rf"\bstep\s*=\s*(\d+).*?\bmean_Ep\s*=\s*({NUMBER})",
            line,
            re.IGNORECASE,
        )
        if dynamic_ep_match:
            metrics[f"ep_{current_ep_scope}@{dynamic_ep_match.group(1)}"] = float(
                dynamic_ep_match.group(2)
            )

        reward_match = re.search(rf"Expected Reward\s*:\s*({NUMBER})", line, re.IGNORECASE)
        if reward_match:
            metrics["expected_reward"] = float(reward_match.group(1))

        mean_reward_match = re.search(
            rf"mean rewards?.*?\brewards?\s*:\s*\[\s*({NUMBER})",
            line,
            re.IGNORECASE,
        )
        if not mean_reward_match:
            mean_reward_match = re.search(rf"\bmean rewards?\s*[:=]\s*({NUMBER})", line, re.IGNORECASE)
        if mean_reward_match:
            metrics["mean_reward"] = float(mean_reward_match.group(1))

        kg_mean = re.search(rf"mean\s+(ACC|NOV|DIV)\s*=\s*({NUMBER})", line, re.IGNORECASE)
        if kg_mean:
            metrics[kg_mean.group(1).lower()] = float(kg_mean.group(2))
        kg_std = re.search(rf"std\s+(ACC|NOV|DIV)\s*=\s*({NUMBER})", line, re.IGNORECASE)
        if kg_std:
            metrics[f"{kg_std.group(1).lower()}_std"] = float(kg_std.group(2))

        if "ep calculated by [last 20%" in lower:
            current_ep_scope = "portion"
        elif "full knowledge component ep" in lower or "one-time test" in lower:
            current_ep_scope = "all"

    return metrics


def load_metrics(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(key): float(value) for key, value in data.items()}


def save_metrics(path: Path, metrics: Mapping[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(sorted(metrics.items())), handle, ensure_ascii=False, indent=2)


def print_report(model: str, dataset: str, metrics: Mapping[str, float], run_dir: Path) -> None:
    print(f"\nMetrics report | model={model} | dataset={dataset}")
    print("-" * 64)
    if not metrics:
        print("No structured metrics are available yet.")
        print("For path-level models, training and evaluation are integrated in the original runner.")
    else:
        width = max(len(key) for key in metrics)
        for key in sorted(metrics):
            print(f"{key:<{width}}  {metrics[key]:.6f}")
    print("-" * 64)
    print(f"Artifacts: {run_dir}")
