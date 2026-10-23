"""Statistics and baseline-locked margins for FreshQueue confirmation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .queue_regime import PAIR_METRICS


def _canonical_baseline(frame: pd.DataFrame) -> list[dict[str, Any]]:
    columns = ["setting", "seed", "ASW_cell", "knn_celltype_purity_k15"]
    selected = frame.loc[frame["method"] == "NoQueue", columns].copy()
    selected["seed"] = selected["seed"].astype(int)
    return selected.sort_values(["setting", "seed"]).to_dict("records")


def _baseline_sha256(frame: pd.DataFrame) -> str:
    encoded = json.dumps(
        _canonical_baseline(frame), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def compute_baseline_margins(
    method_metrics: pd.DataFrame,
    setting_names: list[str],
    seeds: list[int],
    *,
    multiplier: float = 0.5,
) -> dict[str, Any]:
    """Predefine half-SD margins using only completed NoQueue results."""

    if multiplier <= 0:
        raise ValueError("margin multiplier must be positive")
    baseline = method_metrics.loc[method_metrics["method"] == "NoQueue"].copy()
    expected = {(setting, int(seed)) for setting in setting_names for seed in seeds}
    observed = {
        (str(row.setting), int(row.seed))
        for row in baseline[["setting", "seed"]].itertuples(index=False)
    }
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"NoQueue margin inputs are incomplete: missing={missing}, extra={extra}")
    settings: dict[str, Any] = {}
    for setting in setting_names:
        subset = baseline.loc[baseline["setting"] == setting].sort_values("seed")
        metric_rows = {}
        for metric in ("ASW_cell", "knn_celltype_purity_k15"):
            values = subset[metric].to_numpy(float)
            sample_sd = float(np.std(values, ddof=1))
            metric_rows[metric] = {
                "baseline_values": values.tolist(),
                "baseline_mean": float(np.mean(values)),
                "baseline_sample_sd": sample_sd,
                "margin": float(multiplier * sample_sd),
            }
        settings[setting] = metric_rows
    return {
        "status": "frozen_before_freshqueue",
        "source_method": "NoQueue",
        "source_training_seeds": [int(seed) for seed in seeds],
        "formula": "margin = 0.5 * NoQueue sample SD across five training seeds",
        "sd_ddof": 1,
        "multiplier": float(multiplier),
        "baseline_sha256": _baseline_sha256(method_metrics),
        "settings": settings,
    }


def freeze_or_validate_margins(
    method_metrics: pd.DataFrame,
    setting_names: list[str],
    seeds: list[int],
    path: Path,
    *,
    multiplier: float = 0.5,
) -> dict[str, Any]:
    """Create the immutable margin lock or verify it against resumed baselines."""

    expected = compute_baseline_margins(
        method_metrics, setting_names, seeds, multiplier=multiplier
    )
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            observed = json.load(handle)
        if observed != expected:
            raise RuntimeError("frozen baseline margins do not match current NoQueue results")
        return observed
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(expected, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    return expected


def margin_frame(margins: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for setting, metrics in margins["settings"].items():
        for metric, values in metrics.items():
            rows.append(
                {
                    "setting": setting,
                    "metric": metric,
                    "baseline_mean": values["baseline_mean"],
                    "baseline_sample_sd": values["baseline_sample_sd"],
                    "multiplier": margins["multiplier"],
                    "margin": values["margin"],
                    "baseline_sha256": margins["baseline_sha256"],
                }
            )
    return pd.DataFrame(rows)


def paired_interval(values: np.ndarray, confidence: float = 0.95) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": math.nan, "sd": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    mean = float(np.mean(values))
    if n == 1:
        return {"n": 1, "mean": mean, "sd": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    sd = float(np.std(values, ddof=1))
    critical = float(stats.t.ppf((1 + confidence) / 2, n - 1))
    margin = critical * sd / math.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci_low": mean - margin,
        "ci_high": mean + margin,
    }


def method_summary(method_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (setting, method), subset in method_metrics.groupby(
        ["setting", "method"], sort=False
    ):
        row: dict[str, Any] = {
            "setting": setting,
            "dataset": subset["dataset"].iloc[0],
            "method": method,
            "n_training_seeds": int(subset["seed"].nunique()),
            "ARI_mean": float(subset["ARI_mean"].mean()),
            "ARI_sd_across_training_seeds": float(subset["ARI_mean"].std(ddof=1)),
            "mean_within_embedding_Leiden_ARI_sd": float(subset["ARI_sd"].mean()),
            "NMI_mean": float(subset["NMI_mean"].mean()),
            "NMI_sd_across_training_seeds": float(subset["NMI_mean"].std(ddof=1)),
            "mean_within_embedding_Leiden_NMI_sd": float(subset["NMI_sd"].mean()),
        }
        for metric in (
            "ASW_cell",
            "ASW_batch",
            "graph_connectivity",
            "iLISI_proxy",
            "knn_celltype_purity_k15",
        ):
            row[f"{metric}_mean"] = float(subset[metric].mean())
            row[f"{metric}_sd"] = float(subset[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def confirmatory_summary(
    paired: pd.DataFrame,
    margins: dict[str, Any],
    *,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Summarize paired deltas, CIs, positive counts, and locked-margin checks."""

    rows = []
    for setting, subset in paired.groupby("setting", sort=False):
        row: dict[str, Any] = {
            "setting": setting,
            "dataset": subset["dataset"].iloc[0],
            "n_paired_seeds": int(subset["seed"].nunique()),
        }
        for metric in PAIR_METRICS:
            short_name = metric.removesuffix("_mean")
            column = f"delta_{short_name}"
            interval = paired_interval(subset[column].to_numpy(float), confidence)
            row[f"delta_{short_name}_mean"] = interval["mean"]
            row[f"delta_{short_name}_sd"] = interval["sd"]
            row[f"delta_{short_name}_ci95_low"] = interval["ci_low"]
            row[f"delta_{short_name}_ci95_high"] = interval["ci_high"]
            if short_name in {"ARI", "NMI"}:
                row[f"positive_{short_name}_seeds"] = int((subset[column] > 0).sum())

        for metric in ("ASW_cell", "knn_celltype_purity_k15"):
            margin = float(margins["settings"][setting][metric]["margin"])
            low = float(row[f"delta_{metric}_ci95_low"])
            high = float(row[f"delta_{metric}_ci95_high"])
            row[f"{metric}_noninferiority_margin"] = margin
            row[f"{metric}_noninferior"] = low >= -margin
            row[f"{metric}_equivalent_within_margin"] = low >= -margin and high <= margin

        for descriptor in (
            "mean_pairwise_composition_overlap",
            "mean_pairwise_jensen_shannon_divergence",
            "representation_drift_mean",
            "NoQueue_representation_drift_mean",
            "queue_false_negative_mass",
            "queue_false_negative_mass_pre_mask",
        ):
            row[descriptor] = float(subset[descriptor].mean())
        rows.append(row)
    return pd.DataFrame(rows)
