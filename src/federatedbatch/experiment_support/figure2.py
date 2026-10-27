from __future__ import annotations

import copy

import math

import numpy as np

import pandas as pd

from scipy import stats

from federatedbatch.experiment_support.roundonce import REMOTE_PER_SOURCE, SEEDS

METHOD_NOQUEUE = "NoQueue"

METHOD_REFERENCE = "Remote A/B (sigma=0)"

METHOD_FINAL = "FederatedBatch"

METHODS = (METHOD_NOQUEUE, METHOD_REFERENCE, METHOD_FINAL)

METRICS = ("ARI", "NMI", "ASW_batch", "ASW_cell", "graph_connectivity", "iLISI")

JOINT_C50 = 1206.3391650512762

def schedule_from_frame(frame: pd.DataFrame) -> dict:
    schedule: dict[str, dict[str, dict[str, list[int]]]] = {}
    for (collection_round, slot, source), group in frame.groupby(
        ["collection_round", "interaction_slot", "source_client"], sort=False
    ):
        schedule.setdefault(str(int(collection_round)), {}).setdefault(str(int(slot)), {})[
            str(int(source))
        ] = group["source_local_index"].astype(int).tolist()
    return schedule

def reference_config(base: dict, seed: int, schedule: dict) -> dict:
    configured = copy.deepcopy(base)
    configured["seed"] = int(seed)
    queue = configured["federated"]["queue"]
    queue.update(
        {
            "enabled": True,
            "mode": "round_once_joint_cellclip",
            "size": 20000,
            "samples_per_client": REMOTE_PER_SOURCE,
            "weight": 0.10,
            "temperature": 0.20,
            "mask_fraction": 0.0,
            "warmup_rounds": 3,
            "cross_client_only": True,
            "collection": "after_fedavg_with_aggregated_encoder",
            "ab_max_entries": 4 * REMOTE_PER_SOURCE,
            "joint_clipping_threshold": JOINT_C50,
            "_scheduled_remote_indices": schedule,
        }
    )
    configured["analysis"] = {"leiden_random_states": [0, 1, 2, 3, 4], "purity_k": 15}
    return configured

def t_interval(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return math.nan, math.nan
    mean = float(np.mean(values))
    margin = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
    return mean - margin, mean + margin

def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted_sorted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, original_index in enumerate(order):
        candidate = (total - rank) * p_values[original_index]
        running = max(running, candidate)
        adjusted_sorted[rank] = min(1.0, running)
    adjusted = np.empty(len(p_values), dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted

def method_and_stat_tables(per_seed: pd.DataFrame):
    summary_rows = []
    for method in METHODS:
        subset = per_seed.loc[per_seed["method"] == method].sort_values("seed")
        if list(subset["seed"].astype(int)) != sorted(SEEDS):
            raise ValueError(f"incomplete seven-seed method table for {method}")
        for metric in METRICS:
            values = subset[metric].to_numpy(float)
            ci_low, ci_high = t_interval(values)
            summary_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                }
            )
    comparisons = (
        (METHOD_FINAL, METHOD_NOQUEUE),
        (METHOD_REFERENCE, METHOD_NOQUEUE),
        (METHOD_FINAL, METHOD_REFERENCE),
    )
    indexed = per_seed.set_index(["seed", "method"])
    difference_rows = []
    test_rows = []
    for method_a, method_b in comparisons:
        for metric in METRICS:
            differences = []
            for seed in SEEDS:
                value_a = float(indexed.loc[(seed, method_a), metric])
                value_b = float(indexed.loc[(seed, method_b), metric])
                difference = value_a - value_b
                differences.append(difference)
                difference_rows.append(
                    {
                        "seed": seed,
                        "comparison": f"{method_a} vs {method_b}",
                        "method_a": method_a,
                        "method_b": method_b,
                        "metric": metric,
                        "value_a": value_a,
                        "value_b": value_b,
                        "paired_difference_a_minus_b": difference,
                    }
                )
            values = np.asarray(differences, dtype=float)
            ci_low, ci_high = t_interval(values)
            test = stats.ttest_1samp(values, popmean=0.0)
            try:
                wilcoxon = stats.wilcoxon(values, alternative="two-sided", method="exact")
                w_stat, w_p = float(wilcoxon.statistic), float(wilcoxon.pvalue)
            except ValueError:
                w_stat, w_p = math.nan, math.nan
            test_rows.append(
                {
                    "comparison": f"{method_a} vs {method_b}",
                    "method_a": method_a,
                    "method_b": method_b,
                    "metric": metric,
                    "n_pairs": len(values),
                    "mean_paired_difference": float(np.mean(values)),
                    "sd_paired_difference": float(np.std(values, ddof=1)),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "t_statistic": float(test.statistic),
                    "df": len(values) - 1,
                    "p_raw_two_sided": float(test.pvalue),
                    "wilcoxon_statistic_sensitivity": w_stat,
                    "wilcoxon_p_two_sided_sensitivity": w_p,
                    "positive_seed_count": int(np.sum(values > 0)),
                    "holm_family": "all 18 paired t-tests",
                    "interpretation_note": (
                        "raw ASW(batch): larger is not automatically better"
                        if metric == "ASW_batch"
                        else "difference is method_a minus method_b"
                    ),
                }
            )
    tests = pd.DataFrame(test_rows)
    tests["p_holm_18"] = holm_adjust(tests["p_raw_two_sided"].to_numpy(float))
    tests["significant_holm_0.05"] = tests["p_holm_18"] < 0.05
    return pd.DataFrame(summary_rows), pd.DataFrame(difference_rows), tests
