#!/usr/bin/env python3
"""Export fresh Figure 4 LOBO metrics to the final preprocessing-fixed schema."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


SEEDS = (218, 394, 542, 639, 788, 814, 880)
METRICS = ("ARI", "NMI", "ASW_celltype", "Graph_connectivity")
METHOD_MAP = {
    "Centralized CONCORD": "Centralized CONCORD",
    "FedscGen zero-shot": "FedscGen (without correction)",
    "FedscGen two-stage": "FedscGen (correction workflow)",
    "FederatedBatch": "FederatedBatch (zero-shot)",
    "FedscGen (without correction)": "FedscGen (without correction)",
    "FedscGen (correction workflow)": "FedscGen (correction workflow)",
    "FederatedBatch (zero-shot)": "FederatedBatch (zero-shot)",
}
METHODS = (
    "Centralized CONCORD",
    "FedscGen (without correction)",
    "FedscGen (correction workflow)",
    "FederatedBatch (zero-shot)",
)
REFERENCE = "FederatedBatch (zero-shot)"
COMPARATORS = tuple(method for method in METHODS if method != REFERENCE)


def t_interval(values: np.ndarray) -> tuple[float, float]:
    mean = float(values.mean())
    if len(values) < 2:
        return math.nan, math.nan
    margin = float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean - margin, mean + margin


def holm(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    result = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p_values[index]))
        result[index] = running
    return result


def validate(frame: pd.DataFrame) -> list[str]:
    required = {"held_out_batch", "method", "training_seed", *METRICS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing Figure 4 columns: {sorted(missing)}")
    if set(frame.method) != set(METHODS):
        raise ValueError(f"method mismatch: {sorted(set(frame.method))}")
    if set(frame.training_seed.astype(int)) != set(SEEDS):
        raise ValueError("Figure 4 does not contain the canonical seven seeds")
    batches = list(dict.fromkeys(frame.held_out_batch.astype(str)))
    if len(batches) != 9:
        raise ValueError(f"expected nine held-out batches, found {len(batches)}")
    if frame.duplicated(["held_out_batch", "method", "training_seed"]).any():
        raise ValueError("duplicate Figure 4 scenario rows")
    if len(frame) != len(SEEDS) * len(batches) * len(METHODS):
        raise ValueError(f"incomplete Figure 4 table: {len(frame)} rows")
    return batches


def build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    seed_means = frame.groupby(["training_seed", "method"], as_index=False)[list(METRICS)].mean()
    rows = []
    for method in METHODS:
        subset = seed_means[seed_means.method == method].set_index("training_seed").loc[list(SEEDS)]
        for metric in METRICS:
            values = subset[metric].to_numpy(float)
            low, high = t_interval(values)
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "n_seeds": len(values),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "ci95_low_unadjusted": low,
                    "ci95_high_unadjusted": high,
                }
            )
    return pd.DataFrame(rows)


def build_tests(frame: pd.DataFrame) -> pd.DataFrame:
    seed_means = frame.groupby(["training_seed", "method"], as_index=False)[list(METRICS)].mean()
    indexed = seed_means.set_index(["training_seed", "method"])
    rows = []
    for comparator in COMPARATORS:
        for metric in METRICS:
            differences = np.asarray(
                [
                    indexed.loc[(seed, REFERENCE), metric]
                    - indexed.loc[(seed, comparator), metric]
                    for seed in SEEDS
                ],
                dtype=float,
            )
            low, high = t_interval(differences)
            test = stats.ttest_1samp(differences, 0.0)
            rows.append(
                {
                    "reference": REFERENCE,
                    "comparator": comparator,
                    "metric": metric,
                    "n_pairs": len(differences),
                    "mean_paired_difference": float(differences.mean()),
                    "ci95_low_unadjusted": low,
                    "ci95_high_unadjusted": high,
                    "paired_t_raw_p": float(test.pvalue),
                    "positive_seed_count": int((differences > 0).sum()),
                    "difference_direction": "FederatedBatch minus comparator",
                    "holm_family": "12 paired t-tests: 3 comparators x 4 metrics",
                }
            )
    result = pd.DataFrame(rows)
    result["paired_t_holm_p"] = holm(result.paired_t_raw_p.to_numpy(float))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = pd.read_csv(args.input.resolve())
    source = source[source.method.isin(METHOD_MAP)].copy()
    source["method"] = source.method.map(METHOD_MAP)
    batches = validate(source)
    output = source[["held_out_batch", "method", "training_seed", *METRICS]].copy()
    output["held_out_batch"] = pd.Categorical(
        output.held_out_batch.astype(str), categories=batches, ordered=True
    )
    output = output.sort_values(["training_seed", "held_out_batch", "method"]).reset_index(drop=True)
    output["held_out_batch"] = output.held_out_batch.astype(str)

    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination / "figure4_fixed_per_seed_per_heldout_all_methods.csv", index=False)
    build_summary(output).to_csv(destination / "figure4_fixed_summary.csv", index=False)
    build_tests(output).to_csv(destination / "FIGURE4_PAIRED_TESTS.csv", index=False)
    print(f"Exported final Figure 4 tables to {destination}")


if __name__ == "__main__":
    main()
