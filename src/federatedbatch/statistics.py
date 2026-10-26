"""Statistical summaries for repeated manuscript experiments."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats


METRICS = (
    "ARI",
    "NMI",
    "ASW_batch",
    "ASW_celltype",
    "Graph_connectivity",
    "kNN_batch_mixing_scaled",
    "Label_transfer_accuracy",
    "Label_transfer_macro_F1",
    "Label_transfer_coverage",
)


def summarize_metrics(
    frame: pd.DataFrame,
    group_columns: Sequence[str] = (),
    metrics: Iterable[str] = METRICS,
) -> pd.DataFrame:
    """Return long-form mean, SD, and t-based 95% confidence intervals."""
    group_columns = list(group_columns)
    if group_columns:
        groups = frame.groupby(group_columns, dropna=False, sort=False)
    else:
        groups = [((), frame)]

    rows = []
    for keys, group in groups:
        if not isinstance(keys, tuple):
            keys = (keys,)
        labels = dict(zip(group_columns, keys))
        for metric in (metric for metric in metrics if metric in group.columns):
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            n = len(values)
            mean = float(np.mean(values)) if n else np.nan
            sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
            if n > 1:
                margin = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
                ci_low, ci_high = mean - margin, mean + margin
            else:
                ci_low = ci_high = np.nan
            rows.append(
                {
                    **labels,
                    "metric": metric,
                    "n": n,
                    "mean": mean,
                    "sd": sd,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    adjusted = np.full(len(p_values), np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(p_values))
    if not len(finite_indices):
        return adjusted
    ordered = finite_indices[np.argsort(p_values[finite_indices])]
    running = 0.0
    m = len(ordered)
    for rank, index in enumerate(ordered):
        value = min(1.0, (m - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def paired_method_tests(
    frame: pd.DataFrame,
    *,
    unit_columns: Sequence[str],
    reference: str,
    method_column: str = "Method",
    metrics: Iterable[str] = METRICS,
) -> pd.DataFrame:
    """Run two-sided paired Wilcoxon tests and report paired standardized effects."""
    unit_columns = list(unit_columns)
    duplicated = frame.duplicated([*unit_columns, method_column])
    if duplicated.any():
        raise ValueError("paired comparison units are not unique within method")
    methods = [value for value in frame[method_column].unique() if value != reference]
    rows = []
    for comparator in methods:
        for metric in (metric for metric in metrics if metric in frame.columns):
            subset = frame[frame[method_column].isin([reference, comparator])]
            wide = subset.pivot(index=unit_columns, columns=method_column, values=metric)
            if reference not in wide or comparator not in wide:
                continue
            paired = wide[[reference, comparator]].apply(pd.to_numeric, errors="coerce").dropna()
            difference = paired[reference].to_numpy() - paired[comparator].to_numpy()
            n = len(difference)
            if n == 0:
                statistic = p_value = effect = np.nan
            elif np.allclose(difference, 0):
                statistic, p_value, effect = 0.0, 1.0, 0.0
            else:
                result = stats.wilcoxon(
                    paired[reference],
                    paired[comparator],
                    alternative="two-sided",
                    method="auto",
                )
                statistic, p_value = float(result.statistic), float(result.pvalue)
                difference_sd = np.std(difference, ddof=1) if n > 1 else np.nan
                effect = (
                    float(np.mean(difference) / difference_sd)
                    if n > 1 and difference_sd > 0
                    else np.nan
                )
            rows.append(
                {
                    "reference": reference,
                    "comparator": comparator,
                    "metric": metric,
                    "n_pairs": n,
                    "mean_paired_difference": float(np.mean(difference)) if n else np.nan,
                    "cohen_dz": effect,
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                }
            )
    output = pd.DataFrame(rows)
    if len(output):
        output["p_holm"] = _holm_adjust(output["p_value"].to_numpy(float))
    return output
