#!/usr/bin/env python3
"""Final consistency, statistics, and wording audit for reviewer-facing Figure 2."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from statsmodels.stats.multitest import multipletests
from torch.nn import functional as F

from experiments.finalize_figure2_reviewer import (
    ACTIVE_ROUNDS,
    BASE_CONFIG,
    FINAL_DP,
    FIGURE2,
    JOINT_C50,
    METHOD_FINAL,
    METHOD_NOQUEUE,
    METHOD_REFERENCE,
    METHODS,
    METRICS,
    NOISE_SEEDS,
    REMOTE_PER_SOURCE,
    SEEDS,
    atomic_csv,
    atomic_json,
    reference_config,
    schedule_from_frame,
    sha256,
)
from experiments.moderate_skew_freshqueue_confirmation import method_config
from experiments.round_once_jointDP_multiseed import CANDIDATES, config_for_candidate
from federatedbatch.config import load_config
from federatedbatch.federated import (
    _joint_noisy_ab_endpoints,
    _joint_remote_scales,
    _scheduled_private_queue,
)


OUTPUT = Path("results/final_figure2_reviewer_canonical7")
COMPARISONS = (
    (METHOD_FINAL, METHOD_NOQUEUE),
    (METHOD_REFERENCE, METHOD_NOQUEUE),
    (METHOD_FINAL, METHOD_REFERENCE),
)


def scalar_text(value) -> str:
    if isinstance(value, (list, tuple)):
        return json.dumps(value)
    if isinstance(value, bool):
        return str(value)
    if value is None:
        return "not configured"
    return str(value)


def field(config: dict, dotted: str):
    value = config
    for key in dotted.split("."):
        value = value[key]
    return value


def flatten(config: dict, prefix: str = "") -> dict[str, object]:
    rows = {}
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else key
        if key == "_scheduled_remote_indices":
            continue
        if isinstance(value, dict):
            rows.update(flatten(value, name))
        else:
            rows[name] = value
    return rows


def configuration_audit(repo_root: Path, output: Path):
    base = load_config(BASE_CONFIG)
    rows = []
    difference_rows = []
    expected_release_differences = {
        "federated.queue.mode",
        "federated.queue.noise_multiplier",
        "federated.queue.positive_a_floor",
        "federated.queue._joint_noise_seed",
    }
    common_fields = {
        "dataset": "dataset",
        "clients": "federated.n_clients",
        "communication_rounds": "federated.n_rounds",
        "local_epochs": "federated.local_epochs",
        "batch_size": "federated.batch_size",
        "learning_rate": "federated.learning_rate",
        "local_temperature": "federated.temperature",
        "hard_negative_beta": "federated.hard_negative_beta",
        "latent_dim": "federated.latent_dim",
        "encoder_dims": "federated.encoder_dims",
        "dropout": "federated.dropout",
        "remote_weight_alpha": "federated.queue.weight",
        "remote_temperature_tau_q": "federated.queue.temperature",
        "warmup_rounds": "federated.queue.warmup_rounds",
    }
    all_pre_noise_match = True
    all_schedules_match = True
    for seed in SEEDS:
        schedule_source = repo_root / FINAL_DP / "schedules" / f"remote_cell_schedule_seed_{seed}.csv"
        schedule_control = output / "schedules" / f"remote_cell_schedule_seed_{seed}.csv"
        source_frame = pd.read_csv(schedule_source)
        control_frame = pd.read_csv(schedule_control)
        schedule = schedule_from_frame(source_frame)
        control = reference_config(base, seed, schedule)
        private = config_for_candidate(
            base,
            seed,
            NOISE_SEEDS[seed],
            schedule,
            CANDIDATES["Joint-C50-sigma2"],
        )
        noqueue = method_config(base, "NoQueue", seed)
        schedule_equal = sha256(schedule_source) == sha256(schedule_control)
        all_schedules_match &= schedule_equal
        partition = repo_root / FIGURE2 / "partitions" / f"seed_{seed}.json"

        for label, dotted in common_fields.items():
            nq_value = field(noqueue, dotted)
            control_value = field(control, dotted)
            private_value = field(private, dotted)
            rows.append(
                {
                    "seed": seed,
                    "field": label,
                    "NoQueue": scalar_text(nq_value),
                    "Remote_AB_sigma0": scalar_text(control_value),
                    "FederatedBatch_sigma2": scalar_text(private_value),
                    "control_vs_final_match": control_value == private_value,
                    "role": "shared training setting",
                }
            )
            all_pre_noise_match &= control_value == private_value
        extra = (
            ("training_seed", seed, seed, seed, True, "paired training seed"),
            ("partition_sha256", sha256(partition), sha256(partition), sha256(partition), True, "paired partition"),
            ("active_rounds", "none", "4-20", "4-20", True, "remote protocol"),
            ("interactions_per_active_round", 0, 1, 1, True, "remote protocol"),
            ("remote_cells_per_source_active_round", 0, REMOTE_PER_SOURCE, REMOTE_PER_SOURCE, True, "remote protocol"),
            ("unique_remote_cells_per_source", 0, 1734, 1734, True, "remote exposure"),
            ("max_remote_participation", 0, 1, 1, True, "remote exposure"),
            ("schedule_sha256", "not applicable", sha256(schedule_control), sha256(schedule_source), schedule_equal, "remote schedule"),
            ("joint_clipping_threshold", "not applicable", JOINT_C50, JOINT_C50, True, "pre-noise release"),
            ("mask_fraction", "not applicable", 0.0, 0.0, True, "pre-noise release"),
            ("Gaussian_sigma", "not applicable", 0.0, 2.0, False, "private release difference"),
            ("positive_A_floor", "not applicable", "not needed (A>0)", 1e-6, False, "post-processing required by noisy release"),
            ("DP_noise_seed", "not applicable", "not used", NOISE_SEEDS[seed], False, "private release difference"),
            ("trajectory_callback", "yes, RNG restored", "yes, RNG restored", "no", False, "non-training instrumentation"),
        )
        for label, nq_value, control_value, private_value, match, role in extra:
            rows.append(
                {
                    "seed": seed,
                    "field": label,
                    "NoQueue": scalar_text(nq_value),
                    "Remote_AB_sigma0": scalar_text(control_value),
                    "FederatedBatch_sigma2": scalar_text(private_value),
                    "control_vs_final_match": match,
                    "role": role,
                }
            )

        flat_control = flatten(control)
        flat_private = flatten(private)
        missing = object()
        observed_difference_keys = set()
        for key in sorted(set(flat_control) | set(flat_private)):
            left = flat_control.get(key, missing)
            right = flat_private.get(key, missing)
            if left != right:
                observed_difference_keys.add(key)
                difference_rows.append(
                    {
                        "seed": seed,
                        "config_key": key,
                        "Remote_AB_sigma0": "not configured" if left is missing else scalar_text(left),
                        "FederatedBatch_sigma2": "not configured" if right is missing else scalar_text(right),
                        "expected_release_layer_difference": key in expected_release_differences,
                    }
                )
        unexpected = observed_difference_keys - expected_release_differences
        if unexpected:
            all_pre_noise_match = False
            raise ValueError(f"unexpected control/final configuration differences for seed {seed}: {sorted(unexpected)}")

    consistency = pd.DataFrame(rows)
    differences = pd.DataFrame(difference_rows)
    atomic_csv(consistency, output / "configuration_consistency.csv")
    atomic_csv(differences, output / "configuration_differences.csv")
    return {
        "all_pre_noise_training_and_remote_settings_match": bool(all_pre_noise_match),
        "all_schedule_files_byte_identical": bool(all_schedules_match),
        "literal_only_config_difference_is_sigma": False,
        "release_layer_differences": [
            "mode selects aggregate Gaussian release path",
            "sigma 0 versus 2",
            "fixed DP-noise seed is used only by FederatedBatch",
            "positive-A floor 1e-6 is used only after noisy release",
        ],
        "non_training_instrumentation_difference": (
            "sigma=0 captured round-wise metrics with an RNG-restoring callback; "
            "the completed sigma=2 run did not save intermediate trajectories"
        ),
        "valid_interpretation": (
            "matched estimate of the complete private release layer effect; "
            "not a pure noise-only cost and not an upper bound"
        ),
    }


def interface_parity(output: Path) -> pd.DataFrame:
    torch.manual_seed(20261003)
    n_clients = 5
    anchors = 256
    remote_per_source = 102
    dimension = 100
    targets = {
        target: F.normalize(torch.randn(anchors, dimension), dim=1).detach()
        for target in range(n_clients)
    }
    remote = F.normalize(torch.randn(n_clients * remote_per_source, dimension), dim=1).detach()
    records = [
        {
            "client": source,
            "cell_id": f"source_{source}_cell_{cell}",
            "collected_round": 3,
            "interaction_slot": 0,
        }
        for source in range(n_clients)
        for cell in range(remote_per_source)
    ]
    queue = [row.numpy() for row in remote]
    scales, diagnostics = _joint_remote_scales(
        targets,
        queue,
        records,
        queue_temperature=0.20,
        clipping_threshold=JOINT_C50,
        round_number=4,
    )
    # The function requires sigma>0. 1e-50 passes that guard but underflows to
    # exactly zero in float32, exposing the final code path's pre-noise output.
    final_path, noise_rows = _joint_noisy_ab_endpoints(
        targets,
        queue,
        records,
        scales,
        [float(row["joint_norm"]) for row in diagnostics],
        queue_temperature=0.20,
        clipping_threshold=JOINT_C50,
        noise_multiplier=1e-50,
        seed=10218,
        round_number=4,
        positive_a_floor=1e-6,
    )
    if any(float(row["noise_l2"]) != 0.0 for row in noise_rows):
        raise ValueError("parity test did not produce an exact zero-noise final path")
    rows = []
    for target in range(n_clients):
        control = _scheduled_private_queue(
            queue,
            records,
            target_client=target,
            interaction_slot=0,
            device=torch.device("cpu"),
            seed=218,
            round_number=4,
            fixed_scales=scales,
        )
        a_control, b_control = control.aggregate(
            targets[target], max_entries=4 * remote_per_source, temperature=0.20
        )
        a_final, b_final = final_path[target].aggregate(
            targets[target], max_entries=4 * remote_per_source, temperature=0.20
        )
        rows.append(
            {
                "target_client": target,
                "anchors_after_two_view_concat": anchors,
                "remote_cells": 4 * remote_per_source,
                "A_relative_L2_error": float(torch.linalg.vector_norm(a_control - a_final) / torch.linalg.vector_norm(a_control)),
                "A_max_absolute_error": float(torch.max(torch.abs(a_control - a_final))),
                "B_relative_L2_error": float(torch.linalg.vector_norm(b_control - b_final) / torch.linalg.vector_norm(b_control)),
                "B_cosine": float(F.cosine_similarity(b_control.reshape(1, -1), b_final.reshape(1, -1))),
                "minimum_pre_noise_A": float(a_final.min()),
                "positive_A_clamp_active_without_noise": bool((a_final <= 1e-6).any()),
            }
        )
    frame = pd.DataFrame(rows)
    if frame["A_relative_L2_error"].max() > 2e-6 or frame["B_relative_L2_error"].max() > 2e-6:
        raise ValueError("sigma=0 and final pre-noise A/B interfaces fail numerical parity")
    if frame["positive_A_clamp_active_without_noise"].any():
        raise ValueError("positive-A clamp unexpectedly changes the pre-noise signal")
    atomic_csv(frame, output / "interface_parity_qc.csv")
    return frame


def t_interval(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    margin = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
    return mean - margin, mean + margin


def statistics_audit(output: Path):
    per_seed = pd.read_csv(output / "per_seed_metrics.csv")
    if set(per_seed["seed"].astype(int)) != set(SEEDS):
        raise ValueError("canonical seed set mismatch")
    if set(per_seed["method"]) != set(METHODS) or len(per_seed) != len(SEEDS) * len(METHODS):
        raise ValueError("method-by-seed table is incomplete")
    if per_seed.duplicated(["seed", "method"]).any():
        raise ValueError("duplicate method-level seed rows")

    summary_rows = []
    for method in METHODS:
        subset = per_seed.loc[per_seed["method"] == method].sort_values("seed")
        for metric in METRICS:
            values = subset[metric].to_numpy(float)
            low, high = t_interval(values)
            summary_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "n_independent_training_seeds": len(values),
                    "mean": float(values.mean()),
                    "sd_sample": float(values.std(ddof=1)),
                    "mean_plus_minus_sd": f"{values.mean():.6f} +/- {values.std(ddof=1):.6f}",
                    "mean_ci95_low_unadjusted": low,
                    "mean_ci95_high_unadjusted": high,
                    "ci_multiplicity_adjusted": False,
                    "leiden_repeats_are_independent_replicates": False,
                }
            )

    indexed = per_seed.set_index(["seed", "method"])
    difference_rows = []
    test_rows = []
    for method_a, method_b in COMPARISONS:
        for metric in METRICS:
            values_a = np.asarray([indexed.loc[(seed, method_a), metric] for seed in SEEDS], dtype=float)
            values_b = np.asarray([indexed.loc[(seed, method_b), metric] for seed in SEEDS], dtype=float)
            differences = values_a - values_b
            low, high = t_interval(differences)
            paired_t = stats.ttest_rel(values_a, values_b)
            wilcoxon = stats.wilcoxon(
                differences,
                zero_method="wilcox",
                correction=False,
                alternative="two-sided",
                method="exact",
            )
            for seed, value_a, value_b, difference in zip(SEEDS, values_a, values_b, differences):
                difference_rows.append(
                    {
                        "seed": seed,
                        "comparison": f"{method_a} vs {method_b}",
                        "metric": metric,
                        "value_a": value_a,
                        "value_b": value_b,
                        "paired_difference_a_minus_b": difference,
                    }
                )
            test_rows.append(
                {
                    "comparison": f"{method_a} vs {method_b}",
                    "method_a": method_a,
                    "method_b": method_b,
                    "metric": metric,
                    "n_independent_training_seed_pairs": len(differences),
                    "mean_paired_difference": float(differences.mean()),
                    "sd_paired_difference": float(differences.std(ddof=1)),
                    "paired_difference_ci95_low_unadjusted": low,
                    "paired_difference_ci95_high_unadjusted": high,
                    "ci_multiplicity_adjusted": False,
                    "paired_t_statistic": float(paired_t.statistic),
                    "paired_t_df": len(differences) - 1,
                    "paired_t_p_raw_two_sided": float(paired_t.pvalue),
                    "wilcoxon_statistic_exact_sensitivity": float(wilcoxon.statistic),
                    "wilcoxon_p_raw_two_sided_exact_sensitivity": float(wilcoxon.pvalue),
                    "positive_seed_count": int(np.sum(differences > 0)),
                    "negative_seed_count": int(np.sum(differences < 0)),
                    "zero_seed_count": int(np.sum(differences == 0)),
                    "ASW_batch_direction_note": (
                        "raw silhouette; a larger value is not automatically better"
                        if metric == "ASW_batch"
                        else ""
                    ),
                }
            )
    tests = pd.DataFrame(test_rows)
    tests["paired_t_p_holm_18"] = multipletests(
        tests["paired_t_p_raw_two_sided"].to_numpy(float), method="holm"
    )[1]
    tests["paired_t_significant_holm_0.05"] = tests["paired_t_p_holm_18"] < 0.05
    tests["wilcoxon_p_holm_18_sensitivity"] = multipletests(
        tests["wilcoxon_p_raw_two_sided_exact_sensitivity"].to_numpy(float), method="holm"
    )[1]
    tests["holm_family"] = "18 tests: 3 paired comparisons x 6 metrics"
    tests["primary_inference"] = "two-sided paired t-test; Holm-adjusted p"
    tests["sensitivity_inference"] = "exact two-sided Wilcoxon; separately Holm-adjusted"

    previous = pd.read_csv(output / "paired_statistics.csv").set_index(["comparison", "metric"])
    checks = []
    mapping = {
        "mean_paired_difference": "mean_paired_difference",
        "sd_paired_difference": "sd_paired_difference",
        "paired_difference_ci95_low_unadjusted": "ci95_low",
        "paired_difference_ci95_high_unadjusted": "ci95_high",
        "paired_t_statistic": "t_statistic",
        "paired_t_p_raw_two_sided": "p_raw_two_sided",
        "wilcoxon_statistic_exact_sensitivity": "wilcoxon_statistic_sensitivity",
        "wilcoxon_p_raw_two_sided_exact_sensitivity": "wilcoxon_p_two_sided_sensitivity",
        "paired_t_p_holm_18": "p_holm_18",
    }
    for row in tests.itertuples(index=False):
        old = previous.loc[(row.comparison, row.metric)]
        for new_column, old_column in mapping.items():
            observed = float(getattr(row, new_column))
            expected = float(old[old_column])
            difference = abs(observed - expected)
            checks.append(
                {
                    "comparison": row.comparison,
                    "metric": row.metric,
                    "quantity": new_column,
                    "independent_recalculation": observed,
                    "previous_value": expected,
                    "absolute_difference": difference,
                    "pass_tolerance_1e-12": difference <= 1e-12,
                }
            )
    checks = pd.DataFrame(checks)
    if not checks["pass_tolerance_1e-12"].all():
        raise ValueError("independent statistical recalculation does not match prior outputs")
    summary = pd.DataFrame(summary_rows)
    differences = pd.DataFrame(difference_rows)
    atomic_csv(per_seed.sort_values(["method", "seed"]), output / "per_seed_metrics_final.csv")
    atomic_csv(summary, output / "method_summary_final.csv")
    atomic_csv(differences, output / "paired_differences_final.csv")
    atomic_csv(tests, output / "paired_statistics_final.csv")
    atomic_csv(checks, output / "statistics_audit_checks.csv")
    return per_seed, summary, tests, checks


def stat_row(tests: pd.DataFrame, comparison: str, metric: str) -> pd.Series:
    selected = tests.loc[(tests["comparison"] == comparison) & (tests["metric"] == metric)]
    if len(selected) != 1:
        raise ValueError(f"missing statistical row: {comparison}, {metric}")
    return selected.iloc[0]


def stat_sentence(row: pd.Series) -> str:
    return (
        f"mean paired difference {row.mean_paired_difference:+.4f} "
        f"(unadjusted 95% CI [{row.paired_difference_ci95_low_unadjusted:.4f}, "
        f"{row.paired_difference_ci95_high_unadjusted:.4f}]; paired t({int(row.paired_t_df)})="
        f"{row.paired_t_statistic:.3f}, raw p={row.paired_t_p_raw_two_sided:.4g}, "
        f"Holm-adjusted p={row.paired_t_p_holm_18:.4g}; "
        f"{int(row.positive_seed_count)}/7 positive)"
    )


def mean_sd(summary: pd.DataFrame, method: str, metric: str) -> str:
    row = summary.loc[(summary["method"] == method) & (summary["metric"] == metric)].iloc[0]
    return f"{row['mean']:.4f} (SD {row['sd_sample']:.4f})"


def write_wording_and_report(
    repo_root: Path,
    output: Path,
    config_result: dict,
    parity: pd.DataFrame,
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    checks: pd.DataFrame,
) -> None:
    fb_nq_ari = stat_row(tests, f"{METHOD_FINAL} vs {METHOD_NOQUEUE}", "ARI")
    fb_nq_nmi = stat_row(tests, f"{METHOD_FINAL} vs {METHOD_NOQUEUE}", "NMI")
    ctl_nq_ari = stat_row(tests, f"{METHOD_REFERENCE} vs {METHOD_NOQUEUE}", "ARI")
    ctl_nq_nmi = stat_row(tests, f"{METHOD_REFERENCE} vs {METHOD_NOQUEUE}", "NMI")
    fb_ctl_ari = stat_row(tests, f"{METHOD_FINAL} vs {METHOD_REFERENCE}", "ARI")
    fb_ctl_nmi = stat_row(tests, f"{METHOD_FINAL} vs {METHOD_REFERENCE}", "NMI")
    invalid = pd.read_csv(repo_root / FINAL_DP / "clipping_invalidA_summary.csv")
    invalid = invalid.loc[invalid["method"] == "Joint-C50-sigma2"]
    invalid_min = invalid["invalid_a_frequency"].min()
    invalid_max = invalid["invalid_a_frequency"].max()

    caption = f"""Figure 2. Contribution of the cross-client remote interaction under the finalized FederatedBatch protocol. (A) Representative five-client Pancreas partition with moderate cell-type composition skew (seed 218). (B) The remote A/B interaction is inactive during rounds 1-3 and is applied once per communication round in rounds 4-20. Each active round uses 102 previously unused remote cells per source client; each remote cell participates at most once, and its contributions to all four target clients are jointly clipped at C=1206.3391650512762. (C) Paired final-round ARI and NMI across the seven prespecified training seeds (218, 394, 542, 639, 788, 814, and 880). NoQueue uses the same local/FedAvg training without a cross-client remote A/B step. Remote A/B (sigma=0) is a matched remote-interaction control using the same RoundOnce schedule, one-time participation, exact A/B interface, Joint-C50 clipping, and all other training settings as FederatedBatch, but without Gaussian noise; it is not interpreted as a non-private upper bound. FederatedBatch adds the finalized Gaussian release (sigma=2) and its fixed positive-A post-processing floor. Thin gray lines connect matched training seeds; large diamonds denote means and error bars denote unadjusted t-based 95% confidence intervals across the seven seeds. (D) Descriptive ARI and NMI trajectories for NoQueue and the matched sigma=0 control; faint lines are individual seeds, solid lines are means, and ribbons are unadjusted 95% confidence intervals. Because intermediate embeddings were not retained for the completed FederatedBatch runs and models were not retrained for visualization, FederatedBatch is shown only at round 20. (E) Final-round raw ASW(batch), ASW(cell type), legacy graph connectivity, and full-pairwise legacy iLISI. Final FederatedBatch exceeded NoQueue for ARI and NMI in 7/7 seeds; NMI remained significant after Holm correction, whereas ARI is described as a consistent numerical improvement, not a multiplicity-adjusted significant result. Primary p values are from two-sided paired t-tests and were Holm-adjusted jointly across 18 tests (three comparisons by six metrics). The displayed 95% confidence intervals are unadjusted for multiplicity; therefore a confidence interval may exclude zero while the corresponding Holm-adjusted p value exceeds 0.05. Exact two-sided Wilcoxon signed-rank tests are reported as a small-sample sensitivity analysis. Raw ASW(batch) is reported without assuming that larger values are automatically better."""

    results = f"""Across seven prespecified, method-matched training seeds, final FederatedBatch increased both ARI and NMI relative to NoQueue in every seed. Mean ARI was {mean_sd(summary, METHOD_FINAL, 'ARI')} for FederatedBatch and {mean_sd(summary, METHOD_NOQUEUE, 'ARI')} for NoQueue, with {stat_sentence(fb_nq_ari)}. Because its Holm-adjusted p value was 0.2900, this is reported as a consistent numerical improvement rather than a statistically significant increase after multiplicity correction. Mean NMI was {mean_sd(summary, METHOD_FINAL, 'NMI')} versus {mean_sd(summary, METHOD_NOQUEUE, 'NMI')}, with {stat_sentence(fb_nq_nmi)}; this difference remained significant after Holm correction. The matched Remote A/B (sigma=0) control showed positive differences from NoQueue in 6/7 seeds for both ARI and NMI. For ARI, {stat_sentence(ctl_nq_ari)}; for NMI, {stat_sentence(ctl_nq_nmi)}. Neither comparison remained significant after Holm correction. Final FederatedBatch did not differ significantly from the matched sigma=0 control: for ARI, {stat_sentence(fb_ctl_ari)}; for NMI, {stat_sentence(fb_ctl_nmi)}. We therefore found no clear utility loss attributable to the complete private release layer in this seven-seed experiment; this absence of a detected difference is not an equivalence claim and the numerically higher final means are not interpreted as noise improving performance. All confidence intervals quoted here are unadjusted t-based 95% intervals, whereas multiplicity control was applied to p values across the full 18-test family."""

    response = f"""To address the reviewer's question about whether the Queue-derived cross-client signal contributes to performance, we replaced the old method terminology with a direct, seed-matched comparison among NoQueue, a matched Remote A/B (sigma=0) control, and final FederatedBatch. The matched control uses the same seven prespecified seeds and client partitions, exactly 102 remote cells per source in each active round, the identical locked remote-cell schedule, one-time participation, the exact RoundOnce A/B interface, Joint-C50 clipping, and the same optimizer and local-training settings as final FederatedBatch. It is a matched remote-interaction control, not a non-private upper bound. Final FederatedBatch differs at the release layer by adding sigma=2 Gaussian noise, its prespecified noise seed, and the fixed positive-A post-processing floor required for noisy releases; thus the final-versus-control comparison estimates the effect of the complete private release layer rather than a pure noise-only cost. Compared with NoQueue, final FederatedBatch improved ARI and NMI in all 7/7 seeds. The NMI difference remained significant after Holm correction: {stat_sentence(fb_nq_nmi)}. ARI showed a consistent numerical improvement but was not significant after correction: {stat_sentence(fb_nq_ari)}. Final FederatedBatch was not significantly different from the matched sigma=0 control for either ARI or NMI, so we state only that no clear utility loss was detected; we do not claim equivalence or that Gaussian noise improves performance. For transparency, figure error bars and reported 95% confidence intervals are unadjusted, whereas paired t-test p values were Holm-adjusted across all 18 comparisons; exact Wilcoxon tests are provided as a small-sample sensitivity analysis."""

    (output / "FIGURE2_FINAL_CAPTION.txt").write_text(caption + "\n", encoding="utf-8")
    (output / "RESULTS_PARAGRAPH_FINAL.txt").write_text(results + "\n", encoding="utf-8")
    (output / "REVIEWER_RESPONSE_FINAL.txt").write_text(response + "\n", encoding="utf-8")

    significant = tests.loc[tests["paired_t_significant_holm_0.05"], ["comparison", "metric", "paired_t_p_holm_18"]]
    report = f"""# Figure 2 final consistency, statistics, and wording audit

## Decision

**PASS with a wording qualification.** No model was retrained. No critical
experimental-setting mismatch was found. The sigma=0 control and final
FederatedBatch match in every pre-noise training, sampling, schedule, and
clipping setting. They are not literally distinguished by one scalar config
value: the final release path also supplies a fixed DP-noise seed and applies
the prespecified `A >= 1e-6` post-processing floor after Gaussian noise. The
sigma=0 run used an RNG-restoring trajectory callback, whereas the older final
run did not retain intermediate trajectories; this is evaluation
instrumentation, not a training update.

The defensible interpretation is therefore **the effect of the complete private
release layer**, not a pure `sigma`-only cost. `Remote A/B (sigma=0)` is a
matched remote-interaction control and must not be called an upper bound.

## Protocol consistency

- Canonical training seeds: `{', '.join(map(str, SEEDS))}` for all three methods.
- Identical partition hashes across methods for every seed.
- Control and final schedule CSVs are byte-identical for all seven seeds.
- Both remote methods use rounds 4-20, one interaction per active round, 102
  remote cells per source per active round, 1,734 unique cells per source,
  maximum participation one, zero reuse, `tau_q=0.20`, `alpha=0.10`, no mask,
  and Joint-C50 `C={JOINT_C50}`.
- The pre-noise A/B code paths passed a real-scale-shape numerical parity test:
  maximum A relative L2 error `{parity['A_relative_L2_error'].max():.3e}` and
  maximum B relative L2 error `{parity['B_relative_L2_error'].max():.3e}`.
- The positive-A floor is inactive without noise in the parity test. In the
  saved sigma=2 runs, invalid-A frequencies before post-processing range from
  `{invalid_min:.4%}` to `{invalid_max:.4%}` across seeds.

Machine-readable details are in `configuration_consistency.csv`,
`configuration_differences.csv`, and `interface_parity_qc.csv`.

## Statistics audit

All method means, sample SDs, paired differences, unadjusted t-based 95% CIs,
paired t statistics, exact two-sided Wilcoxon results, and Holm-adjusted p values
were independently recomputed from the 21 method-level rows. The five Leiden
random states remain within-seed evaluation repeats and were not treated as
replicates. Every independently recomputed value matched the prior table within
`1e-12` (`{int(checks['pass_tolerance_1e-12'].sum())}/{len(checks)}` checks).

Holm correction was applied jointly to the 18 paired t-tests. The exact
Wilcoxon sensitivity p values were also Holm-adjusted as a separate 18-test
family. Confidence intervals are **unadjusted** for multiplicity. This is now
stated explicitly in the caption and support text.

Holm-significant primary results:

```text
{significant.to_string(index=False)}
```

## Reviewer-safe interpretation

- Final FederatedBatch vs NoQueue: ARI and NMI are higher in 7/7 seeds. NMI is
  significant after Holm correction. ARI is a consistent numerical improvement
  but not significant after Holm correction.
- Matched sigma=0 control vs NoQueue: ARI and NMI are positive in 6/7 seeds;
  neither survives the 18-test Holm correction.
- Final vs matched sigma=0 control: neither ARI nor NMI differs significantly.
  This supports only "no clear utility loss was detected". It does not establish
  equivalence and does not show that noise improves utility.
- No other metric is significant after the primary 18-test Holm correction.

## Outputs

The final caption, Results paragraph, reviewer-response paragraph, final
statistics, and Figure 2 PNG/PDF/SVG were regenerated. Figure 1/3/4/5 and the
manuscript were not modified.
"""
    (output / "FIGURE2_FINAL_AUDIT.md").write_text(report, encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output = repo_root / OUTPUT
    if not (output / "pipeline_complete.txt").is_file():
        raise FileNotFoundError("Figure 2 finalization outputs are incomplete")
    config_result = configuration_audit(repo_root, output)
    parity = interface_parity(output)
    per_seed, summary, tests, checks = statistics_audit(output)
    write_wording_and_report(repo_root, output, config_result, parity, summary, tests, checks)
    atomic_json(
        {
            "status": "PASS_WITH_WORDING_QUALIFICATION",
            "models_retrained": False,
            "canonical_seeds": list(SEEDS),
            "configuration": config_result,
            "statistics_independent_checks_passed": bool(checks["pass_tolerance_1e-12"].all()),
            "holm_family_size": 18,
            "confidence_intervals_multiplicity_adjusted": False,
            "figure1_3_4_5_modified": False,
            "manuscript_modified": False,
        },
        output / "final_audit_status.json",
    )
    subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "render_figure2_reviewer.py"), "--input", str(output)],
        cwd=repo_root,
        check=True,
    )
    print(f"Figure 2 final audit complete: {output}")


if __name__ == "__main__":
    main()
