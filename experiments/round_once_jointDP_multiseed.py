#!/usr/bin/env python3
"""Seven-seed validation of the two pre-specified RoundOnce joint-DP candidates."""

from __future__ import annotations

import copy
import gc
import hashlib
import importlib.util
import json
import math
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.optimize import brentq
from scipy.stats import norm

from experiments.fresh_queue_regime import evaluate_embedding
from federatedbatch.config import load_config, resolve_repo_path
from federatedbatch.data import normalize, read_dataset
from federatedbatch.federated import train_federated


OUTPUT = Path("results/round_once_jointDP_multiseed_102")
FIGURE2 = Path("results/figure2_unified_7seed_original_environment_20260923")
LEGACY_VALUES = Path(
    "figure2_data/unified_7seed_original_environment_metrics/legacy_metric_seed_values.csv"
)
LEGACY_SCRIPT = Path(
    "external_artifacts/figure1/scripts/recompute_legacy_metrics.py"
)
BASE_CONFIG = "configs/figure2_unified_7seed_original_environment.yaml"
SEEDS = (218, 394, 542, 639, 788, 814, 880)
NOISE_SEEDS = {
    218: 10218,
    394: 10394,
    542: 10542,
    639: 10639,
    788: 10788,
    814: 10814,
    880: 10880,
}
ACTIVE_ROUNDS = tuple(range(4, 21))
REMOTE_PER_SOURCE = 102
N_CLIENTS = 5
EXPECTED_DATA_SHA256 = "97e6dfd65553e4d10aa3ef5d904362970a75c677c31d70fabc9234191a09db8c"
METRICS = ("ARI", "NMI", "graph_connectivity", "ASW_batch", "ASW_cell", "iLISI")
CANDIDATES = {
    "Joint-C50-sigma2": {"C_label": "C50", "C": 1206.3391650512762, "sigma": 2.0},
    "Joint-C75-sigma4": {"C_label": "C75", "C": 1669.772952068228, "sigma": 4.0},
}
BASELINES = ("Official-NoQueue", "Official-VanillaFreshQueue")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_protocol(config: dict) -> None:
    fed = config["federated"]
    queue = fed["queue"]
    observed = (
        config["dataset"],
        fed["n_clients"],
        fed["n_rounds"],
        fed["local_epochs"],
        fed["batch_size"],
        fed["learning_rate"],
        fed["temperature"],
        fed["latent_dim"],
        fed["encoder_dims"],
        queue["weight"],
        queue["temperature"],
        queue["warmup_rounds"],
    )
    expected = ("pancreas", 5, 20, 5, 128, 0.001, 0.07, 100, [128], 0.1, 0.2, 3)
    if observed != expected:
        raise ValueError(f"Figure 2 protocol mismatch: {observed}")
    if tuple(config["confirmation"]["training_seeds"]) != SEEDS:
        raise ValueError("Figure 2 seed order does not match the pre-specified seven seeds")
    if NOISE_SEEDS != {seed: 10000 + seed for seed in SEEDS}:
        raise ValueError("DP noise-seed mapping was altered")


def load_partition(path: Path, seed: int, n_obs: int) -> list[np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload["seed"]) != seed or int(payload["n_clients"]) != N_CLIENTS:
        raise ValueError(f"partition metadata mismatch for seed {seed}")
    groups = [np.asarray(values, dtype=np.int64) for values in payload["client_indices"]]
    flat = np.concatenate(groups)
    if len(flat) != n_obs or len(np.unique(flat)) != n_obs:
        raise ValueError(f"partition for seed {seed} does not cover the dataset exactly once")
    return groups


def build_schedule(
    groups: list[np.ndarray], cell_ids: np.ndarray, seed: int
) -> tuple[dict, pd.DataFrame]:
    """Freeze the RoundOnce schedule; exhaustion is an error rather than implicit reuse."""

    schedule: dict[str, dict[str, dict[str, list[int]]]] = {}
    rows = []
    required_per_source = len(ACTIVE_ROUNDS) * REMOTE_PER_SOURCE
    for source, group in enumerate(groups):
        if len(group) < required_per_source:
            raise ValueError(
                f"seed {seed} client {source} has {len(group)} cells but needs "
                f"{required_per_source} unique remote cells"
            )
        rng = np.random.default_rng(seed * 1009 + source * 7919 + 104729)
        permutation = rng.permutation(len(group))[:required_per_source]
        cursor = 0
        for active_round in ACTIVE_ROUNDS:
            collection_round = active_round - 1
            selected = permutation[cursor : cursor + REMOTE_PER_SOURCE].astype(int).tolist()
            cursor += REMOTE_PER_SOURCE
            schedule.setdefault(str(collection_round), {}).setdefault("0", {})[
                str(source)
            ] = selected
            for local_index in selected:
                global_index = int(group[local_index])
                rows.append(
                    {
                        "seed": seed,
                        "active_round": active_round,
                        "collection_round": collection_round,
                        "interaction_slot": 0,
                        "source_client": source,
                        "source_local_index": local_index,
                        "global_index": global_index,
                        "cell_id": str(cell_ids[global_index]),
                    }
                )
    frame = pd.DataFrame(rows)
    expected_rows = len(ACTIVE_ROUNDS) * N_CLIENTS * REMOTE_PER_SOURCE
    if len(frame) != expected_rows or frame["cell_id"].nunique() != expected_rows:
        raise ValueError(f"seed {seed} schedule is not globally one-participation")
    if int(frame.groupby("cell_id").size().max()) != 1:
        raise ValueError(f"seed {seed} schedule reuses a remote cell")
    per_round_source = frame.groupby(["active_round", "source_client"]).size()
    if set(per_round_source.astype(int)) != {REMOTE_PER_SOURCE}:
        raise ValueError(
            f"seed {seed} schedule is not {REMOTE_PER_SOURCE} cells per round/source"
        )
    return schedule, frame


def config_for_candidate(
    base: dict,
    seed: int,
    noise_seed: int,
    schedule: dict,
    candidate: dict,
) -> dict:
    configured = copy.deepcopy(base)
    configured["seed"] = seed
    queue = configured["federated"]["queue"]
    queue.update(
        {
            "enabled": True,
            "mode": "round_once_joint_cellclip_noise",
            "size": 20000,
            "samples_per_client": REMOTE_PER_SOURCE,
            "weight": 0.10,
            "temperature": 0.20,
            "mask_fraction": 0.0,
            "warmup_rounds": 3,
            "cross_client_only": True,
            "collection": "after_fedavg_with_aggregated_encoder",
            "ab_max_entries": 4 * REMOTE_PER_SOURCE,
            "joint_clipping_threshold": float(candidate["C"]),
            "noise_multiplier": float(candidate["sigma"]),
            "positive_a_floor": 1e-6,
            "_joint_noise_seed": int(noise_seed),
            "_scheduled_remote_indices": schedule,
        }
    )
    configured["analysis"] = {"leiden_random_states": [0, 1, 2, 3, 4], "purity_k": 15}
    return configured


def load_legacy(repo_root: Path):
    path = (repo_root / LEGACY_SCRIPT).resolve()
    spec = importlib.util.spec_from_file_location("legacy_joint_dp_multiseed", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load legacy evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gaussian_delta(epsilon: float, sigma: float) -> float:
    return max(
        0.0,
        float(
            norm.cdf(0.5 / sigma - epsilon * sigma)
            - math.exp(epsilon) * norm.cdf(-0.5 / sigma - epsilon * sigma)
        ),
    )


def analytic_gaussian_epsilon(sigma: float, delta: float) -> float:
    upper = 1.0
    while gaussian_delta(upper, sigma) > delta:
        upper *= 2.0
    return float(brentq(lambda value: gaussian_delta(value, sigma) - delta, 0.0, upper))


def privacy_table() -> pd.DataFrame:
    rows = []
    for method, candidate in CANDIDATES.items():
        for delta in (1e-5, 1e-6):
            sigma = float(candidate["sigma"])
            rows.append(
                {
                    "method": method,
                    "C_label": candidate["C_label"],
                    "C": candidate["C"],
                    "sigma": sigma,
                    "delta": delta,
                    "epsilon": analytic_gaussian_epsilon(sigma, delta),
                    "accountant": "exact analytic Gaussian privacy profile",
                    "formula": "delta(eps)=Phi(1/(2sigma)-eps*sigma)-exp(eps)*Phi(-1/(2sigma)-eps*sigma)",
                    "privacy_unit": "one remotely participating cell",
                    "adjacency": "add/remove one remote cell",
                    "l2_sensitivity": "C",
                    "noise_std": "sigma*C",
                    "mechanisms_per_participating_cell": 1,
                    "round_composition_per_cell": 1,
                    "subsampling_amplification": False,
                    "scope": "remote cross-client A/B release channel only",
                }
            )
    return pd.DataFrame(rows)


def load_official_baselines(repo_root: Path) -> pd.DataFrame:
    source = repo_root / FIGURE2
    primary = pd.read_csv(source / "method_metrics.csv")
    legacy = pd.read_csv(repo_root / LEGACY_VALUES)
    rows = []
    method_map = {
        "NoQueue": "Official-NoQueue",
        "FreshQueue": "Official-VanillaFreshQueue",
    }
    for seed in SEEDS:
        partition_path = source / "partitions" / f"seed_{seed}.json"
        partition_hash = sha256(partition_path)
        for source_method, method in method_map.items():
            primary_row = primary.loc[
                (primary.seed == seed) & (primary.method == source_method)
            ]
            legacy_rows = legacy.loc[
                (legacy.seed == seed) & (legacy.method == source_method)
            ]
            if len(primary_row) != 1 or set(legacy_rows.metric) != {
                "ASW_batch",
                "ASW_cell",
                "Graph_connectivity",
                "iLISI",
            }:
                raise ValueError(f"incomplete official baseline for seed {seed} {source_method}")
            primary_row = primary_row.iloc[0]
            values = legacy_rows.set_index("metric")["value"]
            embedding = source / "runs" / f"seed_{seed}" / source_method / "final_embedding.npy"
            expected_hashes = set(legacy_rows.embedding_sha256.astype(str))
            if expected_hashes != {sha256(embedding)}:
                raise ValueError(f"official embedding hash mismatch for seed {seed} {source_method}")
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "ARI": float(primary_row.ARI_mean),
                    "NMI": float(primary_row.NMI_mean),
                    "graph_connectivity": float(values["Graph_connectivity"]),
                    "ASW_batch": float(values["ASW_batch"]),
                    "ASW_cell": float(values["ASW_cell"]),
                    "iLISI": float(values["iLISI"]),
                    "C_label": "",
                    "C": math.nan,
                    "sigma": math.nan,
                    "epsilon_delta_1e5": math.nan,
                    "noise_seed": math.nan,
                    "clipping_fraction": math.nan,
                    "invalid_a_frequency": math.nan,
                    "invalid_a_count": math.nan,
                    "a_count": math.nan,
                    "training_stable": True,
                    "max_remote_participation_count": math.nan,
                    "partition_sha256": partition_hash,
                    "embedding_sha256": sha256(embedding),
                    "source": str(embedding),
                }
            )
    return pd.DataFrame(rows)


def cleanup_evaluation(adata, identity: str) -> None:
    key = f"X_regime_{identity}"
    neighbors_key = f"neighbors_{key}"
    adata.obsm.pop(key, None)
    adata.uns.pop(neighbors_key, None)
    adata.obsp.pop(f"{neighbors_key}_connectivities", None)
    adata.obsp.pop(f"{neighbors_key}_distances", None)


def evaluate_candidate(legacy, adata, config: dict, embedding: np.ndarray, identity: str) -> dict:
    try:
        repeats, metrics = evaluate_embedding(adata, embedding, config, identity)
    finally:
        cleanup_evaluation(adata, identity)
    batches = np.asarray(adata.obs["batch"]).astype(str)
    labels = np.asarray(adata.obs["celltype"]).astype(str)
    return {
        "ARI": float(metrics["ARI_mean"]),
        "NMI": float(metrics["NMI_mean"]),
        "graph_connectivity": float(legacy.old_graph_connectivity(embedding, labels)),
        "ASW_batch": float(legacy.old_asw(embedding, batches)),
        "ASW_cell": float(legacy.old_asw(embedding, labels)),
        "iLISI": float(legacy.old_ilisi(embedding, batches, block_size=512)),
        "ARI_leiden_sd": float(metrics["ARI_sd"]),
        "NMI_leiden_sd": float(metrics["NMI_sd"]),
        "leiden_repeats_json": json.dumps(repeats, sort_keys=True),
    }


def run_diagnostics(result, method: str, seed: int, run_dir: Path) -> dict:
    clipping = pd.DataFrame(result.joint_clipping_diagnostics)
    noise = pd.DataFrame(result.joint_noise_diagnostics)
    if len(clipping) != len(ACTIVE_ROUNDS) * N_CLIENTS * REMOTE_PER_SOURCE:
        raise ValueError(f"unexpected per-cell clipping row count for seed {seed} {method}")
    source_rows = noise[noise.record_type == "joint_source_release"].copy()
    target_rows = noise[noise.record_type == "target_postprocess"].copy()
    if len(source_rows) != len(ACTIVE_ROUNDS) * N_CLIENTS or len(target_rows) != len(source_rows):
        raise ValueError(f"unexpected joint release count for seed {seed} {method}")
    expected_noise_seed = NOISE_SEEDS[seed]
    expected_standard_seeds = {
        int((expected_noise_seed * 1_000_003 + round_number * 1_009 + source * 97 + 7_919) % (2**63 - 1))
        for round_number in ACTIVE_ROUNDS
        for source in range(N_CLIENTS)
    }
    if set(source_rows.standard_noise_seed.astype(int)) != expected_standard_seeds:
        raise ValueError(f"DP noise seed diagnostics mismatch for seed {seed} {method}")
    invalid_count = int(target_rows.invalid_a_before_clamp_count.sum())
    a_count = int(target_rows.a_count.sum())
    atomic_csv(clipping, run_dir / "per_cell_clipping_diagnostics.csv")
    atomic_csv(noise, run_dir / "clipping_noise_diagnostics.csv")
    return {
        "seed": seed,
        "method": method,
        "C_label": CANDIDATES[method]["C_label"],
        "C": CANDIDATES[method]["C"],
        "sigma": CANDIDATES[method]["sigma"],
        "noise_seed": expected_noise_seed,
        "clipping_fraction": float(clipping.clipped.astype(bool).mean()),
        "median_clipping_scale": float(clipping.clipping_scale.median()),
        "p10_clipping_scale": float(clipping.clipping_scale.quantile(0.10)),
        "p10_joint_norm": float(clipping.joint_norm.quantile(0.10)),
        "median_joint_norm": float(clipping.joint_norm.median()),
        "p90_joint_norm": float(clipping.joint_norm.quantile(0.90)),
        "invalid_a_count": invalid_count,
        "a_count": a_count,
        "invalid_a_frequency": float(invalid_count / a_count),
        "joint_source_release_count": int(len(source_rows)),
        "target_postprocess_count": int(len(target_rows)),
        "training_stable": bool(np.isfinite(result.embeddings).all()),
    }


def t_interval(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        return math.nan, math.nan
    mean = float(np.mean(values))
    sem = float(stats.sem(values))
    critical = float(stats.t.ppf(0.975, len(values) - 1))
    return mean - critical * sem, mean + critical * sem


def statistical_outputs(per_seed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    difference_rows = []
    comparison_pairs = (
        ("Joint-C50-sigma2", "Official-NoQueue"),
        ("Joint-C50-sigma2", "Official-VanillaFreshQueue"),
        ("Joint-C75-sigma4", "Official-NoQueue"),
        ("Joint-C75-sigma4", "Official-VanillaFreshQueue"),
        ("Joint-C50-sigma2", "Joint-C75-sigma4"),
    )
    indexed = per_seed.set_index(["seed", "method"])
    for method_a, method_b in comparison_pairs:
        for metric in ("ARI", "NMI"):
            for seed in SEEDS:
                value_a = float(indexed.loc[(seed, method_a), metric])
                value_b = float(indexed.loc[(seed, method_b), metric])
                difference_rows.append(
                    {
                        "seed": seed,
                        "comparison": f"{method_a} vs {method_b}",
                        "method_a": method_a,
                        "method_b": method_b,
                        "metric": metric,
                        "value_a": value_a,
                        "value_b": value_b,
                        "paired_difference_a_minus_b": value_a - value_b,
                    }
                )
    differences = pd.DataFrame(difference_rows)

    summary_rows = []
    for method in (*BASELINES, *CANDIDATES):
        subset = per_seed[per_seed.method == method]
        for metric in METRICS:
            values = subset.sort_values("seed")[metric].to_numpy(dtype=float)
            ci_low, ci_high = t_interval(values)
            summary_rows.append(
                {
                    "record_type": "method_summary",
                    "method": method,
                    "comparison": "",
                    "metric": metric,
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "mean_paired_difference": math.nan,
                    "sd_paired_difference": math.nan,
                    "paired_difference_ci95_low": math.nan,
                    "paired_difference_ci95_high": math.nan,
                    "t_statistic": math.nan,
                    "t_df": math.nan,
                    "t_p_raw_two_sided": math.nan,
                    "wilcoxon_statistic": math.nan,
                    "wilcoxon_p_two_sided_secondary": math.nan,
                    "positive_seed_count": math.nan,
                }
            )
    for (comparison, metric), group in differences.groupby(["comparison", "metric"], sort=False):
        values = group.sort_values("seed").paired_difference_a_minus_b.to_numpy(dtype=float)
        ci_low, ci_high = t_interval(values)
        t_result = stats.ttest_1samp(values, popmean=0.0)
        try:
            wilcoxon = stats.wilcoxon(values, alternative="two-sided", method="exact")
            w_stat, w_p = float(wilcoxon.statistic), float(wilcoxon.pvalue)
        except ValueError:
            w_stat, w_p = math.nan, math.nan
        summary_rows.append(
            {
                "record_type": "paired_comparison",
                "method": "",
                "comparison": comparison,
                "metric": metric,
                "n": len(values),
                "mean": math.nan,
                "sd": math.nan,
                "ci95_low": math.nan,
                "ci95_high": math.nan,
                "mean_paired_difference": float(np.mean(values)),
                "sd_paired_difference": float(np.std(values, ddof=1)),
                "paired_difference_ci95_low": ci_low,
                "paired_difference_ci95_high": ci_high,
                "t_statistic": float(t_result.statistic),
                "t_df": len(values) - 1,
                "t_p_raw_two_sided": float(t_result.pvalue),
                "wilcoxon_statistic": w_stat,
                "wilcoxon_p_two_sided_secondary": w_p,
                "positive_seed_count": int(np.sum(values > 0)),
            }
        )
    return differences, pd.DataFrame(summary_rows)


def decision_from_results(summary: pd.DataFrame, per_seed: pd.DataFrame) -> tuple[str, str]:
    comparisons = summary[summary.record_type == "paired_comparison"].set_index(
        ["comparison", "metric"]
    )
    assessments = {}
    for candidate in CANDIDATES:
        comparison = f"{candidate} vs Official-NoQueue"
        ari = comparisons.loc[(comparison, "ARI")]
        nmi = comparisons.loc[(comparison, "NMI")]
        stable = bool(per_seed.loc[per_seed.method == candidate, "training_stable"].astype(bool).all())
        mean_positive = ari.mean_paired_difference > 0 and nmi.mean_paired_difference > 0
        consistent = (
            mean_positive
            and int(ari.positive_seed_count) >= 5
            and int(nmi.positive_seed_count) >= 5
            and stable
        )
        ci_positive = (
            ari.paired_difference_ci95_low > 0 and nmi.paired_difference_ci95_low > 0
        )
        assessments[candidate] = {
            "mean_positive": mean_positive,
            "consistent": consistent,
            "ci_positive": ci_positive,
            "stable": stable,
        }
    if assessments["Joint-C75-sigma4"]["consistent"]:
        selected = "Joint-C75-sigma4"
    elif assessments["Joint-C50-sigma2"]["consistent"]:
        selected = "Joint-C50-sigma2"
    elif assessments["Joint-C75-sigma4"]["mean_positive"]:
        selected = "Joint-C75-sigma4"
    elif assessments["Joint-C50-sigma2"]["mean_positive"]:
        selected = "Joint-C50-sigma2"
    else:
        selected = "none"
    if selected != "none" and assessments[selected]["consistent"] and assessments[selected]["ci_positive"]:
        classification = "STRONG GO"
    elif selected != "none" and assessments[selected]["consistent"]:
        classification = "GO"
    elif selected != "none":
        classification = "MARGINAL"
    else:
        classification = "NO-GO"
    return classification, selected


def write_report(
    output: Path,
    per_seed: pd.DataFrame,
    summary: pd.DataFrame,
    accounting: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    classification, selected = decision_from_results(summary, per_seed)
    comparisons = summary[summary.record_type == "paired_comparison"].copy()
    nq = comparisons[comparisons.comparison.str.endswith("vs Official-NoQueue")]
    candidate_lines = []
    for candidate in CANDIDATES:
        rows = nq[nq.comparison == f"{candidate} vs Official-NoQueue"].set_index("metric")
        candidate_lines.append(
            f"- {candidate}: mean paired ARI difference {rows.loc['ARI', 'mean_paired_difference']:.6f} "
            f"(95% CI [{rows.loc['ARI', 'paired_difference_ci95_low']:.6f}, "
            f"{rows.loc['ARI', 'paired_difference_ci95_high']:.6f}]), "
            f"{int(rows.loc['ARI', 'positive_seed_count'])}/7 positive; mean paired NMI difference "
            f"{rows.loc['NMI', 'mean_paired_difference']:.6f} "
            f"(95% CI [{rows.loc['NMI', 'paired_difference_ci95_low']:.6f}, "
            f"{rows.loc['NMI', 'paired_difference_ci95_high']:.6f}]), "
            f"{int(rows.loc['NMI', 'positive_seed_count'])}/7 positive."
        )
    any_unstable = bool((~per_seed[per_seed.method.isin(CANDIDATES)].training_stable.astype(bool)).any())
    invalid = diagnostics.groupby("method").agg(
        invalid_a_count=("invalid_a_count", "sum"),
        a_count=("a_count", "sum"),
        seed_runs=("seed", "count"),
    )
    invalid["invalid_a_frequency"] = invalid.invalid_a_count / invalid.a_count
    epsilon_text = accounting[accounting.delta == 1e-5][
        ["method", "sigma", "epsilon"]
    ].to_string(index=False)
    compact_metrics = per_seed[
        [
            "seed",
            "method",
            "ARI",
            "NMI",
            "graph_connectivity",
            "ASW_batch",
            "ASW_cell",
            "iLISI",
            "clipping_fraction",
            "invalid_a_frequency",
            "training_stable",
            "max_remote_participation_count",
        ]
    ].to_string(index=False)
    report = f"""# RoundOnce joint-DP seven-seed validation

## Scope and locked protocol

Pancreas, five clients, Figure 2 seeds `{', '.join(map(str, SEEDS))}`, 20 rounds,
five local epochs, batch size 128, RoundOnce active only in rounds 4--20,
`tau_q=0.20`, `alpha=0.10`, and no top-5% masking. Each seed reused its exact
Figure 2 partition. Every remote schedule was fixed before candidate training,
used exactly 102 cells per source and active round (1,734 distinct cells per
source and 8,670 per seed), and had maximum participation one. The previous
128-cell seed-218 feasibility pilots are excluded from all multi-seed results.

Only `Joint-C50-sigma2` and `Joint-C75-sigma4` were trained. Noise seeds were
fixed as `{json.dumps(NOISE_SEEDS, sort_keys=True)}` and were never selected or
rerun based on performance. For each source and active round, one Gaussian
vector was added to the complete jointly clipped four-target A/B output before
target blocks were split. There was no adaptive clipping, MPC, HE, or TEE.
Figure 1/2/5 files were not modified.

## Privacy accounting

The independently recomputed exact analytic Gaussian privacy profile is
`delta(epsilon)=Phi(1/(2sigma)-epsilon*sigma)-exp(epsilon)Phi(-1/(2sigma)-epsilon*sigma)`.
Adjacency is add/remove one remote cell, sensitivity is `C`, and noise standard
deviation is `sigma*C`. Reducing the number of summed remote cells from 128 to
102 changes utility and signal-to-noise ratio, but it does not change this
single-cell sensitivity bound because each cell is still clipped once at `C`.
No subsampling amplification or cross-round composition is used because each
remote cell participates at most once. This accounts only for the remote A/B
release channel, not end-to-end model DP.

```text
{epsilon_text}
```

## Per-seed metrics

ARI and NMI are method-level means over the same five Leiden random states;
those five states are not treated as independent replicates. ASW, graph
connectivity, and iLISI use the manuscript's legacy definitions.

```text
{compact_metrics}
```

## Matched primary results

{chr(10).join(candidate_lines)}

Paired t-tests and their t-based confidence intervals are primary descriptive
inference. Exact two-sided Wilcoxon signed-rank tests are included as a
secondary small-sample description in `summary_statistics.csv`. With n=7,
p-values and confidence intervals should not be overinterpreted.

## Numerical QC

```text
{invalid.reset_index().to_string(index=False)}
```

All schedules have maximum remote participation one. Numerical instability in
any candidate run: **{any_unstable}**. Non-positive A values were counted before
the fixed `1e-6` post-processing clamp.

## Answers

1. C50/sigma2 consistency versus NoQueue is quantified above by its paired mean
   differences and positive-seed counts.
2. C75/sigma4 consistency versus NoQueue is quantified above by the same locked
   criteria.
3. The selected privacy-utility candidate is **{selected}**; selection first
   requires consistency, then prefers the lower epsilon C75/sigma4 candidate.
4. Invalid-A frequencies are reported above; training instability was
   **{any_unstable}**.
5. A candidate is considered ready as the final private setting only for a
   `STRONG GO` or `GO` decision under the predeclared rule below.
6. Final decision: **{classification}**.

Decision rule fixed in the run provenance: `STRONG GO` requires a stable
candidate with positive mean ARI/NMI differences versus NoQueue, at least 5/7
positive seeds for each metric, and both paired 95% CI lower bounds above zero.
`GO` relaxes only the CI requirement; `MARGINAL` requires positive mean
differences without consistency; otherwise `NO-GO`. Among consistent
candidates, C75/sigma4 is preferred because it has the smaller epsilon.
"""
    (output / "MULTISEED_102_REPORT.md").write_text(report, encoding="utf-8")


def finalize(output: Path, baseline_rows: pd.DataFrame, accounting: pd.DataFrame) -> None:
    candidate_path = output / "candidate_metrics_partial.csv"
    diagnostics_path = output / "clipping_invalidA_summary_partial.csv"
    candidates = pd.read_csv(candidate_path)
    diagnostics = pd.read_csv(diagnostics_path)
    expected = {(seed, method) for seed in SEEDS for method in CANDIDATES}
    observed = set(zip(candidates.seed.astype(int), candidates.method.astype(str)))
    if observed != expected:
        raise ValueError(f"candidate metrics are incomplete: missing {sorted(expected - observed)}")
    per_seed = pd.concat([baseline_rows, candidates], ignore_index=True, sort=False)
    per_seed = per_seed.sort_values(["seed", "method"]).reset_index(drop=True)
    if len(per_seed) != len(SEEDS) * (len(BASELINES) + len(CANDIDATES)):
        raise ValueError("unexpected per-seed metric row count")
    atomic_csv(per_seed, output / "per_seed_metrics.csv")
    differences, summary = statistical_outputs(per_seed)
    atomic_csv(differences, output / "matched_differences.csv")
    atomic_csv(summary, output / "summary_statistics.csv")
    atomic_csv(diagnostics.sort_values(["seed", "method"]), output / "clipping_invalidA_summary.csv")
    write_report(output, per_seed, summary, accounting, diagnostics)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output = resolve_repo_path(OUTPUT)
    output.mkdir(parents=True, exist_ok=True)
    (output / "training").mkdir(exist_ok=True)
    (output / "schedules").mkdir(exist_ok=True)
    base = load_config(BASE_CONFIG)
    validate_protocol(base)
    accounting = privacy_table()
    atomic_csv(accounting, output / "privacy_accounting.csv")
    epsilon = accounting[accounting.delta == 1e-5].set_index("method").epsilon
    if not math.isclose(float(epsilon["Joint-C50-sigma2"]), 1.9930914044151202, abs_tol=1e-12):
        raise ValueError("independent sigma=2 privacy accounting check failed")
    if not math.isclose(float(epsilon["Joint-C75-sigma4"]), 0.9263415039983884, abs_tol=1e-12):
        raise ValueError("independent sigma=4 privacy accounting check failed")

    adata, data_path = read_dataset(base["dataset"], base["data_dir"], verify_hash=False)
    if sha256(data_path) != EXPECTED_DATA_SHA256:
        raise ValueError("Pancreas data hash does not match Figure 2")
    feature_path = repo_root / FIGURE2 / "features.txt"
    features = feature_path.read_text(encoding="utf-8").splitlines()
    normalize(adata, base["preprocessing"]["normalize_target_sum"], base["preprocessing"]["log1p"])
    legacy = load_legacy(repo_root)
    baseline_rows = load_official_baselines(repo_root)

    schedules = {}
    groups_by_seed = {}
    schedule_rows = []
    exposure_qc = []
    partition_hashes = {}
    for seed in SEEDS:
        partition_path = repo_root / FIGURE2 / "partitions" / f"seed_{seed}.json"
        groups = load_partition(partition_path, seed, adata.n_obs)
        schedule, frame = build_schedule(groups, np.asarray(adata.obs_names), seed)
        schedule_path = output / "schedules" / f"remote_cell_schedule_seed_{seed}.csv"
        atomic_csv(frame, schedule_path)
        schedules[seed] = schedule
        groups_by_seed[seed] = groups
        schedule_rows.append(frame)
        partition_hashes[str(seed)] = sha256(partition_path)
        schedule_hash = sha256(schedule_path)
        for source, source_frame in frame.groupby("source_client", sort=True):
            counts = source_frame.groupby("cell_id").size()
            exposure_qc.append(
                {
                    "seed": seed,
                    "source_client": int(source),
                    "available_training_cells": int(len(groups[int(source)])),
                    "active_rounds": len(ACTIVE_ROUNDS),
                    "remote_cells_per_active_round": REMOTE_PER_SOURCE,
                    "scheduled_selections": int(len(source_frame)),
                    "unique_remote_cells": int(source_frame.cell_id.nunique()),
                    "expected_unique_remote_cells": len(ACTIVE_ROUNDS) * REMOTE_PER_SOURCE,
                    "max_remote_participation_count": int(counts.max()),
                    "remote_cell_reuse_count": int((counts - 1).clip(lower=0).sum()),
                    "zero_remote_cell_reuse": bool((counts == 1).all()),
                    "schedule_sha256": schedule_hash,
                    "partition_sha256": partition_hashes[str(seed)],
                }
            )
    combined_schedule = pd.concat(schedule_rows, ignore_index=True)
    atomic_csv(combined_schedule, output / "remote_cell_schedule.csv")
    exposure_frame = pd.DataFrame(exposure_qc)
    if not (
        (exposure_frame.remote_cells_per_active_round == REMOTE_PER_SOURCE).all()
        and (exposure_frame.scheduled_selections == len(ACTIVE_ROUNDS) * REMOTE_PER_SOURCE).all()
        and (exposure_frame.unique_remote_cells == len(ACTIVE_ROUNDS) * REMOTE_PER_SOURCE).all()
        and (exposure_frame.max_remote_participation_count == 1).all()
        and (exposure_frame.remote_cell_reuse_count == 0).all()
        and exposure_frame.zero_remote_cell_reuse.astype(bool).all()
    ):
        raise ValueError("unified 102-cell exposure QC failed")
    atomic_csv(exposure_frame, output / "exposure_QC.csv")

    atomic_json(
        {
            "protocol": "RoundOnce joint-DP seven-seed validation",
            "seeds": list(SEEDS),
            "noise_seeds": NOISE_SEEDS,
            "candidates": CANDIDATES,
            "dataset": str(data_path),
            "dataset_sha256": sha256(data_path),
            "features": str(feature_path),
            "features_sha256": sha256(feature_path),
            "partition_sha256": partition_hashes,
            "same_figure2_partitions": True,
            "fixed_C": True,
            "fixed_sigma": True,
            "fixed_noise_seed_mapping": True,
            "adaptive_tuning": False,
            "remote_cells_per_source_per_active_round": REMOTE_PER_SOURCE,
            "unique_remote_cells_per_source": len(ACTIVE_ROUNDS) * REMOTE_PER_SOURCE,
            "previous_128_cell_seed218_pilots_excluded": True,
            "max_remote_participation_required": 1,
            "joint_noise_before_target_split": True,
            "no_mpc_he_tee": True,
            "figures_modified": False,
            "decision_rule": {
                "STRONG GO": "stable, positive mean ARI/NMI vs NoQueue, >=5/7 positive for each, both paired CI lower bounds >0",
                "GO": "same as STRONG GO without both CI lower bounds >0",
                "MARGINAL": "positive mean ARI/NMI vs NoQueue without consistency",
                "NO-GO": "no candidate has positive mean ARI/NMI vs NoQueue",
                "preference": "among consistent candidates prefer C75/sigma4 because epsilon is lower",
            },
        },
        output / "provenance.json",
    )

    candidate_path = output / "candidate_metrics_partial.csv"
    diagnostics_path = output / "clipping_invalidA_summary_partial.csv"
    status_path = output / "training_status.csv"
    candidate_rows = pd.read_csv(candidate_path).to_dict("records") if candidate_path.is_file() else []
    diagnostic_rows = pd.read_csv(diagnostics_path).to_dict("records") if diagnostics_path.is_file() else []
    statuses = pd.read_csv(status_path).to_dict("records") if status_path.is_file() else []
    completed = {(int(row["seed"]), str(row["method"])) for row in candidate_rows}

    for seed in SEEDS:
        for method, candidate in CANDIDATES.items():
            if (seed, method) in completed:
                continue
            run_dir = output / "training" / f"seed_{seed}" / method
            run_dir.mkdir(parents=True, exist_ok=True)
            configured = config_for_candidate(
                base, seed, NOISE_SEEDS[seed], schedules[seed], candidate
            )
            stable = False
            error = ""
            try:
                result = train_federated(
                    adata,
                    features,
                    None,
                    configured,
                    run_dir,
                    domain_key="batch",
                    client_indices=groups_by_seed[seed],
                )
                embedding = result.embeddings.astype(np.float32)
                np.save(run_dir / "final_embedding.npy", embedding)
                diagnostic = run_diagnostics(result, method, seed, run_dir)
                stable = bool(diagnostic["training_stable"])
                identity = f"jointDP_multiseed_{seed}_{method}"
                metric_values = evaluate_candidate(
                    legacy, adata, configured, embedding, identity
                )
                stable = stable and all(
                    math.isfinite(float(metric_values[name])) for name in METRICS
                )
                metric_row = {
                    "seed": seed,
                    "method": method,
                    **{name: metric_values[name] for name in METRICS},
                    "C_label": candidate["C_label"],
                    "C": candidate["C"],
                    "sigma": candidate["sigma"],
                    "epsilon_delta_1e5": float(epsilon[method]),
                    "noise_seed": NOISE_SEEDS[seed],
                    "clipping_fraction": diagnostic["clipping_fraction"],
                    "invalid_a_frequency": diagnostic["invalid_a_frequency"],
                    "invalid_a_count": diagnostic["invalid_a_count"],
                    "a_count": diagnostic["a_count"],
                    "training_stable": stable,
                    "max_remote_participation_count": 1,
                    "partition_sha256": partition_hashes[str(seed)],
                    "embedding_sha256": sha256(run_dir / "final_embedding.npy"),
                    "source": str(run_dir / "final_embedding.npy"),
                    "ARI_leiden_sd": metric_values["ARI_leiden_sd"],
                    "NMI_leiden_sd": metric_values["NMI_leiden_sd"],
                    "leiden_repeats_json": metric_values["leiden_repeats_json"],
                }
                candidate_rows.append(metric_row)
                diagnostic["training_stable"] = stable
                diagnostic["max_remote_participation_count"] = 1
                diagnostic_rows.append(diagnostic)
                atomic_csv(pd.DataFrame(candidate_rows), candidate_path)
                atomic_csv(pd.DataFrame(diagnostic_rows), diagnostics_path)
                completed.add((seed, method))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                (run_dir / "error.txt").write_text(
                    error + "\n\n" + traceback.format_exc(), encoding="utf-8"
                )
            statuses = [
                row
                for row in statuses
                if not (int(row["seed"]) == seed and str(row["method"]) == method)
            ]
            statuses.append(
                {
                    "seed": seed,
                    "method": method,
                    "noise_seed": NOISE_SEEDS[seed],
                    "training_completed": (seed, method) in completed,
                    "training_stable": stable,
                    "error": error,
                }
            )
            atomic_csv(pd.DataFrame(statuses), status_path)
            if error:
                raise RuntimeError(error)
            del result, embedding
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    finalize(output, baseline_rows, accounting)
    print(f"RoundOnce joint-DP multi-seed validation complete: {output}")


if __name__ == "__main__":
    main()
