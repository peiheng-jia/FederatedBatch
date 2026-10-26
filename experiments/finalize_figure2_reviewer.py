#!/usr/bin/env python3
"""Finalize reviewer-facing Figure 2 from locked canonical-seven experiments.

Only the missing non-private RoundOnce Joint-C50 sigma=0 reference is trained.
NoQueue and the final private FederatedBatch are reused with hash-based provenance
checks.  The script is restartable at the seed level.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch
from scipy import stats

from experiments.moderate_skew_freshqueue_confirmation import trajectory_metrics
from experiments.round_once_jointDP_multiseed import (
    ACTIVE_ROUNDS,
    BASE_CONFIG,
    EXPECTED_DATA_SHA256,
    FIGURE2,
    LEGACY_VALUES,
    N_CLIENTS,
    NOISE_SEEDS,
    REMOTE_PER_SOURCE,
    SEEDS,
    atomic_csv,
    atomic_json,
    build_schedule,
    evaluate_candidate,
    load_legacy,
    load_official_baselines,
    load_partition,
    sha256,
    validate_protocol,
)
from federatedbatch.config import load_config, resolve_repo_path
from federatedbatch.data import normalize, read_dataset
from federatedbatch.federated import predict_embeddings, train_federated


OUTPUT = Path("results/final_figure2_reviewer_canonical7")
FINAL_DP = Path("results/round_once_jointDP_multiseed_102")
METHOD_NOQUEUE = "NoQueue"
METHOD_REFERENCE = "Remote A/B (sigma=0)"
METHOD_FINAL = "FederatedBatch"
METHODS = (METHOD_NOQUEUE, METHOD_REFERENCE, METHOD_FINAL)
METRICS = ("ARI", "NMI", "ASW_batch", "ASW_cell", "graph_connectivity", "iLISI")
JOINT_C50 = 1206.3391650512762
EXPECTED_FEATURE_SHA256 = "ff729379f06798ee96d232f3ff9d2263c56741abd78614b263440fe0f01458f2"
EXPECTED_PYTHON = "3.10.13"
EXPECTED_TORCH = "2.1.2"
EXPECTED_TORCH_CUDA = "11.8"


def save_embedding_atomic(path: Path, embedding: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(embedding, dtype=np.float32), allow_pickle=False)
    temporary.replace(path)


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


def safe_round_callback(adata, config: dict, seed: int, rows: list[dict]):
    def callback(round_number, model, _diagnostics):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        row = {
            "seed": int(seed),
            "method": METHOD_REFERENCE,
            "round": int(round_number),
            "trajectory_leiden_seed": int(config["confirmation"]["trajectory_leiden_seed"]),
        }
        try:
            embedding = predict_embeddings(model, adata)
            values = trajectory_metrics(
                adata,
                embedding,
                config,
                f"figure2_reviewer_{seed}_{round_number}",
            )
            row.update(values)
            row["trajectory_stable"] = True
            row["trajectory_error"] = ""
        except Exception as exc:
            row.update(
                {
                    "ARI": math.nan,
                    "NMI": math.nan,
                    "trajectory_stable": False,
                    "trajectory_error": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            rows.append(row)
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    return callback


def runtime_record() -> dict:
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }


def assert_runtime() -> dict:
    observed = runtime_record()
    checks = {
        "python": observed["python"] == EXPECTED_PYTHON,
        "torch": observed["torch"] == EXPECTED_TORCH,
        "torch_cuda": observed["torch_cuda"] == EXPECTED_TORCH_CUDA,
        "cuda_available": observed["cuda_available"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"frozen Figure 2 runtime mismatch: {checks}; observed={observed}")
    return {"status": "PASS", "checks": checks, "observed": observed}


def load_final_private(repo_root: Path) -> pd.DataFrame:
    source = repo_root / FINAL_DP
    frame = pd.read_csv(source / "per_seed_metrics.csv")
    selected = frame.loc[frame["method"] == "Joint-C50-sigma2"].copy()
    if set(selected["seed"].astype(int)) != set(SEEDS) or len(selected) != len(SEEDS):
        raise ValueError("final FederatedBatch canonical-seven metrics are incomplete")
    if not selected["training_stable"].astype(bool).all():
        raise ValueError("at least one final FederatedBatch run is not stable")
    if set(selected["max_remote_participation_count"].astype(int)) != {1}:
        raise ValueError("final FederatedBatch violates one-time participation")
    for row in selected.itertuples(index=False):
        embedding = source / "training" / f"seed_{int(row.seed)}" / "Joint-C50-sigma2" / "final_embedding.npy"
        if sha256(embedding) != str(row.embedding_sha256):
            raise ValueError(f"final FederatedBatch embedding hash mismatch for seed {row.seed}")
        partition = repo_root / FIGURE2 / "partitions" / f"seed_{int(row.seed)}.json"
        if sha256(partition) != str(row.partition_sha256):
            raise ValueError(f"final FederatedBatch partition hash mismatch for seed {row.seed}")
    selected["method"] = METHOD_FINAL
    selected["source_method"] = "Joint-C50-sigma2"
    selected["retrained_for_figure2_reviewer"] = False
    return selected


def compare_locked_schedule(rebuilt: pd.DataFrame, locked: pd.DataFrame, seed: int) -> None:
    columns = [
        "seed",
        "active_round",
        "collection_round",
        "interaction_slot",
        "source_client",
        "source_local_index",
        "global_index",
        "cell_id",
    ]
    left = rebuilt[columns].reset_index(drop=True).copy()
    right = locked[columns].reset_index(drop=True).copy()
    for column in columns[:-1]:
        left[column] = left[column].astype(int)
        right[column] = right[column].astype(int)
    left["cell_id"] = left["cell_id"].astype(str)
    right["cell_id"] = right["cell_id"].astype(str)
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False)
    except AssertionError as exc:
        raise ValueError(f"rebuilt schedule differs from locked final schedule for seed {seed}") from exc


def prepare_preflight(repo_root: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    (output / "training").mkdir(exist_ok=True)
    (output / "schedules").mkdir(exist_ok=True)
    runtime = assert_runtime()
    atomic_json(runtime, output / "runtime_verification.json")

    base = load_config(BASE_CONFIG)
    validate_protocol(base)
    adata, data_path = read_dataset(base["dataset"], base["data_dir"], verify_hash=False)
    if sha256(data_path) != EXPECTED_DATA_SHA256:
        raise ValueError("Pancreas data hash does not match locked Figure 2")
    feature_path = repo_root / FIGURE2 / "features.txt"
    if sha256(feature_path) != EXPECTED_FEATURE_SHA256:
        raise ValueError("Figure 2 feature-list hash mismatch")
    features = feature_path.read_text(encoding="utf-8").splitlines()
    normalize(adata, base["preprocessing"]["normalize_target_sum"], base["preprocessing"]["log1p"])

    official = load_official_baselines(repo_root)
    noqueue = official.loc[official["method"] == "Official-NoQueue"].copy()
    noqueue["method"] = METHOD_NOQUEUE
    noqueue["source_method"] = "Official-NoQueue"
    noqueue["retrained_for_figure2_reviewer"] = False
    final_private = load_final_private(repo_root)

    schedules: dict[int, dict] = {}
    groups_by_seed: dict[int, list[np.ndarray]] = {}
    schedule_frames = []
    qc_rows = []
    provenance_rows = []
    for seed in SEEDS:
        partition_path = repo_root / FIGURE2 / "partitions" / f"seed_{seed}.json"
        groups = load_partition(partition_path, seed, adata.n_obs)
        rebuilt_schedule, rebuilt = build_schedule(groups, np.asarray(adata.obs_names), seed)
        locked_path = repo_root / FINAL_DP / "schedules" / f"remote_cell_schedule_seed_{seed}.csv"
        locked = pd.read_csv(locked_path)
        compare_locked_schedule(rebuilt, locked, seed)
        locked_output = output / "schedules" / f"remote_cell_schedule_seed_{seed}.csv"
        atomic_csv(locked, locked_output)
        schedule = schedule_from_frame(locked)
        if schedule != rebuilt_schedule:
            raise ValueError(f"schedule dictionary mismatch for seed {seed}")
        schedules[seed] = schedule
        groups_by_seed[seed] = groups
        schedule_frames.append(locked)
        for source, group in locked.groupby("source_client", sort=True):
            counts = group.groupby("cell_id").size()
            qc_rows.append(
                {
                    "seed": seed,
                    "source_client": int(source),
                    "available_cells": len(groups[int(source)]),
                    "active_rounds": len(ACTIVE_ROUNDS),
                    "remote_cells_per_active_round": REMOTE_PER_SOURCE,
                    "scheduled_cells": len(group),
                    "unique_remote_cells": group["cell_id"].nunique(),
                    "max_remote_participation_count": int(counts.max()),
                    "remote_cell_reuse_count": int((counts - 1).clip(lower=0).sum()),
                    "schedule_sha256": sha256(locked_path),
                    "partition_sha256": sha256(partition_path),
                }
            )
        nq = noqueue.loc[noqueue["seed"] == seed].iloc[0]
        fb = final_private.loc[final_private["seed"] == seed].iloc[0]
        provenance_rows.extend(
            [
                {
                    "seed": seed,
                    "method": METHOD_NOQUEUE,
                    "training_status": "reused",
                    "source": nq["source"],
                    "embedding_sha256": nq["embedding_sha256"],
                    "partition_sha256": nq["partition_sha256"],
                    "protocol_match": True,
                },
                {
                    "seed": seed,
                    "method": METHOD_REFERENCE,
                    "training_status": "missing; scheduled for training",
                    "source": str(output / "training" / f"seed_{seed}" / "RemoteAB-sigma0" / "final_embedding.npy"),
                    "embedding_sha256": "",
                    "partition_sha256": sha256(partition_path),
                    "protocol_match": True,
                },
                {
                    "seed": seed,
                    "method": METHOD_FINAL,
                    "training_status": "reused",
                    "source": fb["source"],
                    "embedding_sha256": fb["embedding_sha256"],
                    "partition_sha256": fb["partition_sha256"],
                    "protocol_match": True,
                },
            ]
        )

    exposure = pd.DataFrame(qc_rows)
    if not (
        (exposure["remote_cells_per_active_round"] == 102).all()
        and (exposure["scheduled_cells"] == 1734).all()
        and (exposure["unique_remote_cells"] == 1734).all()
        and (exposure["max_remote_participation_count"] == 1).all()
        and (exposure["remote_cell_reuse_count"] == 0).all()
    ):
        raise ValueError("locked schedule exposure QC failed")
    atomic_csv(exposure, output / "exposure_QC.csv")
    atomic_csv(pd.concat(schedule_frames, ignore_index=True), output / "remote_cell_schedule.csv")
    atomic_csv(pd.DataFrame(provenance_rows), output / "provenance_table.csv")

    atomic_json(
        {
            "status": "PASS",
            "scope": "Figure 2 reviewer-facing canonical-seven finalization",
            "seeds": list(SEEDS),
            "methods": list(METHODS),
            "reused_methods": [METHOD_NOQUEUE, METHOD_FINAL],
            "only_trained_method": METHOD_REFERENCE,
            "reference_protocol": {
                "mode": "RoundOnce exact A/B with joint cell-level clipping",
                "joint_C": JOINT_C50,
                "sigma": 0.0,
                "remote_cells_per_source_active_round": REMOTE_PER_SOURCE,
                "active_rounds": list(ACTIVE_ROUNDS),
                "unique_remote_cells_per_source": 1734,
                "max_remote_participation": 1,
                "tau_q": 0.20,
                "alpha": 0.10,
            },
            "dataset": str(data_path),
            "dataset_sha256": sha256(data_path),
            "features": str(feature_path),
            "features_sha256": sha256(feature_path),
            "final_dp_source": str(repo_root / FINAL_DP),
            "noqueue_source": str(repo_root / FIGURE2),
            "old_vanilla_freshqueue_excluded_from_final_comparison": True,
            "figure1_3_4_5_modified": False,
            "manuscript_modified": False,
        },
        output / "provenance.json",
    )
    return base, adata, features, schedules, groups_by_seed, noqueue, final_private


def reference_diagnostics(result, seed: int, run_dir: Path) -> dict:
    clipping = pd.DataFrame(result.joint_clipping_diagnostics)
    expected = len(ACTIVE_ROUNDS) * N_CLIENTS * REMOTE_PER_SOURCE
    if len(clipping) != expected:
        raise ValueError(f"seed {seed}: expected {expected} joint clipping rows, found {len(clipping)}")
    if set(clipping["targets_covered"].astype(int)) != {N_CLIENTS - 1}:
        raise ValueError(f"seed {seed}: clipping was not joint across all other targets")
    if not np.allclose(clipping["threshold"].astype(float), JOINT_C50):
        raise ValueError(f"seed {seed}: adaptive or incorrect clipping threshold detected")
    if not np.allclose(clipping["noise_multiplier"].astype(float), 0.0):
        raise ValueError(f"seed {seed}: sigma=0 reference contains nonzero noise")
    if result.joint_noise_diagnostics:
        raise ValueError(f"seed {seed}: sigma=0 reference used the Gaussian release path")
    atomic_csv(clipping, run_dir / "joint_clipping_diagnostics.csv")
    return {
        "seed": seed,
        "method": METHOD_REFERENCE,
        "joint_clipping_rows": len(clipping),
        "clipping_fraction": float(clipping["clipped"].astype(bool).mean()),
        "median_clipping_scale": float(clipping["clipping_scale"].median()),
        "p10_clipping_scale": float(clipping["clipping_scale"].quantile(0.10)),
        "median_joint_norm": float(clipping["joint_norm"].median()),
        "p90_joint_norm": float(clipping["joint_norm"].quantile(0.90)),
        "invalid_a_frequency": 0.0,
        "training_stable": bool(np.isfinite(result.embeddings).all()),
        "max_remote_participation_count": 1,
    }


def train_reference(
    repo_root: Path,
    output: Path,
    base: dict,
    adata,
    features: list[str],
    schedules: dict[int, dict],
    groups_by_seed: dict[int, list[np.ndarray]],
) -> None:
    legacy = load_legacy(repo_root)
    metrics_path = output / "remote_ab_sigma0_metrics_partial.csv"
    diagnostics_path = output / "remote_ab_sigma0_diagnostics_partial.csv"
    trajectory_path = output / "remote_ab_sigma0_trajectory_partial.csv"
    status_path = output / "training_status.csv"
    metric_rows = pd.read_csv(metrics_path).to_dict("records") if metrics_path.is_file() else []
    diagnostic_rows = pd.read_csv(diagnostics_path).to_dict("records") if diagnostics_path.is_file() else []
    trajectory_rows = pd.read_csv(trajectory_path).to_dict("records") if trajectory_path.is_file() else []
    statuses = pd.read_csv(status_path).to_dict("records") if status_path.is_file() else []
    completed = {int(row["seed"]) for row in metric_rows}

    for seed in SEEDS:
        if seed in completed:
            continue
        run_dir = output / "training" / f"seed_{seed}" / "RemoteAB-sigma0"
        run_dir.mkdir(parents=True, exist_ok=True)
        configured = reference_config(base, seed, schedules[seed])
        seed_trajectory: list[dict] = []
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
                round_callback=safe_round_callback(adata, configured, seed, seed_trajectory),
            )
            embedding = result.embeddings.astype(np.float32)
            save_embedding_atomic(run_dir / "final_embedding.npy", embedding)
            diagnostic = reference_diagnostics(result, seed, run_dir)
            stable = bool(diagnostic["training_stable"])
            if len(seed_trajectory) != 20 or not all(
                bool(row["trajectory_stable"]) for row in seed_trajectory
            ):
                raise ValueError(f"seed {seed}: incomplete or unstable round trajectory")
            metric_values = evaluate_candidate(
                legacy,
                adata,
                configured,
                embedding,
                f"figure2_reviewer_sigma0_{seed}",
            )
            stable = stable and all(math.isfinite(float(metric_values[name])) for name in METRICS)
            metric_rows.append(
                {
                    "seed": seed,
                    "method": METHOD_REFERENCE,
                    **{metric: metric_values[metric] for metric in METRICS},
                    "C_label": "Joint-C50",
                    "C": JOINT_C50,
                    "sigma": 0.0,
                    "clipping_fraction": diagnostic["clipping_fraction"],
                    "invalid_a_frequency": 0.0,
                    "training_stable": stable,
                    "max_remote_participation_count": 1,
                    "partition_sha256": sha256(repo_root / FIGURE2 / "partitions" / f"seed_{seed}.json"),
                    "embedding_sha256": sha256(run_dir / "final_embedding.npy"),
                    "source": str(run_dir / "final_embedding.npy"),
                    "source_method": "RoundOnce-Joint-C50-sigma0-102",
                    "retrained_for_figure2_reviewer": True,
                    "ARI_leiden_sd": metric_values["ARI_leiden_sd"],
                    "NMI_leiden_sd": metric_values["NMI_leiden_sd"],
                    "leiden_repeats_json": metric_values["leiden_repeats_json"],
                }
            )
            diagnostic["training_stable"] = stable
            diagnostic_rows.append(diagnostic)
            trajectory_rows.extend(seed_trajectory)
            atomic_csv(pd.DataFrame(metric_rows), metrics_path)
            atomic_csv(pd.DataFrame(diagnostic_rows), diagnostics_path)
            atomic_csv(pd.DataFrame(trajectory_rows), trajectory_path)
            completed.add(seed)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            (run_dir / "error.txt").write_text(
                error + "\n\n" + traceback.format_exc(), encoding="utf-8"
            )
        statuses = [row for row in statuses if int(row["seed"]) != seed]
        statuses.append(
            {
                "seed": seed,
                "method": METHOD_REFERENCE,
                "training_completed": seed in completed,
                "training_stable": stable,
                "error": error,
            }
        )
        atomic_csv(pd.DataFrame(statuses).sort_values("seed"), status_path)
        if error:
            raise RuntimeError(error)
        del result, embedding
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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


def leiden_seed_zero(row: pd.Series) -> tuple[float, float]:
    repeats = json.loads(row["leiden_repeats_json"])
    selected = [item for item in repeats if int(item["leiden_random_state"]) == 0]
    if len(selected) != 1:
        raise ValueError("method-level result lacks one Leiden random-state-zero record")
    return float(selected[0]["ARI"]), float(selected[0]["NMI"])


def trajectory_tables(repo_root: Path, output: Path, final_private: pd.DataFrame):
    noqueue = pd.read_csv(repo_root / FIGURE2 / "round_trajectory.csv")
    noqueue = noqueue.loc[
        noqueue["seed"].isin(SEEDS) & (noqueue["method"] == "NoQueue"),
        ["seed", "method", "round", "ARI", "NMI"],
    ].copy()
    noqueue["method"] = METHOD_NOQUEUE
    noqueue["trajectory_leiden_seed"] = 0
    noqueue["trajectory_stable"] = True
    noqueue["trajectory_error"] = ""
    reference = pd.read_csv(output / "remote_ab_sigma0_trajectory_partial.csv")
    if len(noqueue) != len(SEEDS) * 20 or len(reference) != len(SEEDS) * 20:
        raise ValueError("NoQueue or sigma=0 trajectory is incomplete")
    final_rows = []
    for _, row in final_private.sort_values("seed").iterrows():
        ari, nmi = leiden_seed_zero(row)
        final_rows.append(
            {
                "seed": int(row["seed"]),
                "method": METHOD_FINAL,
                "round": 20,
                "ARI": ari,
                "NMI": nmi,
                "trajectory_leiden_seed": 0,
                "trajectory_stable": True,
                "trajectory_error": "",
            }
        )
    trajectories = pd.concat([noqueue, reference, pd.DataFrame(final_rows)], ignore_index=True)
    trajectories = trajectories.sort_values(["method", "seed", "round"]).reset_index(drop=True)

    summary_rows = []
    for (method, round_number), group in trajectories.groupby(["method", "round"], sort=False):
        for metric in ("ARI", "NMI"):
            values = group[metric].to_numpy(float)
            ci_low, ci_high = t_interval(values)
            summary_rows.append(
                {
                    "method": method,
                    "round": int(round_number),
                    "metric": metric,
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "trajectory_leiden_seed": 0,
                }
            )
    summary = pd.DataFrame(summary_rows)

    wide = trajectories.loc[trajectories["method"].isin([METHOD_NOQUEUE, METHOD_REFERENCE])].pivot(
        index=["seed", "round"], columns="method", values=["ARI", "NMI"]
    )
    difference_rows = []
    for metric in ("ARI", "NMI"):
        for (seed, round_number), row in wide[metric].iterrows():
            difference_rows.append(
                {
                    "seed": int(seed),
                    "round": int(round_number),
                    "metric": metric,
                    "remote_ab_minus_noqueue": float(row[METHOD_REFERENCE] - row[METHOD_NOQUEUE]),
                }
            )
    differences = pd.DataFrame(difference_rows)
    availability = pd.DataFrame(
        [
            {
                "method": METHOD_NOQUEUE,
                "available_rounds": "1-20",
                "seed_count": 7,
                "source": str(repo_root / FIGURE2 / "round_trajectory.csv"),
                "note": "reused complete trajectory",
            },
            {
                "method": METHOD_REFERENCE,
                "available_rounds": "1-20",
                "seed_count": 7,
                "source": str(output / "remote_ab_sigma0_trajectory_partial.csv"),
                "note": "captured during the only newly trained group",
            },
            {
                "method": METHOD_FINAL,
                "available_rounds": "20 only",
                "seed_count": 7,
                "source": str(repo_root / FINAL_DP / "per_seed_metrics.csv"),
                "note": "completed final runs did not save intermediate embeddings; not retrained",
            },
        ]
    )
    return trajectories, summary, differences, availability


def composition_plot_data(adata, groups_by_seed: dict[int, list[np.ndarray]]) -> pd.DataFrame:
    labels = np.asarray(adata.obs["celltype"]).astype(str)
    rows = []
    representative_seed = SEEDS[0]
    for client, indices in enumerate(groups_by_seed[representative_seed], start=1):
        values, counts = np.unique(labels[indices], return_counts=True)
        for label, count in zip(values, counts):
            rows.append(
                {
                    "seed": representative_seed,
                    "client": client,
                    "cell_type": label,
                    "cell_count": int(count),
                    "fraction": float(count / len(indices)),
                }
            )
    return pd.DataFrame(rows)


def write_report(
    output: Path,
    per_seed: pd.DataFrame,
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    trajectory_differences: pd.DataFrame,
    availability: pd.DataFrame,
) -> None:
    compact = summary.pivot(index="method", columns="metric", values=["mean", "sd"])
    final_tests = tests.loc[tests["comparison"] == f"{METHOD_FINAL} vs {METHOD_NOQUEUE}"].copy()
    reference_tests = tests.loc[tests["comparison"] == f"{METHOD_REFERENCE} vs {METHOD_NOQUEUE}"].copy()
    noise_tests = tests.loc[tests["comparison"] == f"{METHOD_FINAL} vs {METHOD_REFERENCE}"].copy()
    endpoint = trajectory_differences.loc[trajectory_differences["round"] == 20].groupby("metric")[
        "remote_ab_minus_noqueue"
    ].mean()

    def stat_text(frame: pd.DataFrame, metric: str) -> str:
        row = frame.loc[frame["metric"] == metric].iloc[0]
        return (
            f"Delta={row.mean_paired_difference:.4f}, 95% CI "
            f"[{row.ci95_low:.4f}, {row.ci95_high:.4f}], "
            f"p={row.p_raw_two_sided:.4g}, Holm p={row.p_holm_18:.4g}, "
            f"{int(row.positive_seed_count)}/7 positive"
        )

    results_paragraph = (
        "Across the seven prespecified and method-matched seeds, the cross-client remote "
        "A/B reference (Joint-C50, sigma=0) was compared directly with NoQueue, while the "
        "final FederatedBatch used the identical 102-cell one-time-participation schedule "
        "with sigma=2 Gaussian noise. For ARI, the sigma=0 reference versus NoQueue gave "
        f"{stat_text(reference_tests, 'ARI')}; final FederatedBatch versus NoQueue gave "
        f"{stat_text(final_tests, 'ARI')}. For NMI, the corresponding results were "
        f"{stat_text(reference_tests, 'NMI')} and {stat_text(final_tests, 'NMI')}. "
        "Direct comparison of final FederatedBatch with the sigma=0 reference quantified "
        f"the effect of adding privacy noise (ARI: {stat_text(noise_tests, 'ARI')}; NMI: "
        f"{stat_text(noise_tests, 'NMI')}). The same paired analysis was applied to raw "
        "ASW(batch), ASW(cell type), legacy graph connectivity, and full-pairwise legacy "
        "iLISI, with Holm correction across all 18 tests; raw ASW(batch) was reported "
        "without assuming that larger values are intrinsically better."
    )
    response_paragraph = (
        "To address the reviewer's question about whether the Queue-derived cross-client "
        "signal contributes to performance, we added a matched ablation that removes the "
        "remote A/B channel (NoQueue) and a non-private reference that retains the exact "
        "RoundOnce remote interaction but sets sigma=0. All three methods use the same seven "
        "prespecified seeds and client partitions. The sigma=0-versus-NoQueue paired result "
        f"was {stat_text(reference_tests, 'ARI')} for ARI and "
        f"{stat_text(reference_tests, 'NMI')} for NMI. At round 20, the descriptive "
        f"trajectory differences were {endpoint['ARI']:+.4f} ARI and {endpoint['NMI']:+.4f} "
        "NMI. These comparisons isolate the contribution of the cross-client remote signal, "
        "whereas the final-versus-sigma=0 comparison isolates the effect of the privacy "
        "mechanism. We use 'remote interaction' or 'remote A/B channel' for the finalized "
        "method and reserve 'Queue contribution' for framing the reviewer's original question."
    )
    report = f"""# Figure 2 reviewer-facing canonical-seven finalization

## Provenance decision

The canonical-seven NoQueue results and final FederatedBatch (`RoundOnce +
Joint-C50 + sigma=2`) were reused without retraining after dataset, feature,
partition, embedding, schedule, stability, and one-time-participation checks.
The only newly trained group was the previously missing `Remote A/B (sigma=0)`
reference, using Joint-C50, exactly 102 remote cells per source in rounds 4--20,
1,734 unique remote cells per source, and the identical locked schedules used by
the final method. Old Vanilla FreshQueue is not a final method in this analysis.

## Mean and SD

```text
{compact.to_string()}
```

## Primary paired comparisons

All differences are method A minus method B. Paired t-tests and t-based 95%
confidence intervals use the seven method-level seed values. The five Leiden
random states are averaged within a seed and are not independent replicates.
Holm correction is joint across the 18 prespecified tests (three comparisons by
six metrics). Exact Wilcoxon results are retained as a small-sample sensitivity
analysis in `paired_statistics.csv`.

```text
{tests.to_string(index=False)}
```

## Trajectory availability

```text
{availability.to_string(index=False)}
```

NoQueue and the new sigma=0 reference have complete rounds 1--20 trajectories,
using the locked trajectory Leiden seed 0. The completed final FederatedBatch
runs did not save intermediate embeddings, so only their genuine round-20
endpoints are shown. They were not retrained merely to manufacture a trajectory.
Trajectory curves are descriptive and are not used as independent inferential
replicates.

## Results-ready text

{results_paragraph}

## Response-letter text

{response_paragraph}
"""
    (output / "FIGURE2_REVIEWER_REPORT.md").write_text(report, encoding="utf-8")
    (output / "RESULTS_PARAGRAPH.txt").write_text(results_paragraph + "\n", encoding="utf-8")
    (output / "RESPONSE_LETTER_PARAGRAPH.txt").write_text(response_paragraph + "\n", encoding="utf-8")


def finalize(
    repo_root: Path,
    output: Path,
    adata,
    groups_by_seed: dict[int, list[np.ndarray]],
    noqueue: pd.DataFrame,
    final_private: pd.DataFrame,
) -> None:
    reference = pd.read_csv(output / "remote_ab_sigma0_metrics_partial.csv")
    if set(reference["seed"].astype(int)) != set(SEEDS) or len(reference) != len(SEEDS):
        raise ValueError("sigma=0 reference is incomplete")
    keep = [
        "seed",
        "method",
        *METRICS,
        "C_label",
        "C",
        "sigma",
        "clipping_fraction",
        "invalid_a_frequency",
        "training_stable",
        "max_remote_participation_count",
        "partition_sha256",
        "embedding_sha256",
        "source",
        "source_method",
        "retrained_for_figure2_reviewer",
        "ARI_leiden_sd",
        "NMI_leiden_sd",
        "leiden_repeats_json",
    ]
    for frame in (noqueue, final_private, reference):
        for column in keep:
            if column not in frame:
                frame[column] = math.nan
    per_seed = pd.concat([noqueue[keep], reference[keep], final_private[keep]], ignore_index=True)
    per_seed = per_seed.sort_values(["seed", "method"]).reset_index(drop=True)
    summary, differences, tests = method_and_stat_tables(per_seed)
    trajectories, trajectory_summary, trajectory_differences, availability = trajectory_tables(
        repo_root, output, final_private
    )
    composition = composition_plot_data(adata, groups_by_seed)
    protocol = pd.DataFrame(
        {
            "round": np.arange(1, 21),
            "remote_interaction_active": [round_number in ACTIVE_ROUNDS for round_number in range(1, 21)],
            "remote_cells_per_source": [0 if round_number < 4 else REMOTE_PER_SOURCE for round_number in range(1, 21)],
        }
    )

    atomic_csv(per_seed, output / "per_seed_metrics.csv")
    atomic_csv(summary, output / "method_summary.csv")
    atomic_csv(differences, output / "paired_differences.csv")
    atomic_csv(tests, output / "paired_statistics.csv")
    atomic_csv(trajectories, output / "round_trajectory.csv")
    atomic_csv(trajectory_summary, output / "round_trajectory_summary.csv")
    atomic_csv(trajectory_differences, output / "round_trajectory_paired_differences.csv")
    atomic_csv(availability, output / "trajectory_availability.csv")
    atomic_csv(composition, output / "plot_client_composition.csv")
    atomic_csv(protocol, output / "plot_protocol_timeline.csv")
    atomic_csv(per_seed, output / "plot_final_seed_values.csv")
    atomic_csv(tests, output / "plot_paired_effects.csv")
    write_report(output, per_seed, summary, tests, trajectory_differences, availability)

    provenance = pd.read_csv(output / "provenance_table.csv")
    reference_hashes = reference.set_index("seed")["embedding_sha256"]
    for seed in SEEDS:
        mask = (provenance["seed"] == seed) & (provenance["method"] == METHOD_REFERENCE)
        provenance.loc[mask, "training_status"] = "newly trained; complete"
        provenance.loc[mask, "embedding_sha256"] = reference_hashes.loc[seed]
    atomic_csv(provenance, output / "provenance_table.csv")

    renderer = repo_root / "scripts" / "render_figure2_reviewer.py"
    subprocess.run([sys.executable, str(renderer), "--input", str(output)], cwd=repo_root, check=True)
    (output / "pipeline_complete.txt").write_text("complete\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output = resolve_repo_path(OUTPUT)
    base, adata, features, schedules, groups, noqueue, final_private = prepare_preflight(
        repo_root, output
    )
    print(f"Preflight PASS: {output}", flush=True)
    if args.preflight_only:
        return
    train_reference(repo_root, output, base, adata, features, schedules, groups)
    finalize(repo_root, output, adata, groups, noqueue, final_private)
    print(f"Figure 2 reviewer-facing finalization complete: {output}", flush=True)


if __name__ == "__main__":
    main()
