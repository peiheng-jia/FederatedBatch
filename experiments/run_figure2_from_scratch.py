#!/usr/bin/env python3
"""Run the final Figure 2 three-arm protocol from the staged Pancreas data.

This is the authoritative public Figure 2 workflow. It does not execute the
historical FIFO or FreshQueue candidates. Runs are restartable at method/seed
granularity and write only below the selected output directory.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.finalize_figure2_reviewer import (  # noqa: E402
    JOINT_C50,
    METHOD_FINAL,
    METHOD_NOQUEUE,
    METHOD_REFERENCE,
    METHODS,
    METRICS,
    method_and_stat_tables,
    reference_config,
    schedule_from_frame,
)
from experiments.moderate_skew_freshqueue_confirmation import (  # noqa: E402
    _partition_hash,
    partition_for_seed,
    trajectory_metrics,
)
from experiments.round_once_jointDP_multiseed import (  # noqa: E402
    ACTIVE_ROUNDS,
    N_CLIENTS,
    NOISE_SEEDS,
    REMOTE_PER_SOURCE,
    SEEDS,
    build_schedule,
    config_for_candidate,
    evaluate_candidate,
    load_legacy,
    load_partition,
)
from federatedbatch.config import load_config, resolve_repo_path  # noqa: E402
from federatedbatch.data import normalize, preprocess_dataset, read_dataset  # noqa: E402
from federatedbatch.federated import (  # noqa: E402
    predict_embeddings,
    save_checkpoint,
    train_federated,
)


DEFAULT_CONFIG = ROOT / "configs" / "figure2_training.yaml"
FINAL_SIGMA = 2.0


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_config(config: dict) -> None:
    fed = config["federated"]
    remote = config["remote_interaction"]
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
        remote["remote_cells_per_source"],
        remote["clipping_C"],
        remote["tau_q"],
        remote["alpha"],
        remote["final_sigma"],
    )
    expected = (
        "pancreas", 5, 20, 5, 128, 0.001, 0.07, 100, [128], 102,
        JOINT_C50, 0.20, 0.10, 2.0,
    )
    if observed != expected:
        raise ValueError(f"Figure 2 protocol mismatch: {observed}")
    if tuple(config["confirmation"]["training_seeds"]) != SEEDS:
        raise ValueError("Figure 2 canonical seed list changed")
    if tuple(remote["active_rounds"]) != ACTIVE_ROUNDS:
        raise ValueError("Figure 2 active rounds changed")
    if tuple(config["confirmation"]["methods"]) != METHODS:
        raise ValueError("Figure 2 method list changed")


def save_partition(path: Path, groups: list[np.ndarray], seed: int) -> None:
    payload = {
        "protocol": "label_skew",
        "seed": seed,
        "n_clients": N_CLIENTS,
        "label_skew_fraction": 0.70,
        "primary_clients_per_label": 2,
        "client_sizes": [int(len(group)) for group in groups],
        "partition_sha256": _partition_hash(groups),
        "client_indices": [np.asarray(group, dtype=int).tolist() for group in groups],
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != serialized:
        raise RuntimeError(f"partition changed for seed {seed}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def preflight(config: dict, output: Path, verify_hash: bool) -> None:
    adata, data_path = read_dataset(config["dataset"], config["data_dir"], verify_hash=verify_hash)
    features = preprocess_dataset(adata, config)
    features_path = output / "features.txt"
    features_path.parent.mkdir(parents=True, exist_ok=True)
    feature_payload = "\n".join(features) + "\n"
    if features_path.exists() and features_path.read_text(encoding="utf-8") != feature_payload:
        raise RuntimeError("Figure 2 feature list changed")
    features_path.write_text(feature_payload, encoding="utf-8")

    partition_rows = []
    for seed in SEEDS:
        groups = partition_for_seed(adata, config, seed)
        partition_path = output / "partitions" / f"seed_{seed}.json"
        save_partition(partition_path, groups, seed)
        schedule, frame = build_schedule(groups, np.asarray(adata.obs_names).astype(str), seed)
        del schedule
        schedule_path = output / "schedules" / f"remote_cell_schedule_seed_{seed}.csv"
        atomic_csv(frame, schedule_path)
        partition_rows.extend(
            {
                "seed": seed,
                "client": client,
                "cell_count": int(len(group)),
                "required_unique_remote_cells": len(ACTIVE_ROUNDS) * REMOTE_PER_SOURCE,
                "passes_remote_capacity": len(group) >= len(ACTIVE_ROUNDS) * REMOTE_PER_SOURCE,
                "partition_sha256": sha256(partition_path),
                "schedule_sha256": sha256(schedule_path),
            }
            for client, group in enumerate(groups)
        )
    capacity = pd.DataFrame(partition_rows)
    if not capacity.passes_remote_capacity.astype(bool).all():
        raise RuntimeError("one or more Figure 2 clients cannot satisfy one-time participation")
    atomic_csv(capacity, output / "preflight_client_capacity.csv")
    atomic_json(
        {
            "status": "PASS",
            "dataset": str(data_path),
            "dataset_sha256": sha256(data_path),
            "features_sha256": sha256(features_path),
            "canonical_seeds": list(SEEDS),
            "methods": list(METHODS),
            "historical_fifo_or_freshqueue_included": False,
            "n_clients": N_CLIENTS,
            "active_rounds": list(ACTIVE_ROUNDS),
            "remote_cells_per_source_active_round": REMOTE_PER_SOURCE,
            "one_time_participation": True,
        },
        output / "preflight.json",
    )
    print(f"Figure 2 preflight PASS: {output}")


def noqueue_config(base: dict, seed: int) -> dict:
    configured = copy.deepcopy(base)
    configured["seed"] = int(seed)
    configured["federated"]["queue"]["enabled"] = False
    configured["analysis"] = {"leiden_random_states": [0, 1, 2, 3, 4], "purity_k": 15}
    return configured


def final_config(base: dict, seed: int, schedule: dict) -> dict:
    configured = config_for_candidate(
        base,
        seed,
        NOISE_SEEDS[seed],
        schedule,
        {"C_label": "C50", "C": JOINT_C50, "sigma": FINAL_SIGMA},
    )
    configured["federated"]["queue"]["active_rounds"] = list(ACTIVE_ROUNDS)
    return configured


def sigma0_config(base: dict, seed: int, schedule: dict) -> dict:
    configured = reference_config(base, seed, schedule)
    configured["federated"]["queue"]["active_rounds"] = list(ACTIVE_ROUNDS)
    return configured


def round_callback(adata, config: dict, seed: int, method: str, rows: list[dict]):
    def callback(round_number, model, _diagnostics):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        row = {"seed": seed, "method": method, "round": int(round_number)}
        try:
            embedding = predict_embeddings(model, adata)
            row.update(trajectory_metrics(adata, embedding, config, f"fig2_{seed}_{method}_{round_number}"))
            row.update({"trajectory_leiden_seed": 0, "trajectory_stable": True, "trajectory_error": ""})
        except Exception as exc:
            row.update(
                {
                    "ARI": math.nan,
                    "NMI": math.nan,
                    "trajectory_leiden_seed": 0,
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


def replace_row(frame: pd.DataFrame, row: dict) -> pd.DataFrame:
    if not frame.empty:
        frame = frame[
            ~((frame.seed.astype(int) == int(row["seed"])) & (frame.method == row["method"]))
        ]
    return pd.concat([frame, pd.DataFrame([row])], ignore_index=True)


def train(config: dict, output: Path, verify_hash: bool) -> None:
    if not (output / "preflight.json").is_file():
        raise FileNotFoundError("run Figure 2 preflight before training")
    adata, _ = read_dataset(config["dataset"], config["data_dir"], verify_hash=verify_hash)
    features = (output / "features.txt").read_text(encoding="utf-8").splitlines()
    normalize(adata, config["preprocessing"]["normalize_target_sum"], config["preprocessing"]["log1p"])
    legacy = load_legacy(ROOT)
    metrics_path = output / "per_seed_metrics_partial.csv"
    trajectories_path = output / "round_trajectory_partial.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    trajectories = pd.read_csv(trajectories_path) if trajectories_path.exists() else pd.DataFrame()

    for seed in SEEDS:
        groups = load_partition(output / "partitions" / f"seed_{seed}.json", seed, adata.n_obs)
        schedule_frame = pd.read_csv(output / "schedules" / f"remote_cell_schedule_seed_{seed}.csv")
        schedule = schedule_from_frame(schedule_frame)
        configurations = {
            METHOD_NOQUEUE: noqueue_config(config, seed),
            METHOD_REFERENCE: sigma0_config(config, seed, schedule),
            METHOD_FINAL: final_config(config, seed, schedule),
        }
        for method in METHODS:
            if not metrics.empty and ((metrics.seed.astype(int) == seed) & (metrics.method == method)).any():
                print(f"reuse Figure 2 seed={seed} method={method}")
                continue
            directory_name = {
                METHOD_NOQUEUE: "NoQueue",
                METHOD_REFERENCE: "RemoteAB-sigma0",
                METHOD_FINAL: "Joint-C50-sigma2",
            }[method]
            run_dir = output / "training" / f"seed_{seed}" / directory_name
            run_dir.mkdir(parents=True, exist_ok=True)
            round_rows: list[dict] = []
            callback = (
                round_callback(adata, configurations[method], seed, method, round_rows)
                if method in {METHOD_NOQUEUE, METHOD_REFERENCE}
                else None
            )
            started = time.perf_counter()
            try:
                result = train_federated(
                    adata,
                    features,
                    None,
                    configurations[method],
                    run_dir,
                    domain_key="batch",
                    client_indices=groups,
                    round_callback=callback,
                )
                embedding = np.asarray(result.embeddings, dtype=np.float32)
                embedding_path = run_dir / "final_embedding.npy"
                np.save(embedding_path, embedding)
                save_checkpoint(
                    run_dir / "model_checkpoint.pt",
                    result,
                    features,
                    configurations[method],
                    domain_key="batch",
                )
                values = evaluate_candidate(
                    legacy, adata, configurations[method], embedding, f"figure2_{seed}_{method}"
                )
                clipping = pd.DataFrame(result.joint_clipping_diagnostics)
                noise = pd.DataFrame(result.joint_noise_diagnostics)
                if not clipping.empty:
                    atomic_csv(clipping, run_dir / "joint_clipping_diagnostics.csv")
                    if method == METHOD_FINAL:
                        atomic_csv(clipping, run_dir / "per_cell_clipping_diagnostics.csv")
                if not noise.empty:
                    atomic_csv(noise, run_dir / "joint_noise_diagnostics.csv")
                row = {
                    "seed": seed,
                    "method": method,
                    **{metric: values[metric] for metric in METRICS},
                    "C_label": "Joint-C50" if method != METHOD_NOQUEUE else "",
                    "C": JOINT_C50 if method != METHOD_NOQUEUE else math.nan,
                    "sigma": (FINAL_SIGMA if method == METHOD_FINAL else 0.0) if method != METHOD_NOQUEUE else math.nan,
                    "clipping_fraction": float(clipping.clipped.astype(bool).mean()) if not clipping.empty else math.nan,
                    "training_stable": bool(np.isfinite(embedding).all()),
                    "max_remote_participation_count": 1 if method != METHOD_NOQUEUE else math.nan,
                    "partition_sha256": sha256(output / "partitions" / f"seed_{seed}.json"),
                    "embedding_sha256": sha256(embedding_path),
                    "source": str(embedding_path),
                    "source_method": method,
                    "ARI_leiden_sd": values["ARI_leiden_sd"],
                    "NMI_leiden_sd": values["NMI_leiden_sd"],
                    "leiden_repeats_json": values["leiden_repeats_json"],
                    "training_seconds": float(result.training_seconds),
                    "wall_seconds": time.perf_counter() - started,
                }
                metrics = replace_row(metrics, row)
                if round_rows:
                    if not trajectories.empty:
                        trajectories = trajectories[
                            ~((trajectories.seed.astype(int) == seed) & (trajectories.method == method))
                        ]
                    trajectories = pd.concat([trajectories, pd.DataFrame(round_rows)], ignore_index=True)
                atomic_csv(metrics.sort_values(["method", "seed"]), metrics_path)
                atomic_csv(trajectories.sort_values(["method", "seed", "round"]), trajectories_path)
            except Exception:
                (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                raise
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    print(f"Figure 2 training complete: {output}")


def trajectory_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, round_number), group in frame.groupby(["method", "round"], sort=False):
        for metric in ("ARI", "NMI"):
            values = group[metric].to_numpy(float)
            mean = float(values.mean())
            sd = float(values.std(ddof=1))
            margin = float(stats.t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values)))
            rows.append(
                {
                    "method": method,
                    "round": int(round_number),
                    "metric": metric,
                    "n": len(values),
                    "mean": mean,
                    "sd": sd,
                    "ci95_low": mean - margin,
                    "ci95_high": mean + margin,
                    "trajectory_leiden_seed": 0,
                }
            )
    return pd.DataFrame(rows)


def summarize(output: Path) -> None:
    metrics = pd.read_csv(output / "per_seed_metrics_partial.csv")
    expected = {(seed, method) for seed in SEEDS for method in METHODS}
    observed = set(zip(metrics.seed.astype(int), metrics.method.astype(str)))
    if observed != expected:
        raise RuntimeError(f"incomplete Figure 2 metrics: missing={sorted(expected - observed)}")
    summary, differences, tests = method_and_stat_tables(metrics)
    trajectories = pd.read_csv(output / "round_trajectory_partial.csv")
    expected_trajectory = {
        (seed, method, round_number)
        for seed in SEEDS
        for method in (METHOD_NOQUEUE, METHOD_REFERENCE)
        for round_number in range(1, 21)
    }
    observed_trajectory = set(
        zip(trajectories.seed.astype(int), trajectories.method.astype(str), trajectories["round"].astype(int))
    )
    if observed_trajectory != expected_trajectory:
        raise RuntimeError("NoQueue/Remote A/B trajectories are incomplete")
    atomic_csv(metrics.sort_values(["method", "seed"]), output / "per_seed_metrics_final.csv")
    atomic_csv(summary, output / "figure2_summary_table.csv")
    atomic_csv(differences, output / "paired_differences.csv")
    atomic_csv(tests, output / "paired_statistics_final.csv")
    atomic_csv(trajectories.sort_values(["method", "seed", "round"]), output / "round_trajectory.csv")
    atomic_csv(trajectory_summary(trajectories), output / "round_trajectory_summary.csv")
    exposure_rows = []
    for seed in SEEDS:
        schedule = pd.read_csv(output / "schedules" / f"remote_cell_schedule_seed_{seed}.csv")
        for source, group in schedule.groupby("source_client", sort=True):
            counts = group.groupby("cell_id").size()
            exposure_rows.append(
                {
                    "seed": seed,
                    "source_client": int(source),
                    "scheduled_rows": int(len(group)),
                    "unique_remote_cells": int(group.cell_id.nunique()),
                    "active_rounds": int(group.active_round.nunique()),
                    "max_remote_participation_count": int(counts.max()),
                    "remote_cell_reuse_count": int(len(group) - group.cell_id.nunique()),
                    "zero_remote_cell_reuse": bool(len(group) == group.cell_id.nunique()),
                }
            )
    exposure = pd.DataFrame(exposure_rows)
    atomic_csv(exposure, output / "exposure_QC.csv")
    atomic_json(
        {
            "status": "COMPLETE",
            "protocol": "Figure 2 final RoundOnce three-arm reproduction",
            "seeds": list(SEEDS),
            "methods": list(METHODS),
            "active_rounds": list(ACTIVE_ROUNDS),
            "remote_cells_per_source_per_active_round": REMOTE_PER_SOURCE,
            "unique_remote_cells_per_source": len(ACTIVE_ROUNDS) * REMOTE_PER_SOURCE,
            "max_remote_participation_required": 1,
            "noise_seeds": {str(seed): NOISE_SEEDS[seed] for seed in SEEDS},
            "candidates": {
                "Joint-C50-sigma2": {"C": JOINT_C50, "sigma": FINAL_SIGMA}
            },
            "dp_scope": "cross-client A/B release channel only",
            "historical_fifo_or_freshqueue_included": False,
        },
        output / "provenance.json",
    )
    atomic_json(
        {
            "status": "COMPLETE",
            "methods": list(METHODS),
            "canonical_seeds": list(SEEDS),
            "holm_family": "18 paired t-tests: 3 comparisons x 6 metrics",
            "historical_fifo_or_freshqueue_included": False,
        },
        output / "completion.json",
    )
    print(f"Figure 2 summaries complete: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", choices=("preflight", "train", "summarize", "all"), required=True)
    parser.add_argument("--no-verify-hash", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(str(args.config))
    config["data_dir"] = str(resolve_repo_path(config["data_dir"]))
    validate_config(config)
    output = args.output.resolve() if args.output else resolve_repo_path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    serialized_config = {key: value for key, value in config.items() if not key.startswith("_")}
    (output / "config.yaml").write_text(
        yaml.safe_dump(serialized_config, sort_keys=False), encoding="utf-8"
    )
    if args.stage in {"preflight", "all"}:
        preflight(config, output, not args.no_verify_hash)
    if args.stage in {"train", "all"}:
        train(config, output, not args.no_verify_hash)
    if args.stage in {"summarize", "all"}:
        summarize(output)


if __name__ == "__main__":
    main()
