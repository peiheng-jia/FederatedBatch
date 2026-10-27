#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
import sys
import time
import traceback
from argparse import Namespace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

SEEDS = (218, 394, 542, 639, 788, 814, 880)
DATASETS = ("pancreas", "immune", "lung")
N_CLIENTS = 3
ACTIVE_ROUNDS = tuple(range(4, 21))
REMOTE_PER_SOURCE = 102
REQUIRED_REMOTE_PER_SOURCE = len(ACTIVE_ROUNDS) * REMOTE_PER_SOURCE
C = 1206.3391650512762
SIGMA = 2.0
NOISE_SEEDS = {seed: 10000 + seed for seed in SEEDS}
FIG3 = ROOT / "results" / "reproduction" / "figure3"
FIG4 = ROOT / "results" / "reproduction" / "figure4"
DATA_DIR = ROOT / "data"
BASE_CONFIG = ROOT / "configs" / "figure3_4.yaml"
FEDSCGEN_CONFIG = ROOT / "configs" / "figure3_4_fedscgen.yaml"
LEGACY_SCRIPT = ROOT / "external_artifacts" / "figure1" / "scripts" / "recompute_legacy_metrics.py"
METRICS = ("ARI", "NMI", "ASW_batch", "ASW_celltype", "Graph_connectivity", "iLISI")
FIG3_METHODS = ("PCA", "Harmony", "Centralized CONCORD", "FedscGen", "FederatedBatch")
FIG4_METHODS = (
    "PCA",
    "Centralized CONCORD",
    "FedscGen zero-shot",
    "FedscGen two-stage",
    "FederatedBatch",
)


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


@lru_cache(maxsize=None)
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def write_text_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(map(str, values)) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise ValueError(f"saved text manifest differs from requested values: {path}")
    path.write_text(payload, encoding="utf-8")


def load_legacy():
    spec = importlib.util.spec_from_file_location("final_figure3_4_legacy", LEGACY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load legacy evaluator: {LEGACY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_exact_ilisi_implementation()
    return module


def load_base_config() -> dict:
    from federatedbatch.config import load_config

    config = load_config(str(BASE_CONFIG))
    config["data_dir"] = str(DATA_DIR)
    config["device"] = "auto"
    config["evaluation"]["save_embeddings"] = True
    expected = (
        config["preprocessing"]["n_hvg"],
        config["preprocessing"]["normalize_target_sum"],
        config["federated"]["n_clients"],
        config["federated"]["n_rounds"],
        config["federated"]["local_epochs"],
        config["federated"]["batch_size"],
        config["federated"]["latent_dim"],
        config["centralized"]["epochs"],
    )
    if expected != (2000, 10000, 3, 20, 5, 128, 100, 100):
        raise ValueError(f"Figure 3/4 base protocol mismatch: {expected}")
    return config


def private_config(base: dict, seed: int, schedule: dict) -> dict:
    configured = copy.deepcopy(base)
    configured["seed"] = int(seed)
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
            "active_rounds": list(ACTIVE_ROUNDS),
            "cross_client_only": True,
            "collection": "after_fedavg_with_aggregated_encoder",
            "ab_max_entries": (N_CLIENTS - 1) * REMOTE_PER_SOURCE,
            "joint_clipping_threshold": C,
            "noise_multiplier": SIGMA,
            "positive_a_floor": 1e-6,
            "_joint_noise_seed": NOISE_SEEDS[seed],
            "_scheduled_remote_indices": schedule,
        }
    )
    return configured


def client_partition(adata, seed: int) -> tuple[list[list[str]], list[np.ndarray]]:
    from federatedbatch.partition import partition_batches

    batch_groups = partition_batches(adata, N_CLIENTS, scheme="batch", seed=seed)
    observed = np.asarray(adata.obs["batch"].astype(str))
    index_groups = [
        np.flatnonzero(np.isin(observed, np.asarray(group).astype(str))).astype(np.int64)
        for group in batch_groups
    ]
    flattened = np.concatenate(index_groups)
    if len(flattened) != adata.n_obs or not np.array_equal(np.sort(flattened), np.arange(adata.n_obs)):
        raise ValueError("batch-to-client partition is not exhaustive and disjoint")
    return [[str(value) for value in group] for group in batch_groups], index_groups


def remote_schedule(
    groups: list[np.ndarray], cell_ids: np.ndarray, seed: int
) -> tuple[dict, pd.DataFrame]:
    schedule: dict[str, dict[str, dict[str, list[int]]]] = {}
    rows: list[dict] = []
    for source, group in enumerate(groups):
        if len(group) < REQUIRED_REMOTE_PER_SOURCE:
            raise ValueError(
                f"seed {seed} source {source} has {len(group)} cells; "
                f"requires {REQUIRED_REMOTE_PER_SOURCE}"
            )
        rng = np.random.default_rng(seed * 1009 + source * 7919 + 104729)
        selected = rng.permutation(len(group))[:REQUIRED_REMOTE_PER_SOURCE]
        cursor = 0
        for active_round in ACTIVE_ROUNDS:
            collection_round = active_round - 1
            local_indices = selected[cursor : cursor + REMOTE_PER_SOURCE].astype(int).tolist()
            cursor += REMOTE_PER_SOURCE
            schedule.setdefault(str(collection_round), {}).setdefault("0", {})[
                str(source)
            ] = local_indices
            for local_index in local_indices:
                global_index = int(group[local_index])
                rows.append(
                    {
                        "training_seed": seed,
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
    expected = len(ACTIVE_ROUNDS) * N_CLIENTS * REMOTE_PER_SOURCE
    counts = frame.groupby("cell_id").size()
    if len(frame) != expected or frame.cell_id.nunique() != expected or int(counts.max()) != 1:
        raise ValueError(f"seed {seed} remote schedule violates one-time participation")
    if set(frame.groupby(["active_round", "source_client"]).size()) != {REMOTE_PER_SOURCE}:
        raise ValueError(f"seed {seed} remote schedule does not use exactly 102 cells/source/round")
    return schedule, frame


def partition_payload(
    adata, batch_groups: list[list[str]], index_groups: list[np.ndarray], seed: int
) -> dict:
    return {
        "training_seed": seed,
        "partition_seed": seed,
        "partition_scheme": "seeded non-overlapping whole-batch assignment",
        "n_clients": N_CLIENTS,
        "n_cells": int(adata.n_obs),
        "client_batches": batch_groups,
        "client_cell_counts": [int(len(group)) for group in index_groups],
        "client_cell_id_sha256": [
            stable_hash(adata.obs_names[group].astype(str).tolist()) for group in index_groups
        ],
        "minimum_client_cells": int(min(map(len, index_groups))),
        "required_unique_remote_cells_per_source": REQUIRED_REMOTE_PER_SOURCE,
    }


def status_update(path: Path, row: dict, keys: tuple[str, ...]) -> None:
    frame = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not frame.empty:
        keep = np.ones(len(frame), dtype=bool)
        for key in keys:
            keep &= frame[key].astype(str).to_numpy() == str(row[key])
        frame = frame.loc[~keep].copy()
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    atomic_csv(frame, path)


def release_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def save_private_diagnostics(result, directory: Path, schedule_frame: pd.DataFrame) -> dict:
    clipping = pd.DataFrame(result.joint_clipping_diagnostics)
    noise = pd.DataFrame(result.joint_noise_diagnostics)
    expected_clipping = len(ACTIVE_ROUNDS) * N_CLIENTS * REMOTE_PER_SOURCE
    source_rows = noise[noise.record_type == "joint_source_release"].copy()
    target_rows = noise[noise.record_type == "target_postprocess"].copy()
    if len(clipping) != expected_clipping:
        raise ValueError(f"unexpected clipping diagnostic count: {len(clipping)}")
    if len(source_rows) != len(ACTIVE_ROUNDS) * N_CLIENTS:
        raise ValueError(f"unexpected joint release count: {len(source_rows)}")
    if len(target_rows) != len(ACTIVE_ROUNDS) * N_CLIENTS:
        raise ValueError(f"unexpected target post-processing count: {len(target_rows)}")
    atomic_csv(clipping, directory / "clipping_diagnostics.csv")
    atomic_csv(noise, directory / "noise_diagnostics.csv")
    invalid = int(target_rows.invalid_a_before_clamp_count.sum())
    a_count = int(target_rows.a_count.sum())
    return {
        "clipping_fraction": float(clipping.clipped.astype(bool).mean()),
        "invalid_a_count": invalid,
        "a_count": a_count,
        "invalid_a_frequency": float(invalid / a_count),
        "max_remote_participation_count": int(schedule_frame.groupby("cell_id").size().max()),
        "remote_cell_reuse_count": int(len(schedule_frame) - schedule_frame.cell_id.nunique()),
        "joint_release_events": int(len(source_rows)),
    }


def train_figure3_native() -> None:
    from federatedbatch.baselines import run_centralized, run_harmony, run_pca
    from federatedbatch.data import normalize, preprocess_dataset, read_dataset
    from federatedbatch.federated import save_checkpoint, train_federated

    root = FIG3 / "training" / "native"
    status_path = FIG3 / "training_status_native.csv"
    base = load_base_config()
    for seed in SEEDS:
        for dataset in DATASETS:
            scenario = root / f"seed_{seed}" / dataset
            scenario.mkdir(parents=True, exist_ok=True)
            config = copy.deepcopy(base)
            config["seed"] = seed
            config["dataset"] = dataset
            adata, data_path = read_dataset(dataset, config["data_dir"], verify_hash=False)
            features = preprocess_dataset(adata, config)
            write_text_lines(scenario / "features.txt", features)
            batch_groups, index_groups = client_partition(adata, seed)
            schedule, schedule_frame = remote_schedule(
                index_groups, np.asarray(adata.obs_names).astype(str), seed
            )
            atomic_json(
                partition_payload(adata, batch_groups, index_groups, seed),
                scenario / "partition.json",
            )
            atomic_csv(schedule_frame, scenario / "remote_cell_schedule.csv")
            atomic_json(
                {
                    "dataset": dataset,
                    "data_path": str(data_path),
                    "data_sha256": sha256(data_path),
                    "training_seed": seed,
                    "partition_seed": seed,
                    "leiden_seed": seed,
                    "dp_noise_seed": NOISE_SEEDS[seed],
                    "preprocessing": config["preprocessing"],
                },
                scenario / "provenance.json",
            )

            pca_path = scenario / "pca" / "final_embedding.npy"
            if not pca_path.exists():
                pca_path.parent.mkdir(parents=True, exist_ok=True)
                started = time.perf_counter()
                embedding, elapsed = run_pca(adata, features, config)
                np.save(pca_path, np.asarray(embedding, dtype=np.float32))
                status_update(
                    status_path,
                    {"dataset": dataset, "training_seed": seed, "method": "PCA", "completed": True, "stable": bool(np.isfinite(embedding).all()), "training_seconds": elapsed, "wall_seconds": time.perf_counter() - started, "error": ""},
                    ("dataset", "training_seed", "method"),
                )

            harmony_path = scenario / "harmony" / "final_embedding.npy"
            if not harmony_path.exists():
                harmony_path.parent.mkdir(parents=True, exist_ok=True)
                started = time.perf_counter()
                try:
                    embedding, elapsed = run_harmony(adata, features, config)
                    np.save(harmony_path, np.asarray(embedding, dtype=np.float32))
                    status_update(status_path, {"dataset": dataset, "training_seed": seed, "method": "Harmony", "completed": True, "stable": bool(np.isfinite(embedding).all()), "training_seconds": elapsed, "wall_seconds": time.perf_counter() - started, "error": ""}, ("dataset", "training_seed", "method"))
                except Exception as exc:
                    status_update(status_path, {"dataset": dataset, "training_seed": seed, "method": "Harmony", "completed": False, "stable": False, "training_seconds": math.nan, "wall_seconds": time.perf_counter() - started, "error": f"{type(exc).__name__}: {exc}"}, ("dataset", "training_seed", "method"))
                    raise
                release_memory()

            concord_path = scenario / "centralized_concord" / "final_embedding.npy"
            if not concord_path.exists():
                concord_path.parent.mkdir(parents=True, exist_ok=True)
                started = time.perf_counter()
                try:
                    embedding, elapsed = run_centralized(
                        adata, features, config, concord_path.parent / "model"
                    )
                    np.save(concord_path, np.asarray(embedding, dtype=np.float32))
                    status_update(status_path, {"dataset": dataset, "training_seed": seed, "method": "Centralized CONCORD", "completed": True, "stable": bool(np.isfinite(embedding).all()), "training_seconds": elapsed, "wall_seconds": time.perf_counter() - started, "error": ""}, ("dataset", "training_seed", "method"))
                except Exception as exc:
                    status_update(status_path, {"dataset": dataset, "training_seed": seed, "method": "Centralized CONCORD", "completed": False, "stable": False, "training_seconds": math.nan, "wall_seconds": time.perf_counter() - started, "error": f"{type(exc).__name__}: {exc}"}, ("dataset", "training_seed", "method"))
                    raise
                release_memory()

            fb_path = scenario / "federatedbatch" / "final_embedding.npy"
            if not fb_path.exists():
                fb_path.parent.mkdir(parents=True, exist_ok=True)
                started = time.perf_counter()
                try:
                    configured = private_config(config, seed, schedule)
                    result = train_federated(
                        adata,
                        features,
                        None,
                        configured,
                        fb_path.parent,
                        domain_key="batch",
                        client_indices=index_groups,
                    )
                    embedding = np.asarray(result.embeddings, dtype=np.float32)
                    np.save(fb_path, embedding)
                    save_checkpoint(
                        fb_path.parent / "global_model.pt",
                        result,
                        features,
                        configured,
                        domain_key="batch",
                    )
                    qc = save_private_diagnostics(result, fb_path.parent, schedule_frame)
                    if qc["max_remote_participation_count"] != 1 or qc["remote_cell_reuse_count"] != 0:
                        raise ValueError("private FederatedBatch remote exposure QC failed")
                    status_update(status_path, {"dataset": dataset, "training_seed": seed, "method": "FederatedBatch", "completed": True, "stable": bool(np.isfinite(embedding).all()), "training_seconds": result.training_seconds, "wall_seconds": time.perf_counter() - started, "error": "", **qc}, ("dataset", "training_seed", "method"))
                except Exception as exc:
                    (fb_path.parent / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    status_update(status_path, {"dataset": dataset, "training_seed": seed, "method": "FederatedBatch", "completed": False, "stable": False, "training_seconds": math.nan, "wall_seconds": time.perf_counter() - started, "error": f"{type(exc).__name__}: {exc}"}, ("dataset", "training_seed", "method"))
                    raise
                release_memory()


def train_figure4_native() -> None:
    from federatedbatch.baselines import run_centralized_zero_shot, run_pca_zero_shot
    from federatedbatch.data import normalize, read_dataset, select_features
    from federatedbatch.federated import predict_embeddings, save_checkpoint, train_federated

    root = FIG4 / "training" / "native"
    status_path = FIG4 / "training_status_native.csv"
    base = load_base_config()
    base["dataset"] = "pancreas"
    source, data_path = read_dataset("pancreas", base["data_dir"], verify_hash=False)
    held_out_batches = [str(value) for value in source.obs["batch"].unique()]
    if len(held_out_batches) != 9:
        raise ValueError(f"expected nine Pancreas batches, found {len(held_out_batches)}")

    for seed in SEEDS:
        for held_out in held_out_batches:
            scenario = root / f"seed_{seed}" / held_out
            scenario.mkdir(parents=True, exist_ok=True)
            config = copy.deepcopy(base)
            config["seed"] = seed
            train = source[source.obs["batch"].astype(str) != held_out].copy()
            test = source[source.obs["batch"].astype(str) == held_out].copy()
            features = select_features(train, 2000, "seurat_v3")
            normalize(train, 10000, True)
            normalize(test, 10000, True)
            write_text_lines(scenario / "features.txt", features)
            write_text_lines(scenario / "test_cell_ids.txt", test.obs_names.astype(str).tolist())
            batch_groups, index_groups = client_partition(train, seed)
            schedule, schedule_frame = remote_schedule(
                index_groups, np.asarray(train.obs_names).astype(str), seed
            )
            partition = partition_payload(train, batch_groups, index_groups, seed)
            partition.update(
                {
                    "held_out_batch": held_out,
                    "held_out_participates_in_training": False,
                    "held_out_cell_count": int(test.n_obs),
                    "held_out_cell_id_sha256": stable_hash(test.obs_names.astype(str).tolist()),
                    "training_held_out_overlap": int(len(set(train.obs_names) & set(test.obs_names))),
                }
            )
            atomic_json(partition, scenario / "partition.json")
            atomic_csv(schedule_frame, scenario / "remote_cell_schedule.csv")
            atomic_json(
                {
                    "dataset": "pancreas",
                    "data_path": str(data_path),
                    "data_sha256": sha256(data_path),
                    "training_seed": seed,
                    "partition_seed": seed,
                    "leiden_seed": seed,
                    "dp_noise_seed": NOISE_SEEDS[seed],
                    "held_out_batch": held_out,
                    "held_out_participates_in_training": False,
                    "held_out_participates_in_correction": False,
                    "additional_communication_rounds": 0,
                },
                scenario / "provenance.json",
            )

            pca_path = scenario / "pca" / "final_embedding.npy"
            if not pca_path.exists():
                pca_path.parent.mkdir(parents=True, exist_ok=True)
                started = time.perf_counter()
                embedding, elapsed = run_pca_zero_shot(train, test, features, config)
                np.save(pca_path, np.asarray(embedding, dtype=np.float32))
                status_update(status_path, {"held_out_batch": held_out, "training_seed": seed, "method": "PCA", "completed": True, "stable": bool(np.isfinite(embedding).all()), "training_seconds": elapsed, "wall_seconds": time.perf_counter() - started, "held_out_participates_in_training": False, "held_out_participates_in_correction": False, "additional_communication_rounds": 0, "error": ""}, ("held_out_batch", "training_seed", "method"))

            concord_path = scenario / "centralized_concord" / "final_embedding.npy"
            if not concord_path.exists():
                concord_path.parent.mkdir(parents=True, exist_ok=True)
                started = time.perf_counter()
                try:
                    embedding, elapsed = run_centralized_zero_shot(
                        train, test, features, config, concord_path.parent / "model"
                    )
                    np.save(concord_path, np.asarray(embedding, dtype=np.float32))
                    status_update(status_path, {"held_out_batch": held_out, "training_seed": seed, "method": "Centralized CONCORD", "completed": True, "stable": bool(np.isfinite(embedding).all()), "training_seconds": elapsed, "wall_seconds": time.perf_counter() - started, "held_out_participates_in_training": False, "held_out_participates_in_correction": False, "additional_communication_rounds": 0, "error": ""}, ("held_out_batch", "training_seed", "method"))
                except Exception as exc:
                    status_update(status_path, {"held_out_batch": held_out, "training_seed": seed, "method": "Centralized CONCORD", "completed": False, "stable": False, "training_seconds": math.nan, "wall_seconds": time.perf_counter() - started, "held_out_participates_in_training": False, "held_out_participates_in_correction": False, "additional_communication_rounds": 0, "error": f"{type(exc).__name__}: {exc}"}, ("held_out_batch", "training_seed", "method"))
                    raise
                release_memory()

            fb_path = scenario / "federatedbatch" / "final_embedding.npy"
            if not fb_path.exists():
                fb_path.parent.mkdir(parents=True, exist_ok=True)
                started = time.perf_counter()
                try:
                    configured = private_config(config, seed, schedule)
                    result = train_federated(
                        train,
                        features,
                        None,
                        configured,
                        fb_path.parent,
                        domain_key=None,
                        client_indices=index_groups,
                    )
                    embedding = np.asarray(predict_embeddings(result.model, test), dtype=np.float32)
                    np.save(fb_path, embedding)
                    save_checkpoint(
                        fb_path.parent / "global_model.pt",
                        result,
                        features,
                        configured,
                        domain_key=None,
                    )
                    qc = save_private_diagnostics(result, fb_path.parent, schedule_frame)
                    if qc["max_remote_participation_count"] != 1 or qc["remote_cell_reuse_count"] != 0:
                        raise ValueError("LOBO private FederatedBatch remote exposure QC failed")
                    status_update(status_path, {"held_out_batch": held_out, "training_seed": seed, "method": "FederatedBatch", "completed": True, "stable": bool(np.isfinite(embedding).all()), "training_seconds": result.training_seconds, "wall_seconds": time.perf_counter() - started, "held_out_participates_in_training": False, "held_out_participates_in_correction": False, "additional_communication_rounds": 0, "error": "", **qc}, ("held_out_batch", "training_seed", "method"))
                except Exception as exc:
                    (fb_path.parent / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    status_update(status_path, {"held_out_batch": held_out, "training_seed": seed, "method": "FederatedBatch", "completed": False, "stable": False, "training_seconds": math.nan, "wall_seconds": time.perf_counter() - started, "held_out_participates_in_training": False, "held_out_participates_in_correction": False, "additional_communication_rounds": 0, "error": f"{type(exc).__name__}: {exc}"}, ("held_out_batch", "training_seed", "method"))
                    raise
                release_memory()


def fedscgen_args(seed: int, output: Path, workflow: str) -> Namespace:
    return Namespace(
        config=str(FEDSCGEN_CONFIG),
        seed=seed,
        output_dir=str(output),
        device=None,
        n_rounds=None,
        local_epochs=None,
        n_hvg=None,
        max_cells=None,
        no_verify_hash=False,
        resume=True,
        datasets=list(DATASETS),
        held_out_batches=None,
    )


def train_fedscgen(selected_figures: tuple[str, ...] = ("3", "4")) -> None:
    # The pinned upstream uses ``import __init__`` inside scgen_utils.py and
    # therefore requires its package directory on sys.path in addition to the
    # repository root. This is import compatibility only; upstream code and
    # the experimental protocol remain unchanged.
    upstream_package = ROOT / "baselines" / "fedscgen" / "upstream" / "fedscgen"
    if str(upstream_package) not in sys.path:
        sys.path.insert(0, str(upstream_package))
    from baselines.fedscgen.run_official import run_lobo, run_multi

    for seed in SEEDS if "3" in selected_figures else ():
        multi = FIG3 / "training" / "fedscgen" / f"seed_{seed}"
        multi.mkdir(parents=True, exist_ok=True)
        result_path = multi / "fedscgen_multi_dataset_results.csv"
        if not result_path.exists() or len(pd.read_csv(result_path)) != len(DATASETS):
            run_multi(fedscgen_args(seed, multi, "multi_dataset"))

    for seed in SEEDS if "4" in selected_figures else ():
        lobo = FIG4 / "training" / "fedscgen" / f"seed_{seed}"
        lobo.mkdir(parents=True, exist_ok=True)
        result_path = lobo / "fedscgen_lobo_summary.csv"
        if not result_path.exists() or len(pd.read_csv(result_path)) != 18:
            run_lobo(fedscgen_args(seed, lobo, "lobo"))


def native_embedding_figure3(seed: int, dataset: str, method: str) -> Path:
    names = {
        "PCA": "pca",
        "Harmony": "harmony",
        "Centralized CONCORD": "centralized_concord",
        "FederatedBatch": "federatedbatch",
    }
    return FIG3 / "training" / "native" / f"seed_{seed}" / dataset / names[method] / "final_embedding.npy"


def fedscgen_embedding_figure3(seed: int, dataset: str) -> tuple[Path, str]:
    return (
        FIG3 / "training" / "fedscgen" / f"seed_{seed}" / dataset / "adata_with_embeddings.h5ad",
        "X_fedscgen",
    )


def native_embedding_figure4(seed: int, held_out: str, method: str) -> Path:
    names = {
        "PCA": "pca",
        "Centralized CONCORD": "centralized_concord",
        "FederatedBatch": "federatedbatch",
    }
    return FIG4 / "training" / "native" / f"seed_{seed}" / held_out / names[method] / "final_embedding.npy"


def fedscgen_embedding_figure4(seed: int, held_out: str, method: str) -> tuple[Path, str]:
    if method == "FedscGen zero-shot":
        name, key = "held_out_zero_shot_embedding.h5ad", "X_fedscgen_zero_shot"
    else:
        name, key = "held_out_two_stage_embedding.h5ad", "X_fedscgen_two_stage"
    return FIG4 / "training" / "fedscgen" / f"seed_{seed}" / held_out / name, key


def read_h5ad_embedding(path: Path, key: str, expected_ids: pd.Index) -> np.ndarray:
    import anndata as ad

    loaded = ad.read_h5ad(path)
    if not loaded.obs_names.equals(expected_ids):
        raise ValueError(f"embedding cell order mismatch: {path}")
    return np.asarray(loaded.obsm[key], dtype=np.float32)


def evaluate_embedding(legacy, adata, embedding: np.ndarray, seed: int, identity: str) -> dict:
    from federatedbatch.evaluation import clear_evaluation_artifacts
    from federatedbatch.evaluation import evaluate_embedding as current_evaluate

    config = {
        "seed": int(seed),
        "evaluation": {"n_neighbors": 15, "leiden_resolution": 1.0, "save_embeddings": False},
    }
    key = f"X_final_{identity}"
    try:
        current = current_evaluate(adata, embedding, config, embedding_key=key)
    finally:
        clear_evaluation_artifacts(adata, key)
    batches = np.asarray(adata.obs["batch"]).astype(str)
    labels = np.asarray(adata.obs["celltype"]).astype(str)
    return {
        "ARI": float(current["ARI"]),
        "NMI": float(current["NMI"]),
        "ASW_batch": float(legacy.old_asw(embedding, batches)) if len(np.unique(batches)) > 1 else math.nan,
        "ASW_celltype": float(legacy.old_asw(embedding, labels)) if len(np.unique(labels)) > 1 else math.nan,
        "Graph_connectivity": float(legacy.old_graph_connectivity(embedding, labels)),
        "iLISI": float(legacy.old_ilisi(embedding, batches, block_size=512)),
    }


def metric_checkpoint(path: Path, build) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    value = build()
    atomic_json(value, path)
    return value


def collect_figure3_metrics(legacy) -> pd.DataFrame:
    from federatedbatch.data import read_dataset

    rows = []
    for seed in SEEDS:
        for dataset in DATASETS:
            adata, _ = read_dataset(dataset, str(DATA_DIR), verify_hash=False)
            native_features = FIG3 / "training" / "native" / f"seed_{seed}" / dataset / "features.txt"
            fedscgen_features = FIG3 / "training" / "fedscgen" / f"seed_{seed}" / dataset / "features.txt"
            if native_features.read_bytes() != fedscgen_features.read_bytes():
                raise ValueError(f"Figure 3 feature mismatch: seed {seed} {dataset}")
            for method in FIG3_METHODS:
                if method == "FedscGen":
                    source, key = fedscgen_embedding_figure3(seed, dataset)
                    embedding = read_h5ad_embedding(source, key, adata.obs_names)
                else:
                    source = native_embedding_figure3(seed, dataset, method)
                    embedding = np.load(source).astype(np.float32, copy=False)
                checkpoint = FIG3 / "metric_checkpoints" / f"seed_{seed}" / dataset / f"{method.replace(' ', '_').lower()}.json"
                values = metric_checkpoint(
                    checkpoint,
                    lambda e=embedding, s=seed, i=f"fig3_{dataset}_{method}_{seed}": evaluate_embedding(legacy, adata, e, s, i),
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "training_seed": seed,
                        "partition_seed": seed,
                        "leiden_seed": seed,
                        "DP_noise_seed": NOISE_SEEDS[seed] if method == "FederatedBatch" else math.nan,
                        **values,
                        "training_stable": bool(np.isfinite(embedding).all()),
                        "embedding_path": str(source),
                        "embedding_sha256": sha256(source),
                    }
                )
            release_memory()
    frame = pd.DataFrame(rows)
    atomic_csv(frame, FIG3 / "per_seed_per_dataset_metrics.csv")
    return frame


def collect_figure4_metrics(legacy) -> pd.DataFrame:
    from federatedbatch.data import read_dataset

    source_data, _ = read_dataset("pancreas", str(DATA_DIR), verify_hash=False)
    batches = [str(value) for value in source_data.obs["batch"].unique()]
    rows = []
    deployment = {
        "PCA": (False, False, 0),
        "Centralized CONCORD": (False, False, 0),
        "FederatedBatch": (False, False, 0),
        "FedscGen zero-shot": (False, False, 0),
        "FedscGen two-stage": (False, True, 2),
    }
    for seed in SEEDS:
        for held_out in batches:
            test = source_data[source_data.obs["batch"].astype(str) == held_out].copy()
            native_features = FIG4 / "training" / "native" / f"seed_{seed}" / held_out / "features.txt"
            fedscgen_features = FIG4 / "training" / "fedscgen" / f"seed_{seed}" / held_out / "features.txt"
            if native_features.read_bytes() != fedscgen_features.read_bytes():
                raise ValueError(f"Figure 4 feature mismatch: seed {seed} held-out {held_out}")
            for method in FIG4_METHODS:
                if method.startswith("FedscGen"):
                    path, key = fedscgen_embedding_figure4(seed, held_out, method)
                    embedding = read_h5ad_embedding(path, key, test.obs_names)
                    source = path
                else:
                    source = native_embedding_figure4(seed, held_out, method)
                    embedding = np.load(source).astype(np.float32, copy=False)
                checkpoint = FIG4 / "metric_checkpoints" / f"seed_{seed}" / held_out / f"{method.replace(' ', '_').lower()}.json"
                values = metric_checkpoint(
                    checkpoint,
                    lambda e=embedding, s=seed, i=f"fig4_{held_out}_{method}_{seed}": evaluate_embedding(legacy, test, e, s, i),
                )
                in_training, in_correction, rounds = deployment[method]
                rows.append(
                    {
                        "held_out_batch": held_out,
                        "method": method,
                        "training_seed": seed,
                        "partition_seed": seed,
                        "leiden_seed": seed,
                        "DP_noise_seed": NOISE_SEEDS[seed] if method == "FederatedBatch" else math.nan,
                        "held_out_participates_in_training": in_training,
                        "held_out_participates_in_correction": in_correction,
                        "additional_communication_rounds": rounds,
                        "deployment_mode": "label-informed two-stage correction" if in_correction else "direct zero-shot encoding",
                        **values,
                        "training_stable": bool(np.isfinite(embedding).all()),
                        "embedding_path": str(source),
                        "embedding_sha256": sha256(source),
                    }
                )
            release_memory()
    frame = pd.DataFrame(rows)
    atomic_csv(frame, FIG4 / "per_seed_per_heldout_metrics.csv")
    return frame


def t_interval(values: np.ndarray) -> tuple[float, float]:
    from scipy import stats

    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return math.nan, math.nan
    mean = float(values.mean())
    margin = float(stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))
    return mean - margin, mean + margin


def summarize(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for keys, part in frame.groupby(groups, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for metric in METRICS:
            values = part[metric].dropna().to_numpy(float)
            low, high = t_interval(values)
            rows.append(
                {
                    **dict(zip(groups, keys)),
                    "metric": metric,
                    "n": len(values),
                    "mean": float(values.mean()) if len(values) else math.nan,
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else math.nan,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(rows)


def holm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    adjusted = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    order = valid[np.argsort(values[valid])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def paired_tests(
    frame: pd.DataFrame,
    unit: str,
    group: str | None,
    reference: str,
    comparators: tuple[str, ...],
) -> pd.DataFrame:
    from scipy import stats

    rows = []
    groups = [("all", frame)] if group is None else list(frame.groupby(group, sort=False))
    for group_value, subset in groups:
        indexed = subset.set_index([unit, "method"])
        for comparator in comparators:
            for metric in METRICS:
                units = sorted(
                    set(subset.loc[subset.method == reference, unit])
                    & set(subset.loc[subset.method == comparator, unit])
                )
                a = np.asarray([indexed.loc[(value, reference), metric] for value in units], dtype=float)
                b = np.asarray([indexed.loc[(value, comparator), metric] for value in units], dtype=float)
                valid = np.isfinite(a) & np.isfinite(b)
                differences = a[valid] - b[valid]
                if len(differences) < 2:
                    continue
                low, high = t_interval(differences)
                t_result = stats.ttest_rel(a[valid], b[valid])
                try:
                    wilcoxon = stats.wilcoxon(differences, alternative="two-sided", method="exact")
                    w_stat, w_p = float(wilcoxon.statistic), float(wilcoxon.pvalue)
                except ValueError:
                    w_stat, w_p = math.nan, math.nan
                rows.append(
                    {
                        "group": group_value,
                        "reference": reference,
                        "comparator": comparator,
                        "metric": metric,
                        "n_pairs": len(differences),
                        "paired_units": ";".join(map(str, np.asarray(units)[valid])),
                        "mean_paired_difference": float(differences.mean()),
                        "sd_paired_difference": float(differences.std(ddof=1)),
                        "ci95_low": low,
                        "ci95_high": high,
                        "paired_t_statistic": float(t_result.statistic),
                        "paired_t_df": len(differences) - 1,
                        "paired_t_raw_p": float(t_result.pvalue),
                        "wilcoxon_statistic": w_stat,
                        "wilcoxon_raw_p": w_p,
                        "positive_pair_count": int((differences > 0).sum()),
                        "difference_direction": f"{reference} minus {comparator}",
                    }
                )
    result = pd.DataFrame(rows)
    result["paired_t_holm_p"] = holm(result.paired_t_raw_p.to_numpy(float))
    result["wilcoxon_holm_p"] = holm(result.wilcoxon_raw_p.to_numpy(float))
    result["ci_note"] = "Unadjusted t-based 95% CI; p values are additionally Holm-adjusted across this table."
    return result


def figure4_seed_means(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(["training_seed", "method"], as_index=False)[list(METRICS)].mean()


def render_figures(fig3_summary: pd.DataFrame, fig4_frame: pd.DataFrame, fig4_overall: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch, Rectangle

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    colors = {
        "PCA": "#A7A9AC",
        "Harmony": "#4C78A8",
        "Centralized CONCORD": "#59A14F",
        "FedscGen": "#B07AA1",
        "FedscGen zero-shot": "#B07AA1",
        "FedscGen two-stage": "#7F4E92",
        "FederatedBatch": "#E76F51",
    }
    labels = {
        "ASW_batch": "ASW (batch)",
        "ASW_celltype": "ASW (cell type)",
        "Graph_connectivity": "Graph connectivity",
        "iLISI": "iLISI",
    }

    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.1), constrained_layout=True)
    x = np.arange(len(DATASETS))
    offsets = np.linspace(-0.28, 0.28, len(FIG3_METHODS))
    for ax, metric, panel in zip(axes.flat, METRICS, "ABCDEF"):
        for offset, method in zip(offsets, FIG3_METHODS):
            rows = fig3_summary[(fig3_summary.method == method) & (fig3_summary.metric == metric)].set_index("dataset")
            means = np.asarray([rows.loc[d, "mean"] for d in DATASETS], dtype=float)
            lows = np.asarray([rows.loc[d, "ci95_low"] for d in DATASETS], dtype=float)
            highs = np.asarray([rows.loc[d, "ci95_high"] for d in DATASETS], dtype=float)
            ax.errorbar(x + offset, means, yerr=np.vstack([means - lows, highs - means]), fmt="o", ms=4, capsize=2, lw=0.8, color=colors[method], label=method)
        ax.set_xticks(x, [value.title() for value in DATASETS])
        ax.set_title(labels.get(metric, metric), fontsize=9, fontweight="bold")
        ax.text(-0.12, 1.08, panel, transform=ax.transAxes, fontsize=11, fontweight="bold")
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.5, zorder=0)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[m], markeredgecolor=colors[m], label=m, markersize=5) for m in FIG3_METHODS]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(FIG3 / f"figure3_canonical7.{suffix}", dpi=400 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    batches = [str(value) for value in fig4_frame.held_out_batch.drop_duplicates()]
    fig = plt.figure(figsize=(11.2, 6.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.78, 1.35], width_ratios=[0.92, 1.65])
    ax = fig.add_subplot(grid[0, 0])
    ax.axis("off")
    ax.text(-0.05, 1.05, "A", transform=ax.transAxes, fontsize=11, fontweight="bold")
    for index in range(4):
        ax.add_patch(Rectangle((0.05 + index * 0.14, 0.62), 0.09, 0.13, facecolor="#EEF1F4", edgecolor="#777777", lw=0.6))
    ax.add_patch(Rectangle((0.72, 0.62), 0.16, 0.13, facecolor="#F9E2DC", edgecolor=colors["FederatedBatch"], lw=0.8))
    ax.add_patch(FancyArrowPatch((0.60, 0.685), (0.70, 0.685), arrowstyle="->", mutation_scale=10, lw=0.8))
    ax.text(0.32, 0.52, "Train on eight batches", ha="center")
    ax.text(0.80, 0.52, "Held-out batch", ha="center")
    ax.text(0.47, 0.27, "Direct encoder inference: 0 additional rounds", ha="center")
    ax.text(0.47, 0.12, "FedscGen two-stage correction: 2 additional rounds", ha="center")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax = fig.add_subplot(grid[0, 1])
    ax.text(-0.08, 1.05, "B", transform=ax.transAxes, fontsize=11, fontweight="bold")
    selected_metrics = ("ARI", "NMI", "ASW_celltype")
    positions = np.arange(len(selected_metrics))
    offsets = np.linspace(-0.27, 0.27, len(FIG4_METHODS))
    for offset, method in zip(offsets, FIG4_METHODS):
        rows = fig4_overall[(fig4_overall.method == method)].set_index("metric")
        means = np.asarray([rows.loc[m, "mean"] for m in selected_metrics])
        lows = np.asarray([rows.loc[m, "ci95_low"] for m in selected_metrics])
        highs = np.asarray([rows.loc[m, "ci95_high"] for m in selected_metrics])
        ax.errorbar(positions + offset, means, yerr=np.vstack([means-lows, highs-means]), fmt="o", ms=4, capsize=2, lw=0.8, color=colors[method], label=method)
    ax.set_xticks(positions, [labels.get(m, m) for m in selected_metrics])
    ax.set_ylabel("Mean across held-out batches")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)

    ax = fig.add_subplot(grid[1, :])
    ax.text(-0.04, 1.04, "C", transform=ax.transAxes, fontsize=11, fontweight="bold")
    by_batch = summarize(fig4_frame, ["held_out_batch", "method"])
    by_batch = by_batch[by_batch.metric == "ARI"]
    for method in FIG4_METHODS:
        rows = by_batch[by_batch.method == method].set_index("held_out_batch")
        means = np.asarray([rows.loc[b, "mean"] for b in batches])
        lows = np.asarray([rows.loc[b, "ci95_low"] for b in batches])
        highs = np.asarray([rows.loc[b, "ci95_high"] for b in batches])
        ax.errorbar(np.arange(len(batches)), means, yerr=np.vstack([means-lows, highs-means]), marker="o", ms=3.5, lw=0.9, capsize=1.5, color=colors[method], label=method)
    ax.set_xticks(np.arange(len(batches)), batches, rotation=35, ha="right")
    ax.set_ylabel("ARI")
    ax.set_xlabel("Held-out technical batch")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)
    handles = [Line2D([0], [0], marker="o", color=colors[m], label=m, markersize=4, lw=1) for m in FIG4_METHODS]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(FIG4 / f"figure4_canonical7.{suffix}", dpi=400 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_reports(fig3: pd.DataFrame, fig3_summary: pd.DataFrame, fig3_tests: pd.DataFrame, fig4: pd.DataFrame, fig4_overall: pd.DataFrame, fig4_tests: pd.DataFrame) -> None:
    fig3_fb = fig3_summary[fig3_summary.method == "FederatedBatch"]
    fig3_fedscgen = fig3_tests[fig3_tests.comparator == "FedscGen"]
    fig3_text = f"""# Final Figure 3 canonical-seven report

All stochastic methods use training seeds `{', '.join(map(str, SEEDS))}`. PCA,
Harmony, centralized CONCORD, official FedscGen, and final private
FederatedBatch use the same 2,000-HVG preprocessing and seed-matched whole-batch
three-client partitions. FederatedBatch is the locked RoundOnce Joint-C50,
sigma=2 protocol with 102 unique remote cells per source in rounds 4--20.
Every saved schedule has maximum remote participation one and zero reuse.

`per_seed_per_dataset_metrics.csv` is the authoritative raw table. ARI/NMI use
the original Figure 3 Leiden setting with the training seed as Leiden seed.
ASW values are raw silhouettes; graph connectivity and iLISI use the submitted
legacy definitions, including full-pairwise Gaussian iLISI. Paired tests use
the seven seed identifiers and never mix the former 42--46 block.

## Reviewer-facing summary

Across Pancreas, Immune, and Lung, the locked private FederatedBatch result is
reported without assuming superiority. Dataset-specific mean, SD, and 95% CI
are in `summary_statistics.csv`; direct seed-matched comparisons against
official FedscGen are in `paired_statistics.csv`. The comparison therefore
addresses cross-tissue generalization and the FedscGen baseline under identical
preprocessing, partitions, and canonical training seeds.

## Results-ready numeric extract

```text
{fig3_fb.to_string(index=False)}
```

## FederatedBatch versus FedscGen

```text
{fig3_fedscgen.to_string(index=False)}
```
"""
    (FIG3 / "FIGURE3_CANONICAL7_REPORT.md").write_text(fig3_text, encoding="utf-8")

    deploy = fig4[["method", "held_out_participates_in_training", "held_out_participates_in_correction", "additional_communication_rounds", "deployment_mode"]].drop_duplicates()
    fig4_text = f"""# Final Figure 4 canonical-seven report

Nine Pancreas leave-one-batch-out scenarios were evaluated for each canonical
training seed. The held-out batch is absent from every model-training run.
Final FederatedBatch and FedscGen zero-shot use direct encoder inference with
zero additional communication rounds. FedscGen two-stage is kept as a distinct,
label-informed correction deployment and uses two additional rounds; it is not
pooled with the zero-shot condition.

Inferential comparisons first average the nine held-out batches within each
training seed, so n=7 remains the independent replicate count. Raw per-scenario
values are in `per_seed_per_heldout_metrics.csv`, and the performance/deployment
trade-off is explicit in `deployment_communication_cost.csv`.

## Deployment conditions

```text
{deploy.to_string(index=False)}
```

## Overall metrics

```text
{fig4_overall.to_string(index=False)}
```

## Reviewer-facing summary

The final table reports true zero-shot FederatedBatch separately from
FedscGen's zero-shot encoder and optional two-stage correction. This makes both
predictive performance and post-training communication cost visible and avoids
interpreting label-informed correction as the same deployment condition as
zero-shot inference.
"""
    (FIG4 / "FIGURE4_CANONICAL7_REPORT.md").write_text(fig4_text, encoding="utf-8")


def finalize(selected_figures: tuple[str, ...] = ("3", "4")) -> None:
    legacy = load_legacy()
    completed = []
    if "3" in selected_figures:
        fig3 = collect_figure3_metrics(legacy)
        fig3_summary = summarize(fig3, ["dataset", "method"])
        atomic_csv(fig3_summary, FIG3 / "summary_statistics.csv")
        fig3_tests = paired_tests(
            fig3,
            unit="training_seed",
            group="dataset",
            reference="FederatedBatch",
            comparators=("PCA", "Harmony", "Centralized CONCORD", "FedscGen"),
        )
        atomic_csv(fig3_tests, FIG3 / "paired_statistics.csv")
        atomic_csv(fig3.copy(), FIG3 / "figure3_plot_data.csv")
        completed.append((FIG3, "Figure 3", fig3))

    if "4" in selected_figures:
        fig4 = collect_figure4_metrics(legacy)
        atomic_csv(fig4.copy(), FIG4 / "figure4_plot_data.csv")
        # Only the final four methods/four held-out metrics enter inference.
        # The exporter enforces nine batches within each seed and 12 tests.
        from scripts.summarize.export_figure4_final_tables import (
            METHOD_MAP, build_summary, build_tests, validate,
        )
        final = fig4[fig4.method.isin(METHOD_MAP)].copy()
        final["method"] = final.method.map(METHOD_MAP)
        validate(final)
        atomic_csv(build_summary(final), FIG4 / "summary_statistics.csv")
        atomic_csv(build_tests(final), FIG4 / "paired_statistics.csv")
        deployment = fig4[["method", "held_out_participates_in_training", "held_out_participates_in_correction", "additional_communication_rounds", "deployment_mode"]].drop_duplicates()
        atomic_csv(deployment, FIG4 / "deployment_communication_cost.csv")
        completed.append((FIG4, "Figure 4", fig4))

    # Publication renderers are separate registered stages. Do not call the
    # historical render_figures/write_reports helpers or generate obsolete panels.
    for output, figure, rows in completed:
        atomic_json(
            {
                "status": "COMPLETE",
                "figure": figure,
                "canonical_training_seeds": list(SEEDS),
                "row_count": int(len(rows)),
                "completed_at_unix": time.time(),
                "old_results_overwritten": False,
                "manuscript_modified": False,
                "other_figures_modified": False,
            },
            output / "completion.json",
        )


def preflight(selected_figures: tuple[str, ...] = ("3", "4")) -> None:
    from federatedbatch.data import read_dataset

    base = load_base_config()
    capacity_rows = []
    for dataset in DATASETS if "3" in selected_figures else ("pancreas",):
        adata, data_path = read_dataset(dataset, str(DATA_DIR), verify_hash=True)
        for seed in SEEDS if "3" in selected_figures else ():
            batch_groups, index_groups = client_partition(adata, seed)
            for source, (batches, indices) in enumerate(zip(batch_groups, index_groups)):
                capacity_rows.append(
                    {
                        "figure": "Figure 3",
                        "dataset": dataset,
                        "held_out_batch": "",
                        "training_seed": seed,
                        "source_client": source,
                        "client_batches": ";".join(batches),
                        "available_training_cells": len(indices),
                        "required_unique_remote_cells": REQUIRED_REMOTE_PER_SOURCE,
                        "passes": len(indices) >= REQUIRED_REMOTE_PER_SOURCE,
                    }
                )
        if dataset == "pancreas" and "4" in selected_figures:
            for held_out in adata.obs["batch"].astype(str).unique():
                train = adata[adata.obs["batch"].astype(str) != held_out].copy()
                for seed in SEEDS:
                    batch_groups, index_groups = client_partition(train, seed)
                    for source, (batches, indices) in enumerate(zip(batch_groups, index_groups)):
                        capacity_rows.append(
                            {
                                "figure": "Figure 4",
                                "dataset": "pancreas",
                                "held_out_batch": held_out,
                                "training_seed": seed,
                                "source_client": source,
                                "client_batches": ";".join(batches),
                                "available_training_cells": len(indices),
                                "required_unique_remote_cells": REQUIRED_REMOTE_PER_SOURCE,
                                "passes": len(indices) >= REQUIRED_REMOTE_PER_SOURCE,
                            }
                        )
    capacity = pd.DataFrame(capacity_rows)
    if not capacity.passes.astype(bool).all():
        failures = capacity[~capacity.passes.astype(bool)]
        raise RuntimeError(f"remote capacity preflight failed:\n{failures.to_string(index=False)}")

    for output, figure in ((FIG3, "Figure 3"), (FIG4, "Figure 4")):
        if figure[-1] not in selected_figures:
            continue
        output.mkdir(parents=True, exist_ok=True)
        atomic_csv(capacity[capacity.figure == figure], output / "preflight_remote_capacity.csv")
        atomic_json(
            {
                "status": "PASS",
                "canonical_training_seeds": list(SEEDS),
                "n_clients": N_CLIENTS,
                "active_rounds": list(ACTIVE_ROUNDS),
                "remote_cells_per_source_per_active_round": REMOTE_PER_SOURCE,
                "required_unique_remote_cells_per_source": REQUIRED_REMOTE_PER_SOURCE,
                "minimum_available_training_cells": int(capacity.loc[capacity.figure == figure, "available_training_cells"].min()),
                "final_private_method": {
                    "mode": "RoundOnce exact A/B with joint clipping and one joint Gaussian release",
                    "C": C,
                    "sigma": SIGMA,
                    "tau_q": 0.20,
                    "alpha": 0.10,
                    "top5_mask": False,
                },
                "DP_noise_seed_mapping": NOISE_SEEDS,
                "base_config": str(BASE_CONFIG),
                "fedscgen_config": str(FEDSCGEN_CONFIG),
                "data_directory": str(DATA_DIR),
                "output_is_independent": True,
                "old_results_overwritten": False,
            },
            output / "preflight.json",
        )
    for figure, group in capacity.groupby("figure"):
        print(f"Preflight PASS: {figure} minimum client={group.available_training_cells.min()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run and finalize canonical-seven Figure 3 and Figure 4 experiments.')
    parser.add_argument(
        "--stage",
        choices=("preflight", "native", "fedscgen", "finalize"),
        required=True,
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--figure3-output", type=Path, default=FIG3)
    parser.add_argument("--figure4-output", type=Path, default=FIG4)
    parser.add_argument("--figure", choices=("3", "4", "both"), default="both")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global DATA_DIR, FIG3, FIG4
    DATA_DIR = args.data_dir.resolve()
    FIG3 = args.figure3_output.resolve()
    FIG4 = args.figure4_output.resolve()
    selected_figures = ("3", "4") if args.figure == "both" else (args.figure,)
    if args.stage == "preflight":
        preflight(selected_figures)
    elif args.stage == "native":
        if "3" in selected_figures:
            train_figure3_native()
        if "4" in selected_figures:
            train_figure4_native()
    elif args.stage == "fedscgen":
        train_fedscgen(selected_figures)
    else:
        finalize(selected_figures)


if __name__ == "__main__":
    main()
