from __future__ import annotations

import copy

import importlib.util

import json

from pathlib import Path

import numpy as np

import pandas as pd

from federatedbatch.experiment_support.evaluation import evaluate_embedding

LEGACY_SCRIPT = Path(
    "external_artifacts/figure1/scripts/recompute_legacy_metrics.py"
)

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
