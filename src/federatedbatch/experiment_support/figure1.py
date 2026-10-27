from __future__ import annotations

import copy

import importlib.util

import json

from pathlib import Path

import numpy as np

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

from federatedbatch.experiment_support.evaluation import evaluate_embedding

REPRO = ROOT / "external_artifacts" / "figure1"

N_CLIENTS = 6

ACTIVE_ROUNDS = tuple(range(4, 21))

REMOTE_PER_SOURCE = 102

C = 1206.3391650512762

SIGMA = 2.0

FIGURE1_CONFIG = ROOT / "configs" / "figure1.yaml"

PARTITION = ROOT / "partitions" / "figure1" / "pancreas_moderate_label_skew_6clients.json"

FEATURES = REPRO / "results" / "fresh_queue_operating_window_30seed" / "features_pancreas.txt"

LEGACY_SCRIPT = REPRO / "scripts" / "recompute_legacy_metrics.py"

def load_legacy():
    spec = importlib.util.spec_from_file_location("legacy_fig1", LEGACY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import legacy evaluator: {LEGACY_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def load_partition(n_obs: int):
    payload = json.loads(PARTITION.read_text(encoding="utf-8"))
    groups = [np.asarray(x, dtype=np.int64) for x in payload["client_indices"]]
    if len(groups) != N_CLIENTS:
        raise ValueError(f"expected {N_CLIENTS} clients, got {len(groups)}")
    flat = np.concatenate(groups)
    if len(flat) != n_obs or len(np.unique(flat)) != n_obs:
        raise ValueError("Figure 1 partition does not cover cells exactly once")
    return groups, payload

def build_schedule(groups, cell_ids):
    schedule = {}
    rows = []
    required = len(ACTIVE_ROUNDS) * REMOTE_PER_SOURCE
    for source, group in enumerate(groups):
        if len(group) < required:
            raise ValueError(f"client {source} has {len(group)} cells; needs {required}")
        rng = np.random.default_rng(20260918 * 1009 + source * 7919 + 104729)
        perm = rng.permutation(len(group))[:required]
        cursor = 0
        for active_round in ACTIVE_ROUNDS:
            collection_round = active_round - 1
            selected = perm[cursor:cursor + REMOTE_PER_SOURCE].astype(int).tolist()
            cursor += REMOTE_PER_SOURCE
            schedule.setdefault(str(collection_round), {}).setdefault("0", {})[str(source)] = selected
            for local_idx in selected:
                global_idx = int(group[local_idx])
                rows.append({
                    "active_round": active_round,
                    "collection_round": collection_round,
                    "interaction_slot": 0,
                    "source_client": source,
                    "source_local_index": int(local_idx),
                    "global_index": global_idx,
                    "cell_id": str(cell_ids[global_idx]),
                })
    frame = pd.DataFrame(rows)
    counts = frame.groupby("cell_id").size()
    if len(frame) != len(ACTIVE_ROUNDS) * N_CLIENTS * REMOTE_PER_SOURCE:
        raise ValueError("unexpected schedule size")
    if counts.max() != 1:
        raise ValueError("remote-cell participation exceeds one")
    return schedule, frame

def configure(base, seed, schedule):
    cfg = copy.deepcopy(base)
    cfg["seed"] = int(seed)
    cfg["federated"]["n_clients"] = N_CLIENTS
    q = cfg["federated"]["queue"]
    q.update({
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
        "_joint_noise_seed": 20000 + int(seed),
        "_scheduled_remote_indices": schedule,
    })
    cfg["analysis"] = {"leiden_random_states": [0, 1, 2, 3, 4], "purity_k": 15}
    return cfg

def cleanup(adata, identity):
    key = f"X_regime_{identity}"
    neighbors = f"neighbors_{key}"
    adata.obsm.pop(key, None)
    adata.uns.pop(neighbors, None)
    adata.obsp.pop(f"{neighbors}_connectivities", None)
    adata.obsp.pop(f"{neighbors}_distances", None)

def evaluate(legacy, adata, cfg, embedding, identity):
    try:
        repeats, metrics = evaluate_embedding(adata, embedding, cfg, identity)
    finally:
        cleanup(adata, identity)
    batches = np.asarray(adata.obs["batch"]).astype(str)
    labels = np.asarray(adata.obs["celltype"]).astype(str)
    return {
        "ARI": float(metrics["ARI_mean"]),
        "NMI": float(metrics["NMI_mean"]),
        "ARI_sd": float(metrics["ARI_sd"]),
        "NMI_sd": float(metrics["NMI_sd"]),
        "ASW_batch": float(legacy.old_asw(embedding, batches)),
        "ASW_cell": float(legacy.old_asw(embedding, labels)),
        "Graph_connectivity": float(legacy.old_graph_connectivity(embedding, labels)),
        "iLISI": float(legacy.old_ilisi(embedding, batches, block_size=512)),
        "leiden_repeats": json.dumps(repeats, sort_keys=True),
    }
