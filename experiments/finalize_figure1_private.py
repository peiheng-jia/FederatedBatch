#!/usr/bin/env python3
"""Helper functions retained for the Figure 1 canonical-seven driver.

The legacy main routine uses an obsolete seed block and is intentionally not a
public entry point. Use `run_figure1_canonical7.py` for the locked protocol.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.fresh_queue_regime import evaluate_embedding
from federatedbatch.config import load_config, resolve_repo_path
from federatedbatch.data import normalize, read_dataset
from federatedbatch.federated import train_federated


OUT = ROOT / "results" / "final_federatedbatch_figure1"
REPRO = ROOT / "external_artifacts" / "figure1"
SEEDS = (3205, 9048, 7487, 5326, 9364)
N_CLIENTS = 6
ACTIVE_ROUNDS = tuple(range(4, 21))
REMOTE_PER_SOURCE = 102
C = 1206.3391650512762
SIGMA = 2.0
NOISE_SEEDS = {seed: 20000 + seed for seed in SEEDS}
FIGURE1_CONFIG = ROOT / "configs" / "figure1.yaml"
PARTITION = ROOT / "partitions" / "figure1" / "pancreas_moderate_label_skew_6clients.json"
FEATURES = REPRO / "results" / "fresh_queue_operating_window_30seed" / "features_pancreas.txt"
LEGACY_SCRIPT = REPRO / "scripts" / "recompute_legacy_metrics.py"
FIGURE1_SOURCE = REPRO / "results" / "figure1_final_inputs" / "figure1B_source_values.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "training").mkdir(exist_ok=True)
    base = load_config(str(FIGURE1_CONFIG))
    base["data_dir"] = str(ROOT / "data")
    base["output_dir"] = str(OUT)
    adata, data_path = read_dataset("pancreas", base["data_dir"], verify_hash=False)
    features = FEATURES.read_text(encoding="utf-8").splitlines()
    if len(features) != 2000:
        raise ValueError(f"expected 2000 features, got {len(features)}")
    normalize(adata, 10000, True)
    groups, partition_payload = load_partition(adata.n_obs)
    schedule, schedule_frame = build_schedule(groups, np.asarray(adata.obs_names))
    atomic_csv(schedule_frame, OUT / "remote_cell_schedule.csv")
    base_provenance = {
        "dataset": str(data_path),
        "dataset_sha256": sha256(data_path),
        "features": str(FEATURES),
        "features_sha256": sha256(FEATURES),
        "partition": str(PARTITION),
        "partition_sha256": sha256(PARTITION),
        "partition_seed": partition_payload.get("partition_seed", 20260918),
        "training_seeds": list(SEEDS),
        "n_clients": N_CLIENTS,
        "remote_cells_per_source_per_active_round": REMOTE_PER_SOURCE,
        "unique_remote_cells_per_source": len(ACTIVE_ROUNDS) * REMOTE_PER_SOURCE,
        "active_rounds": list(ACTIVE_ROUNDS),
        "joint_clipping_threshold": C,
        "noise_multiplier": SIGMA,
        "tau_q": 0.2,
        "alpha": 0.1,
        "old_metric_definitions": str(LEGACY_SCRIPT),
        "old_figure1_source": str(FIGURE1_SOURCE),
    }
    (OUT / "provenance.json").write_text(json.dumps(base_provenance, indent=2), encoding="utf-8")
    legacy = load_legacy()
    rows = []
    diagnostics = []
    # Resume safely: preserve any completed seed outputs from an interrupted run.
    partial_metrics_path = OUT / "per_seed_metrics_partial.csv"
    diagnostics_path = OUT / "clipping_noise_diagnostics.csv"
    if partial_metrics_path.exists():
        try:
            rows = pd.read_csv(partial_metrics_path).to_dict(orient="records")
        except Exception:
            rows = []
    if diagnostics_path.exists():
        try:
            diagnostics = pd.read_csv(diagnostics_path).to_dict(orient="records")
        except Exception:
            diagnostics = []
    completed_seeds = {int(r["training_seed"]) for r in rows if "training_seed" in r}
    for seed in SEEDS:
        run_dir = OUT / "training" / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        if seed in completed_seeds and (run_dir / "final_embedding.npy").exists():
            print(f"skipping completed Figure 1 final private seed {seed}", flush=True)
            continue
        result = None
        try:
            cfg = configure(base, seed, schedule)
            result = train_federated(
                adata,
                features,
                None,
                cfg,
                run_dir,
                domain_key="batch",
                client_indices=groups,
            )
            embedding = result.embeddings.astype(np.float32)
            np.save(run_dir / "final_embedding.npy", embedding)
            clip = pd.DataFrame(result.joint_clipping_diagnostics)
            noise = pd.DataFrame(result.joint_noise_diagnostics)
            clip.to_csv(run_dir / "clipping_diagnostics.csv", index=False)
            noise.to_csv(run_dir / "noise_diagnostics.csv", index=False)
            if len(clip) != len(ACTIVE_ROUNDS) * N_CLIENTS * REMOTE_PER_SOURCE:
                raise ValueError("unexpected clipping diagnostics row count")
            sources = noise[noise.record_type == "joint_source_release"]
            targets = noise[noise.record_type == "target_postprocess"]
            invalid = int(targets.invalid_a_before_clamp_count.sum()) if len(targets) else 0
            acount = int(targets.a_count.sum()) if len(targets) else 0
            metric = evaluate(legacy, adata, cfg, embedding, f"final_fig1_{seed}")
            row = {
                "method": "FederatedBatch_final_private",
                "training_seed": seed,
                **{k: metric[k] for k in ["ARI", "NMI", "ASW_batch", "ASW_cell", "Graph_connectivity", "iLISI"]},
                "ARI_leiden_sd": metric["ARI_sd"],
                "NMI_leiden_sd": metric["NMI_sd"],
                "clipping_fraction": float(clip.clipped.astype(bool).mean()),
                "invalid_a_frequency": float(invalid / acount) if acount else math.nan,
                "invalid_a_count": invalid,
                "a_count": acount,
                "training_stable": bool(np.isfinite(embedding).all()),
                "max_remote_participation_count": int(schedule_frame.groupby("cell_id").size().max()),
                "embedding_path": str(run_dir / "final_embedding.npy"),
                "embedding_sha256": sha256(run_dir / "final_embedding.npy"),
                "leiden_repeats": metric["leiden_repeats"],
            }
            rows.append(row)
            diagnostics.append({
                "training_seed": seed,
                "clipping_rows": len(clip),
                "joint_source_release_rows": len(sources),
                "target_postprocess_rows": len(targets),
                "clipping_fraction": row["clipping_fraction"],
                "median_clipping_scale": float(clip.clipping_scale.median()),
                "p10_clipping_scale": float(clip.clipping_scale.quantile(0.1)),
                "invalid_a_count": invalid,
                "a_count": acount,
                "invalid_a_frequency": row["invalid_a_frequency"],
                "max_remote_participation_count": row["max_remote_participation_count"],
            })
            atomic_csv(pd.DataFrame(rows), OUT / "per_seed_metrics_partial.csv")
            atomic_csv(pd.DataFrame(diagnostics), OUT / "clipping_noise_diagnostics.csv")
            print(f"completed Figure 1 final private seed {seed}", flush=True)
        except Exception as exc:
            (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise
        finally:
            del result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    metrics = pd.DataFrame(rows).sort_values("training_seed")
    atomic_csv(metrics, OUT / "per_seed_metrics.csv")
    summary_rows = []
    for metric_name in ["ARI", "NMI", "ASW_batch", "ASW_cell", "Graph_connectivity", "iLISI"]:
        values = metrics[metric_name].to_numpy(float)
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        ci = 2.776445105197798 * sd / math.sqrt(len(values))
        summary_rows.append({"method": "FederatedBatch_final_private", "metric": metric_name, "mean": mean, "sd": sd, "n": len(values), "ci95_low": mean - ci, "ci95_high": mean + ci})
    atomic_csv(pd.DataFrame(summary_rows), OUT / "metric_summary.csv")
    old = pd.read_csv(FIGURE1_SOURCE)
    old = old[old.method != "FederatedBatch"].copy()
    replacement = []
    for _, r in metrics.iterrows():
        for metric_name in ["ARI", "NMI", "ASW_batch", "ASW_cell", "Graph_connectivity", "iLISI"]:
            replacement.append({"method": "FederatedBatch", "training_seed": int(r.training_seed), "metric": metric_name, "value": float(r[metric_name]), "source": str(OUT / "per_seed_metrics.csv"), "evaluation_definition": "final RoundOnce joint-cellclip Gaussian release; fixed five training seeds; old Figure 1 metric formula", "value_unit": "method-level value"})
    final_source = pd.concat([old, pd.DataFrame(replacement)], ignore_index=True)
    atomic_csv(final_source, OUT / "figure1B_source_values_final_private.csv")
    (OUT / "FINAL_FIGURE1_REPORT.md").write_text("# Final Figure 1 FederatedBatch rerun\n\nThe five Figure 1 training seeds were rerun with the locked RoundOnce joint-cellclip Gaussian protocol. PCA, Harmony, and centralized CONCORD inputs were preserved from the existing Figure 1 source table. Metrics use the legacy ASW, legacy graph connectivity, and full-pairwise legacy iLISI definitions.\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(
        "This legacy helper is not a Figure 1 entry point; use "
        "experiments/run_figure1_canonical7.py."
    )
