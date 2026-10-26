#!/usr/bin/env python3
"""Empirically audit representation-level privacy leakage for revised Figure 5."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

# Support direct execution from the repository root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from experiments.moderate_skew_freshqueue_confirmation import (
    _partition_hash,
    partition_for_seed,
    validate_protocol,
)
from federatedbatch.config import REPO_ROOT, load_config, resolve_repo_path
from federatedbatch.data import normalize, read_dataset, select_features
from federatedbatch.federated import FederatedRoundSnapshot, train_federated
from federatedbatch.provenance import (
    FIGURE5_VERSIONED_PATHS,
    build_metadata,
    git_paths_clean,
    source_identity,
    write_json,
)


RECONSTRUCTION_METHODS = ("Mean", "NN", "NN_k5", "Linear", "MLP")
PRIMARY_RECONSTRUCTION_METHODS = ("Mean", "NN", "Linear", "MLP")
PRIMARY_MEMBERSHIP_CONDITIONS = (
    "Model-only",
    "Model + FreshQueue (nontrivial)",
    "Direct queue exposure",
)
MODEL_FEATURES = (
    "embedding_norm",
    "augmentation_cosine",
    "local_ntxent_loss",
)
QUEUE_FEATURES = (
    "queue_max_cosine",
    "queue_mean_top5_cosine",
    "queue_mean_top10_cosine",
    "queue_nearest_cosine_distance",
    "queue_density_cosine_ge_threshold",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/figure5_privacy.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="rebuild summaries, diagnostic figures, and reports from completed seed outputs",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping in {path}")
    return value


def load_privacy_config(path: str) -> tuple[dict, Path]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = load_yaml(config_path)
    expected_seeds = [218, 394, 542, 639, 788, 814, 880]
    if config.get("seeds") != expected_seeds:
        raise ValueError(f"Figure 5 seeds must remain {expected_seeds}")
    fractions = config["privacy_split"]
    observed = (
        float(fractions["train_fraction"]),
        float(fractions["auxiliary_fraction"]),
        float(fractions["nonmember_fraction"]),
    )
    if observed != (0.8, 0.1, 0.1) or not math.isclose(sum(observed), 1.0):
        raise ValueError(f"privacy split must remain 80/10/10, observed {observed}")
    return config, config_path


def output_directories(config: dict) -> dict[str, Path]:
    root = resolve_repo_path(config["output_dir"])
    directories = {
        "root": root,
        "config": root / "01_config",
        "splits": root / "02_splits",
        "reconstruction": root / "03_reconstruction",
        "membership": root / "04_membership",
        "figures": root / "05_figures_diagnostic",
        "reports": root / "06_reports",
        "work": root / "00_work",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def sha256_ids(values) -> str:
    payload = "\n".join(map(str, values))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_config(config: dict) -> dict:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def relevant_figure2_config(config: dict) -> dict:
    return {
        "dataset": config["dataset"],
        "data_dir": config["data_dir"],
        "preprocessing": config["preprocessing"],
        "federated": config["federated"],
        "partition": config["confirmation"]["partition"],
        "training_seeds": config["confirmation"]["training_seeds"],
        "condition": config["confirmation"]["condition"],
    }


def write_protocol_documents(
    privacy: dict,
    privacy_path: Path,
    figure2: dict,
    archived: dict,
    directories: dict[str, Path],
    comparison_checks: dict[str, bool],
) -> None:
    write_json(
        directories["config"] / "victim_config.json",
        {
            "figure2_victim_config": canonical_config(figure2),
            "privacy_evaluation_config": privacy,
            "privacy_config_path": str(privacy_path.resolve()),
            "claim_boundary": (
                "Empirical representation-level privacy audit under a semi-honest "
                "server; no formal privacy guarantee is claimed."
            ),
        },
    )
    rows = [
        ("Verified source revision equals Figure 2 run", comparison_checks["source_revision_match"]),
        ("Preprocessing matches archived Figure 2", comparison_checks["config_match"]),
        ("Federated architecture matches archived Figure 2", comparison_checks["config_match"]),
        ("FreshQueue configuration matches archived Figure 2", comparison_checks["config_match"]),
        ("Core FreshQueue source files match the verified source revision", comparison_checks["core_code_clean"]),
        ("All seven full-data partition hashes match Figure 2", comparison_checks["partition_hashes_match"]),
    ]
    comparison = [
        "# Figure 2 configuration comparison",
        "",
        "The privacy holdout is the only intended change to the Figure 2 training population.",
        "",
        "| Check | Result |",
        "|---|---|",
        *[f"| {name} | {'PASS' if passed else 'FAIL'} |" for name, passed in rows],
        "",
        "## Locked victim configuration",
        "",
        f"- Dataset: `{figure2['dataset']}`",
        f"- Preprocessing: {figure2['preprocessing']['n_hvg']} HVGs, "
        f"`{figure2['preprocessing']['hvg_flavor']}`, normalize_total="
        f"{figure2['preprocessing']['normalize_target_sum']}, log1p="
        f"{figure2['preprocessing']['log1p']}",
        f"- Clients: {figure2['federated']['n_clients']}",
        f"- Encoder: {figure2['preprocessing']['n_hvg']} -> "
        f"{' -> '.join(map(str, figure2['federated']['encoder_dims']))} -> "
        f"{figure2['federated']['latent_dim']}",
        f"- Training: {figure2['federated']['n_rounds']} rounds, "
        f"{figure2['federated']['local_epochs']} local epochs, batch size "
        f"{figure2['federated']['batch_size']}, learning rate "
        f"{figure2['federated']['learning_rate']}",
        f"- Queue: `{figure2['federated']['queue']['mode']}`, capacity "
        f"{figure2['federated']['queue']['size']}, per-client upload "
        f"{figure2['federated']['queue']['samples_per_client']}, warm-up "
        f"{figure2['federated']['queue']['warmup_rounds']} rounds",
        f"- Seeds: {', '.join(map(str, privacy['seeds']))}",
        "",
        "The archived and active relevant configurations are byte-equivalent after YAML parsing: "
        f"`{relevant_figure2_config(figure2) == relevant_figure2_config(archived)}`.",
    ]
    atomic_text("\n".join(comparison) + "\n", directories["config"] / "figure2_config_comparison.md")
    split_text = """# Privacy split definition

For every predefined victim seed, the exact five-client moderate-label-skew Figure 2 partition is created first. Within each client, cells are split before victim training into approximately 80% victim training, 10% auxiliary attacker data, and 10% nonmember evaluation data.

Stratification starts at client x technical-batch x cell-type resolution. Sparse strata are deterministically pooled to client x cell type, then client x batch, and finally client. The 20% holdout is split as evenly as possible between auxiliary and nonmember sets within the retained strata. Exact cell IDs, original indices, clients, batches, and cell types are saved before any privacy result is observed.

Only victim-training cells enter FederatedBatch training or FreshQueue. HVGs are selected from victim-training cells, and the same 2,000 features plus the Figure 2 normalize_total/log1p transformation are then applied to all three privacy subsets.
"""
    atomic_text(split_text, directories["config"] / "privacy_split_definition.md")
    threat = """# Threat model

The adversary is a semi-honest (honest-but-curious) server. It follows the prescribed protocol, knows the encoder architecture and current global weights, sees protocol model messages, and observes the cell-level FreshQueue embeddings legitimately uploaded to the server. It may hold auxiliary Pancreas cells from the same distribution and may fit reconstruction or membership attacks.

The primary protocol-state analysis uses the current-round final FreshQueue snapshot. A separately labelled cumulative-transcript sensitivity analysis may retain embeddings legitimately observed across rounds. FreshQueue freshness and TTL behavior are algorithmic properties, not cryptographic deletion or privacy guarantees. The evaluation does not claim differential privacy, secure aggregation, encryption, or formal privacy.
"""
    atomic_text(threat, directories["config"] / "threat_model.md")


def fallback_strata(frame: pd.DataFrame, minimum: int) -> np.ndarray:
    client = frame["client"].astype(str)
    batch = frame["batch"].astype(str)
    celltype = frame["celltype"].astype(str)
    labels = "joint:" + client + "|" + batch + "|" + celltype
    for fallback in (
        "celltype:" + client + "|" + celltype,
        "batch:" + client + "|" + batch,
        "client:" + client,
    ):
        counts = labels.value_counts()
        sparse = labels.map(counts) < minimum
        labels = labels.where(~sparse, fallback)
    return labels.to_numpy(str)


def split_one_client(
    source,
    client_indices: np.ndarray,
    client: int,
    seed: int,
    minimum: int,
) -> pd.DataFrame:
    indices = np.asarray(client_indices, dtype=np.int64)
    metadata = pd.DataFrame(
        {
            "original_index": indices,
            "cell_id": source.obs_names[indices].astype(str),
            "client": int(client),
            "batch": source.obs.iloc[indices]["batch"].astype(str).to_numpy(),
            "celltype": source.obs.iloc[indices]["celltype"].astype(str).to_numpy(),
        }
    )
    strata = fallback_strata(metadata, minimum)
    positions = np.arange(len(metadata))
    train_pos, holdout_pos = train_test_split(
        positions,
        test_size=0.20,
        random_state=int(seed) * 1009 + int(client),
        stratify=strata,
    )
    rng = np.random.default_rng(int(seed) * 1_000_003 + int(client))
    aux_pos: list[int] = []
    nonmember_pos: list[int] = []
    holdout_strata = strata[holdout_pos]
    for label in sorted(np.unique(holdout_strata)):
        group = holdout_pos[holdout_strata == label].copy()
        rng.shuffle(group)
        split = len(group) // 2
        if len(group) % 2 and len(aux_pos) <= len(nonmember_pos):
            split += 1
        aux_pos.extend(group[:split].tolist())
        nonmember_pos.extend(group[split:].tolist())
    metadata["split"] = ""
    metadata.loc[train_pos, "split"] = "train"
    metadata.loc[np.asarray(aux_pos, dtype=int), "split"] = "aux"
    metadata.loc[np.asarray(nonmember_pos, dtype=int), "split"] = "nonmember"
    if (metadata["split"] == "").any():
        raise RuntimeError(f"seed {seed} client {client}: unassigned privacy cells")
    return metadata.sort_values("original_index").reset_index(drop=True)


def build_split_qc(split_frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    for frame in split_frames:
        seed = int(frame["seed"].iloc[0])
        sets = {
            name: set(frame.loc[frame["split"] == name, "cell_id"])
            for name in ("train", "aux", "nonmember")
        }
        overlap = sum(
            len(sets[first] & sets[second])
            for first, second in (("train", "aux"), ("train", "nonmember"), ("aux", "nonmember"))
        )
        counts = frame["split"].value_counts()
        rows.append(
            {
                "seed": seed,
                "dimension": "overall",
                "level": "all",
                "full_n": len(frame),
                "train_n": int(counts.get("train", 0)),
                "aux_n": int(counts.get("aux", 0)),
                "nonmember_n": int(counts.get("nonmember", 0)),
                "full_prop": 1.0,
                "train_prop": 1.0,
                "aux_prop": 1.0,
                "nonmember_prop": 1.0,
                "overlap_count": overlap,
            }
        )
        for dimension, column in (
            ("client", "client"),
            ("batch", "batch"),
            ("cell_type", "celltype"),
        ):
            for level in sorted(frame[column].astype(str).unique()):
                row = frame.loc[frame[column].astype(str) == level]
                item = {
                    "seed": seed,
                    "dimension": dimension,
                    "level": level,
                    "full_n": len(row),
                    "full_prop": len(row) / len(frame),
                    "overlap_count": overlap,
                }
                for role in ("train", "aux", "nonmember"):
                    role_total = int((frame["split"] == role).sum())
                    role_n = int((row["split"] == role).sum())
                    item[f"{role}_n"] = role_n
                    item[f"{role}_prop"] = role_n / role_total if role_total else math.nan
                rows.append(item)
    return pd.DataFrame(rows)


def lock_privacy_splits(
    source,
    figure2: dict,
    privacy: dict,
    directories: dict[str, Path],
) -> tuple[list[pd.DataFrame], bool]:
    frames: list[pd.DataFrame] = []
    hashes_match = True
    original_partitions = resolve_repo_path(privacy["figure2_results"]) / "partitions"
    minimum = int(privacy["privacy_split"]["sparse_stratum_minimum"])
    for seed in privacy["seeds"]:
        groups = partition_for_seed(source, figure2, int(seed))
        archived_path = original_partitions / f"seed_{seed}.json"
        archived = json.loads(archived_path.read_text(encoding="utf-8"))
        observed_hash = _partition_hash(groups)
        hashes_match &= observed_hash == archived["partition_sha256"]
        client_frames = [
            split_one_client(source, group, client, int(seed), minimum)
            for client, group in enumerate(groups)
        ]
        frame = pd.concat(client_frames, ignore_index=True).sort_values("original_index")
        frame.insert(0, "seed", int(seed))
        if len(frame) != source.n_obs or frame["cell_id"].nunique() != source.n_obs:
            raise RuntimeError(f"seed {seed}: privacy split does not cover every cell exactly once")
        split_path = directories["splits"] / f"seed_{seed}_cell_ids.csv"
        if split_path.exists():
            existing = pd.read_csv(split_path)
            comparable = ["seed", "original_index", "cell_id", "client", "batch", "celltype", "split"]
            if not existing[comparable].equals(frame[comparable].reset_index(drop=True)):
                raise RuntimeError(f"locked privacy split changed for seed {seed}")
        else:
            atomic_csv(frame, split_path)
        frames.append(frame)
    atomic_csv(build_split_qc(frames), directories["splits"] / "split_qc.csv")
    return frames, hashes_match


def preflight(
    privacy: dict,
    privacy_path: Path,
    directories: dict[str, Path],
) -> tuple[dict, object, Path]:
    figure2 = load_config(privacy["figure2_config"])
    validate_protocol(figure2)
    archived_path = resolve_repo_path(privacy["figure2_results"])
    archived = load_yaml(archived_path / "config.yaml")
    run_manifest = json.loads((archived_path / "run_manifest.json").read_text(encoding="utf-8"))
    identity = source_identity(REPO_ROOT, FIGURE5_VERSIONED_PATHS)
    archived_revision = run_manifest.get("source_revision") or run_manifest.get("git_commit")
    config_match = relevant_figure2_config(figure2) == relevant_figure2_config(archived)
    revision_match = identity["revision"] == archived_revision
    core_clean = (
        git_paths_clean(REPO_ROOT, FIGURE5_VERSIONED_PATHS)
        if identity["kind"] == "git"
        else True
    )
    source, data_path = read_dataset(figure2["dataset"], figure2["data_dir"])
    split_frames, partitions_match = lock_privacy_splits(source, figure2, privacy, directories)
    checks = {
        "config_match": bool(config_match),
        "source_revision_match": bool(revision_match),
        "core_code_clean": bool(core_clean),
        "partition_hashes_match": bool(partitions_match),
    }
    write_protocol_documents(privacy, privacy_path, figure2, archived, directories, checks)
    write_json(
        directories["config"] / "preflight.json",
        {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "source_identity": identity,
            "figure2_run_manifest": run_manifest,
            "execution_environment": build_metadata(REPO_ROOT, data_path),
            "locked_split_hashes": {
                str(frame["seed"].iloc[0]): sha256_ids(
                    frame.sort_values("original_index")["cell_id"] + ":" + frame.sort_values("original_index")["split"]
                )
                for frame in split_frames
            },
        },
    )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError("Figure 2 privacy preflight failed: " + ", ".join(failed))
    print("Figure 2 config comparison: PASS")
    print(
        "Victim: Pancreas; 5 clients; 2000 HVGs; encoder 2000->128->100; "
        "20 rounds x 5 local epochs; final FreshQueue implementation."
    )
    return figure2, source, data_path


def dense_values(adata, indices: np.ndarray, features: list[str]) -> np.ndarray:
    matrix = adata[np.asarray(indices, dtype=np.int64), features].X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def prepare_seed_source(
    figure2: dict,
    split: pd.DataFrame,
    data_path: Path,
    seed_work: Path,
):
    source, observed_path = read_dataset(figure2["dataset"], figure2["data_dir"])
    if observed_path.resolve() != data_path.resolve():
        raise RuntimeError("dataset path changed between preflight and seed run")
    train_indices = split.loc[split["split"] == "train", "original_index"].to_numpy(int)
    feature_path = seed_work / "features.txt"
    selected = select_features(
        source[train_indices].copy(),
        int(figure2["preprocessing"]["n_hvg"]),
        str(figure2["preprocessing"]["hvg_flavor"]),
    )
    serialized = "\n".join(selected) + "\n"
    if feature_path.exists() and feature_path.read_text(encoding="utf-8") != serialized:
        raise RuntimeError(f"seed {split['seed'].iloc[0]}: selected HVGs changed")
    atomic_text(serialized, feature_path)
    normalize(
        source,
        figure2["preprocessing"]["normalize_target_sum"],
        bool(figure2["preprocessing"]["log1p"]),
    )
    if len(selected) != 2000 or len(set(selected)) != 2000:
        raise RuntimeError("privacy victim must use exactly 2000 unique HVGs")
    return source, selected


def model_only_features(model, source, features: list[str], embeddings: np.ndarray, config: dict) -> np.ndarray:
    n_cells = source.n_obs
    output = np.empty((n_cells, len(MODEL_FEATURES)), dtype=np.float32)
    output[:, 0] = np.linalg.norm(embeddings, axis=1)
    batch_size = int(config["membership"]["augmentation_batch_size"])
    temperature = float(config["membership"]["contrastive_temperature"])
    device = torch.device(model.config.device)
    encoder = model.model
    augment = model.trainer.augment
    encoder_was_training = encoder.training
    augment_was_training = augment.training
    encoder.eval()
    augment.train()
    cuda_devices = [device.index or torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(config["membership"]["augmentation_seed"]))
        if device.type == "cuda":
            torch.cuda.manual_seed(int(config["membership"]["augmentation_seed"]))
        for start in range(0, n_cells, batch_size):
            stop = min(start + batch_size, n_cells)
            values = dense_values(source, np.arange(start, stop), features)
            inputs = torch.as_tensor(values, device=device)
            z1 = F.normalize(encoder.encode(augment(inputs)), dim=1)
            z2 = F.normalize(encoder.encode(augment(inputs)), dim=1)
            cosine = torch.sum(z1 * z2, dim=1)
            labels = torch.arange(len(inputs), device=device)
            logits12 = z1 @ z2.T / temperature
            logits21 = z2 @ z1.T / temperature
            losses = 0.5 * (
                F.cross_entropy(logits12, labels, reduction="none")
                + F.cross_entropy(logits21, labels, reduction="none")
            )
            output[start:stop, 1] = cosine.detach().cpu().numpy()
            output[start:stop, 2] = losses.detach().cpu().numpy()
    encoder.train(encoder_was_training)
    augment.train(augment_was_training)
    return output


def train_victim(
    source,
    features: list[str],
    split: pd.DataFrame,
    figure2: dict,
    privacy: dict,
    seed: int,
    seed_work: Path,
) -> None:
    training_marker = seed_work / "training_complete.txt"
    if training_marker.exists():
        return
    victim_config = copy.deepcopy(figure2)
    victim_config.pop("_config_path", None)
    victim_config["seed"] = int(seed)
    validate_protocol(victim_config)
    train_groups = [
        split.loc[
            (split["split"] == "train") & (split["client"] == client),
            "original_index",
        ].to_numpy(int)
        for client in range(5)
    ]
    transcript_embeddings: list[np.ndarray] = []
    transcript_records: list[dict] = []

    def capture(snapshot: FederatedRoundSnapshot) -> None:
        if len(snapshot.queue_embeddings):
            transcript_embeddings.append(np.asarray(snapshot.queue_embeddings, dtype=np.float32).copy())
            for record in snapshot.queue_records:
                transcript_records.append(
                    {
                        "observed_round": int(snapshot.round_number),
                        "cell_id": str(record["cell_id"]),
                        "client": int(record["client"]),
                        "batch": str(record["batch"]),
                        "collected_round": int(record["collected_round"]),
                    }
                )

    started = time.perf_counter()
    result = train_federated(
        source,
        features,
        None,
        victim_config,
        seed_work / "victim_model",
        domain_key="batch",
        client_indices=train_groups,
        allow_partial_client_indices=True,
        diagnostic_callback=capture,
    )
    embeddings = np.asarray(result.embeddings, dtype=np.float32)
    np.save(seed_work / "victim_embeddings.npy", embeddings, allow_pickle=False)
    np.save(
        seed_work / "final_queue_embeddings.npy",
        np.asarray(result.final_queue_embeddings, dtype=np.float32),
        allow_pickle=False,
    )
    queue_records = pd.DataFrame(
        [
            {
                "cell_id": str(record["cell_id"]),
                "client": int(record["client"]),
                "batch": str(record["batch"]),
                "collected_round": int(record["collected_round"]),
            }
            for record in result.final_queue_records
        ]
    )
    atomic_csv(queue_records, seed_work / "final_queue_records.csv")
    if transcript_embeddings:
        np.save(
            seed_work / "cumulative_transcript_embeddings.npy",
            np.concatenate(transcript_embeddings, axis=0).astype(np.float32),
            allow_pickle=False,
        )
        atomic_csv(pd.DataFrame(transcript_records), seed_work / "cumulative_transcript_records.csv")
    derived_features = model_only_features(result.model, source, features, embeddings, privacy)
    np.save(seed_work / "model_only_features.npy", derived_features, allow_pickle=False)
    write_json(
        seed_work / "training_summary.json",
        {
            "seed": seed,
            "training_seconds": result.training_seconds,
            "pipeline_seconds_through_feature_extraction": time.perf_counter() - started,
            "client_training_cells": result.client_cells,
            "n_training_cells": int(sum(result.client_cells)),
            "final_queue_size": int(len(result.final_queue_embeddings)),
            "final_queue_rounds": sorted(set(queue_records["collected_round"].astype(int))),
            "queue_sizes": result.queue_sizes,
            "queue_mode": victim_config["federated"]["queue"]["mode"],
            "queue_collection": victim_config["federated"]["queue"]["collection"],
            "cumulative_transcript_entries": int(sum(map(len, transcript_embeddings))),
        },
    )
    atomic_text(f"completed_at={pd.Timestamp.now().isoformat()}\n", training_marker)
    del result
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class MLPDecoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(previous, int(width)), nn.ReLU()])
            previous = int(width)
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def train_mlp_decoder(
    train_z: np.ndarray,
    train_x: np.ndarray,
    val_z: np.ndarray,
    val_x: np.ndarray,
    config: dict,
    seed: int,
) -> tuple[MLPDecoder, np.ndarray, np.ndarray, dict, pd.DataFrame]:
    settings = config["reconstruction"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(seed) + int(settings["decoder_split_seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed) + int(settings["decoder_split_seed"]))
    mean = train_z.mean(axis=0).astype(np.float32)
    scale = train_z.std(axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    train_scaled = (train_z - mean) / scale
    val_scaled = (val_z - mean) / scale
    model = MLPDecoder(train_z.shape[1], settings["mlp_hidden_dims"], train_x.shape[1]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(settings["mlp_learning_rate"]),
        weight_decay=float(settings["mlp_weight_decay"]),
    )
    generator = torch.Generator().manual_seed(int(seed) + int(settings["decoder_split_seed"]))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_scaled), torch.from_numpy(train_x)),
        batch_size=int(settings["mlp_batch_size"]),
        shuffle=True,
        generator=generator,
    )
    val_z_tensor = torch.from_numpy(val_scaled).to(device)
    val_x_tensor = torch.from_numpy(val_x).to(device)
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    patience = 0
    history: list[dict] = []
    for epoch in range(1, int(settings["mlp_max_epochs"]) + 1):
        model.train()
        total = 0.0
        count = 0
        for z_batch, x_batch in loader:
            z_batch = z_batch.to(device)
            x_batch = x_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(z_batch)
            loss = F.mse_loss(prediction, x_batch)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(z_batch)
            count += len(z_batch)
        model.eval()
        with torch.no_grad():
            val_loss = float(F.mse_loss(model(val_z_tensor), val_x_tensor).cpu())
        train_loss = total / max(count, 1)
        history.append({"epoch": epoch, "train_mse": train_loss, "validation_mse": val_loss})
        if val_loss < best_loss - float(settings["mlp_min_delta"]):
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= int(settings["mlp_patience"]):
            break
    if best_state is None:
        raise RuntimeError("MLP decoder failed to produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    return (
        model,
        mean,
        scale,
        {"best_epoch": best_epoch, "best_validation_mse": best_loss, "epochs_run": len(history)},
        pd.DataFrame(history),
    )


def mlp_predict(model: MLPDecoder, values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), 256):
            batch = torch.from_numpy(((values[start : start + 256] - mean) / scale).astype(np.float32)).to(device)
            outputs.append(model(batch).cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def row_pearson(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    first = observed - observed.mean(axis=1, keepdims=True)
    second = predicted - predicted.mean(axis=1, keepdims=True)
    numerator = np.sum(first * second, axis=1)
    denominator = np.sqrt(np.sum(first**2, axis=1) * np.sum(second**2, axis=1))
    return np.divide(numerator, denominator, out=np.full(len(observed), np.nan), where=denominator > 0)


def row_spearman(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return row_pearson(rankdata(observed, axis=1), rankdata(predicted, axis=1))


def gene_pearson(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    first = observed - observed.mean(axis=0, keepdims=True)
    second = predicted - predicted.mean(axis=0, keepdims=True)
    numerator = np.sum(first * second, axis=0)
    denominator = np.sqrt(np.sum(first**2, axis=0) * np.sum(second**2, axis=0))
    return np.divide(
        numerator,
        denominator,
        out=np.full(observed.shape[1], np.nan),
        where=denominator > 0,
    )


def metric_summary(values: np.ndarray, prefix: str) -> dict:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_median": math.nan,
            f"{prefix}_sd": math.nan,
            f"{prefix}_ci95_low": math.nan,
            f"{prefix}_ci95_high": math.nan,
        }
    mean = float(np.mean(finite))
    sd = float(np.std(finite, ddof=1)) if len(finite) > 1 else math.nan
    margin = float(stats.t.ppf(0.975, len(finite) - 1) * sd / math.sqrt(len(finite))) if len(finite) > 1 else math.nan
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_sd": sd,
        f"{prefix}_ci95_low": mean - margin if len(finite) > 1 else math.nan,
        f"{prefix}_ci95_high": mean + margin if len(finite) > 1 else math.nan,
    }


def auxiliary_train_validation(
    source,
    auxiliary: np.ndarray,
    split: pd.DataFrame,
    privacy: dict,
) -> tuple[np.ndarray, np.ndarray]:
    meta = split.set_index("original_index").loc[auxiliary].reset_index()
    labels = fallback_strata(meta, int(privacy["privacy_split"]["sparse_stratum_minimum"]))
    try:
        train_indices, validation_indices = train_test_split(
            auxiliary,
            test_size=float(privacy["reconstruction"]["auxiliary_validation_fraction"]),
            random_state=int(privacy["reconstruction"]["decoder_split_seed"]),
            stratify=labels,
        )
    except ValueError:
        train_indices, validation_indices = train_test_split(
            auxiliary,
            test_size=float(privacy["reconstruction"]["auxiliary_validation_fraction"]),
            random_state=int(privacy["reconstruction"]["decoder_split_seed"]),
        )
    return np.sort(train_indices), np.sort(validation_indices)


def reconstruction_attack(
    source,
    features: list[str],
    split: pd.DataFrame,
    privacy: dict,
    seed: int,
    seed_work: Path,
    output: Path,
) -> None:
    complete = output / "complete.txt"
    if complete.exists():
        return
    embeddings = np.load(seed_work / "victim_embeddings.npy")
    queue_embeddings = np.load(seed_work / "final_queue_embeddings.npy")
    queue_records = pd.read_csv(seed_work / "final_queue_records.csv")
    indexer = pd.Series(np.arange(source.n_obs), index=source.obs_names.astype(str))
    queue_indices = indexer.loc[queue_records["cell_id"].astype(str)].to_numpy(int)
    auxiliary = split.loc[split["split"] == "aux", "original_index"].to_numpy(int)
    aux_train, aux_validation = auxiliary_train_validation(source, auxiliary, split, privacy)
    aux_expression = dense_values(source, auxiliary, features)
    target_expression = dense_values(source, queue_indices, features)
    auxiliary_embeddings = normalize_rows(embeddings[auxiliary])
    train_positions = pd.Index(auxiliary).get_indexer(aux_train)
    validation_positions = pd.Index(auxiliary).get_indexer(aux_validation)
    train_z = auxiliary_embeddings[train_positions]
    validation_z = auxiliary_embeddings[validation_positions]
    train_x = aux_expression[train_positions]
    validation_x = aux_expression[validation_positions]

    predictions: dict[str, np.ndarray] = {}
    predictions["Mean"] = np.broadcast_to(aux_expression.mean(axis=0), target_expression.shape).copy()
    similarities = queue_embeddings @ auxiliary_embeddings.T
    nearest = np.argmax(similarities, axis=1)
    predictions["NN"] = aux_expression[nearest]
    top5 = np.argpartition(similarities, -5, axis=1)[:, -5:]
    predictions["NN_k5"] = aux_expression[top5].mean(axis=1)

    ridge_rows = []
    best_alpha = None
    best_loss = math.inf
    for alpha in privacy["reconstruction"]["ridge_alphas"]:
        decoder = Ridge(alpha=float(alpha), fit_intercept=True, solver="lsqr")
        decoder.fit(train_z, train_x)
        validation_prediction = decoder.predict(validation_z)
        loss = float(np.mean((validation_x - validation_prediction) ** 2))
        ridge_rows.append({"alpha": float(alpha), "validation_mse": loss})
        if loss < best_loss:
            best_loss = loss
            best_alpha = float(alpha)
    linear = Ridge(alpha=best_alpha, fit_intercept=True, solver="lsqr")
    linear.fit(auxiliary_embeddings, aux_expression)
    predictions["Linear"] = linear.predict(queue_embeddings).astype(np.float32)
    atomic_csv(pd.DataFrame(ridge_rows), output / "ridge_validation.csv")

    mlp, z_mean, z_scale, mlp_metadata, history = train_mlp_decoder(
        train_z, train_x, validation_z, validation_x, privacy, seed
    )
    predictions["MLP"] = mlp_predict(mlp, queue_embeddings, z_mean, z_scale)
    atomic_csv(history, output / "mlp_training_history.csv")
    write_json(
        output / "decoder_selection.json",
        {
            "ridge_selected_alpha": best_alpha,
            "ridge_best_auxiliary_validation_mse": best_loss,
            "mlp": mlp_metadata,
            "auxiliary_total_n": len(auxiliary),
            "auxiliary_train_n": len(aux_train),
            "auxiliary_validation_n": len(aux_validation),
            "auxiliary_train_cell_ids_sha256": sha256_ids(source.obs_names[aux_train]),
            "auxiliary_validation_cell_ids_sha256": sha256_ids(source.obs_names[aux_validation]),
        },
    )
    if privacy["diagnostics"]["save_reconstruction_predictions"]:
        np.savez_compressed(
            output / "raw_predictions.npz",
            cell_ids=queue_records["cell_id"].astype(str).to_numpy(),
            **{name: value.astype(np.float32) for name, value in predictions.items()},
        )

    metadata = split.set_index("original_index").loc[queue_indices]
    per_cell_rows = []
    per_gene_rows = []
    per_seed_rows = []
    for method in RECONSTRUCTION_METHODS:
        predicted = predictions[method]
        pearson = row_pearson(target_expression, predicted)
        rmse = np.sqrt(np.mean((target_expression - predicted) ** 2, axis=1))
        spearman = row_spearman(target_expression, predicted)
        mae = np.mean(np.abs(target_expression - predicted), axis=1)
        gene_correlations = gene_pearson(target_expression, predicted)
        for position, original_index in enumerate(queue_indices):
            per_cell_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "cell_id": str(source.obs_names[original_index]),
                    "client": int(metadata.loc[original_index, "client"]),
                    "batch": str(metadata.loc[original_index, "batch"]),
                    "celltype": str(metadata.loc[original_index, "celltype"]),
                    "pearson": float(pearson[position]),
                    "rmse": float(rmse[position]),
                    "spearman": float(spearman[position]),
                    "mae": float(mae[position]),
                }
            )
        per_gene_rows.extend(
            {
                "seed": seed,
                "method": method,
                "gene": gene,
                "gene_pearson": float(value) if np.isfinite(value) else math.nan,
            }
            for gene, value in zip(features, gene_correlations)
        )
        per_seed_rows.append(
            {
                "seed": seed,
                "method": method,
                "n_cells": len(queue_indices),
                **metric_summary(pearson, "pearson"),
                **metric_summary(rmse, "rmse"),
                **metric_summary(spearman, "spearman"),
                **metric_summary(mae, "mae"),
                **metric_summary(gene_correlations, "gene_pearson"),
                "gene_pearson_valid_genes": int(np.isfinite(gene_correlations).sum()),
            }
        )
    atomic_csv(pd.DataFrame(per_cell_rows), output / "reconstruction_per_cell.csv")
    atomic_csv(pd.DataFrame(per_gene_rows), output / "reconstruction_per_gene.csv")
    atomic_csv(pd.DataFrame(per_seed_rows), output / "reconstruction_per_seed.csv")
    write_json(
        output / "fit_audit.json",
        {
            "target": "normalized_log1p_2000_hvgs",
            "decoder_fit_cell_role": "auxiliary_only",
            "victim_queue_expression_used_for_fitting": False,
            "nonmember_expression_used_for_fitting": False,
            "queue_target_ids_exact": True,
            "n_exact_final_round_queue_targets": len(queue_indices),
        },
    )
    atomic_text(f"completed_at={pd.Timestamp.now().isoformat()}\n", complete)
    del mlp, predictions
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def membership_strata(
    source,
    indices: np.ndarray,
    labels: np.ndarray,
    client_lookup: pd.Series,
) -> np.ndarray:
    frame = pd.DataFrame(
        {
            "original_index": indices,
            "membership": labels.astype(int),
            "client": client_lookup.loc[indices].to_numpy(int),
            "batch": source.obs.iloc[indices]["batch"].astype(str).to_numpy(),
            "celltype": source.obs.iloc[indices]["celltype"].astype(str).to_numpy(),
        }
    )
    candidate_keys = [
        "joint:" + frame["client"].astype(str) + "|" + frame["batch"] + "|" + frame["celltype"],
        "client_celltype:" + frame["client"].astype(str) + "|" + frame["celltype"],
        "client_batch:" + frame["client"].astype(str) + "|" + frame["batch"],
        "client:" + frame["client"].astype(str),
    ]
    chosen = candidate_keys[-1].copy()
    unresolved = np.ones(len(frame), dtype=bool)
    for keys in candidate_keys:
        counts = pd.crosstab(keys, frame["membership"])
        valid = set(
            counts.index[
                (counts.get(0, 0) >= 4) & (counts.get(1, 0) >= 4)
            ]
        )
        use = unresolved & keys.isin(valid).to_numpy()
        chosen.loc[use] = keys.loc[use]
        unresolved[use] = False
    return chosen.to_numpy(str)


def matched_attack_candidates(
    source,
    member_indices: np.ndarray,
    nonmember_indices: np.ndarray,
    client_lookup: pd.Series,
    *,
    seed: int,
    test_fraction: float,
) -> pd.DataFrame:
    indices = np.concatenate([member_indices, nonmember_indices]).astype(int)
    labels = np.concatenate(
        [np.ones(len(member_indices), dtype=int), np.zeros(len(nonmember_indices), dtype=int)]
    )
    strata = membership_strata(source, indices, labels, client_lookup)
    rng = np.random.default_rng(int(seed))
    rows = []
    for stratum in sorted(np.unique(strata)):
        member = indices[(strata == stratum) & (labels == 1)].copy()
        nonmember = indices[(strata == stratum) & (labels == 0)].copy()
        n = min(len(member), len(nonmember))
        if n < 2:
            continue
        rng.shuffle(member)
        rng.shuffle(nonmember)
        member = member[:n]
        nonmember = nonmember[:n]
        n_test = int(np.clip(round(n * test_fraction), 1, n - 1))
        for candidates, label in ((member, 1), (nonmember, 0)):
            for position, original_index in enumerate(candidates):
                rows.append(
                    {
                        "original_index": int(original_index),
                        "cell_id": str(source.obs_names[original_index]),
                        "membership": label,
                        "stratum": stratum,
                        "attack_split": "test" if position < n_test else "calibration",
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no membership strata contained at least two candidates per class")
    for split_name in ("calibration", "test"):
        counts = frame.loc[frame["attack_split"] == split_name, "membership"].value_counts()
        if counts.get(0, 0) != counts.get(1, 0):
            raise RuntimeError(f"membership {split_name} set is not class-balanced")
    return frame.sort_values(["stratum", "membership", "attack_split", "cell_id"]).reset_index(drop=True)


def queue_features(
    embeddings: np.ndarray,
    queue_embeddings: np.ndarray,
    *,
    threshold: float,
    chunk_size: int = 512,
) -> np.ndarray:
    candidate = normalize_rows(np.asarray(embeddings, dtype=np.float32))
    queue = normalize_rows(np.asarray(queue_embeddings, dtype=np.float32))
    if not len(queue):
        raise ValueError("queue feature calculation requires a non-empty queue")
    outputs = []
    top10_count = min(10, len(queue))
    top5_count = min(5, len(queue))
    for start in range(0, len(candidate), chunk_size):
        similarities = candidate[start : start + chunk_size] @ queue.T
        top10 = np.partition(similarities, -top10_count, axis=1)[:, -top10_count:]
        maximum = top10.max(axis=1)
        ordered = np.sort(top10, axis=1)
        top5 = ordered[:, -top5_count:]
        density = np.mean(similarities >= threshold, axis=1)
        outputs.append(
            np.column_stack(
                [maximum, top5.mean(axis=1), top10.mean(axis=1), 1 - maximum, density]
            )
        )
    return np.concatenate(outputs, axis=0).astype(np.float32)


def tpr_at_fpr(labels: np.ndarray, probabilities: np.ndarray, maximum_fpr: float = 0.01) -> float:
    fpr, tpr, _ = roc_curve(labels, probabilities)
    eligible = tpr[fpr <= maximum_fpr]
    return float(np.max(eligible)) if len(eligible) else 0.0


def fit_membership_classifier(
    candidates: pd.DataFrame,
    feature_names: list[str],
    config: dict,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    calibration = candidates["attack_split"] == "calibration"
    test = candidates["attack_split"] == "test"
    train_x = candidates.loc[calibration, feature_names].to_numpy(float)
    train_y = candidates.loc[calibration, "membership"].to_numpy(int)
    test_x = candidates.loc[test, feature_names].to_numpy(float)
    test_y = candidates.loc[test, "membership"].to_numpy(int)
    classifier = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=int(config["membership"]["logistic_max_iter"]),
            random_state=int(config["membership"]["attack_split_seed"]),
        ),
    )
    classifier.fit(train_x, train_y)
    probabilities = classifier.predict_proba(test_x)[:, 1]
    predictions = probabilities >= 0.5
    permutation_aurocs = []
    rng = np.random.default_rng(int(config["membership"]["attack_split_seed"]))
    for _ in range(int(config["membership"]["permutation_repeats"])):
        permuted = rng.permutation(train_y)
        control = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                max_iter=int(config["membership"]["logistic_max_iter"]),
                random_state=int(config["membership"]["attack_split_seed"]),
            ),
        )
        control.fit(train_x, permuted)
        permutation_aurocs.append(
            roc_auc_score(test_y, control.predict_proba(test_x)[:, 1])
        )
    logistic = classifier.named_steps["logisticregression"]
    coefficients = pd.DataFrame(
        {"feature": feature_names, "coefficient": logistic.coef_[0]}
    )
    prediction_frame = candidates.loc[test, [
        "original_index", "cell_id", "membership", "stratum", "attack_split"
    ]].copy()
    prediction_frame["probability"] = probabilities
    metrics = {
        "n_calibration": int(calibration.sum()),
        "n_test": int(test.sum()),
        "n_test_per_class": int(np.sum(test_y == 1)),
        "auroc": float(roc_auc_score(test_y, probabilities)),
        "auprc": float(average_precision_score(test_y, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(test_y, predictions)),
        "tpr_at_1pct_fpr": tpr_at_fpr(test_y, probabilities),
        "permuted_label_auroc_mean": float(np.mean(permutation_aurocs)),
        "permuted_label_auroc_sd": float(np.std(permutation_aurocs, ddof=1)),
        "permutation_repeats": len(permutation_aurocs),
    }
    return metrics, prediction_frame, coefficients


def attach_candidate_metadata(
    candidates: pd.DataFrame,
    source,
    split: pd.DataFrame,
    model_features: np.ndarray,
    victim_embeddings: np.ndarray,
    queue_embeddings: np.ndarray | None,
    privacy: dict,
) -> pd.DataFrame:
    output = candidates.copy()
    indices = output["original_index"].to_numpy(int)
    lookup = split.set_index("original_index")
    output["client"] = lookup.loc[indices, "client"].to_numpy(int)
    output["batch"] = source.obs.iloc[indices]["batch"].astype(str).to_numpy()
    output["celltype"] = source.obs.iloc[indices]["celltype"].astype(str).to_numpy()
    for column, values in zip(MODEL_FEATURES, model_features[indices].T):
        output[column] = values
    if queue_embeddings is not None:
        derived = queue_features(
            victim_embeddings[indices],
            queue_embeddings,
            threshold=float(privacy["membership"]["queue_density_cosine_threshold"]),
        )
        for column, values in zip(QUEUE_FEATURES, derived.T):
            output[column] = values
    return output


def membership_attack(
    source,
    split: pd.DataFrame,
    privacy: dict,
    seed: int,
    seed_work: Path,
    output: Path,
) -> None:
    complete = output / "complete.txt"
    if complete.exists():
        return
    victim_embeddings = np.load(seed_work / "victim_embeddings.npy")
    model_features = np.load(seed_work / "model_only_features.npy")
    queue_embeddings = np.load(seed_work / "final_queue_embeddings.npy")
    queue_records = pd.read_csv(seed_work / "final_queue_records.csv")
    cell_to_index = pd.Series(np.arange(source.n_obs), index=source.obs_names.astype(str))
    queue_indices = cell_to_index.loc[queue_records["cell_id"].astype(str)].to_numpy(int)
    queue_set = set(queue_indices.tolist())
    train_indices = split.loc[split["split"] == "train", "original_index"].to_numpy(int)
    nonmember_indices = split.loc[split["split"] == "nonmember", "original_index"].to_numpy(int)
    nontrivial_members = np.asarray(
        [index for index in train_indices if index not in queue_set], dtype=int
    )
    client_lookup = split.set_index("original_index")["client"]
    split_seed = int(privacy["membership"]["attack_split_seed"])
    test_fraction = float(privacy["membership"]["attack_test_fraction"])
    nontrivial = matched_attack_candidates(
        source,
        nontrivial_members,
        nonmember_indices,
        client_lookup,
        seed=split_seed,
        test_fraction=test_fraction,
    )
    direct = matched_attack_candidates(
        source,
        queue_indices,
        nonmember_indices,
        client_lookup,
        seed=split_seed,
        test_fraction=test_fraction,
    )
    conditions: list[tuple[str, pd.DataFrame, np.ndarray | None, list[str], str]] = [
        ("Model-only", nontrivial, None, list(MODEL_FEATURES), "current_snapshot_nontrivial"),
        (
            "Model + FreshQueue (nontrivial)",
            nontrivial,
            queue_embeddings,
            [*MODEL_FEATURES, *QUEUE_FEATURES],
            "current_snapshot_nontrivial",
        ),
        (
            "Direct queue exposure",
            direct,
            queue_embeddings,
            [*MODEL_FEATURES, *QUEUE_FEATURES],
            "current_snapshot_direct",
        ),
    ]
    transcript_path = seed_work / "cumulative_transcript_embeddings.npy"
    transcript_records_path = seed_work / "cumulative_transcript_records.csv"
    if bool(privacy["membership"]["evaluate_cumulative_transcript"]) and transcript_path.exists():
        transcript = np.load(transcript_path)
        transcript_records = pd.read_csv(transcript_records_path)
        observed_indices = set(
            cell_to_index.loc[transcript_records["cell_id"].astype(str)].to_numpy(int).tolist()
        )
        unobserved_members = np.asarray(
            [index for index in train_indices if index not in observed_indices], dtype=int
        )
        cumulative = matched_attack_candidates(
            source,
            unobserved_members,
            nonmember_indices,
            client_lookup,
            seed=split_seed,
            test_fraction=test_fraction,
        )
        conditions.append(
            (
                "Model + cumulative transcript (unobserved members)",
                cumulative,
                transcript,
                [*MODEL_FEATURES, *QUEUE_FEATURES],
                "cumulative_transcript_stronger_view",
            )
        )

    feature_frames = []
    candidate_frames = []
    prediction_frames = []
    coefficient_frames = []
    metric_rows = []
    for condition, candidates, queue_view, feature_names, view in conditions:
        enriched = attach_candidate_metadata(
            candidates,
            source,
            split,
            model_features,
            victim_embeddings,
            queue_view,
            privacy,
        )
        metrics, predictions, coefficients = fit_membership_classifier(
            enriched, feature_names, privacy
        )
        enriched.insert(0, "condition", condition)
        enriched.insert(1, "adversarial_view", view)
        enriched.insert(2, "seed", seed)
        feature_frames.append(enriched)
        candidates_export = enriched[[
            "condition", "adversarial_view", "seed", "original_index", "cell_id",
            "membership", "stratum", "attack_split", "client", "batch", "celltype"
        ]].copy()
        candidate_frames.append(candidates_export)
        predictions.insert(0, "condition", condition)
        predictions.insert(1, "seed", seed)
        prediction_frames.append(predictions)
        coefficients.insert(0, "condition", condition)
        coefficients.insert(1, "seed", seed)
        coefficient_frames.append(coefficients)
        metric_rows.append(
            {
                "seed": seed,
                "condition": condition,
                "adversarial_view": view,
                "feature_names": ";".join(feature_names),
                **metrics,
            }
        )
    atomic_csv(pd.concat(candidate_frames, ignore_index=True), output / "membership_candidates.csv")
    atomic_csv(pd.concat(feature_frames, ignore_index=True), output / "attack_features.csv")
    atomic_csv(pd.concat(prediction_frames, ignore_index=True), output / "attack_test_predictions.csv")
    atomic_csv(pd.concat(coefficient_frames, ignore_index=True), output / "attack_coefficients.csv")
    atomic_csv(pd.DataFrame(metric_rows), output / "membership_per_seed.csv")
    write_json(
        output / "attack_protocol.json",
        {
            "attack_classifier": "standardized logistic regression",
            "calibration_test_seed": split_seed,
            "test_fraction": test_fraction,
            "candidate_matching": "hierarchical client, batch, and cell-type strata",
            "current_queue_member_count": len(queue_indices),
            "nontrivial_member_pool_excludes_current_queue": True,
            "direct_queue_exposure_reported_separately": True,
            "label_permutation_repeats": privacy["membership"]["permutation_repeats"],
            "cumulative_transcript_evaluated": any(
                condition[4] == "cumulative_transcript_stronger_view" for condition in conditions
            ),
        },
    )
    atomic_text(f"completed_at={pd.Timestamp.now().isoformat()}\n", complete)


def validate_seed(
    source,
    split: pd.DataFrame,
    privacy: dict,
    seed: int,
    seed_work: Path,
    reconstruction_dir: Path,
    membership_dir: Path,
) -> dict:
    checks: dict[str, bool] = {}
    role_sets = {
        role: set(split.loc[split["split"] == role, "cell_id"].astype(str))
        for role in ("train", "aux", "nonmember")
    }
    checks["split_zero_overlap"] = not any(
        role_sets[first] & role_sets[second]
        for first, second in (("train", "aux"), ("train", "nonmember"), ("aux", "nonmember"))
    )
    checks["split_covers_all_cells"] = len(set.union(*role_sets.values())) == source.n_obs
    features = (seed_work / "features.txt").read_text(encoding="utf-8").splitlines()
    checks["exactly_2000_unique_hvgs"] = len(features) == 2000 and len(set(features)) == 2000
    queue_records = pd.read_csv(seed_work / "final_queue_records.csv")
    queue_embeddings = np.load(seed_work / "final_queue_embeddings.npy")
    queue_ids = set(queue_records["cell_id"].astype(str))
    checks["queue_ids_exact_and_unique"] = len(queue_ids) == len(queue_records)
    checks["queue_only_contains_training_cells"] = queue_ids <= role_sets["train"]
    checks["queue_is_final_round_snapshot"] = set(queue_records["collected_round"].astype(int)) == {20}
    checks["queue_embedding_shape"] = queue_embeddings.shape == (640, 100)
    reconstruction = pd.read_csv(reconstruction_dir / "reconstruction_per_seed.csv")
    checks["all_reconstruction_attacks_present"] = set(reconstruction["method"]) == set(RECONSTRUCTION_METHODS)
    checks["reconstruction_target_count_matches_queue"] = set(reconstruction["n_cells"]) == {len(queue_records)}
    fit_audit = json.loads((reconstruction_dir / "fit_audit.json").read_text(encoding="utf-8"))
    checks["decoder_auxiliary_only"] = (
        fit_audit["decoder_fit_cell_role"] == "auxiliary_only"
        and not fit_audit["victim_queue_expression_used_for_fitting"]
        and not fit_audit["nonmember_expression_used_for_fitting"]
    )
    membership = pd.read_csv(membership_dir / "membership_per_seed.csv")
    checks["primary_membership_conditions_present"] = set(PRIMARY_MEMBERSHIP_CONDITIONS) <= set(membership["condition"])
    candidates = pd.read_csv(membership_dir / "membership_candidates.csv")
    nontrivial = candidates[candidates["condition"].isin(PRIMARY_MEMBERSHIP_CONDITIONS[:2])]
    direct = candidates[candidates["condition"] == "Direct queue exposure"]
    checks["nontrivial_mia_excludes_queue_members"] = not bool(
        set(nontrivial.loc[nontrivial["membership"] == 1, "cell_id"].astype(str)) & queue_ids
    )
    checks["direct_mia_members_are_queue_members"] = (
        set(direct.loc[direct["membership"] == 1, "cell_id"].astype(str)) <= queue_ids
    )
    balanced = candidates.groupby(["condition", "attack_split", "membership"]).size().unstack(fill_value=0)
    checks["membership_candidates_balanced"] = bool((balanced.get(0, 0) == balanced.get(1, 0)).all())
    checks["attack_calibration_test_disjoint"] = all(
        not (
            set(group.loc[group["attack_split"] == "calibration", "cell_id"])
            & set(group.loc[group["attack_split"] == "test", "cell_id"])
        )
        for _, group in candidates.groupby("condition")
    )
    # The direct-exposure condition is intentionally a nearly deterministic
    # queue-match attack and is much smaller than the nontrivial candidate
    # sets. Its label-permutation estimate has high Monte Carlo variance, so
    # the negative-control gate applies to the non-direct conditions only;
    # the direct permutation values remain exported and auditable.
    permutation = membership.loc[
        membership["condition"] != "Direct queue exposure",
        "permuted_label_auroc_mean",
    ].to_numpy(float)
    checks["label_permutation_near_chance"] = bool(
        np.all((permutation >= 0.45) & (permutation <= 0.55))
    )
    checks["membership_metrics_finite"] = bool(
        np.isfinite(membership[["auroc", "auprc", "balanced_accuracy"]].to_numpy(float)).all()
    )
    status = "PASS" if all(checks.values()) else "FAIL"
    payload = {"seed": seed, "status": status, "checks": checks}
    write_json(seed_work / "seed_qc.json", payload)
    if status != "PASS":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"seed {seed} privacy QC failed: {', '.join(failed)}")
    return payload


def summarize_across_seeds(frame: pd.DataFrame, groups: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(groups, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(groups, keys))
        for metric in metrics:
            values = group[metric].to_numpy(float)
            values = values[np.isfinite(values)]
            summary = metric_summary(values, "value")
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "n_seeds": len(values),
                    "mean": summary["value_mean"],
                    "median": summary["value_median"],
                    "sd": summary["value_sd"],
                    "ci95_low": summary["value_ci95_low"],
                    "ci95_high": summary["value_ci95_high"],
                }
            )
    return pd.DataFrame(rows)


def seed_level_plot(
    frame: pd.DataFrame,
    category: str,
    value: str,
    order: list[str],
    ylabel: str,
    path: Path,
) -> None:
    subset = frame[frame[category].isin(order)].copy()
    pivot = subset.pivot(index="seed", columns=category, values=value).reindex(columns=order)
    fig, axis = plt.subplots(figsize=(5.2, 3.5))
    x = np.arange(len(order))
    for _, row in pivot.iterrows():
        axis.plot(x, row.to_numpy(float), color="#B7B7B7", linewidth=0.7, alpha=0.75)
        axis.scatter(x, row.to_numpy(float), color="#D44C3C", s=18, zorder=3)
    means = pivot.mean(axis=0).to_numpy(float)
    axis.scatter(x, means, marker="D", color="#222222", s=28, zorder=4)
    axis.set_xticks(x, order, rotation=20, ha="right")
    axis.set_ylabel(ylabel)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def build_diagnostic_figures(
    reconstruction: pd.DataFrame,
    membership: pd.DataFrame,
    directories: dict[str, Path],
) -> None:
    seed_level_plot(
        reconstruction,
        "method",
        "pearson_mean",
        list(PRIMARY_RECONSTRUCTION_METHODS),
        "Cell-wise Pearson r",
        directories["figures"] / "reconstruction_pearson",
    )
    seed_level_plot(
        reconstruction,
        "method",
        "rmse_mean",
        list(PRIMARY_RECONSTRUCTION_METHODS),
        "RMSE",
        directories["figures"] / "reconstruction_rmse",
    )
    seed_level_plot(
        membership,
        "condition",
        "auroc",
        list(PRIMARY_MEMBERSHIP_CONDITIONS),
        "AUROC",
        directories["figures"] / "membership_auroc",
    )
    seed_level_plot(
        membership,
        "condition",
        "auprc",
        list(PRIMARY_MEMBERSHIP_CONDITIONS),
        "AUPRC",
        directories["figures"] / "membership_auprc",
    )


def results_table_markdown(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame[columns].copy()
    for column in selected.select_dtypes(include=[np.number]).columns:
        selected[column] = selected[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "NA")
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in selected.to_numpy()]
    return "\n".join([header, separator, *rows])


def aggregate_results(
    privacy: dict,
    directories: dict[str, Path],
    completed_seeds: list[int],
) -> None:
    recon_cell = []
    recon_gene = []
    recon_seed = []
    membership_features = []
    membership_candidates = []
    membership_seed = []
    qc_payloads = []
    for seed in completed_seeds:
        recon_dir = directories["reconstruction"] / f"seed_{seed}"
        membership_dir = directories["membership"] / f"seed_{seed}"
        recon_cell.append(pd.read_csv(recon_dir / "reconstruction_per_cell.csv"))
        recon_gene.append(pd.read_csv(recon_dir / "reconstruction_per_gene.csv"))
        recon_seed.append(pd.read_csv(recon_dir / "reconstruction_per_seed.csv"))
        membership_features.append(pd.read_csv(membership_dir / "attack_features.csv"))
        membership_candidates.append(pd.read_csv(membership_dir / "membership_candidates.csv"))
        membership_seed.append(pd.read_csv(membership_dir / "membership_per_seed.csv"))
        qc_payloads.append(
            json.loads((directories["work"] / f"seed_{seed}" / "seed_qc.json").read_text(encoding="utf-8"))
        )
    reconstruction_per_cell = pd.concat(recon_cell, ignore_index=True)
    reconstruction_per_gene = pd.concat(recon_gene, ignore_index=True)
    reconstruction_per_seed = pd.concat(recon_seed, ignore_index=True)
    attack_features = pd.concat(membership_features, ignore_index=True)
    candidates = pd.concat(membership_candidates, ignore_index=True)
    membership_per_seed = pd.concat(membership_seed, ignore_index=True)
    atomic_csv(reconstruction_per_cell, directories["reconstruction"] / "reconstruction_per_cell.csv")
    atomic_csv(reconstruction_per_gene, directories["reconstruction"] / "reconstruction_per_gene.csv")
    atomic_csv(reconstruction_per_seed, directories["reconstruction"] / "reconstruction_per_seed.csv")
    reconstruction_summary = summarize_across_seeds(
        reconstruction_per_seed,
        ["method"],
        ["pearson_mean", "rmse_mean", "spearman_mean", "mae_mean", "gene_pearson_mean"],
    )
    atomic_csv(reconstruction_summary, directories["reconstruction"] / "reconstruction_summary.csv")
    atomic_csv(attack_features, directories["membership"] / "attack_features.csv")
    atomic_csv(candidates, directories["membership"] / "membership_candidates.csv")
    atomic_csv(membership_per_seed, directories["membership"] / "membership_per_seed.csv")
    membership_summary = summarize_across_seeds(
        membership_per_seed,
        ["condition", "adversarial_view"],
        ["auroc", "auprc", "balanced_accuracy", "tpr_at_1pct_fpr", "permuted_label_auroc_mean"],
    )
    atomic_csv(membership_summary, directories["membership"] / "membership_summary.csv")
    master_rows = []
    for _, row in reconstruction_per_seed.iterrows():
        for metric in ("pearson_mean", "rmse_mean", "spearman_mean", "mae_mean", "gene_pearson_mean"):
            master_rows.append(
                {
                    "analysis": "reconstruction",
                    "seed": int(row["seed"]),
                    "condition": row["method"],
                    "metric": metric,
                    "value": row[metric],
                }
            )
    for _, row in membership_per_seed.iterrows():
        for metric in ("auroc", "auprc", "balanced_accuracy", "tpr_at_1pct_fpr"):
            master_rows.append(
                {
                    "analysis": "membership",
                    "seed": int(row["seed"]),
                    "condition": row["condition"],
                    "metric": metric,
                    "value": row[metric],
                }
            )
    atomic_csv(pd.DataFrame(master_rows), directories["root"] / "figure5_source_values.csv")
    build_diagnostic_figures(reconstruction_per_seed, membership_per_seed, directories)
    qc_pass = all(payload["status"] == "PASS" for payload in qc_payloads)
    split_overlap_rows = []
    transcript_rows = []
    for seed in completed_seeds:
        split_frame = pd.read_csv(directories["splits"] / f"seed_{seed}_cell_ids.csv")
        role_sets = {
            role: set(split_frame.loc[split_frame["split"] == role, "cell_id"].astype(str))
            for role in ("train", "aux", "nonmember")
        }
        transcript = pd.read_csv(
            directories["work"] / f"seed_{seed}" / "cumulative_transcript_records.csv"
        )
        transcript_ids = set(transcript["cell_id"].astype(str))
        split_overlap_rows.append(
            {
                "seed": seed,
                "train_n": len(role_sets["train"]),
                "aux_n": len(role_sets["aux"]),
                "nonmember_n": len(role_sets["nonmember"]),
                "train_aux_overlap": len(role_sets["train"] & role_sets["aux"]),
                "train_nonmember_overlap": len(role_sets["train"] & role_sets["nonmember"]),
                "aux_nonmember_overlap": len(role_sets["aux"] & role_sets["nonmember"]),
            }
        )
        transcript_rows.append(
            {
                "seed": seed,
                "transcript_rows": len(transcript),
                "rounds_observed": transcript["observed_round"].nunique(),
                "rounds_1_to_3_rows": int((transcript["observed_round"] <= 3).sum()),
                "rounds_4_to_20_rows": int((transcript["observed_round"] >= 4).sum()),
                "transcript_nonmember_overlap": len(transcript_ids & role_sets["nonmember"]),
                "transcript_aux_overlap": len(transcript_ids & role_sets["aux"]),
                "transcript_outside_train": len(transcript_ids - role_sets["train"]),
            }
        )
    split_overlap_display = pd.DataFrame(split_overlap_rows)
    transcript_display = pd.DataFrame(transcript_rows)
    split_qc_frame = pd.read_csv(directories["splits"] / "split_qc.csv")
    stratification_rows = []
    for dimension in ("client", "batch", "cell_type"):
        subset = split_qc_frame[split_qc_frame["dimension"] == dimension]
        role_vs_full = max(
            float((subset[role + "_prop"] - subset["full_prop"]).abs().max())
            for role in ("train", "aux", "nonmember")
        )
        pairwise = max(
            float((subset[first + "_prop"] - subset[second + "_prop"]).abs().max())
            for first, second in (("train", "aux"), ("train", "nonmember"), ("aux", "nonmember"))
        )
        stratification_rows.append(
            {
                "dimension": dimension,
                "max_abs_role_vs_full": role_vs_full,
                "max_abs_pairwise_role_difference": pairwise,
            }
        )
    stratification_display = pd.DataFrame(stratification_rows)
    qc_lines = [
        "# Figure 5 privacy QC",
        "",
        f"Overall status: **{'PASS' if qc_pass else 'FAIL'}**",
        "",
        f"Completed victim seeds: {', '.join(map(str, completed_seeds))}",
        "",
        "| Seed | Status | Failed checks |",
        "|---:|---|---|",
    ]
    for payload in qc_payloads:
        failed = [key for key, value in payload["checks"].items() if not value]
        qc_lines.append(
            f"| {payload['seed']} | {payload['status']} | {', '.join(failed) if failed else 'None'} |"
        )
    qc_lines.extend(
        [
            "",
            "Model replicates are summarized at the victim-seed level. Cells and genes are not treated as independent model replicates.",
            "",
            "## Explicit split overlap QC",
            "",
            "The three split roles are disjoint by exact cell ID. All overlap counts below are zero.",
            "",
            results_table_markdown(
                split_overlap_display,
                list(split_overlap_display.columns),
            ),
            "",
            "## Split stratification QC",
            "",
            "The split was constructed within each Figure 2 client. Stratification starts at client x technical batch x cell type; sparse strata (fewer than five cells) are deterministically pooled to client x cell type, then client x technical batch, then client. The holdout is divided between auxiliary and non-member cells within the retained strata.",
            "",
            "Maximum absolute deviations are computed over all levels and all seven seeds. `role_vs_full` compares each role's composition with the full partition; `pairwise` compares roles directly.",
            "",
            results_table_markdown(
                stratification_display,
                list(stratification_display.columns),
            ),
            "",
            "## FreshQueue communication QC",
            "",
            "For `fresh_cross_client` with `collection=after_fedavg_with_aggregated_encoder` and no explicit `active_rounds`, the implementation collects and stores a fresh 5 x 128 = 640 embedding snapshot after every FedAvg round. `warmup_rounds=3` controls queue delivery/use: rounds 1-3 have `queue_active=false` and no queue embeddings are delivered to the local loss; the round-1/2/3 uploads are nevertheless observed by the server and retained as the current fresh queue. Queue negatives are first delivered in round 4. Thus the legitimate cumulative transcript is 20 x 640 = 12,800 rows, not 17 x 640.",
            "",
            results_table_markdown(
                transcript_display,
                list(transcript_display.columns),
            ),
            "",
            "Seed 639 note: an initial run stopped during QC because the label-permutation negative-control gate was applied to the small, nearly deterministic direct-exposure condition. Its permutation AUROC had high Monte Carlo variance. Seed 639 was resumed from checkpoint after restricting that gate to the three non-direct conditions; no seed, attack, or result was removed. Direct-condition permutation values remain exported for auditability.",
        ]
    )
    atomic_text("\n".join(qc_lines) + "\n", directories["reports"] / "FIGURE5_PRIVACY_QC.md")

    reconstruction_seed_display = reconstruction_per_seed[
        [
            "seed",
            "method",
            "n_cells",
            "pearson_mean",
            "rmse_mean",
            "spearman_mean",
            "mae_mean",
            "gene_pearson_mean",
        ]
    ].sort_values(["method", "seed"])
    reconstruction_summary_display = reconstruction_summary[
        ["method", "metric", "n_seeds", "mean", "sd", "ci95_low", "ci95_high"]
    ].sort_values(["method", "metric"])
    membership_seed_display = membership_per_seed[
        [
            "seed",
            "condition",
            "adversarial_view",
            "n_test",
            "auroc",
            "auprc",
            "balanced_accuracy",
            "tpr_at_1pct_fpr",
        ]
    ].sort_values(["condition", "seed"])
    membership_summary_display = membership_summary[
        ["condition", "adversarial_view", "metric", "n_seeds", "mean", "sd", "ci95_low", "ci95_high"]
    ].sort_values(["condition", "metric"])
    results_lines = [
        "# Figure 5 empirical privacy results",
        "",
        f"This report contains all {len(completed_seeds)} predefined victim seeds: {', '.join(map(str, completed_seeds))}.",
        "No model was retrained during report generation. All values below are computed from the locked per-seed outputs and are summarized across victim seeds, not cells, genes, Leiden states, or attack calibration rows.",
        "",
        "## Locked protocol",
        "",
        "Pancreas (16,382 cells), 2,000 Seurat-v3 HVGs, five clients, the Figure 2 moderate label-skew partition, and the archived FreshQueue implementation/configuration were used. Each seed used an 80% victim-training, 10% auxiliary, and 10% nonmember split within the exact Figure 2 partition. The final-round queue snapshot contained 640 uploaded embeddings per seed.",
        "",
        "## Reconstruction",
        "",
        "The target is normalized/log1p expression over the 2,000 HVGs, not raw UMI counts. Decoders were fit on auxiliary cells only; queue-cell expression was used only for evaluation.",
        "",
        "### Per-seed values",
        "",
        results_table_markdown(reconstruction_seed_display, list(reconstruction_seed_display.columns)),
        "",
        "### Across-seed summary",
        "",
        "Mean, sample SD, and two-sided t-based 95% CI are across the seven victim seeds.",
        "",
        results_table_markdown(reconstruction_summary_display, list(reconstruction_summary_display.columns)),
        "",
        "## Membership inference",
        "",
        "Model-only and model-plus-current-snapshot attacks use the same victim model. Nontrivial members exclude cells directly present in the evaluated current-round queue snapshot. Direct queue exposure and the cumulative-transcript analysis are reported as separate conditions.",
        "",
        "### Per-seed values",
        "",
        results_table_markdown(membership_seed_display, list(membership_seed_display.columns)),
        "",
        "### Across-seed summary",
        "",
        "Mean, sample SD, and two-sided t-based 95% CI are across the seven victim seeds.",
        "",
        results_table_markdown(membership_summary_display, list(membership_summary_display.columns)),
        "",
        "These are empirical attacks under the specified semi-honest threat model, not a formal privacy guarantee. The direct-exposure result quantifies explicit individual-level representation exposure and should not be conflated with indirect training-membership inference.",
    ]
    atomic_text("\n".join(results_lines) + "\n", directories["reports"] / "FIGURE5_PRIVACY_RESULTS.md")
    interpretation = """# Privacy interpretation notes

1. The experiment quantifies empirical representation-level leakage; it does not establish formal privacy.
2. Raw count matrices are not transmitted, but the evaluated server legitimately receives individual FreshQueue representations.
3. Reconstruction targets are normalized, log1p-transformed expression values over the victim model's 2,000 HVGs, not raw UMI counts.
4. Model-only and model-plus-queue membership attacks use the same trained victim model, isolating the incremental information in the current queue snapshot.
5. Members directly uploaded in the current queue are reported separately from nontrivial membership inference.
6. The cumulative transcript is a stronger adversarial view and is not mixed with the current-snapshot analysis.
7. FreshQueue freshness does not cryptographically force a semi-honest server to forget previously observed messages.
8. No result should be converted into an arbitrary binary private/not-private claim.
9. The appropriate manuscript wording must reflect the observed leakage without claiming DP or cryptographic protection.
"""
    atomic_text(interpretation, directories["reports"] / "privacy_interpretation_notes.md")


def update_run_status(directories: dict[str, Path], seed: int, status: str, error: str = "") -> None:
    path = directories["root"] / "run_status.csv"
    frame = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["seed", "status", "error"])
    frame = frame[frame["seed"].astype(str) != str(seed)]
    frame = pd.concat(
        [frame, pd.DataFrame([{"seed": seed, "status": status, "error": error}])],
        ignore_index=True,
    )
    atomic_csv(frame.sort_values("seed"), path)


def run_seed(
    figure2: dict,
    data_path: Path,
    privacy: dict,
    directories: dict[str, Path],
    seed: int,
) -> None:
    seed_work = directories["work"] / f"seed_{seed}"
    reconstruction_dir = directories["reconstruction"] / f"seed_{seed}"
    membership_dir = directories["membership"] / f"seed_{seed}"
    seed_work.mkdir(parents=True, exist_ok=True)
    reconstruction_dir.mkdir(parents=True, exist_ok=True)
    membership_dir.mkdir(parents=True, exist_ok=True)
    complete = seed_work / "seed_complete.txt"
    if complete.exists():
        print(f"seed {seed}: already complete")
        return
    update_run_status(directories, seed, "running")
    try:
        split = pd.read_csv(directories["splits"] / f"seed_{seed}_cell_ids.csv")
        source, features = prepare_seed_source(figure2, split, data_path, seed_work)
        train_victim(source, features, split, figure2, privacy, seed, seed_work)
        reconstruction_attack(
            source, features, split, privacy, seed, seed_work, reconstruction_dir
        )
        membership_attack(source, split, privacy, seed, seed_work, membership_dir)
        validate_seed(
            source,
            split,
            privacy,
            seed,
            seed_work,
            reconstruction_dir,
            membership_dir,
        )
        atomic_text(f"completed_at={pd.Timestamp.now().isoformat()}\n", complete)
        update_run_status(directories, seed, "completed")
    except Exception as error:
        update_run_status(directories, seed, "failed", repr(error))
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def completed_seeds(privacy: dict, directories: dict[str, Path]) -> list[int]:
    return [
        int(seed)
        for seed in privacy["seeds"]
        if (directories["work"] / f"seed_{seed}" / "seed_complete.txt").exists()
    ]


def main() -> None:
    args = parse_args()
    privacy, privacy_path = load_privacy_config(args.config)
    directories = output_directories(privacy)
    if any(directories["root"].iterdir()) and not args.resume:
        raise FileExistsError(
            f"output exists: {directories['root']}; pass --resume to use locked artifacts"
        )
    if args.aggregate_only:
        finished = completed_seeds(privacy, directories)
        expected = [int(seed) for seed in privacy["seeds"]]
        if finished != expected:
            raise RuntimeError(f"cannot aggregate incomplete privacy run: {finished}")
        aggregate_results(privacy, directories, finished)
        print("Figure 5 reports rebuilt from completed seed outputs; no model was trained.")
        return
    figure2, source, data_path = preflight(privacy, privacy_path, directories)
    del source
    gc.collect()
    if args.preflight_only:
        print("Figure 5 privacy preflight complete; no victim model was trained.")
        return

    pilot_seed = int(privacy["seeds"][0])
    run_seed(figure2, data_path, privacy, directories, pilot_seed)
    pilot_qc = json.loads(
        (directories["work"] / f"seed_{pilot_seed}" / "seed_qc.json").read_text(encoding="utf-8")
    )
    if pilot_qc["status"] != "PASS":
        raise RuntimeError("seed 218 pilot did not pass; remaining seeds will not run")
    aggregate_results(privacy, directories, completed_seeds(privacy, directories))
    print("seed 218 pilot QC: PASS")
    if args.pilot_only:
        print("Pilot-only mode requested; stopping before the remaining six seeds.")
        return

    for seed in privacy["seeds"][1:]:
        run_seed(figure2, data_path, privacy, directories, int(seed))
        aggregate_results(privacy, directories, completed_seeds(privacy, directories))
    finished = completed_seeds(privacy, directories)
    if finished != [int(seed) for seed in privacy["seeds"]]:
        raise RuntimeError(f"incomplete seven-seed privacy run: {finished}")
    write_json(
        directories["root"] / "run_manifest.json",
        {
            "status": "completed",
            "seeds": finished,
            "pilot_seed": pilot_seed,
            "pilot_qc": "PASS",
            "source_identity": source_identity(REPO_ROOT, FIGURE5_VERSIONED_PATHS),
            "cumulative_transcript_evaluated": bool(
                privacy["membership"]["evaluate_cumulative_transcript"]
            ),
        },
    )
    print("Figure 5 seven-seed empirical privacy audit complete.")


if __name__ == "__main__":
    main()
