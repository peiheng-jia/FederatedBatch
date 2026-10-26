#!/usr/bin/env python3
"""Train the full-data explicit FreshQueue comparator used by Figure 5 only.

This historical direct-sharing protocol supplies the matched reconstruction
baseline. It is not a Figure 2 arm and is never used as the final
FederatedBatch mechanism.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.moderate_skew_freshqueue_confirmation import (  # noqa: E402
    _partition_hash,
    method_config,
    partition_for_seed,
    validate_protocol,
)
from federatedbatch.config import load_config, resolve_repo_path  # noqa: E402
from federatedbatch.data import normalize, preprocess_dataset, read_dataset  # noqa: E402
from federatedbatch.federated import save_checkpoint, train_federated  # noqa: E402


SEEDS = (218, 394, 542, 639, 788, 814, 880)
DEFAULT_CONFIG = ROOT / "configs" / "figure2_legacy_freshqueue.yaml"
DEFAULT_OUTPUT = ROOT / "results" / "reproduction" / "figure5" / "explicit_full_data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def save_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(values, dtype=np.float32), allow_pickle=False)
    temporary.replace(path)


def validate_figure5_role(config: dict) -> None:
    validate_protocol(config)
    if tuple(config["confirmation"]["training_seeds"]) != SEEDS:
        raise ValueError("explicit comparator must use the canonical seven seeds")
    queue = config["federated"]["queue"]
    if not queue["enabled"] or not str(queue["mode"]).startswith("fresh"):
        raise ValueError("explicit comparator configuration is not FreshQueue")


def prepare(config: dict, output: Path, verify_hash: bool) -> None:
    adata, data_path = read_dataset(config["dataset"], config["data_dir"], verify_hash=verify_hash)
    features = preprocess_dataset(adata, config)
    atomic_text(output / "features.txt", "\n".join(features) + "\n")
    for seed in SEEDS:
        groups = partition_for_seed(adata, config, seed)
        payload = {
            "seed": seed,
            "n_clients": int(config["federated"]["n_clients"]),
            "partition_sha256": _partition_hash(groups),
            "client_indices": [group.astype(int).tolist() for group in groups],
        }
        atomic_text(
            output / "partitions" / f"seed_{seed}.json",
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )
    atomic_text(
        output / "preflight.json",
        json.dumps(
            {
                "status": "PASS",
                "role": "Figure 5 explicit full-data comparator only",
                "dataset": str(data_path),
                "dataset_sha256": sha256(data_path),
                "seeds": list(SEEDS),
                "model_trained": False,
                "figure2_final_protocol": False,
            },
            indent=2,
        )
        + "\n",
    )
    print(f"Figure 5 explicit full-data preflight PASS: {output}")


def load_partition(path: Path, seed: int, n_obs: int) -> list[np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload["seed"]) != seed:
        raise ValueError(f"partition seed mismatch: {path}")
    groups = [np.asarray(values, dtype=np.int64) for values in payload["client_indices"]]
    flat = np.concatenate(groups)
    if len(flat) != n_obs or len(np.unique(flat)) != n_obs:
        raise ValueError(f"partition does not cover each cell once: {path}")
    return groups


def train(config: dict, output: Path, verify_hash: bool) -> None:
    if not (output / "preflight.json").is_file():
        raise FileNotFoundError("run the explicit full-data preflight before training")
    adata, _ = read_dataset(config["dataset"], config["data_dir"], verify_hash=verify_hash)
    features = (output / "features.txt").read_text(encoding="utf-8").splitlines()
    normalize(adata, config["preprocessing"]["normalize_target_sum"], config["preprocessing"]["log1p"])
    records = []
    for seed in SEEDS:
        run_dir = output / f"seed_{seed}" / "FreshQueue"
        embedding_path = run_dir / "final_embedding.npy"
        if embedding_path.is_file():
            records.append({"seed": seed, "status": "reused", "embedding_sha256": sha256(embedding_path)})
            continue
        groups = load_partition(output / "partitions" / f"seed_{seed}.json", seed, adata.n_obs)
        configured = method_config(copy.deepcopy(config), "FreshQueue", seed)
        result = train_federated(
            adata,
            features,
            None,
            configured,
            run_dir,
            domain_key="batch",
            client_indices=groups,
        )
        save_array(embedding_path, result.embeddings)
        save_checkpoint(
            run_dir / "model_checkpoint.pt",
            result,
            features,
            configured,
            domain_key="batch",
        )
        records.append({"seed": seed, "status": "trained", "embedding_sha256": sha256(embedding_path)})
        del result
        gc.collect()
    atomic_text(
        output / "completion.json",
        json.dumps(
            {
                "status": "COMPLETE",
                "role": "Figure 5 explicit full-data comparator only",
                "seeds": list(SEEDS),
                "records": records,
                "figure2_final_protocol": False,
            },
            indent=2,
        )
        + "\n",
    )
    print(f"Figure 5 explicit full-data comparator complete: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("preflight", "train", "all"), required=True)
    parser.add_argument("--no-verify-hash", action="store_true")
    args = parser.parse_args()
    config = load_config(str(args.config.resolve()))
    config["data_dir"] = str(resolve_repo_path(config["data_dir"]))
    validate_figure5_role(config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_text(
        output / "config.yaml",
        yaml.safe_dump({key: value for key, value in config.items() if not key.startswith("_")}, sort_keys=False),
    )
    if args.stage in {"preflight", "all"}:
        prepare(config, output, not args.no_verify_hash)
    if args.stage in {"train", "all"}:
        train(config, output, not args.no_verify_hash)


if __name__ == "__main__":
    main()
