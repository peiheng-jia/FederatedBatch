#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from federatedbatch.experiment_support.partitions import (  # noqa: E402
    _partition_hash,
    partition_for_seed,
)
from federatedbatch.config import load_config, resolve_repo_path  # noqa: E402
from federatedbatch.data import read_dataset  # noqa: E402
from federatedbatch.provenance import (  # noqa: E402
    FIGURE5_VERSIONED_PATHS,
    git_paths_clean,
    source_identity,
)


SEEDS = (218, 394, 542, 639, 788, 814, 880)
DEFAULT_CONFIG = ROOT / "configs" / "legacy" / "figure2_freshqueue_baseline.yaml"
DEFAULT_OUTPUT = ROOT / "results" / "reproduction" / "figure2_explicit_baseline"


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the partition/provenance snapshot for Figure 5's explicit baseline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-verify-hash", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    serialized_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = load_config(str(config_path))
    expected = list(SEEDS)
    if config["confirmation"]["training_seeds"] != expected:
        raise ValueError("explicit-baseline seed list differs from canonical seven")
    queue = config["federated"]["queue"]
    if not str(queue.get("mode", "")).startswith("fresh") or not queue.get("enabled"):
        raise ValueError("explicit baseline must retain the recorded FreshQueue protocol")

    data_dir = resolve_repo_path(config["data_dir"])
    adata, data_path = read_dataset(
        config["dataset"], data_dir, verify_hash=not args.no_verify_hash
    )
    output = args.output.resolve()
    partitions = output / "partitions"
    for seed in SEEDS:
        groups = partition_for_seed(adata, config, seed)
        payload = {
            "protocol": "label_skew",
            "seed": seed,
            "n_clients": int(config["federated"]["n_clients"]),
            "partition_sha256": _partition_hash(groups),
            "client_indices": [group.astype(int).tolist() for group in groups],
        }
        destination = partitions / f"seed_{seed}.json"
        content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if destination.exists() and destination.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"existing explicit-baseline partition changed: {destination}")
        atomic_text(destination, content)

    atomic_text(output / "config.yaml", yaml.safe_dump(serialized_config, sort_keys=False))
    identity = source_identity(ROOT, FIGURE5_VERSIONED_PATHS)
    if identity["kind"] == "git" and not git_paths_clean(ROOT, FIGURE5_VERSIONED_PATHS):
        raise RuntimeError(
            "Figure 5 source paths differ from HEAD or are untracked; use a clean "
            "release commit before preparing the explicit baseline"
        )
    manifest = {
        "status": "prepared_no_training",
        "role": "Figure 5 historical explicit-embedding comparator only",
        "source_kind": identity["kind"],
        "source_revision": identity["revision"],
        "git_commit": identity["git_commit"],
        "dataset": str(data_path),
        "canonical_seeds": list(SEEDS),
        "partition_count": len(SEEDS),
        "model_trained": False,
        "figure2_final_protocol": False,
    }
    atomic_text(output / "run_manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(f"Figure 5 explicit-baseline provenance prepared: {output}")


if __name__ == "__main__":
    main()
