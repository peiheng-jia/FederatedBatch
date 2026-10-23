"""Configuration loading and validation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"configuration not found: {path}")
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"configuration must contain a mapping: {path}")
    validate_config(config)
    config = copy.deepcopy(config)
    config["_config_path"] = str(path.resolve())
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "protocol",
        "seed",
        "device",
        "data_dir",
        "output_dir",
        "dataset",
        "preprocessing",
        "federated",
        "centralized",
        "evaluation",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"configuration is missing: {', '.join(missing)}")
    if config["schema_version"] != 1:
        raise ValueError(f"unsupported schema_version: {config['schema_version']}")

    fed = config["federated"]
    fed_required = {
        "n_clients",
        "n_rounds",
        "local_epochs",
        "batch_size",
        "learning_rate",
        "temperature",
        "hard_negative_beta",
        "latent_dim",
        "encoder_dims",
        "dropout",
        "queue",
    }
    missing_fed = sorted(fed_required - fed.keys())
    if missing_fed:
        raise ValueError(f"federated configuration is missing: {', '.join(missing_fed)}")
    for key in ("n_clients", "n_rounds", "local_epochs", "batch_size", "latent_dim"):
        if not isinstance(fed[key], int) or fed[key] <= 0:
            raise ValueError(f"federated.{key} must be a positive integer")
    if not fed["encoder_dims"] or any(value <= 0 for value in fed["encoder_dims"]):
        raise ValueError("federated.encoder_dims must contain positive dimensions")
    if fed["hard_negative_beta"] < 0:
        raise ValueError("federated.hard_negative_beta cannot be negative")

    queue = fed["queue"]
    queue_required = {"enabled", "size", "samples_per_client", "weight"}
    missing_queue = sorted(queue_required - queue.keys())
    if missing_queue:
        raise ValueError(f"queue configuration is missing: {', '.join(missing_queue)}")
    if queue["size"] < 0 or queue["samples_per_client"] < 0:
        raise ValueError("queue size values cannot be negative")
    if queue["enabled"] and queue["size"] == 0:
        raise ValueError("queue.enabled is true but queue.size is zero")
    if queue.get("mode", "legacy_fifo") not in {
        "legacy_fifo",
        "cross_client_soft",
        "fresh_cross_client",
        "fresh_cross_client_ab",
        "fresh_cross_client_cellclip",
        "fresh_cross_client_cellclip_noise",
        "fresh_local_memory",
        "last_seen_fresh_cross_client",
        "reliable_negative_fresh_cross_client",
        "random_mask_fresh_cross_client",
        "soft_reliability_fresh_cross_client",
        "random_soft_weight_fresh_cross_client",
    }:
        raise ValueError(
            "federated.queue.mode must be legacy_fifo, cross_client_soft, "
            "fresh_cross_client, fresh_local_memory, last_seen_fresh_cross_client, "
            "fresh_cross_client_ab, fresh_cross_client_cellclip, fresh_cross_client_cellclip_noise, "
            "reliable_negative_fresh_cross_client, random_mask_fresh_cross_client, "
            "soft_reliability_fresh_cross_client, or random_soft_weight_fresh_cross_client"
        )
    if float(queue.get("temperature", 0.2)) <= 0:
        raise ValueError("federated.queue.temperature must be positive")
    if queue.get("collection", "before_fedavg_with_local_encoder") not in {
        "before_fedavg_with_local_encoder",
        "after_fedavg_with_aggregated_encoder",
    }:
        raise ValueError("unsupported federated.queue.collection")
    if float(queue.get("mask_fraction", 0.0)) < 0 or float(
        queue.get("mask_fraction", 0.0)
    ) >= 1:
        raise ValueError("federated.queue.mask_fraction must be in [0, 1)")
    if int(queue.get("warmup_rounds", 0)) < 0:
        raise ValueError("federated.queue.warmup_rounds cannot be negative")
    if float(queue.get("noise_multiplier", 0.0)) < 0:
        raise ValueError("federated.queue.noise_multiplier cannot be negative")


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def with_overrides(config: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        target = updated
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    validate_config(updated)
    return updated
