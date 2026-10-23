"""Synchronous FederatedBatch training with an explicit global embedding queue."""

from __future__ import annotations

import copy
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


@dataclass
class FederatedResult:
    embeddings: np.ndarray
    model: Any
    client_batches: list[list]
    client_cells: list[int]
    queue_sizes: list[int]
    final_queue_embeddings: np.ndarray
    final_queue_records: list[dict[str, Any]]
    skipped_state_keys: list[str]
    training_seconds: float
    queue_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    client_queue_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    local_epoch_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    client_resource_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    local_loss_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    communication_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    joint_clipping_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    joint_noise_diagnostics: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class FederatedRoundSnapshot:
    """Read-only state exposed to post-hoc mechanism diagnostics."""

    round_number: int
    models: tuple[Any, ...]
    clients: tuple[Any, ...]
    queue_embeddings: np.ndarray
    queue_records: tuple[dict[str, Any], ...]
    queue_was_active: bool


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return device


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def split_clients(adata, client_batches: list[list], batch_key: str = "batch") -> list:
    clients = []
    for batches in client_batches:
        client = adata[adata.obs[batch_key].isin(batches)].copy()
        if client.n_obs == 0:
            raise ValueError(f"empty client generated for batches {batches}")
        clients.append(client)
    return clients


def split_client_indices(
    adata,
    client_indices: list[np.ndarray],
    *,
    allow_partial: bool = False,
) -> list:
    if not client_indices:
        raise ValueError("client cell indices must be non-empty")
    flattened = np.concatenate([np.asarray(indices, dtype=np.int64) for indices in client_indices])
    if len(np.unique(flattened)) != len(flattened):
        raise ValueError("client cell indices must not contain duplicates")
    if (flattened < 0).any() or (flattened >= adata.n_obs).any():
        raise ValueError("client cell indices are outside the dataset")
    if not allow_partial and (
        len(flattened) != adata.n_obs
        or not np.array_equal(np.sort(flattened), np.arange(adata.n_obs))
    ):
        raise ValueError("client cell indices must cover every cell exactly once")
    clients = []
    for indices in client_indices:
        client = adata[np.asarray(indices, dtype=np.int64)].copy()
        if client.n_obs == 0:
            raise ValueError("empty client generated from cell indices")
        clients.append(client)
    return clients


def aggregate_state_dicts(
    states: list[dict[str, torch.Tensor]],
    sample_counts: list[int],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Sample-weighted FedAvg over compatible tensors.

    Domain embeddings are local even when they have equal shapes: row zero can
    represent a different biological batch on every client. Those tensors are
    intentionally left untouched.
    """

    if not states or len(states) != len(sample_counts):
        raise ValueError("states and sample_counts must be non-empty and have equal length")
    total = sum(sample_counts)
    if total <= 0:
        raise ValueError("sample counts must sum to a positive value")

    averaged: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key in states[0]:
        if key.startswith("domain_embedding.") or key.startswith("covariate_embeddings."):
            skipped.append(key)
            continue
        tensors = [state[key].detach().cpu() for state in states]
        if len({tuple(tensor.shape) for tensor in tensors}) != 1:
            skipped.append(key)
            continue
        first = tensors[0]
        if first.is_floating_point() or first.is_complex():
            value = torch.zeros_like(first)
            for tensor, count in zip(tensors, sample_counts):
                value.add_(tensor, alpha=count / total)
            averaged[key] = value
        else:
            # Integer counters are not model parameters; preserve a valid dtype.
            averaged[key] = first.clone()
    return averaged, skipped


def distribute_state(models: list, averaged: dict[str, torch.Tensor]) -> None:
    for model in models:
        current = model.model.state_dict()
        for key, value in averaged.items():
            if key in current and current[key].shape == value.shape:
                current[key] = value.to(device=current[key].device, dtype=current[key].dtype)
        model.model.load_state_dict(current)


def predict_embeddings(model, adata) -> np.ndarray:
    working = adata
    domain_key = model.config.domain_key
    if domain_key is not None and domain_key not in working.obs:
        working = working.copy()
        working.obs[domain_key] = "single_domain"
        working.obs[domain_key] = working.obs[domain_key].astype("category")
    model.init_dataloader(adata=working, train_frac=1.0, use_sampler=False)
    result = model.predict(
        model.loader,
        return_decoded=False,
        return_class=False,
        return_class_prob=False,
    )
    return np.asarray(result[0])


def _inject_queue(
    models: list,
    queue: list[np.ndarray],
    records: list[dict[str, Any]],
    device: torch.device,
    *,
    cross_client_only: bool,
) -> list[int]:
    delivered_sizes = []
    for client_index, model in enumerate(models):
        selected = queue
        if cross_client_only:
            selected = [
                embedding
                for embedding, record in zip(queue, records)
                if int(record["client"]) != client_index
            ]
        queue_tensor = None
        if selected:
            queue_tensor = torch.as_tensor(
                np.asarray(selected),
                dtype=torch.float32,
                device=device,
            )
        delivered_sizes.append(0 if queue_tensor is None else len(queue_tensor))
        model.config.global_queue = queue_tensor
        trainer = getattr(model, "trainer", None)
        if trainer is not None:
            trainer.update_global_queue(queue_tensor)
    return delivered_sizes


def _inject_fresh_client_queues(
    models: list,
    queue: list[np.ndarray],
    records: list[dict[str, Any]],
    device: torch.device,
    *,
    seed: int = 0,
    round_number: int = 0,
    active_client_indices: set[int] | None = None,
    interface: str = "explicit",
    clipping_threshold: float | None = None,
    noise_multiplier: float = 0.0,
) -> list[int]:
    from .queue import FreshClientQueue, PrivateRemoteABQueue

    delivered_sizes = []
    for client_index, model in enumerate(models):
        if active_client_indices is not None and client_index not in active_client_indices:
            model.config.global_queue = None
            trainer = getattr(model, "trainer", None)
            if trainer is not None:
                trainer.update_global_queue(None)
            delivered_sizes.append(0)
            continue
        keep = [
            index for index, record in enumerate(records) if int(record["client"]) != client_index
        ]
        client_queue = None
        if keep:
            embeddings = torch.as_tensor(
                np.asarray([queue[index] for index in keep]),
                dtype=torch.float32,
                device=device,
            ).detach()
            source_clients = torch.as_tensor(
                [int(records[index]["client"]) for index in keep],
                dtype=torch.long,
                device=device,
            )
            neighborhood_centroids = None
            if all("neighborhood_centroid" in records[index] for index in keep):
                neighborhood_centroids = torch.as_tensor(
                    np.asarray(
                        [records[index]["neighborhood_centroid"] for index in keep]
                    ),
                    dtype=torch.float32,
                    device=device,
                ).detach()
            sampling_seed = seed * 1_000_003 + round_number * 1_009 + client_index
            if interface == "explicit":
                client_queue = FreshClientQueue(
                    embeddings,
                    source_clients,
                    sampling_seed,
                    neighborhood_centroids,
                )
            elif interface in {"ab", "cellclip", "cellclip_noise"}:
                client_queue = PrivateRemoteABQueue(
                    embeddings,
                    source_clients,
                    sampling_seed,
                    clipping_threshold=(
                        clipping_threshold
                        if interface in {"cellclip", "cellclip_noise"}
                        else None
                    ),
                    noise_multiplier=(
                        noise_multiplier if interface == "cellclip_noise" else 0.0
                    ),
                )
            else:
                raise ValueError(f"unknown queue interface: {interface}")
        delivered_sizes.append(0 if client_queue is None else len(client_queue))
        model.config.global_queue = client_queue
        trainer = getattr(model, "trainer", None)
        if trainer is not None:
            trainer.update_global_queue(client_queue)
    return delivered_sizes


def _inject_fresh_local_memories(
    models: list,
    queue: list[np.ndarray],
    records: list[dict[str, Any]],
    device: torch.device,
    *,
    seed: int = 0,
    round_number: int = 0,
) -> list[int]:
    """Deliver each client only its own fresh, previous-round memory entries."""

    from .queue import FreshClientQueue

    delivered_sizes = []
    for client_index, model in enumerate(models):
        keep = [
            index for index, record in enumerate(records) if int(record["client"]) == client_index
        ]
        client_memory = None
        if keep:
            embeddings = torch.as_tensor(
                np.asarray([queue[index] for index in keep]),
                dtype=torch.float32,
                device=device,
            ).detach()
            source_clients = torch.full(
                (len(keep),), client_index, dtype=torch.long, device=device
            )
            sampling_seed = seed * 1_000_003 + round_number * 1_009 + client_index
            client_memory = FreshClientQueue(embeddings, source_clients, sampling_seed)
        delivered_sizes.append(0 if client_memory is None else len(client_memory))
        model.config.global_queue = client_memory
        trainer = getattr(model, "trainer", None)
        if trainer is not None:
            trainer.update_global_queue(client_memory)
    return delivered_sizes


def _state_payload_bytes(model) -> int:
    """Exact bytes in tensors participating in FedAvg, excluding local embeddings."""

    return int(
        sum(
            tensor.numel() * tensor.element_size()
            for key, tensor in model.model.state_dict().items()
            if not key.startswith("domain_embedding.")
            and not key.startswith("covariate_embeddings.")
        )
    )


def _collect_fresh_queue(
    models: list,
    clients: list,
    queue_rng: np.random.Generator,
    samples_per_client: int,
    round_number: int,
    seed: int = 0,
    client_indices: list[int] | None = None,
    neighborhood_k: int = 20,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Encode a new per-round queue with the post-FedAvg global encoder."""

    embeddings: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    source_clients = list(range(len(clients))) if client_indices is None else client_indices
    for client_index in source_clients:
        model = models[client_index]
        client = clients[client_index]
        n_samples = min(samples_per_client, client.n_obs)
        selected = queue_rng.choice(client.n_obs, size=n_samples, replace=False)
        sampled_client = client[selected].copy()
        encoded = _predict_queue_embeddings(
            model,
            sampled_client,
            seed=seed * 1_000_003 + round_number * 1_009 + client_index,
        )
        normalized = (
            F.normalize(torch.as_tensor(encoded, dtype=torch.float32), dim=1)
            .detach()
            .cpu()
            .numpy()
        )
        from .queue import local_neighborhood_centroids

        centroids = (
            local_neighborhood_centroids(
                torch.as_tensor(normalized, dtype=torch.float32), neighborhood_k
            )
            .cpu()
            .numpy()
        )
        embeddings.extend(normalized)
        for local_index, centroid in zip(selected, centroids):
            records.append(
                {
                    "cell_id": str(client.obs_names[local_index]),
                    "client": client_index,
                    "batch": str(client.obs["batch"].iloc[local_index]),
                    "collected_round": round_number,
                    "neighborhood_centroid": centroid,
                }
            )
    return embeddings, records


def _collect_scheduled_queue(
    models: list,
    clients: list,
    scheduled_indices: dict,
    *,
    collection_round: int,
    seed: int,
    neighborhood_k: int = 20,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Encode the precommitted remote-cell schedule for the next interaction round."""

    embeddings: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    slots = scheduled_indices.get(collection_round, scheduled_indices.get(str(collection_round), {}))
    for slot_key, source_map in sorted(slots.items(), key=lambda item: int(item[0])):
        slot = int(slot_key)
        for source_key, local_values in sorted(source_map.items(), key=lambda item: int(item[0])):
            source_client = int(source_key)
            local_indices = np.asarray(local_values, dtype=np.int64)
            if len(local_indices) == 0:
                continue
            sampled_client = clients[source_client][local_indices].copy()
            encoded = _predict_queue_embeddings(
                models[source_client],
                sampled_client,
                seed=seed * 1_000_003 + collection_round * 1_009 + source_client * 97 + slot,
            )
            normalized = (
                F.normalize(torch.as_tensor(encoded, dtype=torch.float32), dim=1)
                .detach()
                .cpu()
                .numpy()
            )
            from .queue import local_neighborhood_centroids

            centroids = local_neighborhood_centroids(
                torch.as_tensor(normalized, dtype=torch.float32), neighborhood_k
            ).cpu().numpy()
            embeddings.extend(normalized)
            for local_index, centroid in zip(local_indices, centroids):
                records.append(
                    {
                        "cell_id": str(clients[source_client].obs_names[local_index]),
                        "client": source_client,
                        "batch": str(clients[source_client].obs["batch"].iloc[local_index]),
                        "collected_round": collection_round,
                        "interaction_slot": slot,
                        "source_local_index": int(local_index),
                        "neighborhood_centroid": centroid,
                    }
                )
    return embeddings, records


def _scheduled_private_queue(
    queue: list[np.ndarray],
    records: list[dict[str, Any]],
    *,
    target_client: int,
    interaction_slot: int,
    device: torch.device,
    seed: int,
    round_number: int,
    fixed_scales: list[float] | np.ndarray | None = None,
) -> Any:
    """Build a target-specific A/B endpoint without exposing remote embeddings."""

    from .queue import PrivateRemoteABQueue

    keep = [
        index
        for index, record in enumerate(records)
        if int(record["client"]) != target_client
        and int(record["collected_round"]) == int(round_number) - 1
        and int(record.get("interaction_slot", 0)) == int(interaction_slot)
    ]
    if not keep:
        return None
    embeddings = torch.as_tensor(
        np.asarray([queue[index] for index in keep]), dtype=torch.float32, device=device
    ).detach()
    source_clients = torch.as_tensor(
        [int(records[index]["client"]) for index in keep], dtype=torch.long, device=device
    )
    endpoint_scales = None
    if fixed_scales is not None:
        endpoint_scales = torch.as_tensor(
            [float(fixed_scales[index]) for index in keep],
            dtype=torch.float32,
            device=device,
        ).detach()
    return PrivateRemoteABQueue(
        embeddings,
        source_clients,
        seed * 1_000_003 + round_number * 1_009 + target_client * 97 + int(interaction_slot),
        fixed_scales=endpoint_scales,
    )


def _take_anchor_batch(model, target_size: int = 128) -> dict[str, Any]:
    """Take one deterministic-size minibatch from the trained client's loader."""

    if not isinstance(model.loader, list) or not model.loader or model.loader[0][0] is None:
        raise RuntimeError("round-level A/B requires a materialized client train loader")
    loader = model.loader[0][0]
    batches: list[dict[str, Any]] = []
    n_rows = 0
    iterator = iter(loader)
    while n_rows < target_size:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batches.append(batch)
        first_tensor = next(value for value in batch.values() if isinstance(value, torch.Tensor))
        n_rows += int(first_tensor.shape[0])
    result: dict[str, Any] = {}
    for key in batches[0]:
        values = [batch[key] for batch in batches]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.cat(values, dim=0)[:target_size]
        else:
            result[key] = values[0]
    return result


def _anchor_representation(model, batch: dict[str, Any]) -> torch.Tensor:
    """Materialize the two-view representation used to precompute joint scales."""

    trainer = model.trainer
    data_dict = {
        key: value.to(trainer.device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    inputs = data_dict["input"]
    domain_labels = data_dict.get("domain")
    covariate_keys = [
        key for key in data_dict if key not in {"input", "domain", "class", "idx"}
    ]
    covariate_tensors = {key: data_dict[key] for key in covariate_keys}
    model.model.train()
    trainer.augment.train()
    with torch.no_grad():
        x1 = trainer.augment(inputs)
        x2 = trainer.augment(inputs)
        z1 = model.model(x1, domain_labels, covariate_tensors)["encoded"]
        z2 = model.model.encode(x2)
        return F.normalize(torch.cat([z1, z2], dim=0), dim=1)


def _capture_rng_state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return torch.random.get_rng_state(), cuda_state


def _restore_rng_state(
    state: tuple[torch.Tensor, torch.Tensor | None], device: torch.device
) -> None:
    torch.random.set_rng_state(state[0])
    if device.type == "cuda" and state[1] is not None:
        torch.cuda.set_rng_state(state[1], device)


def _joint_remote_scales(
    target_representations: dict[int, torch.Tensor],
    queue: list[np.ndarray],
    records: list[dict[str, Any]],
    *,
    queue_temperature: float,
    clipping_threshold: float,
    round_number: int,
) -> tuple[list[float], list[dict[str, Any]]]:
    """Compute one clipping scale per remote cell over all four target clients."""

    if len(queue) != len(records):
        raise ValueError("remote queue and records must have equal lengths")
    if clipping_threshold <= 0:
        raise ValueError("joint clipping threshold must be positive")
    n_targets = len(target_representations)
    if set(target_representations) != set(range(n_targets)):
        raise ValueError("joint clipping requires an anchor batch for every target client")
    device = next(iter(target_representations.values())).device
    remote = F.normalize(
        torch.as_tensor(np.asarray(queue), dtype=torch.float32, device=device),
        dim=1,
    ).detach()
    squared_norms = torch.zeros(len(queue), dtype=torch.float32, device=remote.device)
    for target, local in target_representations.items():
        keep = [index for index, record in enumerate(records) if int(record["client"]) != target]
        if {int(records[index]["client"]) for index in keep} != set(range(n_targets)) - {target}:
            raise ValueError("every remote cell must reach exactly four other targets")
        weights = torch.exp((local @ remote[keep].T) / float(queue_temperature))
        squared_norms[keep] += 2.0 * torch.square(weights).sum(dim=0)
    norms = torch.sqrt(squared_norms)
    scales = torch.minimum(
        torch.ones_like(norms),
        torch.as_tensor(clipping_threshold, dtype=norms.dtype, device=norms.device)
        / norms.clamp_min(torch.finfo(norms.dtype).tiny),
    )
    rows = []
    for index, record in enumerate(records):
        rows.append(
            {
                "round": int(round_number),
                "source_client": int(record["client"]),
                "cell_id": str(record.get("cell_id", index)),
                "joint_norm": float(norms[index].cpu()),
                "clipping_scale": float(scales[index].cpu()),
                "clipped": bool(scales[index] < 1.0),
                "threshold": float(clipping_threshold),
                "targets_covered": int(n_targets - 1),
                "representation_rows_per_target": int(next(iter(target_representations.values())).shape[0]),
                "noise_multiplier": 0.0,
            }
        )
    return scales.cpu().tolist(), rows


def _joint_noisy_ab_endpoints(
    target_representations: dict[int, torch.Tensor],
    queue: list[np.ndarray],
    records: list[dict[str, Any]],
    scales: list[float],
    joint_norms: list[float],
    *,
    queue_temperature: float,
    clipping_threshold: float,
    noise_multiplier: float,
    seed: int,
    round_number: int,
    positive_a_floor: float = 1e-6,
) -> tuple[dict[int, Any], list[dict[str, Any]]]:
    """Create one noisy four-target joint release per source client."""

    from .queue import PrivateAggregateABQueue

    if noise_multiplier <= 0 or not math.isfinite(noise_multiplier):
        raise ValueError("joint Gaussian noise multiplier must be finite and positive")
    if positive_a_floor <= 0:
        raise ValueError("positive A floor must be positive")
    n_clients = len(target_representations)
    if n_clients < 2 or set(target_representations) != set(range(n_clients)):
        raise ValueError("joint Gaussian release requires every target client")
    if not (len(queue) == len(records) == len(scales) == len(joint_norms)):
        raise ValueError("joint release inputs must have equal lengths")
    device = next(iter(target_representations.values())).device
    remote = F.normalize(
        torch.as_tensor(np.asarray(queue), dtype=torch.float32, device=device), dim=1
    ).detach()
    scale_tensor = torch.as_tensor(scales, dtype=torch.float32, device=device).detach()
    norm_tensor = torch.as_tensor(joint_norms, dtype=torch.float32, device=device).detach()
    representation_rows = next(iter(target_representations.values())).shape[0]
    latent_dim = remote.shape[1]
    if any(value.shape != (representation_rows, latent_dim) for value in target_representations.values()):
        raise ValueError("all target representations must share one shape")

    target_a = {
        target: torch.zeros(representation_rows, dtype=torch.float32, device=device)
        for target in range(n_clients)
    }
    target_b = {
        target: torch.zeros((representation_rows, latent_dim), dtype=torch.float32, device=device)
        for target in range(n_clients)
    }
    target_noise_a = {target: torch.zeros_like(value) for target, value in target_a.items()}
    target_noise_b = {target: torch.zeros_like(value) for target, value in target_b.items()}
    diagnostic_rows: list[dict[str, Any]] = []
    noise_std = float(noise_multiplier * clipping_threshold)
    block_size = representation_rows * (latent_dim + 1)

    for source in range(n_clients):
        source_indices = [
            index for index, record in enumerate(records) if int(record["client"]) == source
        ]
        if not source_indices:
            raise ValueError(f"source client {source} has no scheduled remote cells")
        source_remote = remote[source_indices]
        source_scales = scale_tensor[source_indices]
        target_order = [target for target in range(n_clients) if target != source]
        joint_blocks = []
        for target in target_order:
            weights = torch.exp(
                (target_representations[target] @ source_remote.T) / float(queue_temperature)
            ) * source_scales.unsqueeze(0)
            a = weights.sum(dim=1)
            b = weights @ source_remote
            joint_blocks.extend([a.reshape(-1), b.reshape(-1)])
        joint_signal = torch.cat(joint_blocks).detach()
        if joint_signal.numel() != len(target_order) * block_size:
            raise RuntimeError("unexpected joint A/B release dimension")
        standard_noise_seed = int(
            (seed * 1_000_003 + round_number * 1_009 + source * 97 + 7_919)
            % (2**63 - 1)
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(standard_noise_seed)
        standard_noise = torch.randn(
            joint_signal.shape,
            dtype=joint_signal.dtype,
            device=device,
            generator=generator,
        )
        joint_noise = standard_noise * noise_std
        joint_noisy = joint_signal + joint_noise
        offset = 0
        for target in target_order:
            a_end = offset + representation_rows
            b_end = a_end + representation_rows * latent_dim
            block_a = joint_noisy[offset:a_end]
            block_b = joint_noisy[a_end:b_end].reshape(representation_rows, latent_dim)
            noise_a = joint_noise[offset:a_end]
            noise_b = joint_noise[a_end:b_end].reshape(representation_rows, latent_dim)
            target_a[target] += block_a
            target_b[target] += block_b
            target_noise_a[target] += noise_a
            target_noise_b[target] += noise_b
            offset = b_end
        source_norms = norm_tensor[source_indices]
        diagnostic_rows.append(
            {
                "record_type": "joint_source_release",
                "round": int(round_number),
                "source_client": int(source),
                "target_client": -1,
                "joint_release_id": f"round_{round_number}_source_{source}",
                "remote_cells": int(len(source_indices)),
                "targets_covered": int(n_clients - 1),
                "joint_output_dimension": int(joint_signal.numel()),
                "clipping_threshold": float(clipping_threshold),
                "noise_multiplier": float(noise_multiplier),
                "noise_std": noise_std,
                "standard_noise_seed": standard_noise_seed,
                "standard_noise_l2": float(torch.linalg.vector_norm(standard_noise).cpu()),
                "noise_l2": float(torch.linalg.vector_norm(joint_noise).cpu()),
                "signal_l2": float(torch.linalg.vector_norm(joint_signal).cpu()),
                "noisy_signal_l2": float(torch.linalg.vector_norm(joint_noisy).cpu()),
                "fraction_clipped": float((source_scales < 1.0).float().mean().cpu()),
                "median_clipping_scale": float(source_scales.median().cpu()),
                "p10_clipping_scale": float(torch.quantile(source_scales, 0.10).cpu()),
                "p10_joint_norm": float(torch.quantile(source_norms, 0.10).cpu()),
                "median_joint_norm": float(source_norms.median().cpu()),
                "p90_joint_norm": float(torch.quantile(source_norms, 0.90).cpu()),
                "invalid_a_before_clamp_count": 0,
                "a_count": 0,
                "invalid_a_frequency": 0.0,
                "positive_a_floor": float(positive_a_floor),
            }
        )

    endpoints = {}
    for target in range(n_clients):
        invalid = target_a[target] <= 0
        invalid_count = int(invalid.sum().cpu())
        a_count = int(target_a[target].numel())
        clamped_a = target_a[target].clamp_min(positive_a_floor).detach()
        detached_b = target_b[target].detach()
        target_indices = [
            index for index, record in enumerate(records) if int(record["client"]) != target
        ]
        target_scales = scale_tensor[target_indices]
        target_norms = norm_tensor[target_indices]
        total_noise = torch.cat(
            [target_noise_a[target].reshape(-1), target_noise_b[target].reshape(-1)]
        )
        endpoint_diagnostics = {
            "remote_candidate_cells": float(len(target_indices)),
            "u_norm_mean": float(target_norms.mean().cpu()),
            "u_norm_median": float(target_norms.median().cpu()),
            "clipping_fraction": float((target_scales < 1.0).float().mean().cpu()),
            "clipping_scale_mean": float(target_scales.mean().cpu()),
            "clipping_scale_median": float(target_scales.median().cpu()),
            "clipping_scale_p10": float(torch.quantile(target_scales, 0.10).cpu()),
            "a_remote_mean": float(clamped_a.mean().cpu()),
            "a_remote_median": float(clamped_a.median().cpu()),
            "b_remote_norm_mean": float(torch.linalg.vector_norm(detached_b, dim=1).mean().cpu()),
            "b_remote_norm_median": float(torch.linalg.vector_norm(detached_b, dim=1).median().cpu()),
            "noise_multiplier": float(noise_multiplier),
            "noise_std": float(math.sqrt(n_clients - 1) * noise_std),
            "noise_l2": float(torch.linalg.vector_norm(total_noise).cpu()),
            "invalid_a_count": float(invalid_count),
            "a_count": float(a_count),
            "a_invalid_fraction": float(invalid_count / a_count),
        }
        endpoints[target] = PrivateAggregateABQueue(
            clamped_a,
            detached_b,
            remote_cell_count=len(target_indices),
            diagnostics=endpoint_diagnostics,
        )
        diagnostic_rows.append(
            {
                "record_type": "target_postprocess",
                "round": int(round_number),
                "source_client": -1,
                "target_client": int(target),
                "joint_release_id": f"round_{round_number}_target_{target}",
                "remote_cells": int(len(target_indices)),
                "targets_covered": int(n_clients - 1),
                "joint_output_dimension": int(block_size),
                "clipping_threshold": float(clipping_threshold),
                "noise_multiplier": float(noise_multiplier),
                "noise_std": float(math.sqrt(n_clients - 1) * noise_std),
                "standard_noise_seed": -1,
                "standard_noise_l2": math.nan,
                "noise_l2": float(torch.linalg.vector_norm(total_noise).cpu()),
                "signal_l2": math.nan,
                "noisy_signal_l2": float(
                    torch.linalg.vector_norm(torch.cat([clamped_a, detached_b.reshape(-1)])).cpu()
                ),
                "fraction_clipped": float((target_scales < 1.0).float().mean().cpu()),
                "median_clipping_scale": float(target_scales.median().cpu()),
                "p10_clipping_scale": float(torch.quantile(target_scales, 0.10).cpu()),
                "p10_joint_norm": float(torch.quantile(target_norms, 0.10).cpu()),
                "median_joint_norm": float(target_norms.median().cpu()),
                "p90_joint_norm": float(torch.quantile(target_norms, 0.90).cpu()),
                "invalid_a_before_clamp_count": invalid_count,
                "a_count": a_count,
                "invalid_a_frequency": float(invalid_count / a_count),
                "positive_a_floor": float(positive_a_floor),
            }
        )
    return endpoints, diagnostic_rows


def _run_scheduled_ab_step(
    model,
    remote_queue,
    *,
    queue_temperature: float,
    queue_weight: float,
    max_entries: int,
    anchor_batch: dict[str, Any] | None = None,
    rng_state: tuple[torch.Tensor, torch.Tensor | None] | None = None,
) -> dict[str, Any]:
    """Run one extra validated A/B optimizer step after ordinary local training."""

    from .queue import PrivateABQueueLoss

    trainer = model.trainer
    if rng_state is not None:
        _restore_rng_state(rng_state, trainer.device)
    if remote_queue is None or len(remote_queue) == 0:
        return {"ab_step": False, "anchor_batch_size": 0, "loss": float("nan")}
    trainer.global_queue = remote_queue
    criterion = PrivateABQueueLoss(
        temperature=trainer.clr_temperature,
        beta=trainer.clr_beta,
        queue_weight=queue_weight,
        max_entries=max_entries,
    )
    criterion.queue_temperature = float(queue_temperature)
    trainer.clr_criterion = criterion
    batch = _take_anchor_batch(model, target_size=128) if anchor_batch is None else anchor_batch
    data_dict = {
        key: value.to(trainer.device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    inputs = data_dict["input"]
    domain_labels = data_dict.get("domain")
    class_labels = data_dict.get("class")
    covariate_keys = [key for key in data_dict if key not in {"input", "domain", "class", "idx"}]
    covariate_tensors = {key: data_dict[key] for key in covariate_keys}
    model.model.train()
    trainer.augment.train()
    trainer.optimizer.zero_grad()
    loss, *_ = trainer.forward_pass(inputs, class_labels, domain_labels, covariate_tensors)
    loss.backward()
    trainer.optimizer.step()
    diagnostics = criterion.consume_diagnostic_state()
    trainer.global_queue = None
    from concord.model.loss import NTXent_general

    trainer.clr_criterion = NTXent_general(trainer.clr_temperature, trainer.clr_beta)
    return {
        "ab_step": True,
        "anchor_batch_size": int(inputs.shape[0]),
        "loss": float(loss.detach().cpu()),
        **diagnostics,
    }


def _validate_active_client_schedule(
    schedule: list[list[int]] | None,
    *,
    n_rounds: int,
    n_clients: int,
) -> list[list[int]]:
    if schedule is None:
        return [list(range(n_clients)) for _ in range(n_rounds)]
    if len(schedule) != n_rounds:
        raise ValueError("active client schedule must contain one entry per round")
    normalized = []
    for round_clients in schedule:
        selected = [int(value) for value in round_clients]
        if not selected:
            raise ValueError("every round must have at least one active client")
        if len(selected) != len(set(selected)):
            raise ValueError("active client schedule contains duplicate clients")
        if any(value < 0 or value >= n_clients for value in selected):
            raise ValueError("active client schedule contains an out-of-range client")
        normalized.append(sorted(selected))
    return normalized


def _update_last_seen_memory(
    queue: list[np.ndarray],
    records: list[dict[str, Any]],
    fresh_embeddings: list[np.ndarray],
    fresh_records: list[dict[str, Any]],
    *,
    round_number: int,
    ttl_rounds: int,
    active_clients: set[int],
) -> tuple[list[np.ndarray], list[dict[str, Any]], int]:
    """Expire stale snapshots and replace active clients with current snapshots."""

    retained_embeddings = []
    retained_records = []
    expired = 0
    for embedding, record in zip(queue, records):
        source = int(record["client"])
        age = round_number - int(record["collected_round"])
        if source in active_clients:
            continue
        if age > ttl_rounds:
            expired += 1
            continue
        retained_embeddings.append(embedding)
        retained_records.append(record)
    retained_embeddings.extend(fresh_embeddings)
    retained_records.extend(fresh_records)
    return retained_embeddings, retained_records, expired


def _predict_queue_embeddings(model, adata, *, seed: int) -> np.ndarray:
    """Run queue inference without advancing the local-training RNG streams."""

    device = torch.device(model.config.device)
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        return predict_embeddings(model, adata)


def _collect_round_queue_diagnostics(models: list) -> dict[str, Any]:
    from .queue import summarize_queue_diagnostics

    states = []
    for model in models:
        criterion = getattr(getattr(model, "trainer", None), "clr_criterion", None)
        if hasattr(criterion, "consume_diagnostic_state"):
            states.append(criterion.consume_diagnostic_state())
        elif hasattr(criterion, "diagnostic_state"):
            states.append(criterion.diagnostic_state())
    return summarize_queue_diagnostics(states)


def _queue_is_active(
    enabled: bool,
    round_index: int,
    warmup_rounds: int,
    active_rounds: tuple[int, ...] | None = None,
) -> bool:
    if active_rounds is not None:
        return enabled and round_index + 1 in active_rounds
    return enabled and round_index >= warmup_rounds


def _fresh_queue_should_collect(
    enabled: bool,
    round_index: int,
    active_rounds: tuple[int, ...] | None,
) -> bool:
    """Collect after this round only when the next round will consume the queue."""

    if not enabled:
        return False
    if active_rounds is None:
        return True
    return round_index + 2 in active_rounds


def _store_round_queue(
    queue: list[np.ndarray],
    records: list[dict[str, Any]],
    round_embeddings: list[np.ndarray],
    round_records: list[dict[str, Any]],
    *,
    replace: bool,
    maximum_size: int,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    if replace:
        return list(round_embeddings), list(round_records)
    updated_queue = [*queue, *round_embeddings]
    updated_records = [*records, *round_records]
    if len(updated_queue) > maximum_size:
        updated_queue = updated_queue[-maximum_size:]
        updated_records = updated_records[-maximum_size:]
    return updated_queue, updated_records


def train_federated(
    adata,
    features: list[str],
    client_batches: list[list] | None,
    config: dict,
    output_dir: Path,
    *,
    domain_key: str | None = "batch",
    client_indices: list[np.ndarray] | None = None,
    round_callback: Callable[[int, Any, dict[str, Any]], None] | None = None,
    diagnostic_callback: Callable[[FederatedRoundSnapshot], None] | None = None,
    client_batch_sizes: list[int] | None = None,
    allow_partial_client_indices: bool = False,
    active_client_schedule: list[list[int]] | None = None,
    contrastive_audit_callback: Callable[..., None] | None = None,
) -> FederatedResult:
    import concord as ccd

    fed = config["federated"]
    queue_config = fed["queue"]
    queue_mode = queue_config.get("mode", "legacy_fifo")
    if queue_mode not in {
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
        "cross_client_topology",
        "round_once_ab",
        "epoch_once_ab",
        "round_once_joint_cellclip",
        "round_once_joint_cellclip_noise",
    }:
        raise ValueError(f"unknown queue mode: {queue_mode}")
    fresh_queue = queue_mode in {
        "fresh_cross_client",
        "fresh_cross_client_ab",
        "fresh_cross_client_cellclip",
        "fresh_cross_client_cellclip_noise",
        "reliable_negative_fresh_cross_client",
        "random_mask_fresh_cross_client",
        "soft_reliability_fresh_cross_client",
        "random_soft_weight_fresh_cross_client",
        "cross_client_topology",
    }
    fresh_local_memory = queue_mode == "fresh_local_memory"
    scheduled_ab = queue_mode in {
        "round_once_ab",
        "epoch_once_ab",
        "round_once_joint_cellclip",
        "round_once_joint_cellclip_noise",
    }
    joint_cellclip = queue_mode in {
        "round_once_joint_cellclip",
        "round_once_joint_cellclip_noise",
    }
    joint_gaussian = queue_mode == "round_once_joint_cellclip_noise"
    scheduled_ab_slots = 1 if queue_mode in {
        "round_once_ab",
        "round_once_joint_cellclip",
        "round_once_joint_cellclip_noise",
    } else int(fed["local_epochs"])
    fresh_memory = fresh_queue or fresh_local_memory
    last_seen_queue = queue_mode == "last_seen_fresh_cross_client"
    configured_active_rounds = queue_config.get("active_rounds")
    active_rounds = (
        None
        if configured_active_rounds is None
        else tuple(int(value) for value in configured_active_rounds)
    )
    if active_rounds is not None and (
        not active_rounds
        or any(value < 1 for value in active_rounds)
        or tuple(sorted(set(active_rounds))) != active_rounds
    ):
        raise ValueError("queue active_rounds must be a non-empty sorted unique list")
    cross_client_only = bool(
        queue_config.get("cross_client_only", queue_mode == "cross_client_soft")
    )
    collect_after_aggregation = (
        queue_config.get("collection", "before_fedavg_with_local_encoder")
        == "after_fedavg_with_aggregated_encoder"
    )
    if queue_config["enabled"] and queue_mode == "cross_client_soft":
        from .queue import install_cross_client_soft_queue_loss

        install_cross_client_soft_queue_loss(queue_config.get("temperature", 0.2))
    elif queue_config["enabled"] and queue_mode in {
        "reliable_negative_fresh_cross_client",
        "random_mask_fresh_cross_client",
    }:
        from .queue import install_reliable_negative_queue_loss

        install_reliable_negative_queue_loss(
            queue_config.get("temperature", 0.2),
            neighborhood_k=int(queue_config.get("neighborhood_k", 20)),
            similarity_fraction=float(queue_config.get("similarity_fraction", 0.10)),
            consistency_fraction=float(queue_config.get("consistency_fraction", 0.30)),
            random_control=queue_mode == "random_mask_fresh_cross_client",
        )
    elif queue_config["enabled"] and queue_mode in {
        "soft_reliability_fresh_cross_client",
        "random_soft_weight_fresh_cross_client",
    }:
        from .queue import install_soft_reliability_queue_loss

        install_soft_reliability_queue_loss(
            queue_config.get("temperature", 0.2),
            neighborhood_k=int(queue_config.get("neighborhood_k", 20)),
            reliability_beta=float(queue_config.get("reliability_beta", 0.5)),
            random_control=queue_mode == "random_soft_weight_fresh_cross_client",
        )
    elif queue_config["enabled"] and queue_mode == "cross_client_topology":
        from .queue import install_cross_client_topology_queue_loss

        install_cross_client_topology_queue_loss(
            queue_config.get("temperature", 0.2),
            topology_top_k=int(queue_config.get("topology_top_k", 20)),
            topology_temperature=float(
                queue_config.get("topology_temperature", 0.2)
            ),
            local_relation_temperature=float(
                queue_config.get("local_relation_temperature", 0.2)
            ),
        )
    elif queue_config["enabled"] and queue_mode in {
        "fresh_cross_client_ab",
        "fresh_cross_client_cellclip",
        "fresh_cross_client_cellclip_noise",
        "round_once_ab",
        "epoch_once_ab",
        "round_once_joint_cellclip",
        "round_once_joint_cellclip_noise",
    }:
        from .queue import install_private_ab_queue_loss

        install_private_ab_queue_loss(
            queue_config.get("temperature", 0.2),
            queue_config.get("ab_max_entries"),
        )
    elif queue_config["enabled"] and (fresh_memory or last_seen_queue):
        from .queue import install_fresh_cross_client_queue_loss

        install_fresh_cross_client_queue_loss(
            queue_config.get("temperature", 0.2),
            queue_config.get("mask_fraction", 0.05),
            queue_config.get("negative_ratio"),
        )
    device = resolve_device(config["device"])
    seed = int(config["seed"])
    set_reproducible_seed(seed)
    queue_rng = np.random.default_rng(seed)
    ccd.set_verbose_mode(False)

    if (client_batches is None) == (client_indices is None):
        raise ValueError("provide exactly one of client_batches or client_indices")
    if client_indices is None:
        clients = split_clients(adata, client_batches)
        resolved_client_batches = [list(group) for group in client_batches]
    else:
        clients = split_client_indices(
            adata, client_indices, allow_partial=allow_partial_client_indices
        )
        resolved_client_batches = [list(client.obs["batch"].unique()) for client in clients]
    if len(clients) != fed["n_clients"]:
        raise ValueError(
            f"configuration requests {fed['n_clients']} clients, received {len(clients)}"
        )
    resolved_active_schedule = _validate_active_client_schedule(
        active_client_schedule,
        n_rounds=int(fed["n_rounds"]),
        n_clients=len(clients),
    )
    memory_ttl_rounds = int(queue_config.get("ttl_rounds", 2))
    if last_seen_queue and memory_ttl_rounds < 0:
        raise ValueError("last-seen queue TTL must be non-negative")
    if client_batch_sizes is None:
        client_batch_sizes = [int(fed["batch_size"])] * len(clients)
    if len(client_batch_sizes) != len(clients):
        raise ValueError("one client batch size is required per client")
    if any(size <= 0 for size in client_batch_sizes):
        raise ValueError("client batch sizes must be positive")
    if any(size > client.n_obs for size, client in zip(client_batch_sizes, clients)):
        raise ValueError("client batch size cannot exceed its number of unique cells")
    models = []
    for index, (client, client_batch_size) in enumerate(zip(clients, client_batch_sizes)):
        model = ccd.Concord(
            adata=client,
            input_feature=features,
            seed=seed,
            device=device,
            save_dir=output_dir / f"client_{index}",
            n_epochs=fed["local_epochs"],
            batch_size=int(client_batch_size),
            lr=fed["learning_rate"],
            clr_temperature=fed["temperature"],
            clr_beta=fed["hard_negative_beta"],
            latent_dim=fed["latent_dim"],
            encoder_dims=fed["encoder_dims"],
            dropout_prob=fed["dropout"],
            domain_key=domain_key,
            preload_dense=True,
            verbose=False,
            global_queue=None,
            queue_weight=queue_config["weight"],
        )
        model.init_model()
        models.append(model)

    global_queue: list[np.ndarray] = []
    queue_records: list[dict[str, Any]] = []
    scheduled_remote_indices = queue_config.get("_scheduled_remote_indices", {})
    queue_sizes: list[int] = []
    queue_diagnostics: list[dict[str, Any]] = []
    local_epoch_diagnostics: list[dict[str, Any]] = []
    client_resource_diagnostics: list[dict[str, Any]] = []
    local_loss_diagnostics: list[dict[str, Any]] = []
    client_queue_diagnostics: list[dict[str, Any]] = []
    communication_diagnostics: list[dict[str, Any]] = []
    joint_clipping_diagnostics: list[dict[str, Any]] = []
    joint_noise_diagnostics: list[dict[str, Any]] = []
    skipped_keys: set[str] = set()
    model_payload_bytes = _state_payload_bytes(models[0])
    start = time.perf_counter()

    for round_index in range(fed["n_rounds"]):
        active_client_indices = resolved_active_schedule[round_index]
        active_client_set = set(active_client_indices)
        queue_active = _queue_is_active(
            queue_config["enabled"],
            round_index,
            int(queue_config.get("warmup_rounds", 0)),
            active_rounds,
        )
        collect_fresh_queue = fresh_memory and _fresh_queue_should_collect(
            queue_config["enabled"], round_index, active_rounds
        )
        collect_scheduled_queue = scheduled_ab and (
            (round_index + 1) in {int(key) for key in scheduled_remote_indices}
        )
        round_embeddings = []
        round_records = []
        expired_entries = 0
        if queue_config["enabled"] and last_seen_queue:
            round_embeddings, round_records = _collect_fresh_queue(
                models,
                clients,
                queue_rng,
                queue_config["samples_per_client"],
                round_index + 1,
                seed,
                client_indices=active_client_indices,
                neighborhood_k=int(queue_config.get("neighborhood_k", 20)),
            )
            global_queue, queue_records, expired_entries = _update_last_seen_memory(
                global_queue,
                queue_records,
                round_embeddings,
                round_records,
                round_number=round_index + 1,
                ttl_rounds=memory_ttl_rounds,
                active_clients=active_client_set,
            )
        if scheduled_ab:
            delivered_sizes = [0] * len(models)
        elif queue_active and (fresh_queue or last_seen_queue):
            queue_interface = (
                "ab"
                if queue_mode == "fresh_cross_client_ab"
                else "cellclip_noise"
                if queue_mode == "fresh_cross_client_cellclip_noise"
                else "cellclip"
                if queue_mode == "fresh_cross_client_cellclip"
                else "explicit"
            )
            delivered_sizes = _inject_fresh_client_queues(
                models,
                global_queue,
                queue_records,
                device,
                seed=seed,
                round_number=round_index + 1,
                active_client_indices=(active_client_set if last_seen_queue else None),
                interface=queue_interface,
                clipping_threshold=queue_config.get("fixed_clipping_threshold"),
                noise_multiplier=float(queue_config.get("noise_multiplier", 0.0)),
            )
        elif queue_active and fresh_local_memory:
            delivered_sizes = _inject_fresh_local_memories(
                models,
                global_queue,
                queue_records,
                device,
                seed=seed,
                round_number=round_index + 1,
            )
        elif queue_active:
            delivered_sizes = _inject_queue(
                models, global_queue, queue_records, device, cross_client_only=cross_client_only
            )
        else:
            delivered_sizes = _inject_queue(
                models, [], [], device, cross_client_only=False
            )

        local_states = []
        sample_counts = []
        scheduled_interaction_rows = []
        joint_anchor_batches: dict[int, dict[str, Any]] = {}
        joint_anchor_representations: dict[int, torch.Tensor] = {}
        joint_rng_states: dict[int, tuple[torch.Tensor, torch.Tensor | None]] = {}
        for client_index, (model, client) in enumerate(zip(models, clients)):
            if client_index not in active_client_set:
                continue
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                gpu_allocated_start = int(torch.cuda.memory_allocated(device))
                gpu_reserved_start = int(torch.cuda.memory_reserved(device))
            else:
                gpu_allocated_start = 0
                gpu_reserved_start = 0
            epoch_started = time.perf_counter()

            def record_local_epoch(epoch, seen_indices, *, current_client=client_index):
                nonlocal epoch_started
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    allocated = int(torch.cuda.memory_allocated(device))
                    reserved = int(torch.cuda.memory_reserved(device))
                else:
                    allocated = 0
                    reserved = 0
                epoch_finished = time.perf_counter()
                local_epoch_diagnostics.append(
                    {
                        "round": round_index + 1,
                        "client": current_client,
                        "local_epoch": int(epoch) + 1,
                        "unique_local_cells_seen": len(set(seen_indices)),
                        "epoch_wall_seconds": epoch_finished - epoch_started,
                        "gpu_allocated_bytes_epoch_end": allocated,
                        "gpu_reserved_bytes_epoch_end": reserved,
                    }
                )
                epoch_started = epoch_finished

            client_started = time.perf_counter()
            if contrastive_audit_callback is not None:
                model.contrastive_audit_callback = (
                    lambda z_i, z_j, queue, current_round=round_index + 1,
                    current_client=client_index: contrastive_audit_callback(
                        current_round, current_client, z_i, z_j, queue
                    )
                )
            model.train(save_model=False, epoch_callback=record_local_epoch)
            for epoch_metrics in model.trainer.epoch_metrics[-int(fed["local_epochs"]):]:
                local_loss_diagnostics.append(
                    {
                        "round": round_index + 1,
                        "client": client_index,
                        **epoch_metrics,
                    }
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak_allocated = int(torch.cuda.max_memory_allocated(device))
                peak_reserved = int(torch.cuda.max_memory_reserved(device))
            else:
                peak_allocated = 0
                peak_reserved = 0
            client_resource_diagnostics.append(
                {
                    "round": round_index + 1,
                    "client": client_index,
                    "client_training_wall_seconds": time.perf_counter() - client_started,
                    "gpu_allocated_bytes_start": gpu_allocated_start,
                    "gpu_reserved_bytes_start": gpu_reserved_start,
                    "peak_gpu_allocated_bytes": peak_allocated,
                    "peak_gpu_reserved_bytes": peak_reserved,
                    "incremental_peak_gpu_allocated_bytes": max(
                        0, peak_allocated - gpu_allocated_start
                    ),
                }
            )
            if joint_cellclip and queue_active and global_queue:
                joint_anchor_batches[client_index] = _take_anchor_batch(model, target_size=128)
                joint_rng_states[client_index] = _capture_rng_state(device)
                joint_anchor_representations[client_index] = _anchor_representation(
                    model, joint_anchor_batches[client_index]
                ).detach()
            if scheduled_ab and not joint_cellclip and queue_active and global_queue:
                for interaction_slot in range(scheduled_ab_slots):
                    endpoint = _scheduled_private_queue(
                        global_queue,
                        queue_records,
                        target_client=client_index,
                        interaction_slot=interaction_slot,
                        device=device,
                        seed=seed,
                        round_number=round_index + 1,
                    )
                    step_result = _run_scheduled_ab_step(
                        model,
                        endpoint,
                        queue_temperature=float(queue_config.get("temperature", 0.2)),
                        queue_weight=float(queue_config.get("weight", 0.1)),
                        max_entries=int(queue_config.get("ab_max_entries", len(endpoint) if endpoint is not None else 128)),
                    )
                    scheduled_interaction_rows.append(
                        {
                            "round": round_index + 1,
                            "client": client_index,
                            "interaction_slot": interaction_slot,
                            "remote_queue_entries": 0 if endpoint is None else len(endpoint),
                            **step_result,
                        }
                    )
            local_states.append(
                {key: value.detach().clone() for key, value in model.model.state_dict().items()}
            )
            sample_counts.append(client.n_obs)
            if (
                queue_config["enabled"]
                and not fresh_memory
                and not last_seen_queue
                and not collect_after_aggregation
                and not scheduled_ab
            ):
                embeddings = _predict_queue_embeddings(
                    model,
                    client,
                    seed=seed * 1_000_003 + (round_index + 1) * 1_009 + client_index,
                )
                n_samples = min(queue_config["samples_per_client"], embeddings.shape[0])
                selected = queue_rng.choice(embeddings.shape[0], size=n_samples, replace=False)
                round_embeddings.extend(embeddings[selected])
                for local_index in selected:
                    round_records.append(
                        {
                            "cell_id": str(client.obs_names[local_index]),
                            "client": client_index,
                            "batch": str(client.obs["batch"].iloc[local_index]),
                            "collected_round": round_index + 1,
                        }
                    )

        if joint_cellclip and queue_active and global_queue:
            fixed_scales, clipping_rows = _joint_remote_scales(
                joint_anchor_representations,
                global_queue,
                queue_records,
                queue_temperature=float(queue_config.get("temperature", 0.2)),
                clipping_threshold=float(queue_config["joint_clipping_threshold"]),
                round_number=round_index + 1,
            )
            joint_clipping_diagnostics.extend(clipping_rows)
            noisy_endpoints = None
            if joint_gaussian:
                noisy_endpoints, noise_rows = _joint_noisy_ab_endpoints(
                    joint_anchor_representations,
                    global_queue,
                    queue_records,
                    fixed_scales,
                    [float(row["joint_norm"]) for row in clipping_rows],
                    queue_temperature=float(queue_config.get("temperature", 0.2)),
                    clipping_threshold=float(queue_config["joint_clipping_threshold"]),
                    noise_multiplier=float(queue_config["noise_multiplier"]),
                    seed=int(queue_config.get("_joint_noise_seed", seed)),
                    round_number=round_index + 1,
                    positive_a_floor=float(queue_config.get("positive_a_floor", 1e-6)),
                )
                joint_noise_diagnostics.extend(noise_rows)
            post_precompute_rng_state = _capture_rng_state(device)
            for client_index, model in enumerate(models):
                endpoint = (
                    noisy_endpoints[client_index]
                    if noisy_endpoints is not None
                    else _scheduled_private_queue(
                        global_queue,
                        queue_records,
                        target_client=client_index,
                        interaction_slot=0,
                        device=device,
                        seed=seed,
                        round_number=round_index + 1,
                        fixed_scales=fixed_scales,
                    )
                )
                step_result = _run_scheduled_ab_step(
                    model,
                    endpoint,
                    queue_temperature=float(queue_config.get("temperature", 0.2)),
                    queue_weight=float(queue_config.get("weight", 0.1)),
                    max_entries=int(queue_config.get("ab_max_entries", len(endpoint) if endpoint is not None else 128)),
                    anchor_batch=joint_anchor_batches[client_index],
                    rng_state=joint_rng_states[client_index],
                )
                _restore_rng_state(post_precompute_rng_state, device)
                scheduled_interaction_rows.append(
                    {
                        "round": round_index + 1,
                        "client": client_index,
                        "interaction_slot": 0,
                        "remote_queue_entries": 0 if endpoint is None else len(endpoint),
                        **step_result,
                    }
                )
            # The joint A/B step occurs after all five anchor batches are
            # available; refresh the local states so this extra optimizer step
            # is included in FedAvg.
            local_states = [
                {key: value.detach().clone() for key, value in model.model.state_dict().items()}
                for model in models
            ]
            sample_counts = [client.n_obs for client in clients]

        averaged, skipped = aggregate_state_dicts(local_states, sample_counts)
        client_round_states = []
        for client_index, model in enumerate(models):
            criterion = getattr(getattr(model, "trainer", None), "clr_criterion", None)
            if hasattr(criterion, "consume_diagnostic_state"):
                state = criterion.consume_diagnostic_state()
                client_queue_diagnostics.append(
                    {
                        "round": round_index + 1,
                        "client": client_index,
                        **state,
                    }
                )
                client_round_states.append(state)
        from .queue import summarize_queue_diagnostics

        round_diagnostics = summarize_queue_diagnostics(client_round_states)
        skipped_keys.update(skipped)
        distribute_state(models, averaged)

        if collect_fresh_queue:
            round_embeddings, round_records = _collect_fresh_queue(
                models,
                clients,
                queue_rng,
                queue_config["samples_per_client"],
                round_index + 1,
                seed,
                client_indices=active_client_indices,
                neighborhood_k=int(queue_config.get("neighborhood_k", 20)),
            )
        elif collect_scheduled_queue:
            round_embeddings, round_records = _collect_scheduled_queue(
                models,
                clients,
                scheduled_remote_indices,
                collection_round=round_index + 1,
                seed=seed,
                neighborhood_k=int(queue_config.get("neighborhood_k", 20)),
            )
        elif (
            queue_config["enabled"]
            and not fresh_memory
            and not last_seen_queue
            and collect_after_aggregation
            and not scheduled_ab
        ):
            for client_index, (model, client) in enumerate(zip(models, clients)):
                embeddings = _predict_queue_embeddings(
                    model,
                    client,
                    seed=seed * 1_000_003 + (round_index + 1) * 1_009 + client_index,
                )
                n_samples = min(queue_config["samples_per_client"], embeddings.shape[0])
                selected = queue_rng.choice(embeddings.shape[0], size=n_samples, replace=False)
                round_embeddings.extend(embeddings[selected])
                for local_index in selected:
                    round_records.append(
                        {
                            "cell_id": str(client.obs_names[local_index]),
                            "client": client_index,
                            "batch": str(client.obs["batch"].iloc[local_index]),
                            "collected_round": round_index + 1,
                        }
                    )

        if queue_config["enabled"] and last_seen_queue:
            pass
        elif queue_config["enabled"] and scheduled_ab and collect_scheduled_queue:
            global_queue, queue_records = list(round_embeddings), list(round_records)
        elif queue_config["enabled"] and fresh_memory and not collect_fresh_queue:
            global_queue, queue_records = [], []
        elif queue_config["enabled"]:
            global_queue, queue_records = _store_round_queue(
                global_queue,
                queue_records,
                round_embeddings,
                round_records,
                replace=fresh_memory,
                maximum_size=queue_config["size"],
            )
        queue_upload_bytes = (
            0
            if fresh_local_memory
            else int(sum(np.asarray(value).nbytes for value in round_embeddings))
        )
        queue_download_bytes = (
            0
            if fresh_local_memory
            else int(sum(delivered_sizes) * int(fed["latent_dim"]) * 4)
        )
        model_upload_bytes = int(len(active_client_indices) * model_payload_bytes)
        model_download_bytes = int(len(active_client_indices) * model_payload_bytes)
        memory_ages = [
            round_index + 1 - int(record["collected_round"])
            for record in queue_records
        ]
        active_source_entries = sum(
            int(record["client"]) in active_client_set for record in queue_records
        )
        inactive_source_entries = len(queue_records) - active_source_entries
        queue_sizes.append(len(global_queue))
        queue_diagnostics.append(
            {
                "round": round_index + 1,
                "active_clients": active_client_indices,
                "active_client_count": len(active_client_indices),
                "inactive_client_count": len(clients) - len(active_client_indices),
                "queue_active": queue_active and (any(delivered_sizes) or bool(scheduled_interaction_rows)),
                "configured_alpha": (
                    float(queue_config["weight"])
                    if queue_active and (any(delivered_sizes) or bool(scheduled_interaction_rows))
                    else 0.0
                ),
                "stored_queue_size": len(global_queue),
                "delivered_queue_sizes": delivered_sizes,
                "uploaded_queue_entries": len(round_embeddings),
                "active_source_queue_embeddings": active_source_entries,
                "inactive_last_seen_queue_embeddings": inactive_source_entries,
                "memory_age_0_entries": int(sum(age == 0 for age in memory_ages)),
                "memory_age_1_entries": int(sum(age == 1 for age in memory_ages)),
                "memory_age_2_entries": int(sum(age == 2 for age in memory_ages)),
                "memory_age_mean": float(np.mean(memory_ages)) if memory_ages else np.nan,
                "memory_age_max": int(max(memory_ages)) if memory_ages else np.nan,
                "expired_entries": expired_entries,
                "uploaded_embedding_bytes": queue_upload_bytes,
                "downloaded_queue_entries": sum(delivered_sizes),
                "downloaded_embedding_bytes": queue_download_bytes,
                "scheduled_ab_interactions": len(scheduled_interaction_rows),
                "scheduled_ab_remote_entries": int(
                    sum(row["remote_queue_entries"] for row in scheduled_interaction_rows)
                ),
                **round_diagnostics,
            }
        )
        communication_diagnostics.append(
            {
                "round": round_index + 1,
                "model_tensor_payload_bytes_per_client": model_payload_bytes,
                "model_upload_bytes": model_upload_bytes,
                "model_download_bytes": model_download_bytes,
                "model_communication_bytes": model_upload_bytes + model_download_bytes,
                "embedding_upload_bytes": queue_upload_bytes,
                "embedding_download_bytes": queue_download_bytes,
                "embedding_communication_bytes": queue_upload_bytes
                + queue_download_bytes,
                "total_communication_bytes": model_upload_bytes
                + model_download_bytes
                + queue_upload_bytes
                + queue_download_bytes,
            }
        )
        if diagnostic_callback is not None:
            queue_array = (
                np.asarray(global_queue, dtype=np.float32).copy()
                if global_queue
                else np.empty((0, fed["latent_dim"]), dtype=np.float32)
            )
            queue_array.setflags(write=False)
            diagnostic_callback(
                FederatedRoundSnapshot(
                    round_number=round_index + 1,
                    models=tuple(models),
                    clients=tuple(clients),
                    queue_embeddings=queue_array,
                    queue_records=tuple(dict(record) for record in queue_records),
                    queue_was_active=bool(queue_active and any(delivered_sizes)),
                )
            )
        if round_callback is not None:
            round_callback(round_index + 1, models[0], queue_diagnostics[-1])
        print(
            f"round {round_index + 1:02d}/{fed['n_rounds']}: "
            f"queue={len(global_queue)}, delivered={delivered_sizes}, "
            f"active_clients={active_client_indices}"
        )

    training_seconds = time.perf_counter() - start
    global_model = models[0]
    embeddings = predict_embeddings(global_model, adata)
    return FederatedResult(
        embeddings=embeddings,
        model=global_model,
        client_batches=resolved_client_batches,
        client_cells=[client.n_obs for client in clients],
        queue_sizes=queue_sizes,
        final_queue_embeddings=(
            np.asarray(global_queue, dtype=np.float32)
            if global_queue
            else np.empty((0, fed["latent_dim"]), dtype=np.float32)
        ),
        final_queue_records=queue_records,
        skipped_state_keys=sorted(skipped_keys),
        training_seconds=training_seconds,
        queue_diagnostics=queue_diagnostics,
        local_epoch_diagnostics=local_epoch_diagnostics,
        client_resource_diagnostics=client_resource_diagnostics,
        local_loss_diagnostics=local_loss_diagnostics,
        client_queue_diagnostics=client_queue_diagnostics,
        communication_diagnostics=communication_diagnostics,
        joint_clipping_diagnostics=joint_clipping_diagnostics,
        joint_noise_diagnostics=joint_noise_diagnostics,
    )


def save_checkpoint(
    path: Path,
    result: FederatedResult,
    features: list[str],
    config: dict,
    *,
    domain_key: str | None,
) -> None:
    fed = config["federated"]
    payload = {
        "format_version": 1,
        "state_dict": {
            key: value.detach().cpu() for key, value in result.model.model.state_dict().items()
        },
        "features": list(features),
        "model": {
            "latent_dim": fed["latent_dim"],
            "encoder_dims": copy.deepcopy(fed["encoder_dims"]),
            "dropout": fed["dropout"],
            "domain_key": domain_key,
        },
        "training": {
            "protocol": config["protocol"],
            "seed": config["seed"],
            "client_batches": result.client_batches,
            "queue_sizes": result.queue_sizes,
        },
        "resolved_config": {
            key: copy.deepcopy(value) for key, value in config.items() if not key.startswith("_")
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
