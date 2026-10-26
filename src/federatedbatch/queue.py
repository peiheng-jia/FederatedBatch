"""Queue losses and diagnostics for FederatedBatch experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SIMILARITY_HISTOGRAM_BINS = 40
RELIABILITY_HISTOGRAM_BINS = 1000


@dataclass(frozen=True)
class FreshClientQueue:
    """One client's fresh cross-client queue with source-client metadata."""

    embeddings: torch.Tensor
    source_clients: torch.Tensor
    sampling_seed: int = 0
    neighborhood_centroids: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError("queue embeddings must be a two-dimensional tensor")
        if self.source_clients.ndim != 1:
            raise ValueError("queue source clients must be a one-dimensional tensor")
        if len(self.embeddings) != len(self.source_clients):
            raise ValueError("queue embeddings and source clients must have equal length")
        if self.embeddings.requires_grad:
            raise ValueError("queue embeddings must be detached")
        if self.neighborhood_centroids is not None:
            if self.neighborhood_centroids.shape != self.embeddings.shape:
                raise ValueError("queue neighborhood centroids must match embeddings")
            if self.neighborhood_centroids.requires_grad:
                raise ValueError("queue neighborhood centroids must be detached")

    def __len__(self) -> int:
        return len(self.embeddings)


class PrivateRemoteABQueue:
    """Plaintext simulation of a private remote A/B aggregation endpoint.

    The query-client loss receives this handle and only obtains aggregate A/B
    statistics. Remote embeddings stay inside ``aggregate`` and are never
    returned to the loss or written to diagnostics.
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        source_clients: torch.Tensor,
        sampling_seed: int,
        *,
        clipping_threshold: float | None = None,
        noise_multiplier: float = 0.0,
        fixed_scales: torch.Tensor | None = None,
    ) -> None:
        if embeddings.ndim != 2 or source_clients.ndim != 1:
            raise ValueError("private remote queue has invalid dimensions")
        if len(embeddings) != len(source_clients):
            raise ValueError("private remote queue metadata length mismatch")
        if embeddings.requires_grad or source_clients.requires_grad:
            raise ValueError("private remote queue inputs must be detached")
        if clipping_threshold is not None and clipping_threshold <= 0:
            raise ValueError("clipping threshold must be positive")
        if noise_multiplier < 0 or not math.isfinite(noise_multiplier):
            raise ValueError("noise multiplier must be finite and non-negative")
        if noise_multiplier > 0 and clipping_threshold is None:
            raise ValueError("Gaussian noise requires a fixed clipping threshold")
        if fixed_scales is not None:
            if fixed_scales.ndim != 1 or len(fixed_scales) != len(embeddings):
                raise ValueError("fixed clipping scales must align with remote embeddings")
            if fixed_scales.requires_grad:
                raise ValueError("fixed clipping scales must be detached")
            if torch.any(fixed_scales <= 0) or not torch.isfinite(fixed_scales).all():
                raise ValueError("fixed clipping scales must be finite and positive")
        self.embeddings = embeddings.detach()
        self.source_clients = source_clients.detach()
        self.sampling_seed = int(sampling_seed)
        self.clipping_threshold = clipping_threshold
        self.noise_multiplier = float(noise_multiplier)
        self.fixed_scales = None if fixed_scales is None else fixed_scales.detach()
        self._sampling_generator: torch.Generator | None = None
        self._sampling_seed: int | None = None
        self._noise_generator: torch.Generator | None = None
        self._noise_seed: int | None = None
        self.last_diagnostics: dict[str, float] = {}

    def __len__(self) -> int:
        return len(self.embeddings)

    def _get_sampling_generator(self) -> torch.Generator:
        if self._sampling_generator is None or self._sampling_seed != self.sampling_seed:
            self._sampling_generator = torch.Generator(device=self.embeddings.device)
            self._sampling_generator.manual_seed(self.sampling_seed)
            self._sampling_seed = self.sampling_seed
        return self._sampling_generator

    def sample(self, max_entries: int) -> FreshClientQueue:
        queue = FreshClientQueue(
            self.embeddings,
            self.source_clients,
            self.sampling_seed,
        )
        return balanced_queue_sample(queue, max_entries, self._get_sampling_generator())

    def _get_noise_generator(self) -> torch.Generator:
        """Use an independent stream so Gaussian noise cannot perturb augmentation RNG."""

        if self._noise_generator is None or self._noise_seed != self.sampling_seed:
            self._noise_generator = torch.Generator(device=self.embeddings.device)
            self._noise_generator.manual_seed(
                (self.sampling_seed + 0x5DEECE66D) % (2**63 - 1)
            )
            self._noise_seed = self.sampling_seed
        return self._noise_generator

    def aggregate(
        self,
        representations: torch.Tensor,
        *,
        max_entries: int,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sampled = self.sample(max_entries)
        queue = F.normalize(sampled.embeddings, dim=1).detach()
        similarity = representations @ queue.T
        weights = torch.exp(similarity / temperature)
        scales = torch.ones(
            queue.shape[0], dtype=weights.dtype, device=weights.device
        )
        if self.fixed_scales is not None:
            if len(sampled) != len(self.embeddings):
                raise ValueError(
                    "joint fixed clipping requires the complete target remote endpoint; "
                    "increase max_entries to include every remote cell"
                )
            scales = self.fixed_scales.to(device=weights.device, dtype=weights.dtype)
        elif self.clipping_threshold is not None:
            # ||concat_b [w_bj, w_bj*q_j]|| is evaluated jointly over B.
            joint_norm = torch.sqrt(torch.square(weights).sum(dim=0) * 2.0)
            scales = torch.minimum(
                scales,
                torch.as_tensor(
                    self.clipping_threshold,
                    dtype=weights.dtype,
                    device=weights.device,
                )
                / joint_norm.clamp_min(torch.finfo(weights.dtype).tiny),
            )
        scaled_weights = weights * scales.unsqueeze(0)
        a = scaled_weights.sum(dim=1)
        b = scaled_weights @ queue
        noise_l2 = 0.0
        noise_std = 0.0
        if self.noise_multiplier > 0:
            noise_std = self.noise_multiplier * float(self.clipping_threshold)
            released = torch.cat([a.reshape(-1), b.reshape(-1)])
            noise = torch.randn(
                released.shape,
                dtype=released.dtype,
                device=released.device,
                generator=self._get_noise_generator(),
            ) * noise_std
            noise_l2 = float(torch.linalg.vector_norm(noise).detach().cpu())
            split = a.numel()
            a = a + noise[:split].reshape_as(a)
            b = b + noise[split:].reshape_as(b)
        invalid_a_count = int((a <= 0).sum().detach().cpu())
        a_count = int(a.numel())
        self.last_diagnostics = {
            "remote_candidate_cells": float(queue.shape[0]),
            "u_norm_mean": float(torch.sqrt(torch.square(weights).sum(dim=0) * 2.0).mean().detach().cpu()),
            "u_norm_median": float(torch.sqrt(torch.square(weights).sum(dim=0) * 2.0).median().detach().cpu()),
            "clipping_fraction": float((scales < 1.0).float().mean().detach().cpu()),
            "clipping_scale_mean": float(scales.mean().detach().cpu()),
            "clipping_scale_median": float(scales.median().detach().cpu()),
            "clipping_scale_p10": float(torch.quantile(scales, 0.10).detach().cpu()),
            "a_remote_mean": float(a.mean().detach().cpu()),
            "a_remote_median": float(a.median().detach().cpu()),
            "b_remote_norm_mean": float(torch.linalg.vector_norm(b, dim=1).mean().detach().cpu()),
            "b_remote_norm_median": float(torch.linalg.vector_norm(b, dim=1).median().detach().cpu()),
            "noise_multiplier": float(self.noise_multiplier),
            "noise_std": float(noise_std),
            "noise_l2": float(noise_l2),
            "invalid_a_count": float(invalid_a_count),
            "a_count": float(a_count),
            "a_invalid_fraction": float(invalid_a_count / a_count) if a_count else 0.0,
        }
        return a, b


class PrivateAggregateABQueue:
    """Target endpoint containing only a precomputed aggregate A/B release."""

    def __init__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        remote_cell_count: int,
        diagnostics: dict[str, float] | None = None,
    ) -> None:
        if a.ndim != 1 or b.ndim != 2 or b.shape[0] != a.shape[0]:
            raise ValueError("precomputed A/B endpoint has invalid dimensions")
        if a.requires_grad or b.requires_grad:
            raise ValueError("precomputed A/B endpoint must be detached")
        if remote_cell_count <= 0:
            raise ValueError("remote cell count must be positive")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("precomputed A/B endpoint must be finite")
        self.a = a.detach()
        self.b = b.detach()
        self.remote_cell_count = int(remote_cell_count)
        self.last_diagnostics = dict(diagnostics or {})

    def __len__(self) -> int:
        return self.remote_cell_count

    def aggregate(
        self,
        representations: torch.Tensor,
        *,
        max_entries: int,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if representations.shape != self.b.shape:
            raise ValueError("precomputed A/B endpoint does not match anchor representations")
        if max_entries < self.remote_cell_count:
            raise ValueError("max_entries cannot subsample a precomputed joint release")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        return (
            self.a.to(device=representations.device, dtype=representations.dtype),
            self.b.to(device=representations.device, dtype=representations.dtype),
        )


def balanced_queue_sample(
    queue: FreshClientQueue,
    max_entries: int,
    generator: torch.Generator,
) -> FreshClientQueue:
    """Sample without replacement, cycling evenly over source clients."""

    if max_entries <= 0:
        raise ValueError("max_entries must be positive")
    if len(queue) <= max_entries:
        return queue

    source_values = torch.unique(queue.source_clients, sorted=True)
    source_order = source_values[
        torch.randperm(
            len(source_values),
            device=source_values.device,
            generator=generator,
        )
    ]
    shuffled_by_source = []
    for source in source_order:
        indices = torch.nonzero(queue.source_clients == source, as_tuple=False).flatten()
        shuffled = indices[
            torch.randperm(len(indices), device=indices.device, generator=generator)
        ]
        shuffled_by_source.append(shuffled)

    selected: list[torch.Tensor] = []
    offsets = [0] * len(shuffled_by_source)
    while len(selected) < max_entries:
        made_progress = False
        for position, indices in enumerate(shuffled_by_source):
            if offsets[position] >= len(indices):
                continue
            selected.append(indices[offsets[position]])
            offsets[position] += 1
            made_progress = True
            if len(selected) == max_entries:
                break
        if not made_progress:
            break

    selected_indices = torch.stack(selected)
    return FreshClientQueue(
        embeddings=queue.embeddings[selected_indices],
        source_clients=queue.source_clients[selected_indices],
        sampling_seed=queue.sampling_seed,
        neighborhood_centroids=(
            None
            if queue.neighborhood_centroids is None
            else queue.neighborhood_centroids[selected_indices]
        ),
    )


class _QueueDiagnosticsMixin:
    """Accumulate compact queue diagnostics over mini-batches."""

    def _reset_diagnostics(self) -> None:
        self._diagnostics: dict[str, Any] = {
            "forward_calls": 0,
            "similarity_count": 0,
            "similarity_sum": 0.0,
            "similarity_sum_squares": 0.0,
            "similarity_min": math.inf,
            "similarity_max": -math.inf,
            "similarity_histogram": np.zeros(SIMILARITY_HISTOGRAM_BINS, dtype=np.int64),
            "masked_entries": 0,
            "candidate_entries": 0,
            "queue_contribution_sum": 0.0,
            "queue_contribution_count": 0,
            "queue_loss_increment_sum": 0.0,
            "queue_loss_increment_count": 0,
            "sampled_queue_entries_sum": 0,
            "sampled_queue_entries_min": math.inf,
            "sampled_queue_entries_max": 0,
            "local_negative_entries_sum": 0,
            "local_negative_entries_min": math.inf,
            "local_negative_entries_max": 0,
            "effective_queue_entries_sum": 0.0,
            "queue_candidate_ratio_sum": 0.0,
            "queue_effective_ratio_sum": 0.0,
            "centroid_consistency_count": 0,
            "centroid_consistency_sum": 0.0,
            "centroid_consistency_min": math.inf,
            "centroid_consistency_max": -math.inf,
            "hardest_top1_similarity_sum": 0.0,
            "hardest_top1_similarity_count": 0,
            "hardest_top5_similarity_sum": 0.0,
            "hardest_top5_similarity_count": 0,
            "hardest_top10_similarity_sum": 0.0,
            "hardest_top10_similarity_count": 0,
            "ambiguity_count": 0,
            "ambiguity_sum": 0.0,
            "ambiguity_histogram": np.zeros(
                RELIABILITY_HISTOGRAM_BINS, dtype=np.int64
            ),
            "negative_weight_count": 0,
            "negative_weight_sum": 0.0,
            "negative_weight_histogram": np.zeros(
                RELIABILITY_HISTOGRAM_BINS, dtype=np.int64
            ),
            "effective_queue_mass_sum": 0.0,
            "effective_queue_ratio_sum": 0.0,
            "queue_to_local_denominator_ratio_sum": 0.0,
            "queue_to_local_denominator_ratio_count": 0,
        }

    @torch.no_grad()
    def _record_diagnostics(
        self,
        similarity: torch.Tensor,
        mask: torch.Tensor,
        local_logits: torch.Tensor,
        weighted_queue_logits: torch.Tensor,
        local_loss: torch.Tensor,
        full_loss: torch.Tensor,
        local_negative_count: int,
        centroid_consistency: torch.Tensor | None = None,
        negative_weights: torch.Tensor | None = None,
        ambiguity: torch.Tensor | None = None,
    ) -> None:
        values = similarity.detach().float()
        count = values.numel()
        histogram = torch.histc(
            values.clamp(-1, 1),
            bins=SIMILARITY_HISTOGRAM_BINS,
            min=-1,
            max=1,
        )
        diagnostics = self._diagnostics
        diagnostics["forward_calls"] += 1
        diagnostics["similarity_count"] += count
        diagnostics["similarity_sum"] += float(values.sum().cpu())
        diagnostics["similarity_sum_squares"] += float(values.square().sum().cpu())
        diagnostics["similarity_min"] = min(
            diagnostics["similarity_min"], float(values.min().cpu())
        )
        diagnostics["similarity_max"] = max(
            diagnostics["similarity_max"], float(values.max().cpu())
        )
        diagnostics["similarity_histogram"] += histogram.cpu().numpy().astype(np.int64)
        diagnostics["masked_entries"] += int(mask.sum().cpu())
        diagnostics["candidate_entries"] += mask.numel()
        for percentage in (1, 5, 10):
            top_count = min(
                values.shape[1], max(1, math.ceil(percentage / 100 * values.shape[1]))
            )
            hardest = values.topk(top_count, dim=1).values
            diagnostics[f"hardest_top{percentage}_similarity_sum"] += float(
                hardest.sum().cpu()
            )
            diagnostics[f"hardest_top{percentage}_similarity_count"] += hardest.numel()
        if centroid_consistency is not None:
            consistency = centroid_consistency.detach().float()
            diagnostics["centroid_consistency_count"] += consistency.numel()
            diagnostics["centroid_consistency_sum"] += float(consistency.sum().cpu())
            diagnostics["centroid_consistency_min"] = min(
                diagnostics["centroid_consistency_min"], float(consistency.min().cpu())
            )
            diagnostics["centroid_consistency_max"] = max(
                diagnostics["centroid_consistency_max"], float(consistency.max().cpu())
            )
        if ambiguity is not None:
            ambiguity_values = ambiguity.detach().float().clamp(0, 1)
            diagnostics["ambiguity_count"] += ambiguity_values.numel()
            diagnostics["ambiguity_sum"] += float(ambiguity_values.sum().cpu())
            diagnostics["ambiguity_histogram"] += (
                torch.histc(
                    ambiguity_values,
                    bins=RELIABILITY_HISTOGRAM_BINS,
                    min=0,
                    max=1,
                )
                .cpu()
                .numpy()
                .astype(np.int64)
            )
        if negative_weights is None:
            negative_weights = (~mask).float()
        weight_values = negative_weights.detach().float().clamp(0, 1)
        diagnostics["negative_weight_count"] += weight_values.numel()
        diagnostics["negative_weight_sum"] += float(weight_values.sum().cpu())
        diagnostics["negative_weight_histogram"] += (
            torch.histc(
                weight_values,
                bins=RELIABILITY_HISTOGRAM_BINS,
                min=0,
                max=1,
            )
            .cpu()
            .numpy()
            .astype(np.int64)
        )
        diagnostics["effective_queue_mass_sum"] += float(
            weight_values.sum(dim=1).mean().cpu()
        )
        diagnostics["effective_queue_ratio_sum"] += float(weight_values.mean().cpu())

        local_log_denominator = torch.logsumexp(local_logits, dim=1)
        queue_log_denominator = torch.logsumexp(weighted_queue_logits, dim=1)
        combined = torch.logaddexp(local_log_denominator, queue_log_denominator)
        contribution = torch.exp(queue_log_denominator - combined)
        queue_to_local = torch.exp(queue_log_denominator - local_log_denominator)
        diagnostics["queue_contribution_sum"] += float(contribution.sum().cpu())
        diagnostics["queue_contribution_count"] += contribution.numel()
        diagnostics["queue_to_local_denominator_ratio_sum"] += float(
            queue_to_local.sum().cpu()
        )
        diagnostics["queue_to_local_denominator_ratio_count"] += queue_to_local.numel()
        diagnostics["queue_loss_increment_sum"] += float((full_loss - local_loss).cpu())
        diagnostics["queue_loss_increment_count"] += 1

        sampled = similarity.shape[1]
        diagnostics["sampled_queue_entries_sum"] += sampled
        diagnostics["sampled_queue_entries_min"] = min(
            diagnostics["sampled_queue_entries_min"], sampled
        )
        diagnostics["sampled_queue_entries_max"] = max(
            diagnostics["sampled_queue_entries_max"], sampled
        )
        effective = float(weight_values.sum(dim=1).mean().cpu())
        diagnostics["local_negative_entries_sum"] += int(local_negative_count)
        diagnostics["local_negative_entries_min"] = min(
            diagnostics["local_negative_entries_min"], int(local_negative_count)
        )
        diagnostics["local_negative_entries_max"] = max(
            diagnostics["local_negative_entries_max"], int(local_negative_count)
        )
        diagnostics["effective_queue_entries_sum"] += effective
        diagnostics["queue_candidate_ratio_sum"] += sampled / local_negative_count
        diagnostics["queue_effective_ratio_sum"] += effective / local_negative_count

    def diagnostic_state(self) -> dict[str, Any]:
        state = dict(self._diagnostics)
        state["similarity_histogram"] = self._diagnostics["similarity_histogram"].copy()
        state["ambiguity_histogram"] = self._diagnostics["ambiguity_histogram"].copy()
        state["negative_weight_histogram"] = self._diagnostics[
            "negative_weight_histogram"
        ].copy()
        return state

    def consume_diagnostic_state(self) -> dict[str, Any]:
        """Return and clear diagnostics so each summary describes one round."""

        state = self.diagnostic_state()
        self._reset_diagnostics()
        return state


def _local_concord_logits(
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    temperature: float,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce NTXent_general logits without changing its local objective."""

    batch_size = z_i.size(0)
    representations = F.normalize(torch.cat([z_i, z_j], dim=0), dim=1)
    logits = representations @ representations.T / temperature
    rows = torch.arange(2 * batch_size, device=representations.device)
    positive = (rows + batch_size) % (2 * batch_size)
    if beta > 0:
        negative_mask = torch.ones_like(logits, dtype=torch.bool)
        negative_mask.fill_diagonal_(False)
        negative_mask[rows, positive] = False
        negatives = logits[negative_mask].view(2 * batch_size, -1)
        count = negatives.size(1)
        log_normalizer = (
            torch.logsumexp(beta * negatives, dim=1, keepdim=True) - math.log(count)
        )
        logits[negative_mask] = ((beta + 1) * negatives - log_normalizer).reshape(-1)
    logits.fill_diagonal_(float("-inf"))
    return representations, logits, positive


class CrossClientSoftQueueLoss(_QueueDiagnosticsMixin, nn.Module):
    """Existing soft queue, retained unchanged as the pilot's condition B."""

    queue_temperature = 0.2

    def __init__(self, temperature=0.5, beta=0.0, queue_weight=1.0):
        super().__init__()
        self.local_temperature = temperature
        self.beta = beta
        self.queue_weight = queue_weight
        self._reset_diagnostics()

    def forward(self, z_i, z_j, global_queue=None):
        from concord.model.loss import NTXent_general

        if global_queue is None or len(global_queue) == 0:
            return NTXent_general(self.local_temperature, self.beta)(z_i, z_j)
        if self.queue_weight <= 0:
            raise ValueError("queue_weight must be positive")

        local, local_logits, positive = _local_concord_logits(
            z_i, z_j, self.local_temperature, self.beta
        )
        queue = F.normalize(global_queue, dim=1)
        similarity = local @ queue.T
        queue_logits = similarity / self.queue_temperature
        local_negative_count = max(1, local.size(0) - 2)
        queue_entry_weight = self.queue_weight * local_negative_count / queue.size(0)
        weighted_queue_logits = queue_logits + math.log(queue_entry_weight)
        mask = torch.zeros_like(similarity, dtype=torch.bool)
        full_loss = F.cross_entropy(
            torch.cat([local_logits, weighted_queue_logits], dim=1), positive
        )
        local_loss = F.cross_entropy(local_logits, positive)
        self._record_diagnostics(
            similarity,
            mask,
            local_logits,
            weighted_queue_logits,
            local_loss,
            full_loss,
            local_negative_count,
        )
        return full_loss


class FreshCrossClientQueueLoss(_QueueDiagnosticsMixin, nn.Module):
    """Original CONCORD local loss plus fresh, sampled queue negatives."""

    queue_temperature = 0.2
    mask_fraction = 0.05
    negative_ratio: float | None = None

    def __init__(self, temperature=0.5, beta=0.0, queue_weight=0.1):
        super().__init__()
        self.local_temperature = temperature
        self.beta = beta
        self.alpha = float(queue_weight)
        if self.alpha < 0:
            raise ValueError("queue alpha cannot be negative")
        self._sampling_generator: torch.Generator | None = None
        self._sampling_seed: int | None = None
        self._reset_diagnostics()

    def _get_sampling_generator(self, queue: FreshClientQueue) -> torch.Generator:
        if self._sampling_generator is None or self._sampling_seed != queue.sampling_seed:
            self._sampling_generator = torch.Generator(device=queue.embeddings.device)
            self._sampling_generator.manual_seed(queue.sampling_seed)
            self._sampling_seed = queue.sampling_seed
        return self._sampling_generator

    def _build_mask(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        sampled: FreshClientQueue,
        similarity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        mask = torch.zeros_like(similarity, dtype=torch.bool)
        if self.mask_fraction > 0 and similarity.shape[1] > 0:
            mask_count = min(
                similarity.shape[1],
                max(1, math.ceil(self.mask_fraction * similarity.shape[1])),
            )
            mask.scatter_(1, similarity.topk(mask_count, dim=1).indices, True)
        return mask, None

    def _build_negative_weights(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        sampled: FreshClientQueue,
        similarity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        mask, consistency = self._build_mask(z_i, z_j, sampled, similarity)
        return (~mask).float(), consistency, None

    def forward(self, z_i, z_j, global_queue=None):
        from concord.model.loss import NTXent_general

        if global_queue is None or len(global_queue) == 0 or self.alpha == 0:
            return NTXent_general(self.local_temperature, self.beta)(z_i, z_j)
        if not isinstance(global_queue, FreshClientQueue):
            raise TypeError("fresh queue loss requires FreshClientQueue input")

        local_negative_count = 2 * z_i.size(0) - 2
        max_queue_entries = z_i.size(0)
        if self.negative_ratio is not None:
            max_queue_entries = math.floor(self.negative_ratio * local_negative_count)
        if max_queue_entries <= 0:
            return NTXent_general(self.local_temperature, self.beta)(z_i, z_j)
        sampled = balanced_queue_sample(
            global_queue,
            max_entries=max_queue_entries,
            generator=self._get_sampling_generator(global_queue),
        )
        local, local_logits, positive = _local_concord_logits(
            z_i, z_j, self.local_temperature, self.beta
        )
        queue = F.normalize(sampled.embeddings, dim=1)
        similarity = local @ queue.T
        negative_weights, centroid_consistency, ambiguity = self._build_negative_weights(
            z_i, z_j, sampled, similarity
        )
        mask = negative_weights <= 0

        weighted_queue_logits = (
            similarity / self.queue_temperature
            + math.log(self.alpha)
            + negative_weights.clamp_min(torch.finfo(similarity.dtype).tiny).log()
        )
        weighted_queue_logits = weighted_queue_logits.masked_fill(mask, float("-inf"))
        full_loss = F.cross_entropy(
            torch.cat([local_logits, weighted_queue_logits], dim=1), positive
        )
        local_loss = F.cross_entropy(local_logits, positive)
        self._record_diagnostics(
            similarity,
            mask,
            local_logits,
            weighted_queue_logits,
            local_loss,
            full_loss,
            local_negative_count,
            centroid_consistency,
            negative_weights,
            ambiguity,
        )
        return full_loss


class PrivateABQueueLoss(_QueueDiagnosticsMixin, nn.Module):
    """FreshQueue loss consuming only private aggregate A/B statistics."""

    queue_temperature = 0.2
    default_max_entries = None

    def __init__(self, temperature=0.5, beta=0.0, queue_weight=0.1, max_entries=None):
        super().__init__()
        self.local_temperature = temperature
        self.beta = beta
        self.alpha = float(queue_weight)
        if max_entries is None:
            max_entries = self.default_max_entries
        self.max_entries = None if max_entries is None else int(max_entries)
        if self.max_entries is not None and self.max_entries <= 0:
            raise ValueError("max_entries must be positive when provided")
        self._reset_diagnostics()
        self._ab_diagnostics = {
            "ab_forward_calls": 0,
            "ab_remote_candidate_cells_sum": 0.0,
            "ab_u_norm_mean_sum": 0.0,
            "ab_u_norm_median_sum": 0.0,
            "ab_clipping_fraction_sum": 0.0,
            "ab_clipping_scale_mean_sum": 0.0,
            "ab_clipping_scale_median_sum": 0.0,
            "ab_clipping_scale_p10_sum": 0.0,
            "ab_remote_a_mean_sum": 0.0,
            "ab_remote_a_median_sum": 0.0,
            "ab_remote_b_norm_mean_sum": 0.0,
            "ab_remote_b_norm_median_sum": 0.0,
            "ab_noise_l2_sum": 0.0,
            "ab_noise_std_sum": 0.0,
            "ab_invalid_a_count": 0.0,
            "ab_a_count": 0.0,
            "ab_a_invalid_fraction_sum": 0.0,
        }

    def diagnostic_state(self) -> dict[str, Any]:
        state = super().diagnostic_state()
        state.update(self._ab_diagnostics)
        return state

    def _reset_diagnostics(self) -> None:
        super()._reset_diagnostics()
        if hasattr(self, "_ab_diagnostics"):
            for key in self._ab_diagnostics:
                self._ab_diagnostics[key] = 0.0
            self._ab_diagnostics["ab_forward_calls"] = 0

    def forward(self, z_i, z_j, global_queue=None):
        from concord.model.loss import NTXent_general

        if global_queue is None or len(global_queue) == 0 or self.alpha == 0:
            return NTXent_general(self.local_temperature, self.beta)(z_i, z_j)
        if not isinstance(global_queue, (PrivateRemoteABQueue, PrivateAggregateABQueue)):
            raise TypeError("private A/B loss requires PrivateRemoteABQueue input")

        local, local_logits, positive = _local_concord_logits(
            z_i, z_j, self.local_temperature, self.beta
        )
        max_queue_entries = z_i.size(0) if self.max_entries is None else self.max_entries
        a, b = global_queue.aggregate(
            local,
            max_entries=max_queue_entries,
            temperature=self.queue_temperature,
        )
        tiny = torch.finfo(local.dtype).tiny
        remote_log = torch.log((self.alpha * a).clamp_min(tiny))
        remote_direction = (
            local * (b / a[:, None].clamp_min(tiny) / self.queue_temperature).detach()
        ).sum(dim=1)
        # Forward is log(alpha*A); backward additionally injects dA/dz=B/tau.
        remote_log_st = remote_log.detach() + remote_direction - remote_direction.detach()
        local_log_denominator = torch.logsumexp(local_logits, dim=1)
        combined = torch.logaddexp(local_log_denominator, remote_log_st)
        rows = torch.arange(local.shape[0], device=local.device)
        full_loss = (combined - local_logits[rows, positive]).mean()
        local_loss = (local_log_denominator - local_logits[rows, positive]).mean()

        # Keep the established compact queue diagnostics populated without
        # exposing individual remote keys to the query-client interface.
        zero_similarity = torch.zeros(
            (local.shape[0], 1), dtype=local.dtype, device=local.device
        )
        self._record_diagnostics(
            zero_similarity,
            torch.zeros_like(zero_similarity, dtype=torch.bool),
            local_logits,
            remote_log[:, None],
            local_loss,
            full_loss,
            max(1, local.size(0) - 2),
        )
        stats = global_queue.last_diagnostics
        self._ab_diagnostics["ab_forward_calls"] += 1
        diagnostic_keys = {
            "remote_candidate_cells": "ab_remote_candidate_cells_sum",
            "u_norm_mean": "ab_u_norm_mean_sum",
            "u_norm_median": "ab_u_norm_median_sum",
            "clipping_fraction": "ab_clipping_fraction_sum",
            "clipping_scale_mean": "ab_clipping_scale_mean_sum",
            "clipping_scale_median": "ab_clipping_scale_median_sum",
            "clipping_scale_p10": "ab_clipping_scale_p10_sum",
            "a_remote_mean": "ab_remote_a_mean_sum",
            "a_remote_median": "ab_remote_a_median_sum",
            "b_remote_norm_mean": "ab_remote_b_norm_mean_sum",
            "b_remote_norm_median": "ab_remote_b_norm_median_sum",
            "noise_l2": "ab_noise_l2_sum",
            "noise_std": "ab_noise_std_sum",
            "invalid_a_count": "ab_invalid_a_count",
            "a_count": "ab_a_count",
            "a_invalid_fraction": "ab_a_invalid_fraction_sum",
        }
        for source_key, target_key in diagnostic_keys.items():
            self._ab_diagnostics[target_key] += stats[source_key]
        return full_loss


def local_neighborhood_centroids(values: torch.Tensor, k: int) -> torch.Tensor:
    """Return normalized local kNN centroids without using labels."""

    normalized = F.normalize(values.detach(), dim=1)
    if len(normalized) <= 1:
        return normalized
    neighbor_count = min(int(k), len(normalized) - 1)
    similarities = normalized @ normalized.T
    similarities.fill_diagonal_(float("-inf"))
    neighbors = similarities.topk(neighbor_count, dim=1).indices
    return F.normalize(normalized[neighbors].mean(dim=1), dim=1)


def reliable_negative_mask(
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    sampled: FreshClientQueue,
    similarity: torch.Tensor,
    *,
    neighborhood_k: int,
    similarity_fraction: float,
    consistency_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Identify label-free ambiguous negatives using pair and neighborhood geometry."""

    if sampled.neighborhood_centroids is None:
        raise ValueError("reliable-negative filtering requires queue neighborhood centroids")
    with torch.no_grad():
        anchor_centroids = torch.cat(
            [
                local_neighborhood_centroids(z_i, neighborhood_k),
                local_neighborhood_centroids(z_j, neighborhood_k),
            ],
            dim=0,
        )
        queue_centroids = F.normalize(sampled.neighborhood_centroids, dim=1)
        consistency = anchor_centroids @ queue_centroids.T
        similarity_count = min(
            similarity.shape[1],
            max(1, math.ceil(similarity_fraction * similarity.shape[1])),
        )
        similarity_top = torch.zeros_like(similarity, dtype=torch.bool)
        similarity_top.scatter_(
            1, similarity.detach().topk(similarity_count, dim=1).indices, True
        )
        consistency_count = min(
            consistency.numel(),
            max(1, math.ceil(consistency_fraction * consistency.numel())),
        )
        consistency_top = torch.zeros_like(consistency, dtype=torch.bool).flatten()
        consistency_top[
            consistency.flatten().topk(consistency_count).indices
        ] = True
        consistency_top = consistency_top.view_as(consistency)
        return similarity_top & consistency_top, consistency


class ReliableNegativeFreshQueueLoss(FreshCrossClientQueueLoss):
    """FreshQueue that removes label-free ambiguous cross-client negatives."""

    neighborhood_k = 20
    similarity_fraction = 0.10
    consistency_fraction = 0.30

    def _build_mask(self, z_i, z_j, sampled, similarity):
        return reliable_negative_mask(
            z_i,
            z_j,
            sampled,
            similarity,
            neighborhood_k=self.neighborhood_k,
            similarity_fraction=self.similarity_fraction,
            consistency_fraction=self.consistency_fraction,
        )


class RandomMaskFreshQueueLoss(ReliableNegativeFreshQueueLoss):
    """Matched-count control that removes random entries per anchor."""

    def _build_mask(self, z_i, z_j, sampled, similarity):
        reliable_mask, consistency = reliable_negative_mask(
            z_i,
            z_j,
            sampled,
            similarity,
            neighborhood_k=self.neighborhood_k,
            similarity_fraction=self.similarity_fraction,
            consistency_fraction=self.consistency_fraction,
        )
        counts = reliable_mask.sum(dim=1)
        generator = self._get_sampling_generator(sampled)
        random_values = torch.rand(
            similarity.shape,
            device=similarity.device,
            generator=generator,
        )
        order = random_values.argsort(dim=1)
        random_mask = torch.zeros_like(reliable_mask)
        for row, count in enumerate(counts.tolist()):
            if count:
                random_mask[row, order[row, :count]] = True
        return random_mask, consistency


def percentile_ranks(values: torch.Tensor) -> torch.Tensor:
    """Return per-row ascending percentile ranks in [0, 1]."""

    if values.ndim != 2:
        raise ValueError("percentile ranks require a two-dimensional tensor")
    if values.shape[1] == 1:
        return torch.ones_like(values)
    order = values.argsort(dim=1, stable=True)
    ranks = torch.empty_like(values)
    rank_values = torch.arange(
        values.shape[1], device=values.device, dtype=values.dtype
    ).expand_as(values)
    ranks.scatter_(1, order, rank_values)
    return ranks / (values.shape[1] - 1)


def soft_reliability_weights(
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    sampled: FreshClientQueue,
    similarity: torch.Tensor,
    *,
    neighborhood_k: int,
    reliability_beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute frozen label-free ambiguity and continuous negative weights."""

    if sampled.neighborhood_centroids is None:
        raise ValueError("soft reliability requires queue neighborhood centroids")
    with torch.no_grad():
        anchor_centroids = torch.cat(
            [
                local_neighborhood_centroids(z_i, neighborhood_k),
                local_neighborhood_centroids(z_j, neighborhood_k),
            ],
            dim=0,
        )
        consistency = anchor_centroids @ F.normalize(
            sampled.neighborhood_centroids, dim=1
        ).T
        similarity_rank = percentile_ranks(similarity.detach())
        neighborhood_rank = percentile_ranks(consistency)
        ambiguity = torch.sqrt(similarity_rank * neighborhood_rank)
        weights = 1.0 - float(reliability_beta) * ambiguity
    return weights, consistency, ambiguity


class SoftReliabilityFreshQueueLoss(FreshCrossClientQueueLoss):
    """Continuously attenuate ambiguous cross-client negatives without deleting them."""

    neighborhood_k = 20
    reliability_beta = 0.5

    def _build_negative_weights(self, z_i, z_j, sampled, similarity):
        return soft_reliability_weights(
            z_i,
            z_j,
            sampled,
            similarity,
            neighborhood_k=self.neighborhood_k,
            reliability_beta=self.reliability_beta,
        )


class RandomSoftWeightFreshQueueLoss(SoftReliabilityFreshQueueLoss):
    """Per-anchor permutation control with the exact SoftRN weight multiset."""

    def __init__(self, temperature=0.5, beta=0.0, queue_weight=0.1):
        super().__init__(temperature=temperature, beta=beta, queue_weight=queue_weight)
        self._permutation_generator: torch.Generator | None = None
        self._permutation_seed: int | None = None

    def _get_permutation_generator(self, queue: FreshClientQueue) -> torch.Generator:
        if (
            self._permutation_generator is None
            or self._permutation_seed != queue.sampling_seed
        ):
            self._permutation_generator = torch.Generator(
                device=queue.embeddings.device
            )
            permutation_seed = int(queue.sampling_seed) ^ 0x5F3759DF
            self._permutation_generator.manual_seed(permutation_seed)
            self._permutation_seed = queue.sampling_seed
        return self._permutation_generator

    def _build_negative_weights(self, z_i, z_j, sampled, similarity):
        weights, consistency, ambiguity = soft_reliability_weights(
            z_i,
            z_j,
            sampled,
            similarity,
            neighborhood_k=self.neighborhood_k,
            reliability_beta=self.reliability_beta,
        )
        generator = self._get_permutation_generator(sampled)
        permutations = torch.rand(
            weights.shape, device=weights.device, generator=generator
        ).argsort(dim=1)
        return torch.gather(weights, 1, permutations), consistency, ambiguity


def cross_client_topology(
    local_embeddings: torch.Tensor,
    queue_embeddings: torch.Tensor,
    *,
    top_k: int,
    queue_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a detached local-local topology from sparse queue affinities."""

    if top_k <= 0:
        raise ValueError("topology top-k must be positive")
    if queue_temperature <= 0:
        raise ValueError("queue topology temperature must be positive")
    local = F.normalize(local_embeddings.detach(), dim=1)
    queue = F.normalize(queue_embeddings.detach(), dim=1)
    similarity = local @ queue.T
    neighbor_count = min(int(top_k), similarity.shape[1])
    top_values, top_indices = similarity.topk(neighbor_count, dim=1)
    top_weights = F.softmax(top_values / float(queue_temperature), dim=1)
    affinities = torch.zeros_like(similarity)
    affinities.scatter_(1, top_indices, top_weights)
    topology = F.normalize(affinities, dim=1) @ F.normalize(affinities, dim=1).T
    return topology.detach(), top_values.detach(), top_weights.detach()


class CrossClientTopologyQueueLoss(_QueueDiagnosticsMixin, nn.Module):
    """Regularize local relations using detached cross-client queue topology."""

    queue_temperature = 0.2
    topology_temperature = 0.2
    local_relation_temperature = 0.2
    topology_top_k = 20

    def __init__(self, temperature=0.5, beta=0.0, queue_weight=0.05):
        super().__init__()
        self.local_temperature = float(temperature)
        self.beta = float(beta)
        self.topology_weight = float(queue_weight)
        if self.topology_weight < 0:
            raise ValueError("CTQ lambda cannot be negative")
        self._reset_diagnostics()
        self._diagnostics.update(
            {
                "ctq_loss_sum": 0.0,
                "ctq_loss_count": 0,
                "ctq_topk_similarity_sum": 0.0,
                "ctq_topk_similarity_count": 0,
                "ctq_affinity_entropy_sum": 0.0,
                "ctq_effective_neighbors_sum": 0.0,
                "ctq_anchor_count": 0,
                "ctq_relation_spearman_sum": 0.0,
                "ctq_relation_spearman_count": 0,
            }
        )

    @torch.no_grad()
    def _record_ctq(
        self,
        topology: torch.Tensor,
        relation: torch.Tensor,
        top_values: torch.Tensor,
        top_weights: torch.Tensor,
        topology_loss: torch.Tensor,
        queue_size: int,
    ) -> None:
        diagnostics = self._diagnostics
        entropy = -(
            top_weights
            * top_weights.clamp_min(torch.finfo(top_weights.dtype).tiny).log()
        ).sum(dim=1)
        off_diagonal = ~torch.eye(
            topology.shape[0], dtype=torch.bool, device=topology.device
        )
        first = topology[off_diagonal].float()
        second = relation.detach()[off_diagonal].float()
        if first.numel() > 1 and first.std() > 0 and second.std() > 0:
            first_rank = first.argsort().argsort().float()
            second_rank = second.argsort().argsort().float()
            correlation = torch.corrcoef(torch.stack([first_rank, second_rank]))[0, 1]
            if torch.isfinite(correlation):
                diagnostics["ctq_relation_spearman_sum"] += float(correlation.cpu())
                diagnostics["ctq_relation_spearman_count"] += 1
        diagnostics["forward_calls"] += 1
        diagnostics["ctq_loss_sum"] += float(topology_loss.detach().cpu())
        diagnostics["ctq_loss_count"] += 1
        diagnostics["ctq_topk_similarity_sum"] += float(top_values.sum().cpu())
        diagnostics["ctq_topk_similarity_count"] += top_values.numel()
        diagnostics["ctq_affinity_entropy_sum"] += float(entropy.sum().cpu())
        diagnostics["ctq_effective_neighbors_sum"] += float(entropy.exp().sum().cpu())
        diagnostics["ctq_anchor_count"] += len(entropy)
        diagnostics["sampled_queue_entries_sum"] += int(queue_size)
        diagnostics["sampled_queue_entries_min"] = min(
            diagnostics["sampled_queue_entries_min"], int(queue_size)
        )
        diagnostics["sampled_queue_entries_max"] = max(
            diagnostics["sampled_queue_entries_max"], int(queue_size)
        )

    def forward(self, z_i, z_j, global_queue=None):
        from concord.model.loss import NTXent_general

        local_loss = NTXent_general(self.local_temperature, self.beta)(z_i, z_j)
        if (
            global_queue is None
            or len(global_queue) == 0
            or self.topology_weight == 0
            or z_i.shape[0] <= 1
        ):
            return local_loss
        if not isinstance(global_queue, FreshClientQueue):
            raise TypeError("CTQ requires FreshClientQueue input")

        topology, top_values, top_weights = cross_client_topology(
            z_i,
            global_queue.embeddings,
            top_k=self.topology_top_k,
            queue_temperature=self.queue_temperature,
        )
        local = F.normalize(z_i, dim=1)
        relation = local @ local.T
        diagonal = torch.eye(len(local), dtype=torch.bool, device=local.device)
        excluded = torch.finfo(relation.dtype).min
        topology_logits = (topology / self.topology_temperature).masked_fill(
            diagonal, excluded
        )
        relation_logits = (relation / self.local_relation_temperature).masked_fill(
            diagonal, excluded
        )
        target = F.softmax(topology_logits, dim=1).detach()
        topology_loss = F.kl_div(
            F.log_softmax(relation_logits, dim=1), target, reduction="batchmean"
        )
        self._record_ctq(
            topology,
            relation,
            top_values,
            top_weights,
            topology_loss,
            len(global_queue),
        )
        return local_loss + self.topology_weight * topology_loss


def _install_queue_loss(loss_class: type[nn.Module]) -> None:
    import concord.model.loss as loss_module
    import concord.model.trainer as trainer_module

    loss_module.NTXent_with_queue = loss_class
    trainer_module.NTXent_with_queue = loss_class


def install_cross_client_soft_queue_loss(queue_temperature: float) -> None:
    """Install the existing soft queue loss used by condition B."""

    CrossClientSoftQueueLoss.queue_temperature = float(queue_temperature)
    _install_queue_loss(CrossClientSoftQueueLoss)


def install_fresh_cross_client_queue_loss(
    queue_temperature: float,
    mask_fraction: float,
    negative_ratio: float | None = None,
) -> None:
    """Install the fresh queue loss with protocol-level settings."""

    if queue_temperature <= 0:
        raise ValueError("queue temperature must be positive")
    if not 0 <= mask_fraction < 1:
        raise ValueError("mask fraction must be in [0, 1)")
    if negative_ratio is not None and (
        not math.isfinite(negative_ratio) or not 0 < negative_ratio <= 1
    ):
        raise ValueError("queue negative ratio must be finite and in (0, 1]")
    FreshCrossClientQueueLoss.queue_temperature = float(queue_temperature)
    FreshCrossClientQueueLoss.mask_fraction = float(mask_fraction)
    FreshCrossClientQueueLoss.negative_ratio = (
        None if negative_ratio is None else float(negative_ratio)
    )
    _install_queue_loss(FreshCrossClientQueueLoss)


def install_private_ab_queue_loss(
    queue_temperature: float,
    max_entries: int | None = None,
) -> None:
    """Install exact aggregate A/B or fixed cell-clipped A/B loss."""

    if queue_temperature <= 0:
        raise ValueError("queue temperature must be positive")
    PrivateABQueueLoss.queue_temperature = float(queue_temperature)
    PrivateABQueueLoss.default_max_entries = max_entries
    _install_queue_loss(PrivateABQueueLoss)


def install_reliable_negative_queue_loss(
    queue_temperature: float,
    *,
    neighborhood_k: int,
    similarity_fraction: float,
    consistency_fraction: float,
    random_control: bool = False,
) -> None:
    """Install the frozen RN filter or its matched-count random control."""

    if queue_temperature <= 0:
        raise ValueError("queue temperature must be positive")
    if neighborhood_k <= 0:
        raise ValueError("neighborhood k must be positive")
    if not 0 < similarity_fraction < 1 or not 0 < consistency_fraction < 1:
        raise ValueError("reliability fractions must be between zero and one")
    loss_class = RandomMaskFreshQueueLoss if random_control else ReliableNegativeFreshQueueLoss
    loss_class.queue_temperature = float(queue_temperature)
    loss_class.mask_fraction = 0.0
    loss_class.negative_ratio = None
    loss_class.neighborhood_k = int(neighborhood_k)
    loss_class.similarity_fraction = float(similarity_fraction)
    loss_class.consistency_fraction = float(consistency_fraction)
    _install_queue_loss(loss_class)


def install_soft_reliability_queue_loss(
    queue_temperature: float,
    *,
    neighborhood_k: int,
    reliability_beta: float,
    random_control: bool = False,
) -> None:
    """Install frozen SoftRN weighting or its matched-distribution random control."""

    if queue_temperature <= 0:
        raise ValueError("queue temperature must be positive")
    if neighborhood_k <= 0:
        raise ValueError("neighborhood k must be positive")
    if not 0 <= reliability_beta <= 1:
        raise ValueError("reliability beta must be in [0, 1]")
    loss_class = (
        RandomSoftWeightFreshQueueLoss
        if random_control
        else SoftReliabilityFreshQueueLoss
    )
    loss_class.queue_temperature = float(queue_temperature)
    loss_class.mask_fraction = 0.0
    loss_class.negative_ratio = None
    loss_class.neighborhood_k = int(neighborhood_k)
    loss_class.reliability_beta = float(reliability_beta)
    _install_queue_loss(loss_class)


def install_cross_client_topology_queue_loss(
    queue_temperature: float,
    *,
    topology_top_k: int,
    topology_temperature: float,
    local_relation_temperature: float,
) -> None:
    """Install the frozen topology-only queue regularizer."""

    if min(queue_temperature, topology_temperature, local_relation_temperature) <= 0:
        raise ValueError("all CTQ temperatures must be positive")
    if topology_top_k <= 0:
        raise ValueError("CTQ top-k must be positive")
    CrossClientTopologyQueueLoss.queue_temperature = float(queue_temperature)
    CrossClientTopologyQueueLoss.topology_temperature = float(topology_temperature)
    CrossClientTopologyQueueLoss.local_relation_temperature = float(
        local_relation_temperature
    )
    CrossClientTopologyQueueLoss.topology_top_k = int(topology_top_k)
    _install_queue_loss(CrossClientTopologyQueueLoss)


def summarize_queue_diagnostics(states: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge loss-level diagnostic states for one communication round."""

    histogram = np.zeros(SIMILARITY_HISTOGRAM_BINS, dtype=np.int64)
    ambiguity_histogram = np.zeros(RELIABILITY_HISTOGRAM_BINS, dtype=np.int64)
    negative_weight_histogram = np.zeros(RELIABILITY_HISTOGRAM_BINS, dtype=np.int64)
    totals: dict[str, Any] = {
        "forward_calls": 0,
        "similarity_count": 0,
        "similarity_sum": 0.0,
        "similarity_sum_squares": 0.0,
        "similarity_min": math.inf,
        "similarity_max": -math.inf,
        "masked_entries": 0,
        "candidate_entries": 0,
        "queue_contribution_sum": 0.0,
        "queue_contribution_count": 0,
        "queue_loss_increment_sum": 0.0,
        "queue_loss_increment_count": 0,
        "sampled_queue_entries_sum": 0,
        "sampled_queue_entries_min": math.inf,
        "sampled_queue_entries_max": 0,
        "local_negative_entries_sum": 0,
        "local_negative_entries_min": math.inf,
        "local_negative_entries_max": 0,
        "effective_queue_entries_sum": 0.0,
        "queue_candidate_ratio_sum": 0.0,
        "queue_effective_ratio_sum": 0.0,
        "centroid_consistency_count": 0,
        "centroid_consistency_sum": 0.0,
        "centroid_consistency_min": math.inf,
        "centroid_consistency_max": -math.inf,
        "hardest_top1_similarity_sum": 0.0,
        "hardest_top1_similarity_count": 0,
        "hardest_top5_similarity_sum": 0.0,
        "hardest_top5_similarity_count": 0,
        "hardest_top10_similarity_sum": 0.0,
        "hardest_top10_similarity_count": 0,
        "ambiguity_count": 0,
        "ambiguity_sum": 0.0,
        "negative_weight_count": 0,
        "negative_weight_sum": 0.0,
        "effective_queue_mass_sum": 0.0,
        "effective_queue_ratio_sum": 0.0,
        "queue_to_local_denominator_ratio_sum": 0.0,
        "queue_to_local_denominator_ratio_count": 0,
        "ctq_loss_sum": 0.0,
        "ctq_loss_count": 0,
        "ctq_topk_similarity_sum": 0.0,
        "ctq_topk_similarity_count": 0,
        "ctq_affinity_entropy_sum": 0.0,
        "ctq_effective_neighbors_sum": 0.0,
        "ctq_anchor_count": 0,
        "ctq_relation_spearman_sum": 0.0,
        "ctq_relation_spearman_count": 0,
        "ab_forward_calls": 0,
        "ab_remote_candidate_cells": 0.0,
        "ab_u_norm_mean_sum": 0.0,
        "ab_u_norm_median_sum": 0.0,
        "ab_clipping_fraction_sum": 0.0,
        "ab_clipping_scale_mean_sum": 0.0,
        "ab_clipping_scale_median_sum": 0.0,
        "ab_clipping_scale_p10_sum": 0.0,
        "ab_remote_a_mean_sum": 0.0,
        "ab_remote_a_median_sum": 0.0,
        "ab_remote_b_norm_mean_sum": 0.0,
        "ab_remote_b_norm_median_sum": 0.0,
        "ab_noise_l2_sum": 0.0,
        "ab_noise_std_sum": 0.0,
        "ab_invalid_a_count": 0.0,
        "ab_a_count": 0.0,
        "ab_a_invalid_fraction_sum": 0.0,
    }
    for state in states:
        histogram += np.asarray(state["similarity_histogram"], dtype=np.int64)
        ambiguity_histogram += np.asarray(
            state.get(
                "ambiguity_histogram",
                np.zeros(RELIABILITY_HISTOGRAM_BINS, dtype=np.int64),
            ),
            dtype=np.int64,
        )
        negative_weight_histogram += np.asarray(
            state.get(
                "negative_weight_histogram",
                np.zeros(RELIABILITY_HISTOGRAM_BINS, dtype=np.int64),
            ),
            dtype=np.int64,
        )
        for key in (
            "forward_calls",
            "similarity_count",
            "similarity_sum",
            "similarity_sum_squares",
            "masked_entries",
            "candidate_entries",
            "queue_contribution_sum",
            "queue_contribution_count",
            "queue_loss_increment_sum",
            "queue_loss_increment_count",
            "sampled_queue_entries_sum",
            "local_negative_entries_sum",
            "effective_queue_entries_sum",
            "queue_candidate_ratio_sum",
            "queue_effective_ratio_sum",
            "centroid_consistency_count",
            "centroid_consistency_sum",
            "hardest_top1_similarity_sum",
            "hardest_top1_similarity_count",
            "hardest_top5_similarity_sum",
            "hardest_top5_similarity_count",
            "hardest_top10_similarity_sum",
            "hardest_top10_similarity_count",
            "ambiguity_count",
            "ambiguity_sum",
            "negative_weight_count",
            "negative_weight_sum",
            "effective_queue_mass_sum",
            "effective_queue_ratio_sum",
            "queue_to_local_denominator_ratio_sum",
            "queue_to_local_denominator_ratio_count",
            "ctq_loss_sum",
            "ctq_loss_count",
            "ctq_topk_similarity_sum",
            "ctq_topk_similarity_count",
            "ctq_affinity_entropy_sum",
            "ctq_effective_neighbors_sum",
            "ctq_anchor_count",
            "ctq_relation_spearman_sum",
            "ctq_relation_spearman_count",
            "ab_forward_calls",
            "ab_remote_candidate_cells",
            "ab_u_norm_mean_sum",
            "ab_u_norm_median_sum",
            "ab_clipping_fraction_sum",
            "ab_clipping_scale_mean_sum",
            "ab_clipping_scale_median_sum",
            "ab_clipping_scale_p10_sum",
            "ab_remote_a_mean_sum",
            "ab_remote_a_median_sum",
            "ab_remote_b_norm_mean_sum",
            "ab_remote_b_norm_median_sum",
            "ab_noise_l2_sum",
            "ab_noise_std_sum",
            "ab_invalid_a_count",
            "ab_a_count",
            "ab_a_invalid_fraction_sum",
        ):
            totals[key] += state.get(key, 0)
        totals["similarity_min"] = min(totals["similarity_min"], state["similarity_min"])
        totals["similarity_max"] = max(totals["similarity_max"], state["similarity_max"])
        totals["sampled_queue_entries_min"] = min(
            totals["sampled_queue_entries_min"], state["sampled_queue_entries_min"]
        )
        totals["sampled_queue_entries_max"] = max(
            totals["sampled_queue_entries_max"], state["sampled_queue_entries_max"]
        )
        totals["local_negative_entries_min"] = min(
            totals["local_negative_entries_min"], state["local_negative_entries_min"]
        )
        totals["local_negative_entries_max"] = max(
            totals["local_negative_entries_max"], state["local_negative_entries_max"]
        )
        totals["centroid_consistency_min"] = min(
            totals["centroid_consistency_min"], state["centroid_consistency_min"]
        )
        totals["centroid_consistency_max"] = max(
            totals["centroid_consistency_max"], state["centroid_consistency_max"]
        )

    count = totals["similarity_count"]
    mean = totals["similarity_sum"] / count if count else math.nan
    variance = (
        max(0.0, totals["similarity_sum_squares"] / count - mean**2)
        if count
        else math.nan
    )
    calls = totals["forward_calls"]
    edges = np.linspace(-1.0, 1.0, SIMILARITY_HISTOGRAM_BINS + 1)
    reliability_edges = np.linspace(0.0, 1.0, RELIABILITY_HISTOGRAM_BINS + 1)

    def histogram_median(counts: np.ndarray) -> float:
        """Approximate the median by the midpoint of its 0.001-wide bin."""

        total = int(counts.sum())
        if not total:
            return math.nan
        index = int(np.searchsorted(np.cumsum(counts), (total + 1) / 2))
        return float((reliability_edges[index] + reliability_edges[index + 1]) / 2)

    ambiguity_count = totals["ambiguity_count"]
    weight_count = totals["negative_weight_count"]
    queue_denominator_mass = (
        totals["queue_contribution_sum"] / totals["queue_contribution_count"]
        if totals["queue_contribution_count"]
        else 0.0
    )
    return {
        "forward_calls": calls,
        "similarity_count": count,
        "similarity_mean": mean,
        "similarity_std": math.sqrt(variance) if count else math.nan,
        "similarity_min": totals["similarity_min"] if count else math.nan,
        "similarity_max": totals["similarity_max"] if count else math.nan,
        "similarity_histogram_edges": edges.tolist(),
        "similarity_histogram_counts": histogram.tolist(),
        "masked_entries": totals["masked_entries"],
        "candidate_entries": totals["candidate_entries"],
        "masked_fraction": (
            totals["masked_entries"] / totals["candidate_entries"]
            if totals["candidate_entries"]
            else 0.0
        ),
        "queue_contribution": queue_denominator_mass,
        "queue_denominator_mass_fraction": queue_denominator_mass,
        "queue_to_local_denominator_mass_ratio": (
            totals["queue_to_local_denominator_ratio_sum"]
            / totals["queue_to_local_denominator_ratio_count"]
            if totals["queue_to_local_denominator_ratio_count"]
            else 0.0
        ),
        "queue_loss_increment": (
            totals["queue_loss_increment_sum"] / totals["queue_loss_increment_count"]
            if totals["queue_loss_increment_count"]
            else 0.0
        ),
        "sampled_queue_entries_mean": (
            totals["sampled_queue_entries_sum"] / calls if calls else 0.0
        ),
        "sampled_queue_entries_min": (
            totals["sampled_queue_entries_min"] if calls else 0
        ),
        "sampled_queue_entries_max": totals["sampled_queue_entries_max"],
        "local_negative_entries_mean": (
            totals["local_negative_entries_sum"] / calls if calls else 0.0
        ),
        "local_negative_entries_min": (
            totals["local_negative_entries_min"] if calls else 0
        ),
        "local_negative_entries_max": totals["local_negative_entries_max"],
        "effective_queue_entries_mean": (
            totals["effective_queue_entries_sum"] / calls if calls else 0.0
        ),
        "actual_queue_candidate_ratio_mean": (
            totals["queue_candidate_ratio_sum"] / calls if calls else 0.0
        ),
        "actual_queue_effective_ratio_mean": (
            totals["queue_effective_ratio_sum"] / calls if calls else 0.0
        ),
        "ambiguity_count": ambiguity_count,
        "ambiguity_mean": (
            totals["ambiguity_sum"] / ambiguity_count
            if ambiguity_count
            else math.nan
        ),
        "ambiguity_median_approx": histogram_median(ambiguity_histogram),
        "ambiguity_histogram_edges": reliability_edges.tolist(),
        "ambiguity_histogram_counts": ambiguity_histogram.tolist(),
        "negative_weight_count": weight_count,
        "negative_weight_mean": (
            totals["negative_weight_sum"] / weight_count if weight_count else math.nan
        ),
        "negative_weight_median_approx": histogram_median(
            negative_weight_histogram
        ),
        "negative_weight_histogram_edges": reliability_edges.tolist(),
        "negative_weight_histogram_counts": negative_weight_histogram.tolist(),
        "effective_queue_mass_mean": (
            totals["effective_queue_mass_sum"] / calls if calls else 0.0
        ),
        "effective_queue_ratio_mean": (
            totals["effective_queue_ratio_sum"] / calls if calls else 0.0
        ),
        "centroid_consistency_mean": (
            totals["centroid_consistency_sum"] / totals["centroid_consistency_count"]
            if totals["centroid_consistency_count"]
            else math.nan
        ),
        "centroid_consistency_min": (
            totals["centroid_consistency_min"]
            if totals["centroid_consistency_count"]
            else math.nan
        ),
        "ctq_loss": (
            totals["ctq_loss_sum"] / totals["ctq_loss_count"]
            if totals["ctq_loss_count"]
            else 0.0
        ),
        "ctq_mean_topk_queue_similarity": (
            totals["ctq_topk_similarity_sum"] / totals["ctq_topk_similarity_count"]
            if totals["ctq_topk_similarity_count"]
            else math.nan
        ),
        "ctq_affinity_entropy": (
            totals["ctq_affinity_entropy_sum"] / totals["ctq_anchor_count"]
            if totals["ctq_anchor_count"]
            else math.nan
        ),
        "ctq_effective_queue_neighbors": (
            totals["ctq_effective_neighbors_sum"] / totals["ctq_anchor_count"]
            if totals["ctq_anchor_count"]
            else math.nan
        ),
        "ctq_local_queue_relation_spearman": (
            totals["ctq_relation_spearman_sum"]
            / totals["ctq_relation_spearman_count"]
            if totals["ctq_relation_spearman_count"]
            else math.nan
        ),
        "ab_remote_candidate_cells_mean": (
            totals["ab_remote_candidate_cells"] / totals["ab_forward_calls"]
            if totals["ab_forward_calls"]
            else 0.0
        ),
        "ab_u_norm_mean": (
            totals["ab_u_norm_mean_sum"] / totals["ab_forward_calls"]
            if totals["ab_forward_calls"]
            else math.nan
        ),
        "ab_u_norm_median": (
            totals["ab_u_norm_median_sum"] / totals["ab_forward_calls"]
            if totals["ab_forward_calls"]
            else math.nan
        ),
        "ab_clipping_fraction": (
            totals["ab_clipping_fraction_sum"] / totals["ab_forward_calls"]
            if totals["ab_forward_calls"]
            else math.nan
        ),
        "ab_clipping_scale_mean": (
            totals["ab_clipping_scale_mean_sum"] / totals["ab_forward_calls"]
            if totals["ab_forward_calls"]
            else math.nan
        ),
        "ab_clipping_scale_median": (
            totals["ab_clipping_scale_median_sum"] / totals["ab_forward_calls"]
            if totals["ab_forward_calls"]
            else math.nan
        ),
        "ab_clipping_scale_p10": (
            totals["ab_clipping_scale_p10_sum"] / totals["ab_forward_calls"]
            if totals["ab_forward_calls"]
            else math.nan
        ),
        "ab_remote_a_mean": (
            totals["ab_remote_a_mean_sum"] / totals["ab_forward_calls"]
            if totals["ab_forward_calls"]
            else math.nan
        ),
        "ab_remote_a_median": (
            totals["ab_remote_a_median_sum"] / totals["ab_forward_calls"]
            if totals["ab_forward_calls"]
            else math.nan
        ),
        "ab_remote_b_norm_mean": (
            totals["ab_remote_b_norm_mean_sum"] / totals["ab_forward_calls"]
            if totals["ab_forward_calls"]
            else math.nan
        ),
        "ab_remote_b_norm_median": (
            totals["ab_remote_b_norm_median_sum"] / totals["ab_forward_calls"]
            if totals["ab_forward_calls"]
            else math.nan
        ),
        "ab_noise_l2_mean": (
            totals["ab_noise_l2_sum"] / totals["ab_forward_calls"]
            if totals["ab_forward_calls"]
            else 0.0
        ),
        "ab_noise_std_mean": (
            totals["ab_noise_std_sum"] / totals["ab_forward_calls"]
            if totals["ab_forward_calls"]
            else 0.0
        ),
        "ab_invalid_a_count": totals["ab_invalid_a_count"],
        "ab_a_count": totals["ab_a_count"],
        "ab_a_invalid_fraction": (
            totals["ab_invalid_a_count"] / totals["ab_a_count"]
            if totals["ab_a_count"]
            else 0.0
        ),
        "centroid_consistency_max": (
            totals["centroid_consistency_max"]
            if totals["centroid_consistency_count"]
            else math.nan
        ),
        **{
            f"hardest_top{percentage}_similarity_mean": (
                totals[f"hardest_top{percentage}_similarity_sum"]
                / totals[f"hardest_top{percentage}_similarity_count"]
                if totals[f"hardest_top{percentage}_similarity_count"]
                else math.nan
            )
            for percentage in (1, 5, 10)
        },
    }
