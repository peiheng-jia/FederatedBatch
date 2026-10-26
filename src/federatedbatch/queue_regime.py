"""Post-hoc diagnostics for FreshQueue applicability experiments."""

from __future__ import annotations

import math
import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from torch.nn import functional as F

from .federated import FederatedRoundSnapshot, _predict_queue_embeddings


PAIR_METRICS = (
    "ARI_mean",
    "NMI_mean",
    "ASW_cell",
    "ASW_batch",
    "graph_connectivity",
    "iLISI_proxy",
    "knn_celltype_purity_k15",
)


@contextmanager
def preserve_random_states():
    """Prevent read-only inference diagnostics from advancing training RNGs."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _normalized_rows(values: np.ndarray) -> np.ndarray:
    tensor = F.normalize(torch.as_tensor(values, dtype=torch.float32), dim=1)
    return tensor.cpu().numpy()


def composition_statistics(
    labels: np.ndarray,
    client_indices: list[np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Return long-form client compositions and pairwise heterogeneity descriptors."""

    labels = np.asarray(labels).astype(str)
    cell_types = np.unique(labels)
    distributions = []
    rows = []
    for client, indices in enumerate(client_indices):
        client_labels = labels[np.asarray(indices, dtype=np.int64)]
        counts = np.asarray([(client_labels == label).sum() for label in cell_types], dtype=float)
        proportions = counts / counts.sum()
        distributions.append(proportions)
        for label, count, proportion in zip(cell_types, counts, proportions):
            rows.append(
                {
                    "client": client,
                    "cell_type": label,
                    "cell_count": int(count),
                    "proportion": float(proportion),
                }
            )

    overlaps = []
    divergences = []
    for first in range(len(distributions)):
        for second in range(first + 1, len(distributions)):
            overlaps.append(float(np.minimum(distributions[first], distributions[second]).sum()))
            divergences.append(
                float(jensenshannon(distributions[first], distributions[second], base=2.0) ** 2)
            )
    return rows, {
        "mean_pairwise_composition_overlap": float(np.mean(overlaps)),
        "mean_pairwise_jensen_shannon_divergence": float(np.mean(divergences)),
    }


def mean_cosine_distance(previous: np.ndarray, current: np.ndarray) -> float:
    if previous.shape != current.shape:
        raise ValueError("adjacent reference embeddings must have identical shapes")
    first = _normalized_rows(previous)
    second = _normalized_rows(current)
    return float(np.mean(1.0 - np.sum(first * second, axis=1)))


def balanced_sample_indices(
    source_clients: torch.Tensor,
    max_entries: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Reproduce FreshQueue's balanced source-client sampling and retain indices."""

    if max_entries <= 0:
        raise ValueError("max_entries must be positive")
    if len(source_clients) <= max_entries:
        return torch.arange(len(source_clients), device=source_clients.device)
    source_values = torch.unique(source_clients, sorted=True)
    source_order = source_values[
        torch.randperm(
            len(source_values), device=source_values.device, generator=generator
        )
    ]
    shuffled_by_source = []
    for source in source_order:
        indices = torch.nonzero(source_clients == source, as_tuple=False).flatten()
        shuffled_by_source.append(
            indices[
                torch.randperm(
                    len(indices), device=indices.device, generator=generator
                )
            ]
        )
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
    return torch.stack(selected)


def queue_false_negative_mass(
    anchor_embeddings: np.ndarray,
    anchor_clients: np.ndarray,
    anchor_labels: np.ndarray,
    queue_embeddings: np.ndarray,
    queue_source_clients: np.ndarray,
    queue_labels: np.ndarray,
    *,
    batch_size: int | list[int],
    temperature: float,
    mask_fraction: float,
    seed: int,
    sampling_repeats: int,
    queue_negative_ratio: float | None = None,
    queue_entries_used: int | None = None,
) -> dict[str, float | int]:
    """Estimate same-label softmax mass using labels only in this post-hoc function."""

    if len(anchor_embeddings) != len(anchor_clients) or len(anchor_embeddings) != len(
        anchor_labels
    ):
        raise ValueError("anchor embeddings, clients, and labels must align")
    if len(queue_embeddings) != len(queue_source_clients) or len(queue_embeddings) != len(
        queue_labels
    ):
        raise ValueError("queue embeddings, source clients, and labels must align")
    if sampling_repeats <= 0:
        raise ValueError("sampling_repeats must be positive")
    anchors = F.normalize(torch.as_tensor(anchor_embeddings, dtype=torch.float32), dim=1)
    anchor_clients = torch.as_tensor(anchor_clients, dtype=torch.long)
    anchor_labels = np.asarray(anchor_labels).astype(str)
    queue = F.normalize(torch.as_tensor(queue_embeddings, dtype=torch.float32), dim=1)
    queue_sources = torch.as_tensor(queue_source_clients, dtype=torch.long)
    queue_labels = np.asarray(queue_labels).astype(str)
    pre_mask_masses = []
    post_mask_masses = []
    same_type_fractions = []
    sampled_entry_counts = []
    evaluated_anchors = 0
    for target_client in torch.unique(anchor_clients, sorted=True).tolist():
        anchor_indices = torch.nonzero(
            anchor_clients == target_client, as_tuple=False
        ).flatten()
        candidate_indices = torch.nonzero(
            queue_sources != target_client, as_tuple=False
        ).flatten()
        if not len(anchor_indices) or not len(candidate_indices):
            continue
        client_anchors = anchors[anchor_indices]
        candidate_sources = queue_sources[candidate_indices]
        client_anchor_labels = anchor_labels[anchor_indices.cpu().numpy()]
        client_batch_size = (
            int(batch_size[target_client])
            if isinstance(batch_size, list)
            else int(batch_size)
        )
        max_queue_entries = client_batch_size
        if queue_entries_used is not None:
            max_queue_entries = int(queue_entries_used)
        elif queue_negative_ratio is not None:
            local_negative_count = 2 * client_batch_size - 2
            max_queue_entries = math.floor(queue_negative_ratio * local_negative_count)
        if max_queue_entries <= 0:
            continue
        for repeat in range(sampling_repeats):
            generator = torch.Generator().manual_seed(
                int(seed) + int(target_client) * 10_007 + repeat * 1_000_003
            )
            local_indices = balanced_sample_indices(
                candidate_sources, max_queue_entries, generator
            )
            sampled_indices = candidate_indices[local_indices]
            sampled = queue[sampled_indices]
            sampled_entry_counts.append(len(sampled))
            sampled_labels = queue_labels[sampled_indices.cpu().numpy()]
            logits = client_anchors @ sampled.T / float(temperature)
            same_type = torch.as_tensor(
                client_anchor_labels[:, None] == sampled_labels[None, :],
                dtype=torch.bool,
            )
            same_type_fractions.extend(
                same_type.float().mean(dim=1).cpu().numpy().tolist()
            )
            pre_weights = torch.softmax(logits, dim=1)
            pre_mask_masses.extend(
                (pre_weights * same_type).sum(dim=1).cpu().numpy().tolist()
            )

            mask = torch.zeros_like(logits, dtype=torch.bool)
            if mask_fraction > 0 and logits.shape[1] > 0:
                mask_count = min(
                    logits.shape[1], max(1, math.ceil(mask_fraction * logits.shape[1]))
                )
                mask.scatter_(1, logits.topk(mask_count, dim=1).indices, True)
            post_logits = logits.masked_fill(mask, float("-inf"))
            post_weights = torch.softmax(post_logits, dim=1)
            post_mask_masses.extend(
                (post_weights * same_type).sum(dim=1).cpu().numpy().tolist()
            )
            evaluated_anchors += len(anchor_indices)
    return {
        "queue_false_negative_fraction": float(np.mean(same_type_fractions)),
        "queue_false_negative_fraction_median": float(np.median(same_type_fractions)),
        "queue_false_negative_mass_pre_mask": float(np.mean(pre_mask_masses)),
        "queue_false_negative_mass": float(np.mean(post_mask_masses)),
        "queue_false_negative_mass_sd": float(np.std(post_mask_masses, ddof=1)),
        "queue_false_negative_anchor_evaluations": evaluated_anchors,
        "queue_false_negative_sampled_entries_mean": float(
            np.mean(sampled_entry_counts)
        ) if sampled_entry_counts else math.nan,
    }


@dataclass
class _QueueSnapshot:
    round_number: int
    embeddings: np.ndarray
    records: tuple[dict[str, Any], ...]


@dataclass
class _ReferenceSnapshot:
    round_number: int
    embeddings: np.ndarray
    client_ids: np.ndarray
    cell_ids: np.ndarray


class RoundMechanismRecorder:
    """Collect label-free round snapshots, then evaluate labels after training."""

    def __init__(
        self,
        reference_ids_by_client: list[np.ndarray],
        *,
        method: str,
        training_seed: int,
    ):
        self.reference_ids_by_client = reference_ids_by_client
        self.method = method
        self.training_seed = int(training_seed)
        self.previous_reference: np.ndarray | None = None
        self.rows: list[dict[str, Any]] = []
        self._queue_snapshots: list[_QueueSnapshot] = []
        self._reference_snapshots: list[_ReferenceSnapshot] = []

    def __call__(self, snapshot: FederatedRoundSnapshot) -> None:
        references = []
        with preserve_random_states(), torch.no_grad():
            for client_index, (model, client, reference_ids) in enumerate(
                zip(snapshot.models, snapshot.clients, self.reference_ids_by_client)
            ):
                positions = client.obs_names.get_indexer(reference_ids)
                if (positions < 0).any():
                    raise RuntimeError("fixed reference cell is absent from its client")
                reference = client[positions].copy()
                encoded = _predict_queue_embeddings(
                    model,
                    reference,
                    seed=(
                        self.training_seed * 1_000_003
                        + snapshot.round_number * 1_009
                        + client_index
                        + 100_000_007
                    ),
                )
                references.append(_normalized_rows(encoded))
        current_reference = np.concatenate(references, axis=0)
        drift = (
            math.nan
            if self.previous_reference is None
            else mean_cosine_distance(self.previous_reference, current_reference)
        )
        self.previous_reference = current_reference
        reference_clients = np.concatenate(
            [
                np.full(len(ids), client, dtype=np.int64)
                for client, ids in enumerate(self.reference_ids_by_client)
            ]
        )
        reference_ids = np.concatenate(self.reference_ids_by_client)
        self._reference_snapshots.append(
            _ReferenceSnapshot(
                round_number=snapshot.round_number,
                embeddings=current_reference.copy(),
                client_ids=reference_clients,
                cell_ids=reference_ids.copy(),
            )
        )
        self.rows.append(
            {
                "method": self.method,
                "round": snapshot.round_number,
                "representation_drift": drift,
                "queue_was_active": snapshot.queue_was_active,
                "generated_queue_size": len(snapshot.queue_embeddings),
            }
        )
        if len(snapshot.queue_embeddings):
            self._queue_snapshots.append(
                _QueueSnapshot(
                    round_number=snapshot.round_number,
                    embeddings=np.asarray(snapshot.queue_embeddings).copy(),
                    records=snapshot.queue_records,
                )
            )

    def finalize(
        self,
        adata,
        *,
        n_rounds: int,
        warmup_rounds: int,
        batch_size: int | list[int],
        temperature: float,
        mask_fraction: float,
        sampling_repeats: int,
        queue_negative_ratio: float | None = None,
        queue_candidates_by_round: dict[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        """Use ground-truth labels only after local/federated training has ended."""

        label_lookup = adata.obs["celltype"].astype(str).to_dict()
        by_round = {int(row["round"]): row for row in self.rows}
        queue_by_round = {snapshot.round_number: snapshot for snapshot in self._queue_snapshots}
        for reference in self._reference_snapshots:
            row = by_round[reference.round_number]
            if not bool(row["queue_was_active"]):
                continue
            queue_snapshot = queue_by_round.get(reference.round_number - 1)
            if queue_snapshot is None:
                raise RuntimeError("active FreshQueue round has no preceding queue snapshot")
            anchor_labels = np.asarray(
                [label_lookup[cell_id] for cell_id in reference.cell_ids]
            )
            queue_labels = np.asarray(
                [label_lookup[record["cell_id"]] for record in queue_snapshot.records]
            )
            queue_sources = np.asarray(
                [int(record["client"]) for record in queue_snapshot.records],
                dtype=np.int64,
            )
            mass = queue_false_negative_mass(
                reference.embeddings,
                reference.client_ids,
                anchor_labels,
                queue_snapshot.embeddings,
                queue_sources,
                queue_labels,
                batch_size=batch_size,
                temperature=temperature,
                mask_fraction=mask_fraction,
                seed=self.training_seed * 10_009 + reference.round_number,
                sampling_repeats=sampling_repeats,
                queue_negative_ratio=queue_negative_ratio,
                queue_entries_used=(
                    queue_candidates_by_round.get(reference.round_number)
                    if queue_candidates_by_round is not None
                    else None
                ),
            )
            row.update(
                {
                    **mass,
                    "queue_source_round": queue_snapshot.round_number,
                }
            )
        for row in self.rows:
            row.setdefault("queue_false_negative_fraction", math.nan)
            row.setdefault("queue_false_negative_fraction_median", math.nan)
            row.setdefault("queue_false_negative_mass_pre_mask", math.nan)
            row.setdefault("queue_false_negative_mass", math.nan)
            row.setdefault("queue_false_negative_mass_sd", math.nan)
            row.setdefault("queue_false_negative_anchor_evaluations", 0)
            row.setdefault("queue_false_negative_sampled_entries_mean", math.nan)
            row.setdefault("queue_source_round", math.nan)
        self._queue_snapshots.clear()
        self._reference_snapshots.clear()
        return self.rows


def paired_setting_results(
    method_metrics: pd.DataFrame,
    descriptors: pd.DataFrame,
    round_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Produce one analysis row per setting and paired training seed."""

    rows = []
    identifiers = ["setting", "dataset", "seed", "partition", "n_clients", "batch_size"]
    if method_metrics.empty:
        return pd.DataFrame()
    for keys, subset in method_metrics.groupby(identifiers, sort=False):
        if set(subset["method"]) != {"NoQueue", "FreshQueue"}:
            continue
        indexed = subset.set_index("method")
        row = dict(zip(identifiers, keys))
        for metric in PAIR_METRICS:
            no_queue = float(indexed.loc["NoQueue", metric])
            fresh_queue = float(indexed.loc["FreshQueue", metric])
            row[f"NoQueue_{metric}"] = no_queue
            row[f"FreshQueue_{metric}"] = fresh_queue
            row[f"delta_{metric.removesuffix('_mean')}"] = fresh_queue - no_queue
        descriptor = descriptors.loc[
            (descriptors["setting"] == row["setting"])
            & (descriptors["seed"] == row["seed"])
        ]
        if len(descriptor) != 1:
            raise RuntimeError("one composition descriptor row is required per pair")
        row.update(
            descriptor.iloc[0][
                [
                    "mean_pairwise_composition_overlap",
                    "mean_pairwise_jensen_shannon_divergence",
                ]
            ].to_dict()
        )
        diagnostics = round_diagnostics.loc[
            (round_diagnostics["setting"] == row["setting"])
            & (round_diagnostics["seed"] == row["seed"])
        ]
        fresh = diagnostics.loc[diagnostics["method"] == "FreshQueue"]
        baseline = diagnostics.loc[diagnostics["method"] == "NoQueue"]
        active = fresh.loc[
            fresh["queue_was_active"].astype(str).str.lower().isin({"true", "1"})
        ]
        used_queues = active
        matched_rounds = baseline.loc[baseline["round"].isin(active["round"])]
        row["representation_drift_mean"] = float(active["representation_drift"].mean())
        row["NoQueue_representation_drift_mean"] = float(
            matched_rounds["representation_drift"].mean()
        )
        row["queue_false_negative_mass"] = float(
            used_queues["queue_false_negative_mass"].mean()
        )
        row["queue_false_negative_mass_pre_mask"] = float(
            used_queues["queue_false_negative_mass_pre_mask"].mean()
        )
        row["queue_false_negative_fraction"] = (
            float(used_queues["queue_false_negative_fraction"].mean())
            if "queue_false_negative_fraction" in used_queues
            else math.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def setting_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if paired.empty:
        return pd.DataFrame()
    scalar_columns = [
        column
        for column in paired.columns
        if column.startswith("delta_")
        or column
        in {
            "mean_pairwise_composition_overlap",
            "mean_pairwise_jensen_shannon_divergence",
            "representation_drift_mean",
            "NoQueue_representation_drift_mean",
            "queue_false_negative_mass",
            "queue_false_negative_mass_pre_mask",
            "queue_false_negative_fraction",
        }
    ]
    for (setting, dataset), subset in paired.groupby(["setting", "dataset"], sort=False):
        row: dict[str, Any] = {
            "setting": setting,
            "dataset": dataset,
            "n_paired_seeds": int(subset["seed"].nunique()),
            "positive_ARI_seeds": int((subset["delta_ARI"] > 0).sum()),
        }
        for column in scalar_columns:
            row[f"{column}_mean"] = float(subset[column].mean())
            row[f"{column}_sd"] = float(subset[column].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def candidate_settings(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    output = summary.copy()
    output["check_complete_3_seeds"] = output["n_paired_seeds"] == 3
    output["check_ARI_positive_3_of_3"] = output["positive_ARI_seeds"] == 3
    output["check_mean_delta_ARI_at_least_0.02"] = output["delta_ARI_mean"] >= 0.02
    output["check_mean_delta_NMI_not_lower"] = output["delta_NMI_mean"] >= 0
    output["check_mean_delta_ASW_cell_not_lower"] = output["delta_ASW_cell_mean"] >= 0
    output["check_mean_delta_knn_purity_not_lower"] = (
        output["delta_knn_celltype_purity_k15_mean"] >= 0
    )
    checks = [column for column in output if column.startswith("check_")]
    output["confirmation_candidate"] = output[checks].all(axis=1)
    return output


def regime_associations(paired: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    descriptors = (
        "mean_pairwise_composition_overlap",
        "mean_pairwise_jensen_shannon_divergence",
        "representation_drift_mean",
        "queue_false_negative_mass",
    )
    rows = []
    if paired.empty or summary.empty:
        return pd.DataFrame()
    setting_level = summary.rename(
        columns={
            **{descriptor + "_mean": descriptor for descriptor in descriptors},
            "delta_ARI_mean": "delta_ARI",
        }
    )
    for level, frame in (("setting_seed", paired), ("setting_mean", setting_level)):
        for descriptor in descriptors:
            values = frame[[descriptor, "delta_ARI"]].dropna()
            if len(values) < 3 or values[descriptor].nunique() < 2:
                rho, p_value = math.nan, math.nan
            else:
                result = spearmanr(values[descriptor], values["delta_ARI"])
                rho, p_value = float(result.statistic), float(result.pvalue)
            rows.append(
                {
                    "analysis_level": level,
                    "descriptor": descriptor,
                    "n": len(values),
                    "spearman_rho_with_delta_ARI": rho,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(rows)
