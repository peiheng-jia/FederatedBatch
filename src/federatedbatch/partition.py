"""Client partition strategies used by the manuscript experiments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


def seeded_batch_partition(
    adata,
    n_clients: int,
    *,
    seed: int,
    batch_key: str = "batch",
) -> list[list]:
    batches = np.asarray(list(adata.obs[batch_key].unique()), dtype=object)
    if n_clients > len(batches):
        raise ValueError(f"{n_clients} clients requested for only {len(batches)} batches")
    random_state = np.random.RandomState(seed)
    random_state.shuffle(batches)
    quotient, remainder = divmod(len(batches), n_clients)
    groups = []
    start = 0
    for client_index in range(n_clients):
        size = quotient + (1 if client_index < remainder else 0)
        groups.append(batches[start : start + size].tolist())
        start += size
    return groups


def partition_batches(
    adata,
    n_clients: int,
    scheme: str = "batch",
    *,
    seed: int,
) -> list[list]:
    if scheme != "batch":
        raise ValueError(f"unknown batch partition scheme: {scheme}")
    groups = seeded_batch_partition(adata, n_clients, seed=seed)
    validate_partition(adata, groups)
    return groups


def validate_partition(adata, groups: Sequence[Sequence], batch_key: str = "batch") -> None:
    if any(not group for group in groups):
        raise ValueError("client partition contains an empty client")
    flattened = [batch for group in groups for batch in group]
    expected = list(adata.obs[batch_key].unique())
    if len(flattened) != len(set(flattened)):
        raise ValueError("a batch was assigned to more than one client")
    if set(flattened) != set(expected):
        raise ValueError("client partition does not cover every batch")


def partition_cells(
    adata,
    n_clients: int,
    scheme: str,
    *,
    seed: int,
    label_key: str = "celltype",
    label_skew_fraction: float = 0.7,
    primary_clients_per_label: int = 2,
    quantity_skew_factor: float = 3.0,
) -> list[np.ndarray]:
    """Partition every cell exactly once for client-heterogeneity experiments."""

    if n_clients <= 1:
        raise ValueError("cell partition requires at least two clients")
    if adata.n_obs < n_clients:
        raise ValueError(f"{n_clients} clients requested for only {adata.n_obs} cells")
    rng = np.random.default_rng(seed)

    if scheme == "iid":
        groups = [np.sort(group) for group in np.array_split(rng.permutation(adata.n_obs), n_clients)]
    elif scheme == "label_skew":
        if not 0 < label_skew_fraction < 1:
            raise ValueError("label_skew_fraction must be between zero and one")
        if not 0 < primary_clients_per_label < n_clients:
            raise ValueError("primary_clients_per_label must be between one and n_clients - 1")
        labels = np.asarray(adata.obs[label_key].astype(str))
        grouped: list[list[int]] = [[] for _ in range(n_clients)]
        for label in np.unique(labels):
            indices = np.flatnonzero(labels == label)
            rng.shuffle(indices)
            primary = rng.choice(n_clients, size=primary_clients_per_label, replace=False)
            probabilities = np.full(
                n_clients,
                (1 - label_skew_fraction) / (n_clients - primary_clients_per_label),
            )
            probabilities[primary] = label_skew_fraction / primary_clients_per_label
            counts = rng.multinomial(len(indices), probabilities)
            offset = 0
            for client_index, count in enumerate(counts):
                grouped[client_index].extend(indices[offset : offset + count].tolist())
                offset += count
        groups = [np.sort(np.asarray(group, dtype=np.int64)) for group in grouped]
    elif scheme == "quantity_skew":
        if quantity_skew_factor <= 1:
            raise ValueError("quantity_skew_factor must be greater than one")
        weights = quantity_skew_factor ** np.arange(n_clients, dtype=float)
        expected = (adata.n_obs - n_clients) * weights / weights.sum()
        counts = np.ones(n_clients, dtype=int) + np.floor(expected).astype(int)
        remainder_order = np.argsort(-(expected - counts))
        counts[remainder_order[: adata.n_obs - counts.sum()]] += 1
        shuffled = rng.permutation(adata.n_obs)
        boundaries = np.cumsum(counts)[:-1]
        groups = [np.sort(group) for group in np.split(shuffled, boundaries)]
    else:
        raise ValueError(f"unknown cell partition scheme: {scheme}")

    validate_cell_partition(adata.n_obs, groups)
    return groups


def balanced_dirichlet_partition(
    labels: Sequence,
    n_clients: int,
    *,
    concentration: float,
    seed: int,
    weight_floor: float = 1e-4,
) -> list[np.ndarray]:
    """Generate label skew from a Dirichlet draw while fixing client capacities.

    Matrix scaling preserves each observed label count and equal client-size margins.
    Integer transportation rounding then produces an exact, non-overlapping assignment.
    """

    labels = np.asarray(labels).astype(str)
    if n_clients <= 1:
        raise ValueError("Dirichlet partition requires at least two clients")
    if len(labels) < n_clients:
        raise ValueError(f"{n_clients} clients requested for only {len(labels)} cells")
    if not np.isfinite(concentration) or concentration <= 0:
        raise ValueError("Dirichlet concentration must be positive and finite")
    if not 0 < weight_floor < 1:
        raise ValueError("Dirichlet weight floor must be between zero and one")

    cell_types, encoded = np.unique(labels, return_inverse=True)
    label_counts = np.bincount(encoded, minlength=len(cell_types)).astype(np.int64)
    capacities = np.full(n_clients, len(labels) // n_clients, dtype=np.int64)
    capacities[: len(labels) % n_clients] += 1
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.full(n_clients, concentration), size=len(cell_types))
    # A fixed numerical floor gives Sinkhorn scaling full support at alpha=0.01.
    transport = np.maximum(weights, weight_floor)
    transport *= label_counts[:, None] / transport.sum(axis=1, keepdims=True)

    for _ in range(10_000):
        transport *= capacities[None, :] / transport.sum(axis=0, keepdims=True)
        transport *= label_counts[:, None] / transport.sum(axis=1, keepdims=True)
        row_error = np.max(np.abs(transport.sum(axis=1) - label_counts))
        column_error = np.max(np.abs(transport.sum(axis=0) - capacities))
        if max(row_error, column_error) < 1e-6:
            break
    else:
        raise RuntimeError("balanced Dirichlet matrix scaling did not converge")

    allocation = np.floor(transport).astype(np.int64)
    row_deficit = label_counts - allocation.sum(axis=1)
    column_deficit = capacities - allocation.sum(axis=0)
    remaining = int(row_deficit.sum())
    if remaining:
        n_labels = len(cell_types)
        n_variables = n_labels * n_clients
        constraints = np.zeros((n_labels + n_clients, n_variables), dtype=float)
        for label_index in range(n_labels):
            start = label_index * n_clients
            constraints[label_index, start : start + n_clients] = 1
        for client_index in range(n_clients):
            constraints[n_labels + client_index, client_index::n_clients] = 1
        margins = np.concatenate([row_deficit, column_deficit]).astype(float)
        fractions = (transport - allocation).reshape(-1)
        result = milp(
            c=-fractions,
            integrality=np.ones(n_variables),
            bounds=Bounds(0, 1),
            constraints=LinearConstraint(constraints, margins, margins),
            options={"presolve": True},
        )
        if not result.success or result.x is None:
            raise RuntimeError(f"balanced Dirichlet integer rounding failed: {result.message}")
        allocation += np.rint(result.x).astype(np.int64).reshape(n_labels, n_clients)

    if not np.array_equal(allocation.sum(axis=1), label_counts):
        raise RuntimeError("balanced Dirichlet allocation changed label totals")
    if not np.array_equal(allocation.sum(axis=0), capacities):
        raise RuntimeError("balanced Dirichlet allocation violated client capacities")

    grouped: list[list[int]] = [[] for _ in range(n_clients)]
    for label_index in range(len(cell_types)):
        indices = np.flatnonzero(encoded == label_index)
        rng.shuffle(indices)
        offset = 0
        for client_index, count in enumerate(allocation[label_index]):
            grouped[client_index].extend(indices[offset : offset + count].tolist())
            offset += int(count)
    groups = [np.sort(np.asarray(group, dtype=np.int64)) for group in grouped]
    validate_cell_partition(len(labels), groups)
    if max(map(len, groups)) - min(map(len, groups)) > 1:
        raise RuntimeError("balanced Dirichlet partition introduced quantity skew")
    return groups


def validate_cell_partition(n_cells: int, groups: Sequence[Sequence[int]]) -> None:
    if any(len(group) == 0 for group in groups):
        raise ValueError("cell partition contains an empty client")
    flattened = np.concatenate([np.asarray(group, dtype=np.int64) for group in groups])
    if len(flattened) != n_cells:
        raise ValueError(f"cell partition contains {len(flattened)} assignments for {n_cells} cells")
    if len(np.unique(flattened)) != n_cells:
        raise ValueError("cell partition contains duplicate cell assignments")
    if not np.array_equal(np.sort(flattened), np.arange(n_cells)):
        raise ValueError("cell partition does not cover every cell")
