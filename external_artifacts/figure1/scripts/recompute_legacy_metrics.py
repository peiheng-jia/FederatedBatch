"""Figure 1 metric helpers copied from the accepted evaluation implementation.

The public Figure 1 driver imports only these four functions. Their formulas
are preserved from the source project; the historical command-line workflow
and its unrelated artifact dependencies are intentionally not included.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def old_asw(embedding: np.ndarray, labels: np.ndarray) -> float:
    return float(silhouette_score(embedding, labels))


def old_graph_connectivity(
    embedding: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 15,
) -> float:
    """Match the submitted directed 15-NN induced-subgraph edge density."""
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(embedding)
    _, indices = knn.kneighbors(embedding)
    n_obs = embedding.shape[0]
    rows = np.repeat(np.arange(n_obs), n_neighbors)
    cols = indices.reshape(-1)
    scores = []
    for label in np.unique(labels):
        membership = labels == label
        n_cells = int(np.count_nonzero(membership))
        if n_cells < 2:
            continue
        # The submitted sparse adjacency includes self-neighbors and is
        # counted as nnz / 2 even though the kNN graph is directed.
        row_mask = membership[rows]
        subgraph_nnz = int(np.count_nonzero(membership[cols[row_mask]]))
        n_edges = subgraph_nnz / 2.0
        max_edges = n_cells * (n_cells - 1) / 2.0
        scores.append(n_edges / max_edges)
    return float(np.mean(scores)) if scores else math.nan


def full_square_distance_median(condensed: np.ndarray, n_obs: int) -> float:
    """Return the median of the full square distance matrix without allocating it."""
    expected = n_obs * (n_obs - 1) // 2
    if condensed.ndim != 1 or condensed.size != expected:
        raise ValueError(
            f"Condensed distance size {condensed.size} does not match n={n_obs}"
        )
    total = n_obs * n_obs
    square_ranks = [(total - 1) // 2, total // 2]
    condensed_ranks = sorted(
        {(rank - n_obs) // 2 for rank in square_ranks if rank >= n_obs}
    )
    if condensed_ranks:
        condensed.partition(condensed_ranks)

    values = []
    for rank in square_ranks:
        if rank < n_obs:
            values.append(0.0)
        else:
            values.append(float(condensed[(rank - n_obs) // 2]))
    return float(np.mean(values))


def old_ilisi(
    embedding: np.ndarray,
    batches: np.ndarray,
    perplexity: float = 30.0,
    block_size: int = 512,
) -> float:
    """Match the submitted full-pairwise Gaussian inverse-Simpson iLISI."""
    n_obs = embedding.shape[0]
    if n_obs < 2:
        return math.nan

    condensed = np.empty(n_obs * (n_obs - 1) // 2, dtype=np.float64)
    pdist(embedding, metric="euclidean", out=condensed)
    sigma = full_square_distance_median(condensed, n_obs) / perplexity
    del condensed
    if not np.isfinite(sigma) or sigma <= 0:
        return 1.0

    categories = np.unique(batches)
    category_masks = [batches == category for category in categories]
    denominator = 2.0 * sigma * sigma
    score_sum = 0.0

    for start in range(0, n_obs, block_size):
        stop = min(start + block_size, n_obs)
        distances = cdist(embedding[start:stop], embedding, metric="euclidean")
        weights = np.exp(-(distances * distances) / denominator)
        local_rows = np.arange(stop - start)
        weights[local_rows, np.arange(start, stop)] = 0.0

        row_sums = weights.sum(axis=1)
        valid = row_sums > 0
        weights[valid] /= row_sums[valid, None]
        probabilities = np.column_stack(
            [weights[:, mask].sum(axis=1) for mask in category_masks]
        )
        simpson = np.square(probabilities).sum(axis=1)
        block_scores = np.ones(stop - start, dtype=np.float64)
        positive = valid & (simpson > 0)
        block_scores[positive] = 1.0 / simpson[positive]
        score_sum += float(block_scores.sum())

    return score_sum / n_obs


def literal_submitted_ilisi(
    embedding: np.ndarray,
    batches: np.ndarray,
    perplexity: float = 30.0,
) -> float:
    """Small-input reference copied structurally from the submitted code."""
    distances = squareform(pdist(embedding, metric="euclidean"))
    sigma = np.median(distances) / perplexity
    weights = np.exp(-(distances**2) / (2 * sigma**2))
    np.fill_diagonal(weights, 0)
    weights = weights / weights.sum(axis=1, keepdims=True)
    scores = []
    for row in weights:
        probabilities = np.array(
            [row[batches == category].sum() for category in np.unique(batches)]
        )
        simpson = float(np.square(probabilities).sum())
        scores.append(1.0 / simpson if simpson > 0 else 1.0)
    return float(np.mean(scores))


def validate_exact_ilisi_implementation() -> None:
    embedding = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [0.9, 0.8],
            [1.1, 1.0],
            [2.0, 0.4],
            [2.2, 0.7],
        ],
        dtype=np.float64,
    )
    batches = np.array(["a", "b", "a", "c", "b", "c"])
    expected = literal_submitted_ilisi(embedding, batches)
    observed = old_ilisi(embedding, batches, block_size=2)
    if not np.isclose(observed, expected, rtol=1e-13, atol=1e-13):
        raise RuntimeError(
            f"Blocked iLISI does not match submitted formula: {observed} != {expected}"
        )
