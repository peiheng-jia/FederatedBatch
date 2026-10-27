from __future__ import annotations

import math

import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    silhouette_samples,
    silhouette_score,
)


def _silhouette_or_nan(values: np.ndarray, labels) -> float:
    if len(np.unique(np.asarray(labels))) < 2:
        return math.nan
    return float(silhouette_score(values, labels))


def scib_silhouette_scores(
    values: np.ndarray,
    labels,
    batches,
) -> dict[str, float]:
    """Compute scaled cell-type ASW and cell-type-stratified batch ASW."""
    labels = np.asarray(labels)
    batches = np.asarray(batches)
    raw_celltype = _silhouette_or_nan(values, labels)
    celltype_asw = (raw_celltype + 1) / 2 if np.isfinite(raw_celltype) else math.nan
    batch_group_means = []
    for label in np.unique(labels):
        mask = labels == label
        group_batches = batches[mask]
        n_batches = len(np.unique(group_batches))
        if n_batches == 1 or n_batches == mask.sum():
            continue
        group_silhouettes = silhouette_samples(values[mask], group_batches)
        batch_group_means.append(float(np.mean(1 - np.abs(group_silhouettes))))
    return {
        "ASW_batch": float(np.mean(batch_group_means)) if batch_group_means else math.nan,
        "ASW_celltype": float(celltype_asw),
        "Silhouette_celltype_raw": float(raw_celltype),
    }


def graph_connectivity(connectivities, labels) -> float:
    """Mean largest-component fraction within each biological label."""

    graph = csr_matrix(connectivities)
    labels = np.asarray(labels)
    scores = []
    for label in np.unique(labels):
        mask = labels == label
        subgraph = graph[mask][:, mask]
        if subgraph.shape[0] == 0:
            continue
        _, components = connected_components(subgraph, directed=False)
        largest = np.bincount(components).max()
        scores.append(largest / subgraph.shape[0])
    return float(np.mean(scores)) if scores else math.nan


def knn_inverse_simpson(connectivities, labels) -> float:
    """Weighted inverse Simpson index on an existing kNN graph."""

    graph = csr_matrix(connectivities)
    labels = np.asarray(labels)
    scores = []
    for row_index in range(graph.shape[0]):
        start, stop = graph.indptr[row_index : row_index + 2]
        neighbors = graph.indices[start:stop]
        weights = graph.data[start:stop].astype(float, copy=False)
        keep = neighbors != row_index
        neighbors = neighbors[keep]
        weights = weights[keep]
        total = weights.sum()
        if total <= 0:
            continue
        probabilities = []
        for label in np.unique(labels[neighbors]):
            probabilities.append(weights[labels[neighbors] == label].sum() / total)
        denominator = np.square(probabilities).sum()
        if denominator > 0:
            scores.append(1.0 / denominator)
    return float(np.mean(scores)) if scores else math.nan


def cross_batch_label_transfer(connectivities, labels, batches) -> dict[str, float]:
    """Predict each cell from graph neighbors belonging to other batches only."""
    graph = csr_matrix(connectivities)
    labels = np.asarray(labels).astype(str)
    batches = np.asarray(batches).astype(str)
    observed = []
    predicted = []
    for row_index in range(graph.shape[0]):
        start, stop = graph.indptr[row_index : row_index + 2]
        neighbors = graph.indices[start:stop]
        weights = graph.data[start:stop].astype(float, copy=False)
        keep = (neighbors != row_index) & (batches[neighbors] != batches[row_index])
        neighbors = neighbors[keep]
        weights = weights[keep]
        if not len(neighbors):
            continue
        scores = {}
        for label, weight in zip(labels[neighbors], weights):
            scores[label] = scores.get(label, 0.0) + float(weight)
        prediction = min(scores, key=lambda label: (-scores[label], label))
        observed.append(labels[row_index])
        predicted.append(prediction)
    coverage = len(observed) / graph.shape[0] if graph.shape[0] else math.nan
    if not observed:
        return {
            "Label_transfer_accuracy": math.nan,
            "Label_transfer_macro_F1": math.nan,
            "Label_transfer_coverage": coverage,
        }
    return {
        "Label_transfer_accuracy": float(accuracy_score(observed, predicted)),
        "Label_transfer_macro_F1": float(f1_score(observed, predicted, average="macro")),
        "Label_transfer_coverage": float(coverage),
    }


def evaluate_embedding(
    adata,
    embeddings: np.ndarray,
    config: dict,
    *,
    embedding_key: str = "X_eval",
    label_key: str = "celltype",
    batch_key: str = "batch",
) -> dict[str, float]:
    if embeddings.shape[0] != adata.n_obs:
        raise ValueError(
            f"embedding rows ({embeddings.shape[0]}) do not match cells ({adata.n_obs})"
        )
    adata.obsm[embedding_key] = np.asarray(embeddings)
    evaluation = config["evaluation"]
    neighbors_key = f"neighbors_{embedding_key}"
    sc.pp.neighbors(
        adata,
        use_rep=embedding_key,
        n_neighbors=evaluation["n_neighbors"],
        random_state=config["seed"],
        key_added=neighbors_key,
    )
    cluster_key = f"leiden_{embedding_key}"
    sc.tl.leiden(
        adata,
        neighbors_key=neighbors_key,
        key_added=cluster_key,
        resolution=evaluation["leiden_resolution"],
        random_state=config["seed"],
    )
    connectivities = adata.obsp[f"{neighbors_key}_connectivities"]
    inverse_simpson = knn_inverse_simpson(connectivities, adata.obs[batch_key])
    n_batches = adata.obs[batch_key].nunique()
    scaled_inverse_simpson = (
        (inverse_simpson - 1) / (n_batches - 1) if n_batches > 1 else math.nan
    )
    return {
        "ARI": float(adjusted_rand_score(adata.obs[label_key], adata.obs[cluster_key])),
        "NMI": float(normalized_mutual_info_score(adata.obs[label_key], adata.obs[cluster_key])),
        **scib_silhouette_scores(
            embeddings,
            adata.obs[label_key],
            adata.obs[batch_key],
        ),
        "Graph_connectivity": graph_connectivity(connectivities, adata.obs[label_key]),
        "kNN_batch_mixing_unscaled": inverse_simpson,
        "kNN_batch_mixing_scaled": float(scaled_inverse_simpson),
        **cross_batch_label_transfer(
            connectivities,
            adata.obs[label_key],
            adata.obs[batch_key],
        ),
    }


def clear_evaluation_artifacts(adata, embedding_key: str) -> None:
    """Remove Scanpy artifacts created for one temporary embedding evaluation."""

    neighbors_key = f"neighbors_{embedding_key}"
    cluster_key = f"leiden_{embedding_key}"
    neighbors = adata.uns.get(neighbors_key, {})
    obsp_keys = (
        neighbors.get("connectivities_key", f"{neighbors_key}_connectivities"),
        neighbors.get("distances_key", f"{neighbors_key}_distances"),
    )
    for key in obsp_keys:
        if key in adata.obsp:
            del adata.obsp[key]
    if neighbors_key in adata.uns:
        del adata.uns[neighbors_key]
    if cluster_key in adata.obs:
        del adata.obs[cluster_key]
    if embedding_key in adata.obsm:
        del adata.obsm[embedding_key]
