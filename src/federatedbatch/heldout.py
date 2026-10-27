from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from .evaluation import graph_connectivity, knn_inverse_simpson, scib_silhouette_scores


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.finfo(float).eps)


def exact_knn_purity(embeddings: np.ndarray, labels: Iterable, k: int = 15) -> float:
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels).astype(str)
    if len(embeddings) <= k:
        return math.nan
    neighbors = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="brute")
    raw = neighbors.fit(embeddings).kneighbors(embeddings, return_distance=False)
    selected = np.empty((len(embeddings), k), dtype=np.int64)
    for row_index, candidates in enumerate(raw):
        without_self = candidates[candidates != row_index][:k]
        if len(without_self) != k:
            return math.nan
        selected[row_index] = without_self
    return float(np.mean(labels[selected] == labels[:, None]))


def evaluate_heldout_embedding(
    adata,
    embeddings: np.ndarray,
    *,
    leiden_seeds: Iterable[int],
    resolution: float,
    n_neighbors: int,
    purity_k: int,
    identity: str,
) -> tuple[list[dict], dict[str, float]]:
    """Evaluate held-out cells without treating their batches as a mixing endpoint."""

    embeddings = np.asarray(embeddings)
    if embeddings.shape[0] != adata.n_obs:
        raise ValueError("held-out embedding rows do not match held-out cells")
    key = f"X_heldout_{identity}"
    neighbors_key = f"neighbors_{key}"
    adata.obsm[key] = embeddings
    sc.pp.neighbors(
        adata,
        use_rep=key,
        n_neighbors=min(int(n_neighbors), adata.n_obs - 1),
        random_state=0,
        key_added=neighbors_key,
    )
    labels = np.asarray(adata.obs["celltype"]).astype(str)
    repeats = []
    for seed in leiden_seeds:
        cluster_key = f"leiden_{key}_{seed}"
        sc.tl.leiden(
            adata,
            neighbors_key=neighbors_key,
            key_added=cluster_key,
            resolution=float(resolution),
            random_state=int(seed),
        )
        clusters = np.asarray(adata.obs[cluster_key])
        repeats.append(
            {
                "leiden_random_state": int(seed),
                "ARI": float(adjusted_rand_score(labels, clusters)),
                "NMI": float(normalized_mutual_info_score(labels, clusters)),
                "cluster_number": int(len(np.unique(clusters))),
            }
        )
        del adata.obs[cluster_key]
    connectivities = adata.obsp[f"{neighbors_key}_connectivities"]
    raw_asw = float(silhouette_score(embeddings, labels)) if len(np.unique(labels)) > 1 else math.nan
    metrics = {
        "ARI_mean": float(np.mean([row["ARI"] for row in repeats])),
        "ARI_sd_leiden": float(np.std([row["ARI"] for row in repeats], ddof=1)),
        "NMI_mean": float(np.mean([row["NMI"] for row in repeats])),
        "NMI_sd_leiden": float(np.std([row["NMI"] for row in repeats], ddof=1)),
        "ASW_cell": (raw_asw + 1) / 2 if np.isfinite(raw_asw) else math.nan,
        "graph_connectivity": graph_connectivity(connectivities, labels),
        "knn_celltype_purity_k15": exact_knn_purity(embeddings, labels, purity_k),
    }
    metadata = adata.uns[neighbors_key]
    for graph_key in (metadata["connectivities_key"], metadata["distances_key"]):
        del adata.obsp[graph_key]
    del adata.uns[neighbors_key]
    del adata.obsm[key]
    return repeats, metrics


def evaluate_joint_embedding(
    adata,
    embeddings: np.ndarray,
    *,
    n_neighbors: int,
    purity_k: int,
    identity: str,
) -> dict[str, float]:
    """Evaluate biological structure and batch mixing on train plus held-out cells."""

    embeddings = np.asarray(embeddings)
    key = f"X_joint_{identity}"
    neighbors_key = f"neighbors_{key}"
    adata.obsm[key] = embeddings
    sc.pp.neighbors(
        adata,
        use_rep=key,
        n_neighbors=int(n_neighbors),
        random_state=0,
        key_added=neighbors_key,
    )
    graph = adata.obsp[f"{neighbors_key}_connectivities"]
    scores = scib_silhouette_scores(embeddings, adata.obs["celltype"], adata.obs["batch"])
    ilisi = knn_inverse_simpson(graph, adata.obs["batch"])
    n_batches = int(adata.obs["batch"].nunique())
    metrics = {
        "ASW_cell": scores["ASW_celltype"],
        "ASW_batch": scores["ASW_batch"],
        "graph_connectivity": graph_connectivity(graph, adata.obs["celltype"]),
        "iLISI_proxy": ilisi,
        "iLISI_proxy_scaled": (ilisi - 1) / (n_batches - 1) if n_batches > 1 else math.nan,
        "knn_celltype_purity_k15": exact_knn_purity(
            embeddings, adata.obs["celltype"], purity_k
        ),
    }
    metadata = adata.uns[neighbors_key]
    for graph_key in (metadata["connectivities_key"], metadata["distances_key"]):
        del adata.obsp[graph_key]
    del adata.uns[neighbors_key]
    del adata.obsm[key]
    return metrics


def representation_consistency(
    train_embeddings: np.ndarray,
    train_labels: Iterable,
    train_clients: Iterable[int],
    heldout_embeddings: np.ndarray,
    heldout_labels: Iterable,
) -> tuple[dict[str, float], list[dict]]:
    """Compare held-out cell-type centroids with per-client training centroids.

    Between-type separation is the mean cosine distance from each held-out type
    centroid to every differently labelled training-client centroid. Each
    available held-out-type/training-client match receives equal weight.
    """

    train_embeddings = _normalize_rows(train_embeddings)
    heldout_embeddings = _normalize_rows(heldout_embeddings)
    train_labels = np.asarray(train_labels).astype(str)
    heldout_labels = np.asarray(heldout_labels).astype(str)
    train_clients = np.asarray(train_clients, dtype=int)

    train_centroids: dict[tuple[int, str], np.ndarray] = {}
    for client in np.unique(train_clients):
        for label in np.unique(train_labels[train_clients == client]):
            mask = (train_clients == client) & (train_labels == label)
            train_centroids[(int(client), label)] = _normalize_rows(
                train_embeddings[mask].mean(axis=0, keepdims=True)
            )[0]
    heldout_centroids = {
        label: _normalize_rows(
            heldout_embeddings[heldout_labels == label].mean(axis=0, keepdims=True)
        )[0]
        for label in np.unique(heldout_labels)
    }

    rows = []
    for heldout_type, heldout_centroid in heldout_centroids.items():
        different = [
            1 - float(np.dot(heldout_centroid, centroid))
            for (_, train_type), centroid in train_centroids.items()
            if train_type != heldout_type
        ]
        for (client, train_type), train_centroid in train_centroids.items():
            if train_type != heldout_type:
                continue
            cosine = float(np.clip(np.dot(heldout_centroid, train_centroid), -1, 1))
            rows.append(
                {
                    "celltype": heldout_type,
                    "training_client": client,
                    "heldout_cell_count": int(np.sum(heldout_labels == heldout_type)),
                    "training_cell_count": int(
                        np.sum((train_clients == client) & (train_labels == heldout_type))
                    ),
                    "same_type_alignment": cosine,
                    "same_type_dispersion": 1 - cosine,
                    "between_type_separation": float(np.mean(different)) if different else math.nan,
                }
            )
    if not rows:
        return {
            "SameTypeAlignment": math.nan,
            "SameTypeCrossClientDispersion": math.nan,
            "BetweenTypeSeparation": math.nan,
            "RepresentationScore": math.nan,
            "shared_cell_types": 0,
            "matched_client_type_pairs": 0,
        }, []
    alignment = float(np.mean([row["same_type_alignment"] for row in rows]))
    dispersion = float(np.mean([row["same_type_dispersion"] for row in rows]))
    separation = float(np.mean([row["between_type_separation"] for row in rows]))
    summary = {
        "SameTypeAlignment": alignment,
        "SameTypeCrossClientDispersion": dispersion,
        "BetweenTypeSeparation": separation,
        "RepresentationScore": separation / max(dispersion, np.finfo(float).eps),
        "shared_cell_types": int(len({row["celltype"] for row in rows})),
        "matched_client_type_pairs": int(len(rows)),
    }
    return summary, rows


def paired_summary(values: Iterable[float]) -> dict[str, float | int]:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "n_pairs": 0,
            "mean": math.nan,
            "sd": math.nan,
            "median": math.nan,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
            "positive_count": 0,
            "paired_t_pvalue": math.nan,
            "wilcoxon_pvalue": math.nan,
        }
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
    if len(values) > 1:
        margin = float(stats.t.ppf(0.975, len(values) - 1) * sd / math.sqrt(len(values)))
        t_pvalue = float(stats.ttest_1samp(values, 0).pvalue)
    else:
        margin = math.nan
        t_pvalue = math.nan
    if np.allclose(values, 0):
        wilcoxon_pvalue = 1.0
    else:
        try:
            wilcoxon_pvalue = float(stats.wilcoxon(values, method="auto").pvalue)
        except ValueError:
            wilcoxon_pvalue = math.nan
    return {
        "n_pairs": int(len(values)),
        "mean": mean,
        "sd": sd,
        "median": float(np.median(values)),
        "ci95_low": mean - margin if np.isfinite(margin) else math.nan,
        "ci95_high": mean + margin if np.isfinite(margin) else math.nan,
        "positive_count": int(np.sum(values > 0)),
        "paired_t_pvalue": t_pvalue,
        "wilcoxon_pvalue": wilcoxon_pvalue,
    }
