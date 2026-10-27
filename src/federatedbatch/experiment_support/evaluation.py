from __future__ import annotations

import numpy as np

import scanpy as sc

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from sklearn.neighbors import NearestNeighbors

from federatedbatch.evaluation import graph_connectivity, knn_inverse_simpson, scib_silhouette_scores

def exact_knn_purity(embeddings: np.ndarray, labels, k: int) -> float:
    labels = np.asarray(labels).astype(str)
    model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="brute")
    raw = model.fit(embeddings).kneighbors(embeddings, return_distance=False)
    neighbors = np.empty((len(embeddings), k), dtype=np.int64)
    for index, candidates in enumerate(raw):
        selected = candidates[candidates != index][:k]
        if len(selected) != k:
            raise RuntimeError(f"could not identify {k} non-self neighbors")
        neighbors[index] = selected
    return float(np.mean(labels[neighbors] == labels[:, None]))

def evaluate_embedding(adata, embeddings, config: dict, identity: str):
    key = f"X_regime_{identity}"
    neighbors_key = f"neighbors_{key}"
    adata.obsm[key] = np.asarray(embeddings)
    sc.pp.neighbors(
        adata,
        use_rep=key,
        n_neighbors=int(config["evaluation"]["n_neighbors"]),
        random_state=0,
        key_added=neighbors_key,
    )
    labels = np.asarray(adata.obs["celltype"])
    repeats = []
    for leiden_seed in config["analysis"]["leiden_random_states"]:
        cluster_key = f"leiden_{key}_{leiden_seed}"
        sc.tl.leiden(
            adata,
            neighbors_key=neighbors_key,
            key_added=cluster_key,
            resolution=1.0,
            random_state=int(leiden_seed),
        )
        clusters = np.asarray(adata.obs[cluster_key])
        repeats.append(
            {
                "leiden_random_state": int(leiden_seed),
                "ARI": float(adjusted_rand_score(labels, clusters)),
                "NMI": float(normalized_mutual_info_score(labels, clusters)),
                "cluster_number": int(len(np.unique(clusters))),
            }
        )
        del adata.obs[cluster_key]
    connectivities = adata.obsp[f"{neighbors_key}_connectivities"]
    silhouette = scib_silhouette_scores(
        embeddings, adata.obs["celltype"], adata.obs["batch"]
    )
    ilisi = knn_inverse_simpson(connectivities, adata.obs["batch"])
    n_batches = adata.obs["batch"].nunique()
    metrics = {
        "ARI_mean": float(np.mean([row["ARI"] for row in repeats])),
        "ARI_sd": float(np.std([row["ARI"] for row in repeats], ddof=1)),
        "NMI_mean": float(np.mean([row["NMI"] for row in repeats])),
        "NMI_sd": float(np.std([row["NMI"] for row in repeats], ddof=1)),
        "ASW_cell": silhouette["ASW_celltype"],
        "ASW_batch": silhouette["ASW_batch"],
        "graph_connectivity": graph_connectivity(
            connectivities, adata.obs["celltype"]
        ),
        "iLISI_proxy": ilisi,
        "iLISI_proxy_scaled": (ilisi - 1) / (n_batches - 1),
        "knn_celltype_purity_k15": exact_knn_purity(
            embeddings,
            adata.obs["celltype"],
            int(config["analysis"]["purity_k"]),
        ),
    }
    metadata = adata.uns[neighbors_key]
    for graph_key in (metadata["connectivities_key"], metadata["distances_key"]):
        del adata.obsp[graph_key]
    del adata.uns[neighbors_key]
    del adata.obsm[key]
    return repeats, metrics
