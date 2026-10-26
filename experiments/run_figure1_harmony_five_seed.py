#!/usr/bin/env python3
"""Run Harmony for the five locked Figure 1 seeds on one fixed PCA input."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import anndata as ad
import harmonypy
import numpy as np
import pandas as pd
import scanpy as sc
import umap
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "pancreas.h5ad"
BASELINES = REPO / "results" / "figure1_rebuild" / "deterministic_baseline_embeddings.npz"
OUT = REPO / "results" / "figure1_harmony_5seed"
SEEDS = (3205, 9048, 7487, 5326, 9364)
EVALUATION_SEEDS = (0, 1, 2, 3, 4)
REPRESENTATIVE_SEED = 3205
EXPECTED_DATA_SHA256 = "97e6dfd65553e4d10aa3ef5d904362970a75c677c31d70fabc9234191a09db8c"
SEED_ZERO_REFERENCE_ATOL = 1e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def harmony_embedding(
    pca_embedding: np.ndarray,
    metadata: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    result = harmonypy.run_harmony(
        pca_embedding,
        metadata,
        "batch",
        max_iter_harmony=20,
        verbose=False,
        random_state=seed,
    )
    corrected = np.asarray(result.Z_corr)
    if corrected.shape[0] != pca_embedding.shape[0] and corrected.shape[1] == pca_embedding.shape[0]:
        corrected = corrected.T
    if corrected.shape != pca_embedding.shape:
        raise ValueError(
            f"Unexpected Harmony shape for seed {seed}: {corrected.shape}; "
            f"expected {pca_embedding.shape}"
        )
    return corrected


def clear_graph(adata: ad.AnnData, neighbors_key: str, cluster_key: str) -> None:
    neighbors = adata.uns.get(neighbors_key, {})
    for key in (
        neighbors.get("connectivities_key", f"{neighbors_key}_connectivities"),
        neighbors.get("distances_key", f"{neighbors_key}_distances"),
    ):
        if key in adata.obsp:
            del adata.obsp[key]
    if neighbors_key in adata.uns:
        del adata.uns[neighbors_key]
    if cluster_key in adata.obs:
        del adata.obs[cluster_key]


def evaluate_clustering(
    adata: ad.AnnData,
    embedding: np.ndarray,
    harmony_seed: int,
) -> tuple[list[dict], list[dict]]:
    embedding_key = f"X_harmony_{harmony_seed}"
    adata.obsm[embedding_key] = embedding
    repeat_rows = []
    for evaluation_seed in EVALUATION_SEEDS:
        neighbors_key = f"neighbors_harmony_{harmony_seed}_{evaluation_seed}"
        cluster_key = f"leiden_harmony_{harmony_seed}_{evaluation_seed}"
        sc.pp.neighbors(
            adata,
            use_rep=embedding_key,
            n_neighbors=15,
            random_state=evaluation_seed,
            key_added=neighbors_key,
        )
        sc.tl.leiden(
            adata,
            neighbors_key=neighbors_key,
            key_added=cluster_key,
            resolution=1.0,
            random_state=evaluation_seed,
        )
        repeat_rows.extend(
            [
                {
                    "method": "Harmony",
                    "training_seed": harmony_seed,
                    "evaluation_seed": evaluation_seed,
                    "metric": "ARI",
                    "value": float(
                        adjusted_rand_score(adata.obs["celltype"], adata.obs[cluster_key])
                    ),
                },
                {
                    "method": "Harmony",
                    "training_seed": harmony_seed,
                    "evaluation_seed": evaluation_seed,
                    "metric": "NMI",
                    "value": float(
                        normalized_mutual_info_score(
                            adata.obs["celltype"], adata.obs[cluster_key]
                        )
                    ),
                },
            ]
        )
        clear_graph(adata, neighbors_key, cluster_key)
    del adata.obsm[embedding_key]
    repeat = pd.DataFrame(repeat_rows)
    summary = (
        repeat.groupby(["method", "training_seed", "metric"], as_index=False)["value"]
        .mean()
        .to_dict("records")
    )
    return repeat_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if sha256(DATA) != EXPECTED_DATA_SHA256:
        raise RuntimeError("Pancreas dataset hash does not match the locked Figure 1 input")

    adata = ad.read_h5ad(DATA)
    adata.obs["batch"] = adata.obs["tech"].copy().astype("category")
    adata.obs["celltype"] = adata.obs["celltype"].astype("category")
    with np.load(BASELINES) as cache:
        # Harmony is numerically sensitive to dtype. Preserve the archived
        # float32 PCA array exactly rather than silently promoting it.
        pca_embedding = cache["PCA"].copy()
        seed_zero_reference = cache["Harmony"].copy()
    if pca_embedding.shape != (adata.n_obs, 100):
        raise ValueError(f"Unexpected fixed PCA shape: {pca_embedding.shape}")

    print("Validating the fixed PCA input against the archived seed-0 Harmony embedding", flush=True)
    seed_zero = harmony_embedding(pca_embedding, adata.obs, 0)
    seed_zero_max_abs_difference = float(np.max(np.abs(seed_zero - seed_zero_reference)))
    if not np.allclose(
        seed_zero,
        seed_zero_reference,
        rtol=0,
        atol=SEED_ZERO_REFERENCE_ATOL,
    ):
        raise RuntimeError(
            "Current Harmony execution does not reproduce the archived seed-0 embedding; "
            f"maximum absolute difference={seed_zero_max_abs_difference}"
        )

    repeat_rows: list[dict] = []
    summary_rows: list[dict] = []
    input_records = []
    for index, seed in enumerate(SEEDS, start=1):
        seed_dir = OUT / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        embedding_path = seed_dir / "final_embedding.npy"
        started = time.perf_counter()
        if embedding_path.exists() and not args.force:
            embedding = np.load(embedding_path).astype(np.float64, copy=False)
            action = "reused"
        else:
            embedding = harmony_embedding(pca_embedding, adata.obs, seed)
            np.save(embedding_path, embedding)
            action = "computed"
        if embedding.shape != pca_embedding.shape or not np.isfinite(embedding).all():
            raise ValueError(f"Invalid Harmony embedding for seed {seed}")
        repeats, summary = evaluate_clustering(adata, embedding, seed)
        repeat_rows.extend(repeats)
        summary_rows.extend(summary)
        input_records.append(
            {
                "seed": seed,
                "embedding": str(embedding_path.resolve()),
                "embedding_sha256": sha256(embedding_path),
                "elapsed_seconds": time.perf_counter() - started,
                "action": action,
            }
        )
        print(f"[{index}/5] seed={seed}: {action}, clustering metrics complete", flush=True)

    repeats = pd.DataFrame(repeat_rows)
    summaries = pd.DataFrame(summary_rows)
    repeats.to_csv(OUT / "leiden_repeats.csv", index=False)
    summaries.to_csv(OUT / "clustering_metrics.csv", index=False)

    representative = np.load(
        OUT / f"seed_{REPRESENTATIVE_SEED}" / "final_embedding.npy"
    )
    representative_umap = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
    ).fit_transform(representative)
    np.save(OUT / "representative_seed_3205_umap.npy", representative_umap)

    provenance = {
        "status": "PASS",
        "method": "Harmony",
        "random_seeds": list(SEEDS),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "representative_seed": REPRESENTATIVE_SEED,
        "fixed_pca_embedding": str(BASELINES.resolve()),
        "fixed_pca_array_sha256": hashlib.sha256(pca_embedding.tobytes()).hexdigest(),
        "dataset": str(DATA.resolve()),
        "dataset_sha256": sha256(DATA),
        "batch_key": "batch copied from tech",
        "parameters": {
            "max_iter_harmony": 20,
            "random_state": "one of random_seeds",
            "umap_n_neighbors": 15,
            "umap_min_dist": 0.1,
            "umap_metric": "euclidean",
            "umap_random_state": 42,
        },
        "seed_zero_reference_max_abs_difference": seed_zero_max_abs_difference,
        "seed_zero_reference_absolute_tolerance": SEED_ZERO_REFERENCE_ATOL,
        "seed_zero_reference_matches": True,
        "versions": {
            "python": platform.python_version(),
            "harmonypy": importlib.metadata.version("harmonypy"),
            "scanpy": importlib.metadata.version("scanpy"),
            "scikit-learn": importlib.metadata.version("scikit-learn"),
            "umap-learn": importlib.metadata.version("umap-learn"),
            "numpy": np.__version__,
        },
        "outputs": input_records,
    }
    write_json(OUT / "provenance.json", provenance)
    print(summaries.to_string(index=False), flush=True)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
