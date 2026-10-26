#!/usr/bin/env python3
"""Re-render Figure 1C-D with the legacy reference UMAP visualization protocol.

This reads finalized canonical-seven embeddings and recomputes only the 2-D
visualization. It does not train or evaluate any model.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "figure1_canonical7_final_private"
REPRO = ROOT / "external_artifacts" / "figure1"
OUT = ROOT / "results" / "final_figure1_canonical7" / "reference_style_cd"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(RESULTS / "figure1_umap_coordinates_final.csv")[
        ["cell_id", "batch", "celltype"]
    ]
    embeddings = {
        "PCA": np.load(
            REPRO
            / "results"
            / "figure1_rebuild"
            / "deterministic_baseline_embeddings.npz"
        )["PCA"].astype(np.float32),
        "Harmony": np.load(
            RESULTS / "harmony" / "seed_218" / "final_embedding.npy"
        ).astype(np.float32),
        "CONCORD": np.load(
            RESULTS / "centralized" / "seed_218" / "final_embedding.npy"
        ).astype(np.float32),
        "FederatedBatch": np.load(
            RESULTS
            / "federatedbatch"
            / "training"
            / "seed_218"
            / "final_embedding.npy"
        ).astype(np.float32),
    }

    expected_shape = (len(metadata), 100)
    for method, embedding in embeddings.items():
        if embedding.shape != expected_shape or not np.isfinite(embedding).all():
            raise RuntimeError(
                f"Invalid {method} embedding: {embedding.shape}; expected {expected_shape}"
            )

    output = metadata.copy()
    for method, embedding in embeddings.items():
        # Exact visualization settings used by the reference Figure 1 scripts:
        # sc.pp.neighbors(..., n_neighbors=15) and sc.tl.umap() defaults.
        visual = ad.AnnData(X=np.zeros((len(metadata), 1), dtype=np.float32))
        visual.obsm["X_embedding"] = embedding
        sc.pp.neighbors(
            visual,
            n_neighbors=15,
            use_rep="X_embedding",
            metric="euclidean",
            random_state=0,
        )
        sc.tl.umap(visual, min_dist=0.5, random_state=0)
        coordinates = np.asarray(visual.obsm["X_umap"], dtype=np.float32)
        output[f"UMAP1_{method}"] = coordinates[:, 0]
        output[f"UMAP2_{method}"] = coordinates[:, 1]
        print(f"Completed reference-style UMAP: {method}", flush=True)

    output.to_csv(OUT / "figure1_umap_coordinates_reference_style.csv", index=False)
    (OUT / "UMAP_REFERENCE_STYLE_PROVENANCE.md").write_text(
        "\n".join(
            [
                "# Figure 1 reference-style UMAP provenance",
                "",
                "- Models retrained: no.",
                "- Metrics recomputed: no.",
                "- Embeddings: finalized canonical-seven Figure 1, representative seed 218; deterministic PCA.",
                "- Visualization backend: Scanpy.",
                "- n_neighbors: 15.",
                "- metric: Euclidean.",
                "- min_dist: 0.5 (Scanpy default used by the legacy reference script).",
                "- random_state: 0 (Scanpy default used by the legacy reference script).",
                "- Purpose: reproduce the visual geometry of the specified reference Figure 1 directory.",
                "- The locked min_dist=0.1/random_state=42 coordinate file remains unchanged.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote reference-style coordinates to {OUT}", flush=True)


if __name__ == "__main__":
    main()
