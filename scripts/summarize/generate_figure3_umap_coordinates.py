#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import umap


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DATASETS = ("pancreas", "immune", "lung")
METHODS = ("Centralized CONCORD", "FederatedBatch", "FedscGen", "PCA")
SEED = 218
UMAP_PARAMETERS = {
    "n_neighbors": 15,
    "min_dist": 0.5,
    "metric": "euclidean",
    "random_state": 0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def embedding_paths(root: Path, dataset: str) -> dict[str, tuple[Path, str | None]]:
    native = root / "training" / "native" / f"seed_{SEED}" / dataset
    return {
        "Centralized CONCORD": (native / "centralized_concord" / "final_embedding.npy", None),
        "FederatedBatch": (native / "federatedbatch" / "final_embedding.npy", None),
        "PCA": (native / "pca" / "final_embedding.npy", None),
        "FedscGen": (
            root
            / "training"
            / "fedscgen"
            / f"seed_{SEED}"
            / dataset
            / "adata_with_embeddings.h5ad",
            "X_fedscgen",
        ),
    }


def load_embedding(path: Path, key: str | None, expected_ids: pd.Index) -> np.ndarray:
    if key is None:
        values = np.load(path).astype(np.float32, copy=False)
    else:
        source = ad.read_h5ad(path)
        if not source.obs_names.equals(expected_ids):
            raise ValueError(f"cell order mismatch in {path}")
        if key not in source.obsm:
            raise KeyError(f"missing obsm[{key!r}] in {path}")
        values = np.asarray(source.obsm[key], dtype=np.float32)
    if values.shape[0] != len(expected_ids) or values.ndim != 2:
        raise ValueError(f"invalid embedding shape {values.shape}: {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite embedding values: {path}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate Figure 3 representative-seed UMAP tables from trained embeddings.')
    parser.add_argument(
        "--input", type=Path, required=True, help="fresh Figure 3 result root"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--no-verify-hash", action="store_true")
    args = parser.parse_args()

    from federatedbatch.data import read_dataset

    source_root = args.input.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "representative_training_seed": SEED,
        "umap": UMAP_PARAMETERS,
        "fit_mode": "independent UMAP fit per dataset and method",
        "same_coordinates_for_celltype_and_batch": True,
        "datasets": {},
    }
    for dataset in DATASETS:
        data, data_path = read_dataset(
            dataset, args.data_dir.resolve(), verify_hash=not args.no_verify_hash
        )
        paths = embedding_paths(source_root, dataset)
        missing = [str(path) for path, _ in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing Figure 3 embedding(s): " + ", ".join(missing))

        frame = pd.DataFrame(
            {
                "cell_id": data.obs_names.astype(str),
                "celltype": data.obs["celltype"].astype(str).to_numpy(),
                "batch": data.obs["batch"].astype(str).to_numpy(),
            }
        )
        method_records = {}
        for method in METHODS:
            path, key = paths[method]
            values = load_embedding(path, key, data.obs_names)
            coordinates = umap.UMAP(**UMAP_PARAMETERS).fit_transform(values)
            frame[f"UMAP1_{method}"] = coordinates[:, 0]
            frame[f"UMAP2_{method}"] = coordinates[:, 1]
            method_records[method] = {
                "path": str(path),
                "sha256": sha256(path),
                "shape": list(values.shape),
                "obsm_key": key,
            }
            print(f"Figure 3 UMAP: dataset={dataset} method={method}", flush=True)
        destination = output / f"{dataset}_seed{SEED}_umap_coordinates.csv"
        frame.to_csv(destination, index=False)
        provenance["datasets"][dataset] = {
            "data_path": str(data_path),
            "data_sha256": sha256(data_path),
            "coordinate_table": str(destination),
            "n_cells": int(data.n_obs),
            "methods": method_records,
        }
    (output / "figure3_umap_generation_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
