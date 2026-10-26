"""Dataset loading, annotation harmonization, and preprocessing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import scanpy as sc
import numpy as np
from scipy import sparse

from .config import REPO_ROOT, resolve_repo_path
from .provenance import file_sha256

MANIFEST_PATH = REPO_ROOT / "data" / "datasets.json"


def load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def dataset_metadata(name: str) -> dict[str, Any]:
    datasets = load_manifest()["datasets"]
    if name not in datasets:
        raise KeyError(f"unknown dataset {name!r}; choose from {', '.join(datasets)}")
    return datasets[name]


def dataset_path(name: str, data_dir: str | Path) -> Path:
    return resolve_repo_path(data_dir) / dataset_metadata(name)["filename"]


def verify_dataset_file(name: str, path: Path, verify_hash: bool = True) -> None:
    metadata = dataset_metadata(name)
    if not path.exists():
        raise FileNotFoundError(
            f"dataset not found: {path}. Run scripts/prepare/prepare_data.py before the experiment."
        )
    actual_size = path.stat().st_size
    if actual_size != metadata["size_bytes"]:
        raise ValueError(
            f"{name} size mismatch: expected {metadata['size_bytes']}, got {actual_size}"
        )
    if verify_hash:
        actual_hash = file_sha256(path)
        if actual_hash != metadata["sha256"]:
            raise ValueError(
                f"{name} SHA-256 mismatch: expected {metadata['sha256']}, got {actual_hash}"
            )


def _copy_annotation(adata, destination: str, candidates: list[str]) -> None:
    if destination in adata.obs:
        return
    for candidate in candidates:
        if candidate in adata.obs:
            adata.obs[destination] = adata.obs[candidate].copy()
            return
    raise KeyError(
        f"could not create obs[{destination!r}]; available columns: {list(adata.obs.columns)}"
    )


def harmonize_annotations(adata, metadata: dict[str, Any]):
    batch_candidates = [metadata["batch_key"], *metadata.get("batch_fallbacks", [])]
    cell_candidates = [
        metadata["cell_type_key"],
        *metadata.get("cell_type_fallbacks", []),
    ]
    _copy_annotation(adata, "batch", batch_candidates)
    _copy_annotation(adata, "celltype", cell_candidates)
    adata.obs["batch"] = adata.obs["batch"].astype("category")
    adata.obs["celltype"] = adata.obs["celltype"].astype("category")
    return adata


def restore_input_matrix(adata, metadata: dict[str, Any]):
    """Restore the count-like matrix before repository-controlled preprocessing."""

    layer = metadata.get("input_layer")
    if not layer:
        return adata
    if layer not in adata.layers:
        raise KeyError(f"required input layer {layer!r} is absent")
    matrix = adata.layers[layer]
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"input layer {layer!r} contains invalid values")
    adata.X = matrix.copy()
    return adata


def read_dataset(
    name: str,
    data_dir: str | Path,
    *,
    verify_hash: bool = True,
):
    path = dataset_path(name, data_dir)
    verify_dataset_file(name, path, verify_hash=verify_hash)
    adata = sc.read_h5ad(path)
    harmonize_annotations(adata, dataset_metadata(name))
    restore_input_matrix(adata, dataset_metadata(name))
    expected = dataset_metadata(name)
    observed = {
        "n_cells": adata.n_obs,
        "n_batches": adata.obs["batch"].nunique(),
        "n_cell_types": adata.obs["celltype"].nunique(),
    }
    errors = [
        f"{key}: expected {expected[key]}, got {value}"
        for key, value in observed.items()
        if expected[key] != value
    ]
    if errors:
        raise ValueError(f"{name} metadata mismatch ({'; '.join(errors)})")
    return adata, path


def balanced_subsample(adata, max_cells: int, seed: int):
    """Subsample approximately equal cell counts per batch."""
    if max_cells <= 0 or adata.n_obs <= max_cells:
        return adata.copy()
    rng = np.random.default_rng(seed)
    batches = list(adata.obs["batch"].unique())
    base, remainder = divmod(max_cells, len(batches))
    selected = []
    for index, batch in enumerate(batches):
        batch_indices = np.flatnonzero(np.asarray(adata.obs["batch"] == batch))
        count = min(len(batch_indices), base + int(index < remainder))
        selected.extend(rng.choice(batch_indices, size=count, replace=False).tolist())
    if len(selected) < max_cells:
        remaining = np.setdiff1d(np.arange(adata.n_obs), np.asarray(selected))
        count = min(max_cells - len(selected), len(remaining))
        selected.extend(rng.choice(remaining, size=count, replace=False).tolist())
    return adata[np.sort(selected)].copy()


def select_features(
    adata,
    n_hvg: int,
    flavor: str = "seurat_v3",
    *,
    batch_key: str | None = None,
) -> list[str]:
    sc.pp.highly_variable_genes(
        adata,
        flavor=flavor,
        n_top_genes=n_hvg,
        batch_key=batch_key,
        subset=False,
    )
    features = adata.var_names[adata.var["highly_variable"]].tolist()
    if len(features) != n_hvg:
        raise ValueError(f"expected {n_hvg} highly variable genes, selected {len(features)}")
    return features


def normalize(adata, target_sum: float | None, log1p: bool = True) -> None:
    if target_sum is None:
        sc.pp.normalize_total(adata)
    else:
        sc.pp.normalize_total(adata, target_sum=target_sum)
    if log1p:
        sc.pp.log1p(adata)


def preprocess_dataset(adata, config: dict[str, Any], *, hvg_batch_key: str | None = None):
    prep = config["preprocessing"]
    features = select_features(
        adata,
        n_hvg=prep["n_hvg"],
        flavor=prep["hvg_flavor"],
        batch_key=hvg_batch_key,
    )
    normalize(adata, prep["normalize_target_sum"], prep["log1p"])
    return features
