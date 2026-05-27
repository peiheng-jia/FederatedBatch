#!/usr/bin/env python3
"""Download a small public pancreas dataset for FederatedBatch smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import scanpy as sc


def standardize_obs(adata):
    if "tech" in adata.obs.columns and "batch" not in adata.obs.columns:
        adata.obs["batch"] = adata.obs["tech"]

    if "celltype" not in adata.obs.columns:
        for key in ("cell_type", "celltype1", "louvain", "leiden"):
            if key in adata.obs.columns:
                adata.obs["celltype"] = adata.obs[key]
                break

    missing = [key for key in ("batch", "celltype") if key not in adata.obs.columns]
    if missing:
        raise ValueError(f"Downloaded dataset is missing required obs columns: {missing}")

    return adata


def download_pancreas(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pancreas.h5ad"

    if output_path.exists():
        print(f"Data already exists: {output_path}")
        return output_path

    print("Downloading smoke-test pancreas dataset with scanpy.datasets.pancreas()")
    adata = sc.datasets.pancreas()
    adata = standardize_obs(adata)
    adata.write_h5ad(output_path)

    print(f"Saved: {output_path}")
    print(f"Cells: {adata.n_obs}")
    print(f"Genes: {adata.n_vars}")
    print(f"Batches: {adata.obs['batch'].nunique()}")
    print(f"Cell types: {adata.obs['celltype'].nunique()}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="pancreas", choices=["pancreas"])
    parser.add_argument("--output", default="data", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    if args.dataset == "pancreas":
        download_pancreas(output_dir)


if __name__ == "__main__":
    main()
