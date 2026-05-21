#!/usr/bin/env python3
"""Run the minimal FederatedBatch reproducibility experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baselines import HarmonyBaseline, PCABaseline
from federatedbatch import FederatedBatch
from utils import (
    create_client_dataloaders,
    evaluate_clustering,
    load_pancreas_data,
    partition_data,
    preprocess_data,
)


def run_pca(adata, n_components: int, random_state: int) -> None:
    n_components = min(n_components, adata.n_obs - 1, adata.n_vars - 1)
    PCABaseline(n_components=n_components, random_state=random_state).fit_transform_adata(
        adata, output_key="X_pca"
    )


def run_harmony(adata) -> None:
    if "X_pca" not in adata.obsm:
        return
    try:
        HarmonyBaseline(batch_key="batch").fit_transform_adata(
            adata, use_rep="X_pca", output_key="X_harmony"
        )
    except Exception as exc:
        print(f"Harmony skipped: {exc}")


def run_federatedbatch(adata, client_data, config, device) -> None:
    dataloaders = create_client_dataloaders(
        client_data,
        batch_size=config["batch_size"],
        shuffle=True,
    )
    client_samples = [client.n_obs for client in client_data]

    model = FederatedBatch(
        n_clients=len(client_data),
        input_dim=adata.n_vars,
        latent_dim=config["latent_dim"],
        encoder_dims=config["encoder_dims"],
        n_rounds=config["n_rounds"],
        local_epochs=config["local_epochs"],
        batch_size=config["batch_size"],
        lr=config["lr"],
        temperature=config["temperature"],
        queue_size=config["queue_size"],
        device=device,
        verbose=True,
    )
    model.initialize_clients(client_data)
    model.fit(dataloaders, client_samples)

    full_loader = create_client_dataloaders([adata], batch_size=config["batch_size"], shuffle=False)[0]
    adata.obsm["X_federatedbatch"] = model.transform(full_loader)


def collect_metrics(adata, methods) -> pd.DataFrame:
    rows = []
    for method, embedding_key in methods.items():
        if embedding_key not in adata.obsm:
            continue
        metrics = evaluate_clustering(
            adata,
            embedding_key=embedding_key,
            celltype_key="celltype",
            batch_key="batch",
        )
        metrics["method"] = method
        rows.append(metrics)
    return pd.DataFrame(rows)[["method", "ARI", "NMI", "ASW_batch", "ASW_celltype"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/pancreas.h5ad")
    parser.add_argument("--output", default="results/main")
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--quick", action="store_true", help="Use a small subsample and fewer rounds")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "n_top_genes": args.n_top_genes,
        "latent_dim": 100,
        "encoder_dims": [128],
        "n_rounds": 2 if args.quick else 10,
        "local_epochs": 1 if args.quick else 2,
        "batch_size": 128,
        "lr": 1e-3,
        "temperature": 0.07,
        "queue_size": 1024 if args.quick else 8192,
        "seed": args.seed,
        "quick": args.quick,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adata = load_pancreas_data(args.data)

    if args.quick and adata.n_obs > 1500:
        sc.pp.subsample(adata, n_obs=1500, random_state=args.seed)

    adata, _ = preprocess_data(adata, n_top_genes=config["n_top_genes"])
    adata = adata[:, adata.var["highly_variable"]].copy()

    client_data = partition_data(adata, batch_key="batch")

    run_pca(adata, n_components=50, random_state=args.seed)
    run_harmony(adata)
    run_federatedbatch(adata, client_data, config, device)

    metrics_df = collect_metrics(
        adata,
        {
            "PCA": "X_pca",
            "Harmony": "X_harmony",
            "FederatedBatch": "X_federatedbatch",
        },
    )
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    adata.write_h5ad(output_dir / "adata_with_embeddings.h5ad")

    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print(metrics_df.to_string(index=False))
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
