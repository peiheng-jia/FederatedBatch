#!/usr/bin/env python3
"""Small CPU example using the existing AnnData training/checkpoint interfaces."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
import yaml

from federatedbatch.config import load_config
from federatedbatch.data import normalize
from federatedbatch.federated import save_checkpoint, train_federated
from federatedbatch.inference import load_and_embed
from federatedbatch.partition import partition_batches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="optional small count-like .h5ad; requires obs['batch']")
    parser.add_argument("--output", type=Path, default=Path("results/runs/minimal"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("choose an empty --output directory to preserve existing files")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)

    if args.input is None:
        # Two technical batches, 64 cells and 24 genes; no external download.
        rng = np.random.default_rng(218)
        counts = rng.poisson(3.0, size=(64, 24)).astype(np.float32)
        source = ad.AnnData(
            counts,
            obs=pd.DataFrame(
                {"batch": pd.Categorical(np.repeat(["batch_a", "batch_b"], 32))},
                index=[f"cell_{i}" for i in range(64)],
            ),
            var=pd.DataFrame(index=[f"gene_{i}" for i in range(24)]),
        )
        input_path = args.output / "input_counts.h5ad"
        source.write_h5ad(input_path)
    else:
        input_path = args.input

    # X must be unnormalized count-like input, or supply layers['counts'].
    source = ad.read_h5ad(input_path)
    if "counts" in source.layers:
        source.X = source.layers["counts"].copy()
    if "batch" not in source.obs or source.obs["batch"].isna().any():
        raise ValueError("input requires nonmissing obs['batch'] annotations")
    if not source.obs_names.is_unique or not source.var_names.is_unique:
        raise ValueError("cell and gene names must be unique")
    source.obs["batch"] = source.obs["batch"].astype("category")
    if source.n_vars < 2 or source.obs.groupby("batch", observed=True).size().min() < 8:
        raise ValueError("use at least two genes and eight cells per batch")

    config = load_config("configs/figure2_training.yaml")
    config.update(protocol="minimal_cpu_example", dataset="example", device="cpu",
                  data_dir=str(input_path.parent), output_dir=str(args.output))
    # Do not retain canonical-study metadata in the example's saved config.
    config.pop("confirmation", None)
    config.pop("remote_interaction", None)
    config["preprocessing"]["n_hvg"] = source.n_vars
    fed = config["federated"]
    fed.update(n_clients=2, n_rounds=2, local_epochs=1, batch_size=8,
               latent_dim=8, encoder_dims=[16], dropout=0.0)
    # Reduced demonstration settings, not the canonical Figure 1-5 protocol.
    fed["queue"].update(
        enabled=True, mode="round_once_joint_cellclip_noise", size=16,
        samples_per_client=2, warmup_rounds=1, active_rounds=[2],
        ab_max_entries=2, joint_clipping_threshold=25.0,
        noise_multiplier=2.0, positive_a_floor=1e-6, _joint_noise_seed=10447,
        _scheduled_remote_indices={"1": {"0": {"0": [0, 1], "1": [0, 1]}}},
    )
    groups = partition_batches(source, 2, seed=config["seed"])
    # Keep all genes for this small example; paper runs use configured HVGs.
    features = source.var_names.tolist()
    working = source.copy()
    normalize(working, target_sum=10_000, log1p=True)
    result = train_federated(
        working, features, groups, config, args.output / "training", domain_key=None,
    )
    if not result.joint_clipping_diagnostics or not result.joint_noise_diagnostics:
        raise RuntimeError("the example did not execute the private A/B interaction")
    if result.embeddings.shape != (source.n_obs, 8) or not np.isfinite(result.embeddings).all():
        raise RuntimeError("invalid training embedding")

    checkpoint = args.output / "model.pt"
    save_checkpoint(checkpoint, result, features, config, domain_key=None)
    # The existing inference function normalizes count-like input internally.
    reloaded = load_and_embed(checkpoint, source, device="cpu")
    np.testing.assert_allclose(reloaded, result.embeddings, rtol=1e-5, atol=1e-6)
    source.obsm["X_federatedbatch"] = result.embeddings
    source.write_h5ad(args.output / "embedding.h5ad", compression="gzip")
    (args.output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"PASS: {source.n_obs} cells -> {result.embeddings.shape[1]} dimensions; checkpoint reload matched.")
    print(f"Private A/B release diagnostics: {len(result.joint_noise_diagnostics)} records.")
    print(f"Outputs: {args.output / 'embedding.h5ad'}, {checkpoint}, {args.output / 'config.yaml'}")


if __name__ == "__main__":
    main()
