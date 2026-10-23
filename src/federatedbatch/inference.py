"""Load a deployment checkpoint and embed an unseen AnnData object."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scanpy as sc
import torch

from .data import normalize
from .federated import predict_embeddings, resolve_device


def load_and_embed(
    checkpoint_path: str | Path,
    adata,
    *,
    device: str = "auto",
) -> np.ndarray:
    import concord as ccd

    checkpoint_path = Path(checkpoint_path)
    target_device = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=target_device)
    if checkpoint.get("format_version") != 1:
        raise ValueError(f"unsupported checkpoint format: {checkpoint.get('format_version')}")
    model_config = checkpoint["model"]
    if model_config["domain_key"] is not None:
        raise ValueError(
            "this checkpoint contains training-domain embeddings and is not a "
            "zero-shot deployment checkpoint; use a checkpoint produced by experiments/lobo.py"
        )
    features = checkpoint["features"]
    missing = sorted(set(features) - set(adata.var_names))
    if missing:
        raise ValueError(f"input data is missing {len(missing)} model features: {missing[:5]}")

    resolved = checkpoint["resolved_config"]
    prep = resolved["preprocessing"]
    working = adata.copy()
    normalize(working, prep["normalize_target_sum"], prep["log1p"])
    fed = resolved["federated"]
    model = ccd.Concord(
        adata=working,
        input_feature=features,
        seed=resolved["seed"],
        device=target_device,
        n_epochs=1,
        batch_size=fed["batch_size"],
        lr=fed["learning_rate"],
        clr_temperature=fed["temperature"],
        latent_dim=model_config["latent_dim"],
        encoder_dims=model_config["encoder_dims"],
        dropout_prob=model_config["dropout"],
        domain_key=None,
        preload_dense=True,
        verbose=False,
    )
    model.init_model()
    model.model.load_state_dict(checkpoint["state_dict"])
    return predict_embeddings(model, working)


def embed_h5ad(
    checkpoint_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    *,
    output_key: str = "X_federatedbatch",
    device: str = "auto",
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    adata = sc.read_h5ad(input_path)
    adata.obsm[output_key] = load_and_embed(checkpoint_path, adata, device=device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path, compression="gzip")
    return output_path
